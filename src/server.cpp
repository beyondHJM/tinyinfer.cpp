// tinyinfer-server: OpenAI-compatible HTTP server for tinyinfer.cpp.
// Endpoints mirror llama-server's OpenAI-compatible surface:
//   GET  /health               -> {"status":"ok"}
//   GET  /v1/models            -> model list
//   GET  /props                -> server properties
//   POST /v1/completions       -> text completion (stream + non-stream)
//   POST /v1/chat/completions  -> chat completion (stream + non-stream)
// Zero third-party dependencies: POSIX sockets + a bundled minimal JSON parser.

#include "gguf_reader.h"
#include "http.h"
#include "inference.h"
#include "json.h"
#include "model.h"
#include "tokenizer.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Globals (read-only model shared by all requests; one Inference instance
// guarded by a mutex keeps requests serialized like a single-slot server).
// ---------------------------------------------------------------------------
Qwen3Model g_model;
Tokenizer g_tok;
Inference g_infer;
std::mutex g_infer_mutex;

std::string g_model_path;
std::string g_model_name = "tinyinfer";
int g_max_seq = 2048;
bool g_chat = false;  // not used by the server; kept for symmetry

// ---------------------------------------------------------------------------
// Sampling (mirrors main.cpp).
// ---------------------------------------------------------------------------
struct SampleParams {
    float temp = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    uint64_t seed = 42;
};

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------
struct GenResult {
    std::string text;
    int n_prompt = 0;
    int n_gen = 0;
    std::string finish_reason = "length";
    double prompt_ms = 0.0;
    double predict_ms = 0.0;
};

// Sample from the GPU top-k candidates (at most SAM_CAND_PER_SEQ items).
// Temperature scaling uses p^(1/T) which is proportional to softmax(logit/T),
// so the host only ever sees the reduced candidate set (llama.cpp-style).
int sample_from_candidates(const std::vector<float> & probs,
                           const std::vector<int> & idxs,
                           const SampleParams & sp) {
    std::vector<std::pair<float, int>> cand;
    cand.reserve(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        if (idxs[i] >= 0) cand.push_back({probs[i], idxs[i]});
    }
    std::sort(cand.begin(), cand.end(),
              [](const auto & a, const auto & b) { return a.first > b.first; });
    if (sp.temp <= 0.0f) return cand.empty() ? 0 : cand[0].second;

    const float invT = 1.0f / sp.temp;
    for (auto & c : cand) {
        c.first = expf(logf(fmaxf(c.first, 1e-30f)) * invT);
    }
    const int K = std::max(1, std::min((int) cand.size(),
                                       sp.top_k > 0 ? sp.top_k : (int) cand.size()));
    if ((int) cand.size() > K) cand.resize(K);

    float sum = 0.0f;
    for (auto & c : cand) sum += c.first;
    if (sum <= 0.0f) return cand.empty() ? 0 : cand[0].second;

    if (sp.top_p < 1.0f && !cand.empty()) {
        double acc = 0.0;
        size_t cut = cand.size();
        for (size_t i = 0; i < cand.size(); ++i) {
            acc += cand[i].first / sum;
            if (acc >= sp.top_p) { cut = i + 1; break; }
        }
        cand.resize(cut);
        sum = 0.0f;
        for (auto & c : cand) sum += c.first;
        if (sum <= 0.0f) return cand.empty() ? 0 : cand[0].second;
    }

    static thread_local std::mt19937 rng((uint32_t) sp.seed);
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);
    double acc2 = 0.0;
    for (auto & c : cand) {
        acc2 += c.first;
        if (r <= (float) acc2) return c.second;
    }
    return cand.empty() ? 0 : cand.back().second;
}

// on_token(id) is invoked for every generated token (used by streaming).
// Returns false to stop early (e.g. client disconnect).
template <typename Fn>
GenResult generate(const std::vector<int32_t> & prompt, int n_predict,
                   const SampleParams & sp, const std::vector<std::string> & stops,
                   int max_seq, Fn && on_token) {
    GenResult res;
    res.n_prompt = (int) prompt.size();

    // prefill
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int32_t> batch_tokens = prompt;
    std::vector<int> batch_pos(batch_tokens.size()), batch_seq(batch_tokens.size(), 0);
    for (size_t i = 0; i < batch_tokens.size(); ++i) batch_pos[i] = (int) i;
    std::vector<float> logits;
    std::vector<float> cand_probs;
    std::vector<int> cand_idx;
    g_infer.forward(batch_tokens, batch_pos, batch_seq, 1, logits, &cand_probs, &cand_idx);
    auto t1 = std::chrono::steady_clock::now();
    res.prompt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::string out;
    int n_gen = 0;

    auto emit = [&](int id) -> bool {
        std::string piece = g_tok.token_to_piece(id);
        out += piece;
        // stop-string check (piece-granular; matches llama.cpp behavior closely)
        for (const auto & st : stops) {
            if (!st.empty() && out.size() >= st.size() &&
                out.compare(out.size() - st.size(), st.size(), st) == 0) {
                out.resize(out.size() - st.size());
                res.finish_reason = "stop";
                return false;
            }
        }
        if (g_tok.is_eog(id)) {
            res.finish_reason = "stop";
            return false;
        }
        return true;
    };

    while (n_gen < n_predict) {
        int id;
        if (n_gen == 0) {
            id = sample_from_candidates(cand_probs, cand_idx, sp);
        } else {
            batch_tokens = {(int32_t) id};
            batch_pos = {(int) res.n_prompt + n_gen - 1};
            batch_seq = {0};
            if (!g_infer.forward(batch_tokens, batch_pos, batch_seq, 1, logits,
                                 &cand_probs, &cand_idx)) break;
            id = sample_from_candidates(cand_probs, cand_idx, sp);
        }
        if (!on_token(id)) { res.finish_reason = "abort"; break; }
        ++n_gen;
        if (!emit(id)) break;
    }
    res.text = out;
    res.n_gen = n_gen;
    auto t3 = std::chrono::steady_clock::now();
    res.predict_ms = std::chrono::duration<double, std::milli>(t3 - t1).count();
    if (res.finish_reason == "length" && n_gen < n_predict) res.finish_reason = "stop";
    return res;
}

std::string strip_think_blocks(const std::string & in) {
    std::string s = in;
    size_t a = s.find("<think>");
    while (a != std::string::npos) {
        size_t b = s.find("</think>", a);
        if (b == std::string::npos) {
            s.erase(a);
            break;
        }
        s.erase(a, b + 8 - a);
        a = s.find("<think>");
    }
    return s;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
long long now_epoch() { return (long long) time(nullptr); }

std::string make_id(const char * prefix) {
    static std::atomic<long long> counter{0};
    char buf[64];
    snprintf(buf, sizeof buf, "%s-%lld-%lld", prefix, now_epoch(),
             counter.fetch_add(1));
    return buf;
}

int clamp_int(double v, int lo, int hi) {
    int x = (int) v;
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return x;
}

std::vector<std::string> read_stops(const tj::Json * stop) {
    std::vector<std::string> out;
    if (!stop) return out;
    if (stop->is_string()) {
        out.push_back(stop->str);
    } else if (stop->is_array()) {
        for (auto & s : stop->arr) {
            if (s.is_string()) out.push_back(s.str);
        }
    }
    return out;
}

void send_json_error(StreamWriter & w, int status, const std::string & msg) {
    tj::Json err = tj::Json::make_object();
    err.set("error", tj::Json::make_object());
    err.get("error")->set("message", tj::Json::make_string(msg));
    err.get("error")->set("type", tj::Json::make_string("invalid_request_error"));
    err.get("error")->set("code", tj::Json::make_number(status));
    std::map<std::string, std::string> hdrs{{"Content-Type", "application/json"}};
    w.write_head(status, "Bad Request", hdrs, false);
    w.write(err.dump() + "\n");
    w.finish();
}

tj::Json make_usage(int prompt_tokens, int completion_tokens) {
    tj::Json u = tj::Json::make_object();
    u.set("prompt_tokens", tj::Json::make_number(prompt_tokens));
    u.set("completion_tokens", tj::Json::make_number(completion_tokens));
    u.set("total_tokens", tj::Json::make_number(prompt_tokens + completion_tokens));
    tj::Json det = tj::Json::make_object();
    det.set("cached_tokens", tj::Json::make_number(0));
    u.set("prompt_tokens_details", std::move(det));
    return u;
}

tj::Json make_timings(int prompt_n, double prompt_ms, int predicted_n, double predicted_ms) {
    tj::Json t = tj::Json::make_object();
    t.set("prompt_n", tj::Json::make_number(prompt_n));
    t.set("prompt_ms", tj::Json::make_number(prompt_ms));
    t.set("prompt_per_second", tj::Json::make_number(
        prompt_ms > 0.0 ? prompt_n / (prompt_ms / 1000.0) : 0.0));
    t.set("predicted_n", tj::Json::make_number(predicted_n));
    t.set("predicted_ms", tj::Json::make_number(predicted_ms));
    t.set("predicted_per_second", tj::Json::make_number(
        predicted_ms > 0.0 ? predicted_n / (predicted_ms / 1000.0) : 0.0));
    return t;
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------
void handle_health(const HttpRequest &, StreamWriter & w) {
    std::map<std::string, std::string> hdrs{{"Content-Type", "application/json"}};
    w.write_head(200, "OK", hdrs, false);
    w.write("{\"status\":\"ok\"}\n");
    w.finish();
}

void handle_models(const HttpRequest &, StreamWriter & w) {
    tj::Json m = tj::Json::make_object();
    m.set("id", tj::Json::make_string(g_model_name));
    m.set("object", tj::Json::make_string("model"));
    m.set("created", tj::Json::make_number(0));
    m.set("owned_by", tj::Json::make_string("tinyinfer"));
    tj::Json list = tj::Json::make_array();
    list.push_back(m);
    tj::Json resp = tj::Json::make_object();
    resp.set("object", tj::Json::make_string("list"));
    resp.set("data", std::move(list));
    std::map<std::string, std::string> hdrs{{"Content-Type", "application/json"}};
    w.write_head(200, "OK", hdrs, false);
    w.write(resp.dump() + "\n");
    w.finish();
}

void handle_props(const HttpRequest &, StreamWriter & w) {
    tj::Json p = tj::Json::make_object();
    p.set("total_slots", tj::Json::make_number(1));
    p.set("model_path", tj::Json::make_string(g_model_path));
    p.set("chat_template", tj::Json::make_string("qwen3"));
    p.set("version", tj::Json::make_string("tinyinfer-0.1"));
    std::map<std::string, std::string> hdrs{{"Content-Type", "application/json"}};
    w.write_head(200, "OK", hdrs, false);
    w.write(p.dump() + "\n");
    w.finish();
}

struct CompletionParams {
    std::vector<int32_t> prompt_tokens;
    std::string prompt_text;
    int n_predict = 128;
    SampleParams sp;
    std::vector<std::string> stops;
    bool echo = false;
    bool stream = false;
    int n = 1;
    bool is_chat = false;
};

// Parse common sampling parameters from a JSON request body.
bool parse_params(const tj::Json & body, CompletionParams & cp, std::string & err) {
    if (!body.is_object()) { err = "request body must be a JSON object"; return false; }

    const tj::Json * mp = body.get("max_tokens");
    if (mp && mp->is_number()) cp.n_predict = clamp_int(mp->num, 1, g_max_seq - 1);
    const tj::Json * np = body.get("n_predict");
    if (np && np->is_number()) cp.n_predict = clamp_int(np->num, 1, g_max_seq - 1);

    const tj::Json * t = body.get("temperature");
    if (t && t->is_number()) cp.sp.temp = (float) t->num;
    const tj::Json * tk = body.get("top_k");
    if (tk && tk->is_number()) cp.sp.top_k = clamp_int(tk->num, 0, 100000);
    const tj::Json * tp = body.get("top_p");
    if (tp && tp->is_number()) cp.sp.top_p = (float) tp->num;
    const tj::Json * sd = body.get("seed");
    if (sd && sd->is_number()) cp.sp.seed = (uint64_t) sd->num;

    cp.stops = read_stops(body.get("stop"));
    const tj::Json * ec = body.get("echo");
    if (ec && ec->is_bool()) cp.echo = ec->b;
    const tj::Json * st = body.get("stream");
    if (st && st->is_bool()) cp.stream = st->b;
    const tj::Json * nn = body.get("n");
    if (nn && nn->is_number()) cp.n = clamp_int(nn->num, 1, 8);
    if (cp.stream) cp.n = 1;  // streaming supports a single sequence for now
    return true;
}

// Shared implementation for /v1/completions and /v1/chat/completions.
void run_generation(const HttpRequest & req, StreamWriter & w, const CompletionParams & cp,
                    bool is_chat) {
    const int n_vocab = g_infer.n_vocab();
    const std::string obj_type = is_chat ? "chat.completion" : "text_completion";
    const std::string id = make_id(is_chat ? "chatcmpl" : "cmpl");
    const long long created = now_epoch();

    auto build_base = [&](tj::Json & base) {
        base.set("id", tj::Json::make_string(id));
        base.set("object", tj::Json::make_string(obj_type));
        base.set("created", tj::Json::make_number(created));
        base.set("model", tj::Json::make_string(g_model_name));
        base.set("system_fingerprint", tj::Json::make_string("fp_tinyinfer"));
    };

    if (cp.stream) {
        // SSE streaming.
        tj::Json base = tj::Json::make_object();
        build_base(base);
        tj::Json first = base;
        tj::Json first_choice = tj::Json::make_object();
        first_choice.set("index", tj::Json::make_number(0));
        if (is_chat) {
            tj::Json delta = tj::Json::make_object();
            delta.set("role", tj::Json::make_string("assistant"));
            delta.set("content", tj::Json::make_string(""));
            first_choice.set("delta", std::move(delta));
            first.set("object", tj::Json::make_string("chat.completion.chunk"));
        } else {
            first_choice.set("text", tj::Json::make_string(""));
        }
        first_choice.set("finish_reason", tj::Json::make_null());
        first.set("choices", tj::Json::make_array());
        first.get("choices")->push_back(std::move(first_choice));
        first.set("usage", tj::Json::make_null());

        auto stream_chunk = [&](const std::string & text_delta, bool finish, bool stop_reason) {
            tj::Json chunk = tj::Json::make_object();
            chunk.set("id", tj::Json::make_string(id));
            chunk.set("object", tj::Json::make_string(
                is_chat ? "chat.completion.chunk" : "text_completion"));
            chunk.set("created", tj::Json::make_number(created));
            chunk.set("model", tj::Json::make_string(g_model_name));
            tj::Json c = tj::Json::make_object();
            c.set("index", tj::Json::make_number(0));
            if (is_chat) {
                tj::Json d = tj::Json::make_object();
                if (!text_delta.empty()) d.set("content", tj::Json::make_string(text_delta));
                c.set("delta", std::move(d));
            } else {
                c.set("text", tj::Json::make_string(text_delta));
            }
            c.set("finish_reason", finish
                      ? tj::Json::make_string(stop_reason ? "stop" : "length")
                      : tj::Json::make_null());
            chunk.set("choices", tj::Json::make_array());
            chunk.get("choices")->push_back(std::move(c));
            chunk.set("usage", finish ? make_usage(0, 0) : tj::Json::make_null());
            return w.write("data: " + chunk.dump() + "\n\n");
        };

        std::map<std::string, std::string> hdrs{
            {"Content-Type", "text/event-stream"},
            {"Cache-Control", "no-cache"},
            {"X-Accel-Buffering", "no"},
        };
        w.write_head(200, "OK", hdrs, true);
        w.write("data: " + first.dump() + "\n\n");

        std::lock_guard<std::mutex> lock(g_infer_mutex);
        g_infer.reset_kv();
        bool aborted = false;
        GenResult res = generate(
            cp.prompt_tokens, cp.n_predict, cp.sp, cp.stops, g_max_seq,
            [&](int tok) -> bool {
                if (!aborted) {
                    std::string piece = g_tok.token_to_piece(tok);
                    if (!stream_chunk(piece, false, false)) {
                        aborted = true;
                        return false;
                    }
                }
                return true;
            });
        if (!aborted) {
            tj::Json fin = tj::Json::make_object();
            fin.set("id", tj::Json::make_string(id));
            fin.set("object", tj::Json::make_string(
                is_chat ? "chat.completion.chunk" : "text_completion"));
            fin.set("created", tj::Json::make_number(created));
            fin.set("model", tj::Json::make_string(g_model_name));
            tj::Json c = tj::Json::make_object();
            c.set("index", tj::Json::make_number(0));
            if (is_chat) c.set("delta", tj::Json::make_object());
            else c.set("text", tj::Json::make_string(""));
            c.set("finish_reason", tj::Json::make_string(
                res.finish_reason == "stop" ? "stop" : "length"));
            fin.set("choices", tj::Json::make_array());
            fin.get("choices")->push_back(std::move(c));
            fin.set("usage", std::move(make_usage(res.n_prompt, res.n_gen)));
            fin.set("timings", std::move(make_timings(res.n_prompt, res.prompt_ms,
                                                      res.n_gen, res.predict_ms)));
            w.write("data: " + fin.dump() + "\n\n");
        }
        w.write("data: [DONE]\n\n");
        w.finish();
        return;
    }

    // Non-streaming.
    std::lock_guard<std::mutex> lock(g_infer_mutex);
    tj::Json resp = tj::Json::make_object();
    build_base(resp);
    tj::Json choices = tj::Json::make_array();
    int total_prompt = 0, total_gen = 0;
    double sum_prompt_ms = 0.0, sum_predict_ms = 0.0;
    for (int s = 0; s < cp.n; ++s) {
        g_infer.reset_kv();
        GenResult res = generate(cp.prompt_tokens, cp.n_predict, cp.sp, cp.stops, g_max_seq,
                                 [](int) { return true; });
        std::string text = res.text;
        if (is_chat) text = strip_think_blocks(text);
        if (cp.echo && !is_chat) text = cp.prompt_text + text;

        tj::Json ch = tj::Json::make_object();
        ch.set("index", tj::Json::make_number(s));
        if (is_chat) {
            tj::Json msg = tj::Json::make_object();
            msg.set("role", tj::Json::make_string("assistant"));
            msg.set("content", tj::Json::make_string(text));
            ch.set("message", std::move(msg));
        } else {
            ch.set("text", tj::Json::make_string(text));
        }
        ch.set("finish_reason", tj::Json::make_string(res.finish_reason));
        ch.set("logprobs", tj::Json::make_null());
        choices.push_back(std::move(ch));
        total_prompt += res.n_prompt;
        total_gen += res.n_gen;
        sum_prompt_ms += res.prompt_ms;
        sum_predict_ms += res.predict_ms;
    }
    resp.set("choices", std::move(choices));
    resp.set("usage", std::move(make_usage(total_prompt, total_gen)));
    resp.set("timings", std::move(make_timings(total_prompt, sum_prompt_ms, total_gen,
                                               sum_predict_ms)));

    std::map<std::string, std::string> hdrs{{"Content-Type", "application/json"}};
    w.write_head(200, "OK", hdrs, false);
    w.write(resp.dump() + "\n");
    w.finish();
}

void handle_completions(const HttpRequest & req, StreamWriter & w) {
    bool ok = false;
    tj::Json body = tj::json_parse(req.body, &ok);
    if (!ok || !body.is_object()) { send_json_error(w, 400, "invalid JSON body"); return; }

    CompletionParams cp;
    std::string err;
    if (!parse_params(body, cp, err)) { send_json_error(w, 400, err); return; }

    const tj::Json * pr = body.get("prompt");
    std::string prompt_text;
    if (pr && pr->is_string()) {
        prompt_text = pr->str;
    } else if (pr && pr->is_array() && !pr->arr.empty() && pr->arr[0].is_string()) {
        prompt_text = pr->arr[0].str;
    } else {
        send_json_error(w, 400, "missing 'prompt' (string or array of strings)");
        return;
    }

    cp.is_chat = false;
    cp.prompt_text = prompt_text;
    cp.prompt_tokens = g_tok.tokenize(prompt_text, false, false);
    if ((int) cp.prompt_tokens.size() + cp.n_predict > g_max_seq) {
        send_json_error(w, 400, "prompt + max_tokens exceeds context size (" +
                                    std::to_string(g_max_seq) + ")");
        return;
    }
    run_generation(req, w, cp, false);
}

void handle_chat_completions(const HttpRequest & req, StreamWriter & w) {
    bool ok = false;
    tj::Json body = tj::json_parse(req.body, &ok);
    if (!ok || !body.is_object()) { send_json_error(w, 400, "invalid JSON body"); return; }

    CompletionParams cp;
    std::string err;
    if (!parse_params(body, cp, err)) { send_json_error(w, 400, err); return; }

    const tj::Json * msgs = body.get("messages");
    if (!msgs || !msgs->is_array() || msgs->arr.empty()) {
        send_json_error(w, 400, "missing 'messages' array");
        return;
    }
    // Build the Qwen3 chat prompt from the message list.
    std::string prompt_text;
    for (auto & m : msgs->arr) {
        std::string role = m.is_object() ? m.get("role")->str_or("user") : "user";
        std::string content;
        if (m.is_object()) {
            const tj::Json * c = m.get("content");
            if (c && c->is_string()) content = c->str;
            else if (c && c->is_array()) {
                for (auto & part : c->arr) {
                    const tj::Json * txt = part.get("text");
                    if (txt && txt->is_string()) content += txt->str;
                }
            }
        }
        if (role == "system") prompt_text += "<|im_start|>system\n" + content + "<|im_end|>\n";
        else if (role == "user") prompt_text += "<|im_start|>user\n" + content + "<|im_end|>\n";
        else if (role == "assistant") prompt_text += "<|im_start|>assistant\n" + content + "<|im_end|>\n";
    }
    prompt_text += "<|im_start|>assistant\n<think>\n\n</think>\n\n";

    cp.is_chat = true;
    cp.prompt_text = prompt_text;
    cp.prompt_tokens = g_tok.tokenize(prompt_text, false, true);
    if ((int) cp.prompt_tokens.size() + cp.n_predict > g_max_seq) {
        send_json_error(w, 400, "prompt + max_tokens exceeds context size (" +
                                    std::to_string(g_max_seq) + ")");
        return;
    }
    run_generation(req, w, cp, true);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
void print_usage(const char * prog) {
    fprintf(stderr,
            "usage: %s -m model.gguf [--host HOST] [--port PORT] [-c N] [--alias NAME]\n"
            "  -m, --model PATH      GGUF model file\n"
            "  --host HOST           bind address (accepted for llama-server compat; binds 0.0.0.0)\n"
            "  -p, --port PORT       listen port (default 8080)\n"
            "  -c, --ctx-size N      max context size (default 2048)\n"
            "  --alias NAME          model name reported by /v1/models (default tinyinfer)\n"
            "  -ngl N, -t N          accepted and ignored (GPU-only engine)\n",
            prog);
}

} // namespace

int main(int argc, char ** argv) {
    int port = 8080;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(1); }
            return argv[++i];
        };
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            g_model_path = need("--model");
        } else if (strcmp(argv[i], "--host") == 0) {
            need("--host");  // accepted for compatibility; we bind 0.0.0.0
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--port") == 0) {
            port = atoi(need("--port"));
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--ctx-size") == 0) {
            g_max_seq = atoi(need("--ctx-size"));
        } else if (strcmp(argv[i], "--alias") == 0) {
            g_model_name = need("--alias");
        } else if (strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "-t") == 0 ||
                   strcmp(argv[i], "--threads") == 0) {
            need(argv[i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    if (g_model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Load model + tokenizer (tokenizer data comes from the GGUF reader,
    // mirroring main.cpp).
    if (!g_model.load(g_model_path)) {
        fprintf(stderr, "server: failed to load model %s\n", g_model_path.c_str());
        return 1;
    }
    GgufReader reader;
    if (!reader.load(g_model_path)) return 1;
    if (!g_tok.load_tokens(reader.get_string_array("tokenizer.ggml.tokens"),
                           reader.get_u32_array("tokenizer.ggml.token_type"))) return 1;
    if (!g_tok.load_merges(reader.get_string_array("tokenizer.ggml.merges"))) return 1;
    g_tok.set_eos_id((int32_t) reader.get_u32("tokenizer.ggml.eos_token_id", 151645));

    // One Inference instance, serialized by a mutex (single-slot semantics).
    if (!g_infer.init(g_model, g_max_seq, 1)) {
        fprintf(stderr, "server: inference init failed\n");
        return 1;
    }
    g_infer.set_use_cuda_graph(true);
    fprintf(stderr, "server: model '%s' ready (vocab=%d, ctx=%d)\n",
            g_model_name.c_str(), g_model.cfg.n_vocab, g_max_seq);

    HttpServer srv(port);
    if (!srv.start()) return 1;
    srv.handle("GET", "/health", handle_health);
    srv.handle("GET", "/v1/models", handle_models);
    srv.handle("GET", "/props", handle_props);
    srv.handle("POST", "/v1/completions", handle_completions);
    srv.handle("POST", "/v1/chat/completions", handle_chat_completions);
    srv.run();
    return 0;
}
