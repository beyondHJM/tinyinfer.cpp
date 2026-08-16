// qwen3-gpu-infer: offline batched Qwen3-0.6B (BF16) inference demo.
// CLI skeleton modeled on llama.cpp examples/simple/simple.cpp.

#include "gguf_reader.h"
#include "inference.h"
#include "model.h"
#include "selftest.h"
#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct SampleParams {
    float temp = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    uint64_t seed = 42;
};

int sample_token(const float * logits, int n_vocab, const SampleParams & sp) {
    if (sp.temp <= 0.0f) {
        int best = 0;
        float bestv = -INFINITY;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits[i] > bestv) { bestv = logits[i]; best = i; }
        }
        return best;
    }

    // Top-K selection instead of a full sort of all n_vocab logits
    // (full sort is O(vocab log vocab), which dominates the decode step).
    std::vector<std::pair<float, int>> scored;
    scored.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        if (std::isfinite(logits[i])) scored.push_back({logits[i], i});
    }
    const int K = std::max(1, std::min((int) scored.size(),
                                       sp.top_k > 0 ? sp.top_k : 512));
    if ((int) scored.size() > K) {
        std::nth_element(scored.begin(), scored.begin() + K, scored.end(),
                         [](const auto & a, const auto & b) { return a.first > b.first; });
        scored.resize(K);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto & a, const auto & b) { return a.first > b.first; });

    float maxv = scored.empty() ? 0.0f : scored[0].first;
    double sum = 0.0;
    for (auto & s : scored) {
        s.first = expf(s.first / sp.temp - maxv);
        sum += s.first;
    }
    if (sp.top_p < 1.0f && !scored.empty()) {
        double acc = 0.0;
        size_t cut = scored.size();
        for (size_t i = 0; i < scored.size(); ++i) {
            acc += scored[i].first / sum;
            if (acc >= sp.top_p) { cut = i + 1; break; }
        }
        scored.resize(cut);
        sum = 0.0;
        for (auto & s : scored) sum += s.first;
    }

    static thread_local std::mt19937 rng((uint32_t) sp.seed);
    std::uniform_real_distribution<float> dist(0.0f, (float) sum);
    float r = dist(rng);
    double acc = 0.0;
    for (auto & s : scored) {
        acc += s.first;
        if (r <= (float) acc) return s.second;
    }
    return scored.empty() ? 0 : scored.back().second;
}

void print_usage(const char * prog) {
    fprintf(stderr,
            "usage: %s -m model.gguf -n max_tokens -p \"prompt\" [-p \"prompt2\" ...] [-f prompts.txt]\n"
            "  [-t temp] [--top-k K] [--top-p P] [--seed S] [--max-seq N] [--cuda-graph] [--verbose]\n"
            "  --cuda-graph: replay the decode phase with CUDA Graphs (off by default)\n",
            prog);
}

} // namespace

static bool read_prompts_file(const std::string & path, std::vector<std::string> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return true;
}

static std::string strip_think_blocks(const std::string & in) {
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

int main(int argc, char ** argv) {
    if (getenv("QWEN_SELFTEST")) {
        return run_selftest();
    }

    std::string model_path;
    std::vector<std::string> prompts;
    int n_predict = 128;
    SampleParams sp;
    int max_seq = 2048;
    bool verbose = false;
    bool chat = false;
    bool use_cuda_graph = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompts.emplace_back(argv[++i]);
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            if (!read_prompts_file(argv[++i], prompts)) {
                fprintf(stderr, "main: cannot read prompts file %s\n", argv[i]);
                return 1;
            }
        }
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) sp.temp = (float) atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) sp.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) sp.top_p = (float) atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) sp.seed = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "--max-seq") == 0 && i + 1 < argc) max_seq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cuda-graph") == 0) use_cuda_graph = true;
        else if (strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (strcmp(argv[i], "--chat") == 0) chat = true;
        else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || prompts.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    GgufReader reader;
    if (!reader.load(model_path)) return 1;

    Tokenizer tok;
    if (!tok.load_tokens(reader.get_string_array("tokenizer.ggml.tokens"),
                         reader.get_u32_array("tokenizer.ggml.token_type"))) return 1;
    if (!tok.load_merges(reader.get_string_array("tokenizer.ggml.merges"))) return 1;
    tok.set_eos_id((int32_t) reader.get_u32("tokenizer.ggml.eos_token_id", 151645));

    Qwen3Model model;
    if (!model.load(model_path)) return 1;

    const int n_vocab = model.cfg.n_vocab;
    const int n_seqs = (int) prompts.size();
    if (n_seqs > 16) {
        fprintf(stderr, "main: too many prompts (max 16)\n");
        return 1;
    }

    // Tokenize all prompts (optionally wrapped in the Qwen3 chat template).
    std::vector<std::vector<int32_t>> toks(n_seqs);
    int total_tokens = 0;
    for (int s = 0; s < n_seqs; ++s) {
        std::string text = prompts[s];
        if (chat) {
            // Qwen3 chat format. The empty <think> block disables thinking mode
            // (the model then answers directly instead of rambling inside <think>).
            text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
                   "<think>\n\n</think>\n\n";
        }
        toks[s] = tok.tokenize(text, false, chat);
        if (verbose) {
            fprintf(stderr, "prompt %d tokens: %zu\n", s, toks[s].size());
            for (int t : toks[s]) fprintf(stderr, "  %d '%s'\n", t, tok.token_to_piece(t).c_str());
        }
        total_tokens += (int) toks[s].size();
    }
    if (total_tokens == 0) {
        fprintf(stderr, "main: empty prompts\n");
        return 1;
    }

    Inference infer;
    if (!infer.init(model, max_seq, n_seqs)) return 1;
    infer.set_use_cuda_graph(use_cuda_graph);

    // ---------------- prefill ----------------
    std::vector<int32_t> batch_tokens;
    std::vector<int> batch_pos, batch_seq;
    for (int s = 0; s < n_seqs; ++s) {
        for (size_t p = 0; p < toks[s].size(); ++p) {
            batch_tokens.push_back(toks[s][p]);
            batch_pos.push_back((int) p);
            batch_seq.push_back(s);
        }
    }
    std::vector<float> logits;
    std::vector<int> gen_len(n_seqs, 0);
    std::vector<bool> finished(n_seqs, false);
    std::vector<std::string> output(n_seqs);

    auto t0 = std::chrono::steady_clock::now();
    if (!infer.forward(batch_tokens, batch_pos, batch_seq, n_seqs, logits)) return 1;
    auto t1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (verbose) {
        for (int s = 0; s < n_seqs; ++s) {
            const float * row = &logits[(size_t) s * n_vocab];
            std::vector<std::pair<float, int>> scored;
            for (int i = 0; i < n_vocab; ++i) scored.push_back({row[i], i});
            std::sort(scored.begin(), scored.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });
            fprintf(stderr, "prefill logits seq %d (first 5, top 5):", s);
            for (int i = 0; i < 5; ++i) fprintf(stderr, " %.3f", row[i]);
            fprintf(stderr, " | top5:");
            for (int i = 0; i < 5; ++i) {
                fprintf(stderr, " %d(%s):%.3f", scored[i].second,
                        tok.token_to_piece(scored[i].second).c_str(), scored[i].first);
            }
            fprintf(stderr, "\n");
        }
    }

    int generated = 0;
    for (int s = 0; s < n_seqs; ++s) {
        int id = sample_token(&logits[(size_t) s * n_vocab], n_vocab, sp);
        if (tok.is_eog(id)) { finished[s] = true; continue; }
        output[s] += tok.token_to_piece(id);
        toks[s].push_back(id);
        gen_len[s] = 1;
        generated++;
        if (verbose) fprintf(stderr, "gen[%d] id=%d piece='%s'\n", s, id,
                             tok.token_to_piece(id).c_str());
    }

    // ---------------- decode loop ----------------
    auto t2 = std::chrono::steady_clock::now();
    while (true) {
        std::vector<int32_t> dec_tokens;
        std::vector<int> dec_pos, dec_seq;
        for (int s = 0; s < n_seqs; ++s) {
            if (finished[s]) continue;
            dec_tokens.push_back(toks[s].back());
            dec_pos.push_back((int) toks[s].size() - 1);
            dec_seq.push_back(s);
        }
        if (dec_tokens.empty()) break;
        if (generated >= n_seqs + n_predict) break;

        if (!infer.forward(dec_tokens, dec_pos, dec_seq, n_seqs, logits)) return 1;

        int active = 0;
        for (int s = 0; s < n_seqs; ++s) {
            if (finished[s]) continue;
            if (gen_len[s] >= n_predict) { finished[s] = true; continue; }
            int id = sample_token(&logits[(size_t) s * n_vocab], n_vocab, sp);
            if (tok.is_eog(id)) { finished[s] = true; continue; }
            output[s] += tok.token_to_piece(id);
            toks[s].push_back(id);
            gen_len[s]++;
            generated++;
            active++;
            if (verbose) fprintf(stderr, "gen[%d] id=%d piece='%s'\n", s, id,
                                 tok.token_to_piece(id).c_str());
        }
        if (active == 0) break;
    }
    auto t3 = std::chrono::steady_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ---------------- report ----------------
    for (int s = 0; s < n_seqs; ++s) {
        printf("==================== prompt %d ====================\n", s);
        printf("[prompt] %s\n", prompts[s].c_str());
        printf("[output] %s\n", chat ? strip_think_blocks(output[s]).c_str() : output[s].c_str());
    }
    fprintf(stderr, "main: prefill: %.1f ms (%d tokens)\n", prefill_ms, total_tokens);
    fprintf(stderr, "main: decode: %.1f ms, %d tokens, %.1f tok/s [cuda-graph: %s]\n",
            decode_ms, generated, generated / (decode_ms / 1000.0),
            use_cuda_graph ? "on" : "off");
    return 0;
}
