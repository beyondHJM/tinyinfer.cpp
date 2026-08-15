// Reference tool: dump last-token logits from llama.cpp for one raw prompt.
// Used for M2 numerical alignment of qwen3-gpu-infer against llama.cpp.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf out.txt [prompt]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string out_path = argv[2];
    const std::string prompt = argc > 3 ? argv[3] : "Hello world";

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks;
    const int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, false, false);
    toks.resize(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), false, false);

    fprintf(stderr, "ref tokens:");
    for (int t : toks) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to init context\n");
        return 1;
    }

    llama_batch batch = llama_batch_get_one(toks.data(), toks.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    const float * logits = llama_get_logits(ctx);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    FILE * f = fopen(out_path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", out_path.c_str());
        return 1;
    }
    for (int i = 0; i < n_vocab; ++i) {
        fprintf(f, "%.9g\n", logits[i]);
    }
    fclose(f);

    std::vector<std::pair<float, int>> scored;
    scored.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) scored.push_back({logits[i], i});
    std::sort(scored.begin(), scored.end(),
              [](const auto & a, const auto & b) { return a.first > b.first; });
    fprintf(stderr, "ref top10:");
    for (int i = 0; i < 10; ++i) fprintf(stderr, " %d(%.3f)", scored[i].second, scored[i].first);
    fprintf(stderr, "\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
