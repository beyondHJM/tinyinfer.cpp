#include "model.h"

#include <cstring>

namespace {

constexpr uint32_t GGML_TYPE_F32  = 0;
constexpr uint32_t GGML_TYPE_BF16 = 30;

// Allocate device memory and copy from host. Returns pointer or nullptr.
template <typename T>
T * to_device(const void * host, size_t nbytes) {
    T * d = nullptr;
    if (nbytes == 0) return nullptr;
    QWEN_CU_CHECK(cudaMalloc(&d, nbytes));
    QWEN_CU_CHECK(cudaMemcpy(d, host, nbytes, cudaMemcpyHostToDevice));
    return d;
}

} // namespace

bool Qwen3Model::load(const std::string & gguf_path) {
    GgufReader reader;
    if (!reader.load(gguf_path)) return false;

    const std::string arch = reader.get_string("general.architecture");
    if (arch != "qwen3") {
        fprintf(stderr, "model: unsupported architecture '%s' (only qwen3)\n", arch.c_str());
        return false;
    }

    Qwen3Config c;
    c.n_layer   = (int) reader.get_u32("qwen3.block_count");
    c.n_embd    = (int) reader.get_u32("qwen3.embedding_length");
    c.n_ff      = (int) reader.get_u32("qwen3.feed_forward_length");
    c.n_head    = (int) reader.get_u32("qwen3.attention.head_count");
    c.n_head_kv = (int) reader.get_u32("qwen3.attention.head_count_kv");
    c.head_dim  = (int) reader.get_u32("qwen3.attention.key_length");
    c.rope_theta = reader.get_f32("qwen3.rope.freq_base", 1000000.0f);
    c.norm_eps   = reader.get_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    c.n_vocab    = (int) reader.get_u32("qwen3.vocab_size", 0);

    if (c.n_layer == 0 || c.n_embd == 0 || c.n_head == 0 || c.n_head_kv == 0 ||
        c.head_dim == 0 || c.n_ff == 0) {
        fprintf(stderr, "model: failed to read qwen3 hyperparameters from GGUF\n");
        return false;
    }
    if (c.n_vocab == 0) {
        const GgufTensorInfo * te = reader.find_tensor("token_embd.weight");
        if (!te) { fprintf(stderr, "model: token_embd.weight not found\n"); return false; }
        c.n_vocab = (int) te->ne[1];
    }

    const int nq = c.n_head * c.head_dim;       // 2048
    const int nk = c.n_head_kv * c.head_dim;    // 1024

    cfg = c;
    printf("model: qwen3 config: n_layer=%d n_embd=%d n_head=%d n_head_kv=%d "
           "head_dim=%d n_ff=%d n_vocab=%d rope_theta=%.0f norm_eps=%g\n",
           c.n_layer, c.n_embd, c.n_head, c.n_head_kv, c.head_dim, c.n_ff,
           c.n_vocab, c.rope_theta, c.norm_eps);

    auto load_bf16_tensor = [&](const std::string & name, size_t expect_elems,
                                bf16 * & dst) -> bool {
        const GgufTensorInfo * ti = reader.find_tensor(name);
        if (!ti) {
            fprintf(stderr, "model: tensor not found: %s\n", name.c_str());
            return false;
        }
        if (ti->type != GGML_TYPE_BF16) {
            fprintf(stderr, "model: tensor %s is not BF16 (type %u)\n", name.c_str(), ti->type);
            return false;
        }
        size_t elems = 1;
        for (int d = 0; d < ti->n_dims; ++d) elems *= ti->ne[d];
        if (expect_elems != 0 && elems != expect_elems) {
            fprintf(stderr, "model: tensor %s shape mismatch: got %zu, expected %zu\n",
                    name.c_str(), elems, expect_elems);
            return false;
        }
        std::vector<uint16_t> host(elems);
        if (!reader.read_tensor_data(*ti, host.data(), elems * sizeof(uint16_t))) {
            fprintf(stderr, "model: failed to read tensor %s\n", name.c_str());
            return false;
        }
        dst = to_device<bf16>(host.data(), elems * sizeof(uint16_t));
        printf("model: loaded %s [%llu x %llu] BF16 (%zu MB)\n", name.c_str(),
               (unsigned long long) ti->ne[1], (unsigned long long) ti->ne[0],
               elems * 2 / (1024 * 1024));
        return true;
    };

    auto load_f32_tensor = [&](const std::string & name, size_t expect_elems,
                               float * & dst) -> bool {
        const GgufTensorInfo * ti = reader.find_tensor(name);
        if (!ti) {
            fprintf(stderr, "model: tensor not found: %s\n", name.c_str());
            return false;
        }
        if (ti->type != GGML_TYPE_F32) {
            fprintf(stderr, "model: tensor %s is not F32 (type %u)\n", name.c_str(), ti->type);
            return false;
        }
        size_t elems = 1;
        for (int d = 0; d < ti->n_dims; ++d) elems *= ti->ne[d];
        if (expect_elems != 0 && elems != expect_elems) {
            fprintf(stderr, "model: tensor %s shape mismatch\n", name.c_str());
            return false;
        }
        std::vector<float> host(elems);
        if (!reader.read_tensor_data(*ti, host.data(), elems * sizeof(float))) {
            fprintf(stderr, "model: failed to read tensor %s\n", name.c_str());
            return false;
        }
        dst = to_device<float>(host.data(), elems * sizeof(float));
        return true;
    };

    // token embedding / LM head
    if (!load_bf16_tensor("token_embd.weight", (size_t) c.n_vocab * c.n_embd, tok_embd)) return false;
    if (!load_f32_tensor("output_norm.weight", c.n_embd, output_norm)) return false;

    layers.resize(c.n_layer);
    char name[256];
    for (int i = 0; i < c.n_layer; ++i) {
        Qwen3Layer & L = layers[i];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        if (!load_f32_tensor(name, c.n_embd, L.attn_norm)) return false;
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        if (!load_f32_tensor(name, c.n_embd, L.ffn_norm)) return false;
        snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", i);
        if (!load_f32_tensor(name, c.head_dim, L.q_norm)) return false;
        snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", i);
        if (!load_f32_tensor(name, c.head_dim, L.k_norm)) return false;

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
        if (!load_bf16_tensor(name, (size_t) nq * c.n_embd, L.wq)) return false;
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
        if (!load_bf16_tensor(name, (size_t) nk * c.n_embd, L.wk)) return false;
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
        if (!load_bf16_tensor(name, (size_t) nk * c.n_embd, L.wv)) return false;
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
        if (!load_bf16_tensor(name, (size_t) c.n_embd * nq, L.wo)) return false;

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
        if (!load_bf16_tensor(name, (size_t) c.n_ff * c.n_embd, L.w_gate)) return false;
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        if (!load_bf16_tensor(name, (size_t) c.n_ff * c.n_embd, L.w_up)) return false;
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        if (!load_bf16_tensor(name, (size_t) c.n_embd * c.n_ff, L.w_down)) return false;
    }

    printf("model: all %zu tensors loaded to GPU\n", reader.tensors().size());
    return true;
}

void Qwen3Model::free_all() {
    auto free_dev = [](void * p) { if (p) cudaFree(p); };
    free_dev(tok_embd);
    free_dev(output_norm);
    for (auto & L : layers) {
        free_dev(L.attn_norm); free_dev(L.ffn_norm); free_dev(L.q_norm); free_dev(L.k_norm);
        free_dev(L.wq); free_dev(L.wk); free_dev(L.wv); free_dev(L.wo);
        free_dev(L.w_gate); free_dev(L.w_up); free_dev(L.w_down);
        L.attn_norm = L.ffn_norm = L.q_norm = L.k_norm = nullptr;
        L.wq = L.wk = L.wv = L.wo = nullptr;
        L.w_gate = L.w_up = L.w_down = nullptr;
    }
    layers.clear();
}
