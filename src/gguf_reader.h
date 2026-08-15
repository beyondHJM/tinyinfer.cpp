#pragma once
// Minimal GGUF (v2/v3) reader, self-written.
// Supports only what this project needs: scalar/string/array metadata and
// F32 / BF16 tensor data. No ggml dependency.

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct GgufTensorInfo {
    std::string name;
    int         n_dims = 0;
    uint64_t    ne[4]  = {0, 0, 0, 0};
    uint32_t    type   = 0;   // GGML_TYPE: F32=0, BF16=30
    uint64_t    offset = 0;
};

class GgufReader {
public:
    // Metadata value: stores one scalar or one homogeneous array.
    struct Value {
        uint32_t type = 0;   // GGUF_TYPE
        // scalar
        uint64_t    u = 0;
        int64_t     i = 0;
        float       f = 0.0f;
        bool        b = false;
        std::string s;
        // arrays
        std::vector<std::string> str_arr;
        std::vector<uint64_t>    u_arr;   // UINT32 / UINT64 arrays
        std::vector<float>       f32_arr;
    };

    bool load(const std::string & path);

    const std::map<std::string, Value> & metadata() const { return kv_; }
    bool has(const std::string & key) const { return kv_.count(key) != 0; }

    const Value * get(const std::string & key) const {
        auto it = kv_.find(key);
        return it == kv_.end() ? nullptr : &it->second;
    }

    std::string get_string(const std::string & key, const std::string & def = "") const;
    uint32_t    get_u32(const std::string & key, uint32_t def = 0) const;
    uint64_t    get_u64(const std::string & key, uint64_t def = 0) const;
    float       get_f32(const std::string & key, float def = 0.0f) const;
    bool        get_bool(const std::string & key, bool def = false) const;
    std::vector<std::string> get_string_array(const std::string & key) const;
    std::vector<uint32_t>    get_u32_array(const std::string & key) const;

    const std::vector<GgufTensorInfo> & tensors() const { return tensors_; }
    const GgufTensorInfo * find_tensor(const std::string & name) const;

    // Read raw tensor bytes (BF16/F32) into dst. dst size must be at least
    // nbytes. Returns false on I/O error.
    bool read_tensor_data(const GgufTensorInfo & ti, void * dst, size_t nbytes);

    uint32_t version() const { return version_; }
    uint64_t tensor_count() const { return tensor_count_; }

private:
    bool read_string(std::ifstream & f, std::string & out);
    bool read_value(std::ifstream & f, uint32_t type, Value & out);

    uint32_t version_ = 0;
    uint64_t tensor_count_ = 0;
    uint64_t kv_count_ = 0;
    uint64_t tensor_data_off_ = 0;   // absolute file offset of the tensor data section
    std::map<std::string, Value> kv_;
    std::vector<GgufTensorInfo> tensors_;
    std::string path_;
};
