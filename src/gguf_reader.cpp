#include "gguf_reader.h"

#include <cstring>

namespace {

enum GgufType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

template <typename T>
static bool read_exact(std::ifstream & f, T & out) {
    f.read(reinterpret_cast<char *>(&out), sizeof(T));
    return f.gcount() == (std::streamsize) sizeof(T);
}

} // namespace

bool GgufReader::load(const std::string & path) {
    path_ = path;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "gguf: cannot open %s\n", path.c_str());
        return false;
    }

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "gguf: not a GGUF file: %s\n", path.c_str());
        return false;
    }
    if (!read_exact(f, version_)) return false;
    if (!read_exact(f, tensor_count_)) return false;
    if (!read_exact(f, kv_count_)) return false;
    if (version_ < 2 || version_ > 3) {
        fprintf(stderr, "gguf: unsupported version %u\n", version_);
        return false;
    }

    for (uint64_t i = 0; i < kv_count_; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_string(f, key)) return false;
        if (!read_exact(f, type)) return false;
        Value v;
        if (!read_value(f, type, v)) return false;
        kv_[key] = std::move(v);
    }

    for (uint64_t i = 0; i < tensor_count_; ++i) {
        GgufTensorInfo ti;
        if (!read_string(f, ti.name)) return false;
        uint32_t n_dims = 0;
        if (!read_exact(f, n_dims)) return false;
        ti.n_dims = (int) n_dims;
        for (uint32_t d = 0; d < n_dims; ++d) {
            if (!read_exact(f, ti.ne[d])) return false;
        }
        if (!read_exact(f, ti.type)) return false;
        if (!read_exact(f, ti.offset)) return false;
        tensors_.push_back(std::move(ti));
    }

    // Tensor data starts at the first 32-byte aligned offset after the
    // metadata + tensor-info section. Per-tensor offsets are relative to it.
    const uint64_t pos = (uint64_t) f.tellg();
    tensor_data_off_ = (pos + 31) & ~(uint64_t) 31;
    return true;
}

bool GgufReader::read_string(std::ifstream & f, std::string & out) {
    uint64_t n = 0;
    if (!read_exact(f, n)) return false;
    out.resize(n);
    if (n > 0) {
        f.read(&out[0], n);
        if (f.gcount() != (std::streamsize) n) return false;
    }
    return true;
}

bool GgufReader::read_value(std::ifstream & f, uint32_t type, Value & out) {
    out.type = type;
    switch (type) {
        case GGUF_TYPE_UINT8: {
            uint8_t v = 0; if (!read_exact(f, v)) return false; out.u = v; return true;
        }
        case GGUF_TYPE_INT8: {
            int8_t v = 0; if (!read_exact(f, v)) return false; out.i = v; return true;
        }
        case GGUF_TYPE_UINT16: {
            uint16_t v = 0; if (!read_exact(f, v)) return false; out.u = v; return true;
        }
        case GGUF_TYPE_INT16: {
            int16_t v = 0; if (!read_exact(f, v)) return false; out.i = v; return true;
        }
        case GGUF_TYPE_UINT32: {
            uint32_t v = 0; if (!read_exact(f, v)) return false; out.u = v; return true;
        }
        case GGUF_TYPE_INT32: {
            int32_t v = 0; if (!read_exact(f, v)) return false; out.i = v; return true;
        }
        case GGUF_TYPE_FLOAT32: {
            float v = 0; if (!read_exact(f, v)) return false; out.f = v; return true;
        }
        case GGUF_TYPE_BOOL: {
            bool v = false; if (!read_exact(f, v)) return false; out.b = v; return true;
        }
        case GGUF_TYPE_STRING: {
            return read_string(f, out.s);
        }
        case GGUF_TYPE_UINT64: {
            uint64_t v = 0; if (!read_exact(f, v)) return false; out.u = v; return true;
        }
        case GGUF_TYPE_INT64: {
            int64_t v = 0; if (!read_exact(f, v)) return false; out.i = v; return true;
        }
        case GGUF_TYPE_FLOAT64: {
            double v = 0; if (!read_exact(f, v)) return false; out.f = (float) v; return true;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type = 0;
            uint64_t n = 0;
            if (!read_exact(f, elem_type)) return false;
            if (!read_exact(f, n)) return false;
            out.type = GGUF_TYPE_ARRAY;
            for (uint64_t i = 0; i < n; ++i) {
                Value elem;
                if (!read_value(f, elem_type, elem)) return false;
                if (elem_type == GGUF_TYPE_STRING) {
                    out.str_arr.push_back(std::move(elem.s));
                } else if (elem_type == GGUF_TYPE_UINT32 || elem_type == GGUF_TYPE_UINT64 ||
                           elem_type == GGUF_TYPE_UINT8 || elem_type == GGUF_TYPE_UINT16) {
                    out.u_arr.push_back(elem.u);
                } else if (elem_type == GGUF_TYPE_INT32 || elem_type == GGUF_TYPE_INT64 ||
                           elem_type == GGUF_TYPE_INT8 || elem_type == GGUF_TYPE_INT16) {
                    out.u_arr.push_back((uint64_t) elem.i);
                } else if (elem_type == GGUF_TYPE_FLOAT32 || elem_type == GGUF_TYPE_FLOAT64) {
                    out.f32_arr.push_back(elem.f);
                } else if (elem_type == GGUF_TYPE_BOOL) {
                    out.u_arr.push_back(elem.b ? 1 : 0);
                } else {
                    fprintf(stderr, "gguf: unsupported array element type %u\n", elem_type);
                    return false;
                }
            }
            return true;
        }
        default:
            fprintf(stderr, "gguf: unsupported value type %u\n", type);
            return false;
    }
}

std::string GgufReader::get_string(const std::string & key, const std::string & def) const {
    const Value * v = get(key);
    return v && v->type == GGUF_TYPE_STRING ? v->s : def;
}

uint32_t GgufReader::get_u32(const std::string & key, uint32_t def) const {
    const Value * v = get(key);
    return v && v->type == GGUF_TYPE_UINT32 ? (uint32_t) v->u : def;
}

uint64_t GgufReader::get_u64(const std::string & key, uint64_t def) const {
    const Value * v = get(key);
    return v && v->type == GGUF_TYPE_UINT64 ? v->u : def;
}

float GgufReader::get_f32(const std::string & key, float def) const {
    const Value * v = get(key);
    return v && v->type == GGUF_TYPE_FLOAT32 ? v->f : def;
}

bool GgufReader::get_bool(const std::string & key, bool def) const {
    const Value * v = get(key);
    return v && v->type == GGUF_TYPE_BOOL ? v->b : def;
}

std::vector<std::string> GgufReader::get_string_array(const std::string & key) const {
    const Value * v = get(key);
    return v && v->type == GGUF_TYPE_ARRAY ? v->str_arr : std::vector<std::string>{};
}

std::vector<uint32_t> GgufReader::get_u32_array(const std::string & key) const {
    const Value * v = get(key);
    std::vector<uint32_t> out;
    if (v && v->type == GGUF_TYPE_ARRAY) {
        out.reserve(v->u_arr.size());
        for (uint64_t x : v->u_arr) out.push_back((uint32_t) x);
    }
    return out;
}

const GgufTensorInfo * GgufReader::find_tensor(const std::string & name) const {
    for (const auto & ti : tensors_) {
        if (ti.name == name) return &ti;
    }
    return nullptr;
}

bool GgufReader::read_tensor_data(const GgufTensorInfo & ti, void * dst, size_t nbytes) {
    std::ifstream f(path_, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg((std::streamoff) (tensor_data_off_ + ti.offset), std::ios::beg);
    if (!f.good()) return false;
    f.read(reinterpret_cast<char *>(dst), (std::streamsize) nbytes);
    return f.gcount() == (std::streamsize) nbytes;
}
