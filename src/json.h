#pragma once
// Minimal self-contained JSON value + parser + serializer for tinyinfer-server.
// No third-party dependencies; supports the subset used by OpenAI-compatible
// chat/completions APIs (objects, arrays, strings, numbers, bool, null).

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tj {

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;  // insertion-ordered

    static Json make_null() { return Json(); }
    static Json make_bool(bool v) { Json j; j.type = Bool; j.b = v; return j; }
    static Json make_number(double v) { Json j; j.type = Number; j.num = v; return j; }
    static Json make_string(std::string v) { Json j; j.type = String; j.str = std::move(v); return j; }
    static Json make_array() { Json j; j.type = Array; return j; }
    static Json make_object() { Json j; j.type = Object; return j; }

    bool is_null() const { return type == Null; }
    bool is_bool() const { return type == Bool; }
    bool is_number() const { return type == Number; }
    bool is_string() const { return type == String; }
    bool is_array() const { return type == Array; }
    bool is_object() const { return type == Object; }

    void push_back(Json v) { arr.push_back(std::move(v)); }
    void set(std::string key, Json v) {
        for (auto & kv : obj) {
            if (kv.first == key) { kv.second = std::move(v); return; }
        }
        obj.emplace_back(std::move(key), std::move(v));
    }

    const Json * find(const std::string & key) const {
        for (auto & kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
    Json * find(const std::string & key) {
        for (auto & kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }

    // Convenience getters with defaults.
    double num_or(double dflt) const { return is_number() ? num : dflt; }
    const std::string & str_or(const std::string & dflt) const { return is_string() ? str : dflt; }
    bool bool_or(bool dflt) const { return is_bool() ? b : dflt; }
    const Json * get(const std::string & key) const { return find(key); }
    Json * get(const std::string & key) { return find(key); }

    std::string dump() const;
    size_t size() const { return type == Array ? arr.size() : (type == Object ? obj.size() : 0); }
};

// Parse a JSON document. Returns Null on error; on success the value.
// `ok` is set to true on success.
Json json_parse(const std::string & text, bool * ok = nullptr);

} // namespace tj
