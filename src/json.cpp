#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace tj {
namespace {

struct Parser {
    const char * p;
    const char * end;

    void skip_ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }

    bool consume(char c) {
        if (p < end && *p == c) { ++p; return true; }
        return false;
    }

    bool parse_string(std::string & out) {
        if (!consume('"')) return false;
        out.clear();
        while (p < end) {
            char c = *p++;
            if (c == '"') return true;
            if (c == '\\') {
                if (p >= end) return false;
                char e = *p++;
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        // Decode a single UTF-16 code unit; surrogate pairs are
                        // combined. Produces UTF-8 bytes.
                        if (end - p < 4) return false;
                        unsigned cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = *p++;
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                            else return false;
                        }
                        if (cp >= 0xD800 && cp <= 0xDBFF && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                            p += 2;
                            unsigned lo = 0;
                            for (int i = 0; i < 4; ++i) {
                                char h = *p++;
                                lo <<= 4;
                                if (h >= '0' && h <= '9') lo |= (unsigned)(h - '0');
                                else if (h >= 'a' && h <= 'f') lo |= (unsigned)(h - 'a' + 10);
                                else if (h >= 'A' && h <= 'F') lo |= (unsigned)(h - 'A' + 10);
                                else return false;
                            }
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        // Encode UTF-8.
                        if (cp < 0x80) out += (char) cp;
                        else if (cp < 0x800) {
                            out += (char) (0xC0 | (cp >> 6));
                            out += (char) (0x80 | (cp & 0x3F));
                        } else if (cp < 0x10000) {
                            out += (char) (0xE0 | (cp >> 12));
                            out += (char) (0x80 | ((cp >> 6) & 0x3F));
                            out += (char) (0x80 | (cp & 0x3F));
                        } else {
                            out += (char) (0xF0 | (cp >> 18));
                            out += (char) (0x80 | ((cp >> 12) & 0x3F));
                            out += (char) (0x80 | ((cp >> 6) & 0x3F));
                            out += (char) (0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: return false;
                }
            } else if ((unsigned char) c < 0x20) {
                return false;
            } else {
                out += c;
            }
        }
        return false;
    }

    bool parse_number(Json & j) {
        const char * start = p;
        if (p < end && (*p == '-' || *p == '+')) ++p;
        bool any = false;
        while (p < end && *p >= '0' && *p <= '9') { ++p; any = true; }
        if (p < end && *p == '.') {
            ++p;
            while (p < end && *p >= '0' && *p <= '9') { ++p; any = true; }
        }
        if (any && p < end && (*p == 'e' || *p == 'E')) {
            ++p;
            if (p < end && (*p == '-' || *p == '+')) ++p;
            while (p < end && *p >= '0' && *p <= '9') ++p;
        }
        if (!any) return false;
        j.type = Json::Number;
        j.num = strtod(start, nullptr);
        return true;
    }

    bool parse_value(Json & j) {
        skip_ws();
        if (p >= end) return false;
        if (*p == '{') {
            ++p;
            j = Json::make_object();
            skip_ws();
            if (consume('}')) return true;
            for (;;) {
                skip_ws();
                std::string key;
                if (!parse_string(key)) return false;
                skip_ws();
                if (!consume(':')) return false;
                Json v;
                if (!parse_value(v)) return false;
                j.set(std::move(key), std::move(v));
                skip_ws();
                if (consume('}')) return true;
                if (!consume(',')) return false;
            }
        }
        if (*p == '[') {
            ++p;
            j = Json::make_array();
            skip_ws();
            if (consume(']')) return true;
            for (;;) {
                Json v;
                if (!parse_value(v)) return false;
                j.push_back(std::move(v));
                skip_ws();
                if (consume(']')) return true;
                if (!consume(',')) return false;
            }
        }
        if (*p == '"') {
            j = Json::make_string("");
            return parse_string(j.str);
        }
        if (end - p >= 4 && strncmp(p, "true", 4) == 0) { p += 4; j = Json::make_bool(true); return true; }
        if (end - p >= 5 && strncmp(p, "false", 5) == 0) { p += 5; j = Json::make_bool(false); return true; }
        if (end - p >= 4 && strncmp(p, "null", 4) == 0) { p += 4; j = Json::make_null(); return true; }
        return parse_number(j);
    }
};

void dump_string(const std::string & s, std::string & out) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    out += '"';
}

} // namespace

std::string Json::dump() const {
    std::string out;
    switch (type) {
        case Null: out += "null"; break;
        case Bool: out += b ? "true" : "false"; break;
        case Number: {
            char buf[32];
            if (num == (double) (long long) num && std::fabs(num) < 1e15) {
                snprintf(buf, sizeof buf, "%lld", (long long) num);
            } else {
                snprintf(buf, sizeof buf, "%.9g", num);
            }
            out += buf;
            break;
        }
        case String: dump_string(str, out); break;
        case Array: {
            out += '[';
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i) out += ',';
                out += arr[i].dump();
            }
            out += ']';
            break;
        }
        case Object: {
            out += '{';
            for (size_t i = 0; i < obj.size(); ++i) {
                if (i) out += ',';
                dump_string(obj[i].first, out);
                out += ':';
                out += obj[i].second.dump();
            }
            out += '}';
            break;
        }
    }
    return out;
}

Json json_parse(const std::string & text, bool * ok) {
    Parser ps{text.data(), text.data() + text.size()};
    Json j;
    bool good = ps.parse_value(j);
    if (good) ps.skip_ws();
    if (good && ps.p == ps.end) {
        if (ok) *ok = true;
        return j;
    }
    if (ok) *ok = false;
    return Json::make_null();
}

} // namespace tj
