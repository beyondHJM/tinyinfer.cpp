#include "tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace {

// ---------------------------------------------------------------------------
// UTF-8 helpers (ported from llama.cpp src/unicode.cpp, MIT)
// ---------------------------------------------------------------------------

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    const auto & s = utf8;
    if (!(s[offset + 0] & 0x80)) {
        uint32_t r = (uint8_t) s[offset + 0];
        offset += 1;
        return r;
    }
    if (!(s[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(s[offset + 0] & 0x20)) {
        if (offset + 1 >= s.size() || ((s[offset + 1] & 0xc0) != 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        uint32_t r = ((s[offset + 0] & 0x1f) << 6) | (s[offset + 1] & 0x3f);
        offset += 2;
        return r;
    }
    if (!(s[offset + 0] & 0x10)) {
        if (offset + 2 >= s.size() || ((s[offset + 1] & 0xc0) != 0x80) ||
            ((s[offset + 2] & 0xc0) != 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        uint32_t r = ((s[offset + 0] & 0x0f) << 12) |
                     ((s[offset + 1] & 0x3f) << 6) | (s[offset + 2] & 0x3f);
        offset += 3;
        return r;
    }
    if (!(s[offset + 0] & 0x08)) {
        if (offset + 3 >= s.size() || ((s[offset + 1] & 0xc0) != 0x80) ||
            ((s[offset + 2] & 0xc0) != 0x80) || ((s[offset + 3] & 0xc0) != 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        uint32_t r = ((s[offset + 0] & 0x07) << 18) |
                     ((s[offset + 1] & 0x3f) << 12) | ((s[offset + 2] & 0x3f) << 6) |
                     (s[offset + 3] & 0x3f);
        offset += 4;
        return r;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;
    if (cpt <= 0x7f) {
        result.push_back((char) cpt);
    } else if (cpt <= 0x7ff) {
        result.push_back((char) (0xc0 | ((cpt >> 6) & 0x1f)));
        result.push_back((char) (0x80 | (cpt & 0x3f)));
    } else if (cpt <= 0xffff) {
        result.push_back((char) (0xe0 | ((cpt >> 12) & 0x0f)));
        result.push_back((char) (0x80 | ((cpt >> 6) & 0x3f)));
        result.push_back((char) (0x80 | (cpt & 0x3f)));
    } else {
        result.push_back((char) (0xf0 | ((cpt >> 18) & 0x07)));
        result.push_back((char) (0x80 | ((cpt >> 12) & 0x3f)));
        result.push_back((char) (0x80 | ((cpt >> 6) & 0x3f)));
        result.push_back((char) (0x80 | (cpt & 0x3f)));
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        try {
            result.push_back(unicode_cpt_from_utf8(utf8, offset));
        } catch (const std::invalid_argument &) {
            ++offset;
            result.push_back(0xFFFD);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Simplified Unicode category flags.
// This is an approximation of llama.cpp's unicode-data tables, targeted at the
// characters used by Chinese/English prompts. Categories that matter for the
// Qwen2 pre-tokenizer regex: letter, number, whitespace, and "anything else".
// ---------------------------------------------------------------------------

struct CptFlags {
    uint16_t is_undefined : 1;
    uint16_t is_number    : 1;
    uint16_t is_letter    : 1;
    uint16_t is_punctuation : 1;
    uint16_t is_symbol    : 1;
    uint16_t is_whitespace : 1;

    CptFlags(uint16_t flags = 0) {
        is_undefined   = (flags >> 0) & 1;
        is_number      = (flags >> 1) & 1;
        is_letter      = (flags >> 2) & 1;
        is_punctuation = (flags >> 3) & 1;
        is_symbol      = (flags >> 4) & 1;
        is_whitespace  = (flags >> 5) & 1;
    }

    uint16_t as_uint() const {
        return (uint16_t) (is_undefined | (is_number << 1) | (is_letter << 2) |
                           (is_punctuation << 3) | (is_symbol << 4) |
                           (is_whitespace << 5));
    }
};

static bool cpt_is_letter(uint32_t c) {
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return true;
    // Latin-1 supplement letters
    if ((c >= 0xC0 && c <= 0xD6) || (c >= 0xD8 && c <= 0xF6) || (c >= 0xF8 && c <= 0xFF)) return true;
    // Latin Extended-A / -B
    if ((c >= 0x100 && c <= 0x17F) || (c >= 0x180 && c <= 0x24F)) return true;
    // Greek / Cyrillic
    if ((c >= 0x370 && c <= 0x3FF) || (c >= 0x400 && c <= 0x4FF)) return true;
    // Hebrew / Arabic / Devanagari / Thai
    if ((c >= 0x590 && c <= 0x5FF) || (c >= 0x600 && c <= 0x6FF) ||
        (c >= 0x900 && c <= 0x97F) || (c >= 0xE00 && c <= 0xE7F)) return true;
    // Hangul
    if ((c >= 0x1100 && c <= 0x11FF) || (c >= 0xAC00 && c <= 0xD7AF)) return true;
    // Kana
    if ((c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF)) return true;
    // CJK
    if ((c >= 0x3100 && c <= 0x312F) || (c >= 0x3400 && c <= 0x4DBF) ||
        (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF) ||
        (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) ||
        (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B820 && c <= 0x2CEAF) ||
        (c >= 0x2CEB0 && c <= 0x2EBEF) || (c >= 0x2F800 && c <= 0x2FA1F) ||
        (c >= 0x30000 && c <= 0x3134F)) return true;
    // Fullwidth Latin
    if ((c >= 0xFF21 && c <= 0xFF3A) || (c >= 0xFF41 && c <= 0xFF5A)) return true;
    return false;
}

static bool cpt_is_number(uint32_t c) {
    if (c >= '0' && c <= '9') return true;
    if ((c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9)) return true;
    if (c >= 0x966 && c <= 0x96F) return true;
    if (c >= 0xFF10 && c <= 0xFF19) return true;
    return false;
}

static bool cpt_is_whitespace(uint32_t c) {
    if ((c >= 0x09 && c <= 0x0D) || c == 0x20) return true;
    if (c == 0x85 || c == 0xA0 || c == 0x1680) return true;
    if (c >= 0x2000 && c <= 0x200A) return true;
    if (c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000) return true;
    return false;
}

static bool cpt_is_punctuation(uint32_t c) {
    if ((c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) ||
        (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E)) return true;
    if (c >= 0x2000 && c <= 0x206F) return true;   // general punctuation (whitespace subset handled first)
    if (c >= 0x3000 && c <= 0x303F) return true;   // CJK punctuation
    if ((c >= 0xFF00 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF20) ||
        (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65)) return true;
    return false;
}

static bool cpt_is_symbol(uint32_t c) {
    if ((c >= 0x2190 && c <= 0x21FF) || (c >= 0x2200 && c <= 0x22FF)) return true;
    if ((c >= 0x2600 && c <= 0x27BF) || (c >= 0x1F300 && c <= 0x1FAFF)) return true;
    return false;
}

static CptFlags cpt_flags(uint32_t c) {
    CptFlags f;
    if (cpt_is_whitespace(c)) {
        f.is_whitespace = 1;
    } else if (cpt_is_letter(c)) {
        f.is_letter = 1;
    } else if (cpt_is_number(c)) {
        f.is_number = 1;
    } else if (cpt_is_punctuation(c)) {
        f.is_punctuation = 1;
    } else if (cpt_is_symbol(c)) {
        f.is_symbol = 1;
    } else {
        f.is_undefined = 1;   // defined but not in our tables
    }
    return f;
}

static uint32_t cpt_tolower(uint32_t c) {
    if (c >= 'A' && c <= 'Z') return c + 0x20;
    if (c >= 0xC0 && c <= 0xDE && c != 0xD7) return c + 0x20;
    if ((c & 1) == 0 && c >= 0x100 && c <= 0x17E) return c + 1;
    return c;
}

// ---------------------------------------------------------------------------
// GPT-2 byte encoding tables (same algorithm as llama.cpp/unicode.cpp)
// ---------------------------------------------------------------------------

std::vector<std::pair<uint8_t, std::string>> build_byte_encoder() {
    std::vector<std::pair<uint8_t, std::string>> out;
    std::unordered_set<uint8_t> used;
    auto add = [&](uint8_t b) {
        if (!used.count(b)) {
            used.insert(b);
            out.push_back({b, unicode_cpt_to_utf8(b)});
        }
    };
    for (int ch = 0x21; ch <= 0x7E; ++ch) add((uint8_t) ch);
    for (int ch = 0xA1; ch <= 0xAC; ++ch) add((uint8_t) ch);
    for (int ch = 0xAE; ch <= 0xFF; ++ch) add((uint8_t) ch);
    int n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (!used.count((uint8_t) ch)) {
            out.push_back({(uint8_t) ch, unicode_cpt_to_utf8(256 + n)});
            ++n;
        }
    }
    return out;
}

std::unordered_map<std::string, uint8_t> build_byte_decoder() {
    auto enc = build_byte_encoder();
    std::unordered_map<std::string, uint8_t> dec;
    for (auto & p : enc) dec[p.second] = p.first;
    return dec;
}

// ---------------------------------------------------------------------------
// Qwen2 pre-tokenizer regex splitter.
// Ported 1:1 from llama.cpp src/unicode.cpp
// `unicode_regex_split_custom_qwen2` (MIT).
// Regex: (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|
//        [^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|
//        \s*[\r\n]+|\s+(?!\S)|\s+
// ---------------------------------------------------------------------------

std::vector<size_t> regex_split_qwen2(const std::vector<uint32_t> & cpts) {
    std::vector<size_t> offsets;
    const size_t offset_ini = 0;
    const size_t offset_end = cpts.size();

    auto _get_cpt = [&](size_t pos) -> uint32_t {
        return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : 0xFFFFFFFFu;
    };
    auto _get_flags = [&](size_t pos) -> CptFlags {
        return (offset_ini <= pos && pos < offset_end) ? cpt_flags(cpts[pos]) : CptFlags{};
    };

    size_t _prev_end = offset_ini;
    auto _add_token = [&](size_t end) -> size_t {
        size_t len = end - _prev_end;
        if (len > 0) offsets.push_back(len);
        _prev_end = end;
        return len;
    };

    for (size_t pos = offset_ini; pos < offset_end;) {
        uint32_t cpt = _get_cpt(pos);
        CptFlags flags = _get_flags(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < offset_end) {
            uint32_t cpt_next = cpt_tolower(_get_cpt(pos + 1));
            if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                pos += _add_token(pos + 2);
                continue;
            }
            if (pos + 2 < offset_end) {
                uint32_t cpt_next_next = cpt_tolower(_get_cpt(pos + 2));
                if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                    (cpt_next == 'v' && cpt_next_next == 'e') ||
                    (cpt_next == 'l' && cpt_next_next == 'l')) {
                    pos += _add_token(pos + 3);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
            if (flags.is_letter || _get_flags(pos + 1).is_letter) {
                pos++;
                while (_get_flags(pos).is_letter) pos++;
                _add_token(pos);
                continue;
            }
        }

        // \p{N}
        if (flags.is_number) {
            pos++;
            _add_token(pos);
            continue;
        }

        // ?[^\s\p{L}\p{N}]+[\r\n]*
        CptFlags flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
        if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
            pos += (cpt == ' ');
            while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                flags2 = _get_flags(++pos);
            }
            uint32_t cpt2 = _get_cpt(pos);
            while (cpt2 == '\r' || cpt2 == '\n') cpt2 = _get_cpt(++pos);
            _add_token(pos);
            continue;
        }

        size_t num_whitespaces = 0;
        size_t last_end_r_or_n = 0;
        while (_get_flags(pos + num_whitespaces).is_whitespace) {
            uint32_t cpt2 = _get_cpt(pos + num_whitespaces);
            if (cpt2 == '\r' || cpt2 == '\n') last_end_r_or_n = pos + num_whitespaces + 1;
            num_whitespaces++;
        }

        // \s*[\r\n]+
        if (last_end_r_or_n > 0) {
            pos = last_end_r_or_n;
            _add_token(pos);
            continue;
        }

        // \s+(?!\S)
        if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != 0xFFFFFFFFu) {
            pos += num_whitespaces - 1;
            _add_token(pos);
            continue;
        }

        // \s+
        if (num_whitespaces > 0) {
            pos += num_whitespaces;
            _add_token(pos);
            continue;
        }

        _add_token(++pos);
    }
    return offsets;
}

} // namespace

bool Tokenizer::load_tokens(const std::vector<std::string> & tokens,
                            const std::vector<uint32_t> & types) {
    tokens_ = tokens;
    types_ = types;
    token_to_id_.clear();
    for (size_t i = 0; i < tokens_.size(); ++i) {
        token_to_id_[tokens_[i]] = (int32_t) i;
    }
    utf8_to_byte_ = build_byte_decoder();
    for (auto & p : utf8_to_byte_) byte_to_utf8_[p.second] = p.first;
    return true;
}

bool Tokenizer::load_merges(const std::vector<std::string> & merges) {
    bpe_ranks_.clear();
    for (size_t i = 0; i < merges.size(); ++i) {
        const std::string & word = merges[i];
        const size_t pos = word.find(' ', 1);
        if (pos == std::string::npos) continue;
        std::string first = word.substr(0, pos);
        std::string second = word.substr(pos + 1);
        bpe_ranks_[first + " " + second] = (int) i;
    }
    return true;
}

std::vector<int32_t> Tokenizer::tokenize(const std::string & text,
                                         bool add_special,
                                         bool parse_special) const {
    std::vector<int32_t> output;
    (void) add_special;   // Qwen3 has no BOS token

    // Split the text on special tokens when requested (e.g. <|im_start|>).
    std::vector<std::string> fragments;
    if (parse_special) {
        std::vector<std::pair<std::string, int32_t>> specials;
        for (size_t i = 0; i < tokens_.size(); ++i) {
            if (is_special((int32_t) i)) specials.push_back({tokens_[i], (int32_t) i});
        }
        std::string remaining = text;
        while (!remaining.empty()) {
            size_t best_pos = std::string::npos;
            size_t best_len = 0;
            int32_t best_id = -1;
            for (auto & sp : specials) {
                if (sp.first.empty()) continue;
                size_t p = remaining.find(sp.first);
                if (p != std::string::npos && (p < best_pos || (p == best_pos && sp.first.size() > best_len))) {
                    best_pos = p;
                    best_len = sp.first.size();
                    best_id = sp.second;
                }
            }
            if (best_pos == std::string::npos) {
                fragments.push_back(remaining);
                break;
            }
            if (best_pos > 0) fragments.push_back(remaining.substr(0, best_pos));
            output.push_back(best_id);
            remaining = remaining.substr(best_pos + best_len);
        }
    } else {
        fragments.push_back(text);
    }

    for (const auto & fragment : fragments) {
        if (fragment.empty()) continue;
        const auto cpts = unicode_cpts_from_utf8(fragment);
        const auto offsets = regex_split_qwen2(cpts);

        // Build byte-encoded words.
        std::vector<std::string> words;
        size_t start = 0;
        for (size_t off : offsets) {
            std::string word;
            for (size_t i = start; i < start + off; ++i) word += unicode_cpt_to_utf8(cpts[i]);
            start += off;
            if (word.empty()) continue;
            std::string encoded;
            for (char ch : word) {
                auto it = byte_to_utf8_.find((uint8_t) ch);
                if (it != byte_to_utf8_.end()) {
                    encoded += it->second;
                } else {
                    encoded += ch;
                }
            }
            words.push_back(encoded);
        }

        // BPE merge (ported from llama.cpp llm_tokenizer_bpe_session, MIT).
        std::vector<Symbol> symbols_final;
        int final_prev_index = -1;

        for (const auto & word : words) {
            std::vector<Symbol> symbols;
            auto cmp = [](const Bigram & l, const Bigram & r) {
                return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
            };
            std::priority_queue<Bigram, std::vector<Bigram>, decltype(cmp)> work_queue(cmp);

            int index = 0;
            size_t offset = 0;
            while (offset < word.size()) {
                Symbol sym;
                size_t char_len = std::min(word.size() - offset, unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.push_back(sym);
            }

            auto add_new_bigram = [&](int left, int right) {
                if (left == -1 || right == -1) return;
                std::string left_token(symbols[left].text, symbols[left].n);
                std::string right_token(symbols[right].text, symbols[right].n);
                auto rit = bpe_ranks_.find(left_token + " " + right_token);
                if (rit == bpe_ranks_.end()) return;
                Bigram bg;
                bg.left = left;
                bg.right = right;
                bg.text = left_token + right_token;
                bg.rank = rit->second;
                bg.size = left_token.size() + right_token.size();
                work_queue.push(std::move(bg));
            };

            for (int i = 1; i < (int) symbols.size(); ++i) add_new_bigram(i - 1, i);

            while (!work_queue.empty()) {
                Bigram bigram = std::move(const_cast<Bigram &>(work_queue.top()));
                work_queue.pop();
                Symbol & left_symbol = symbols[bigram.left];
                Symbol & right_symbol = symbols[bigram.right];
                if (left_symbol.n == 0 || right_symbol.n == 0) continue;
                std::string left_token(left_symbol.text, left_symbol.n);
                std::string right_token(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) continue;

                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) symbols[right_symbol.next].prev = bigram.left;
                add_new_bigram(left_symbol.prev, bigram.left);
                add_new_bigram(bigram.left, left_symbol.next);
            }

            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) symbols_final[final_prev_index].next = (int) symbols_final.size();
                    symbols_final.push_back(sym);
                    final_prev_index = (int) symbols_final.size() - 1;
                }
            }
        }

        // Map merged symbols to token ids.
        for (auto & symbol : symbols_final) {
            if (symbol.n == 0) continue;
            std::string str(symbol.text, symbol.n);
            auto it = token_to_id_.find(str);
            if (it != token_to_id_.end()) {
                output.push_back(it->second);
            } else {
                // Byte fallback: one token per byte.
                for (auto j = str.begin(); j != str.end(); ++j) {
                    std::string byte_str(1, *j);
                    auto bit = token_to_id_.find(byte_str);
                    if (bit != token_to_id_.end()) output.push_back(bit->second);
                }
            }
        }
    }
    return output;
}

std::string Tokenizer::token_to_piece(int32_t id, bool special) const {
    if (id < 0 || (size_t) id >= tokens_.size()) return "";
    const std::string & token_text = tokens_[id];
    uint32_t t = types_[id];
    bool attr_special = (t == TOKEN_UNKNOWN || t == TOKEN_CONTROL);
    if (!special && attr_special) return "";

    if (attr_special || t == TOKEN_USER_DEFINED) {
        return token_text;
    }
    if (t == TOKEN_NORMAL) {
        // Byte-decode: each codepoint maps back to one byte.
        std::string result;
        const auto cpts = unicode_cpts_from_utf8(token_text);
        for (uint32_t c : cpts) {
            auto it = utf8_to_byte_.find(unicode_cpt_to_utf8(c));
            if (it != utf8_to_byte_.end()) result.push_back((char) it->second);
            else result += unicode_cpt_to_utf8(c);
        }
        return result;
    }
    if (t == TOKEN_BYTE) {
        // "<0xXX>"
        if (token_text.size() == 6 && token_text.compare(0, 3, "<0x") == 0 && token_text.back() == '>') {
            char c = (char) strtol(token_text.substr(3, 2).c_str(), nullptr, 16);
            return std::string(1, c);
        }
        return "";
    }
    return "";
}
