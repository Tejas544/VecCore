#pragma once

// A deliberately tiny JSON *writer*.  We only ever emit flat records of
// strings, numbers and bools, so a dependency (nlohmann, RapidJSON) would be
// more surface than the problem has.  We never *parse* JSON in C++ -- the
// Python side does that, where a real parser already exists.
//
// D6 boundary note: this is plumbing, not an algorithm.  Nobody will ask you to
// defend a string escaper.

#include <cstdio>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace veccore::json {

inline std::string escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

/// Builds one flat JSON object.  Insertion-ordered, which matters only because
/// diffing two records in git is much easier when the keys do not move.
class Object {
public:
    Object& str(std::string_view key, std::string_view value) {
        return raw(key, "\"" + escape(value) + "\"");
    }

    Object& num(std::string_view key, double value) {
        std::ostringstream os;
        // 17 significant digits round-trips an IEEE754 double exactly.  A
        // truncated latency in a results file is a number you cannot reproduce.
        os.precision(17);
        os << value;
        return raw(key, os.str());
    }

    Object& num(std::string_view key, long long value) {
        return raw(key, std::to_string(value));
    }

    Object& num(std::string_view key, std::size_t value) {
        return raw(key, std::to_string(value));
    }

    Object& boolean(std::string_view key, bool value) {
        return raw(key, value ? "true" : "false");
    }

    Object& array(std::string_view key, const std::vector<double>& values) {
        std::ostringstream os;
        os.precision(17);
        os << '[';
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) os << ',';
            os << values[i];
        }
        os << ']';
        return raw(key, os.str());
    }

    /// Splice in an already-serialised fragment (a nested object, say).
    Object& raw(std::string_view key, std::string_view json_fragment) {
        if (!fields_.empty()) fields_ += ',';
        fields_ += '"';
        fields_ += escape(key);
        fields_ += "\":";
        fields_ += json_fragment;
        return *this;
    }

    [[nodiscard]] std::string str() const { return "{" + fields_ + "}"; }

private:
    std::string fields_;
};

}  // namespace veccore::json
