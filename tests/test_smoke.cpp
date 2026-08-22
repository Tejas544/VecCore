#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "veccore/json.hpp"
#include "veccore/stamp.hpp"
#include "veccore/types.hpp"

#include <cstdint>
#include <limits>
#include <string>

using namespace veccore;

// Phase 0 has no algorithms in it, so these tests exist to prove the *rails*
// work: the build produces a linkable library, the test runner runs, and the
// stamping that every later benchmark record depends on actually populates.

TEST_CASE("json escaping survives the characters that break naive writers") {
    CHECK(json::escape("plain") == "plain");
    CHECK(json::escape("say \"hi\"") == "say \\\"hi\\\"");
    CHECK(json::escape("back\\slash") == "back\\\\slash");
    CHECK(json::escape("line\nbreak") == "line\\nbreak");
    CHECK(json::escape(std::string(1, '\x01')) == "\\u0001");
}

TEST_CASE("json object emits well-formed flat records") {
    json::Object o;
    o.str("name", "veccore").num("recall", 0.95).num("n", static_cast<long long>(1000000))
     .boolean("trusted", false).array("latencies_ms", {0.5, 1.25});

    const std::string s = o.str();
    CHECK(s.front() == '{');
    CHECK(s.back() == '}');
    CHECK(s.find("\"name\":\"veccore\"") != std::string::npos);
    CHECK(s.find("\"n\":1000000") != std::string::npos);
    CHECK(s.find("\"trusted\":false") != std::string::npos);
    CHECK(s.find("\"latencies_ms\":[0.5,1.25]") != std::string::npos);
}

TEST_CASE("environment stamp populates the fields D10 requires") {
    const EnvStamp s = capture_env(".");

    CHECK_FALSE(s.timestamp_utc.empty());
    CHECK_FALSE(s.hostname.empty());
    CHECK_FALSE(s.cpu_model.empty());
    CHECK(s.hw_threads > 0);
    CHECK_FALSE(s.build_type.empty());
    CHECK_FALSE(s.compiler.empty());

    // L-05: the guard that stops a disk-full failure from masquerading as a
    // performance regression only works if the number is actually read.
    CHECK(s.free_disk_gb > 0.0);

    const std::string j = to_json(s);
    CHECK(j.find("\"trusted\":") != std::string::npos);
    CHECK(j.find("\"git_sha\":") != std::string::npos);
}

TEST_CASE("a sanitizer build is never considered publishable") {
    EnvStamp s;
    s.build_type = "Debug";
    s.sanitizers = "address,undefined";
    s.git_sha = "abc1234";
    s.git_dirty = false;
    CHECK_FALSE(s.trusted());
    CHECK_FALSE(s.untrusted_reason().empty());

    s.build_type = "Release";
    CHECK_FALSE(s.trusted());  // still sanitized

    s.sanitizers = "none";
    CHECK(s.trusted());

    s.git_dirty = true;
    CHECK_FALSE(s.trusted());
}

TEST_CASE("offset arithmetic is wide enough for the scale we claim (P-14)") {
    // 1M x 128 is 1.28e8 -- fine in int32.  The point of offset_t is the case
    // beyond it, where int32 silently wraps into memory corruption.
    const offset_t n = 40'000'000;
    const offset_t d = 128;
    const offset_t last = (n - 1) * d;
    CHECK(last > static_cast<offset_t>(std::numeric_limits<std::int32_t>::max()));
    CHECK(last == 5'119'999'872ULL);
}
