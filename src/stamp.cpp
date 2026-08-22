#include "veccore/stamp.hpp"

#include "veccore/build_info.hpp"
#include "veccore/json.hpp"

#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>
#include <thread>

namespace veccore {
namespace {

struct CmdResult {
    int rc = -1;
    std::string first_line;
};

/// Run a command, capturing stdout AND stderr, and return its first line.
///
/// Capturing stderr is deliberate.  The first version of this discarded it, so
/// when git refused to read the repository (safe.directory -- see L-07) the
/// stamp recorded `git_sha: "unknown"` and nothing else.  That is a true
/// statement that hides the entire diagnosis: git had printed exactly what was
/// wrong and exactly how to fix it, and we threw it away.
///
/// A missing stamp field must never take down a benchmark run -- but it must
/// never be silent about why it is missing either.
CmdResult run(const std::string& cmd) {
    CmdResult result;
    std::array<char, 512> buf{};
    FILE* pipe = ::popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) return result;
    if (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
        result.first_line = buf.data();
    }
    result.rc = ::pclose(pipe);
    while (!result.first_line.empty() &&
           (result.first_line.back() == '\n' || result.first_line.back() == '\r' ||
            result.first_line.back() == ' ')) {
        result.first_line.pop_back();
    }
    return result;
}

std::string read_cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (line.rfind("model name", 0) != 0) continue;
        std::string value = line.substr(colon + 1);
        const auto first = value.find_first_not_of(" \t");
        return first == std::string::npos ? std::string{} : value.substr(first);
    }
    return "unknown";
}

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    ::gmtime_r(&t, &tm);
    std::array<char, 32> buf{};
    std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf.data();
}

}  // namespace

bool EnvStamp::trusted() const { return untrusted_reason().empty(); }

std::string EnvStamp::untrusted_reason() const {
    if (build_type != "Release")   return "build type is " + build_type + ", not Release";
    if (sanitizers != "none")      return "built with sanitizers (" + sanitizers + ")";
    if (git_sha == "unknown") {
        return git_note.empty() ? "git SHA could not be resolved"
                                : "git SHA could not be resolved: " + git_note;
    }
    if (git_dirty)                 return "working tree is dirty";
    return {};
}

EnvStamp capture_env(const std::string& disk_path) {
    EnvStamp s;

    s.timestamp_utc = utc_now();

    std::array<char, 256> host{};
    s.hostname = (::gethostname(host.data(), host.size()) == 0) ? host.data() : "unknown";

    utsname uts{};
    s.kernel = (::uname(&uts) == 0) ? uts.release : "unknown";

    s.cpu_model   = read_cpu_model();
    s.hw_threads  = std::thread::hardware_concurrency();

    struct statvfs vfs {};
    if (::statvfs(disk_path.c_str(), &vfs) == 0) {
        const double bytes = static_cast<double>(vfs.f_bavail) * static_cast<double>(vfs.f_frsize);
        s.free_disk_gb = bytes / (1024.0 * 1024.0 * 1024.0);
    }

    // Provenance is resolved at *runtime*, not baked in at configure time -- a
    // SHA captured when you last ran cmake is a SHA that goes stale silently,
    // which is the exact failure D10 rule 7 exists to prevent.
    const std::string git = "git -C \"" VECCORE_SOURCE_DIR "\" ";
    const CmdResult sha = run(git + "rev-parse --short HEAD");
    if (sha.rc == 0 && !sha.first_line.empty()) {
        s.git_sha = sha.first_line;
        // Exclude results/ from the dirty check.  bench appends to
        // results/bench.jsonl, which dirtied the tree and made the NEXT bench
        // run refuse to write -- the harness invalidating its own trust
        // condition by doing its job (B-03).
        //
        // The claim `trusted` makes is "this binary was built from commit X".
        // A modified source file falsifies that.  An appended results file does
        // not: it is the output, not the input.
        const CmdResult status = run(git + "status --porcelain -- ':!results'");
        s.git_dirty = !status.first_line.empty();
    } else {
        s.git_note = sha.first_line;  // git already said what is wrong; keep it
    }

    s.build_type = VECCORE_BUILD_TYPE;
    s.compiler   = std::string(VECCORE_COMPILER_ID) + " " + VECCORE_COMPILER_VER;
    s.cxx_flags  = VECCORE_CXX_FLAGS;
    s.sanitizers = VECCORE_SANITIZERS;

    return s;
}

std::string to_json(const EnvStamp& s) {
    json::Object o;
    o.str("timestamp_utc", s.timestamp_utc)
     .str("hostname", s.hostname)
     .str("kernel", s.kernel)
     .str("cpu_model", s.cpu_model)
     .num("hw_threads", static_cast<long long>(s.hw_threads))
     .num("free_disk_gb", s.free_disk_gb)
     .str("git_sha", s.git_sha)
     .boolean("git_dirty", s.git_dirty)
     .str("git_note", s.git_note)
     .str("veccore_version", VECCORE_VERSION)
     .str("build_type", s.build_type)
     .str("compiler", s.compiler)
     .str("cxx_flags", s.cxx_flags)
     .str("sanitizers", s.sanitizers)
     .boolean("trusted", s.trusted())
     .str("untrusted_reason", s.untrusted_reason());
    return o.str();
}

}  // namespace veccore
