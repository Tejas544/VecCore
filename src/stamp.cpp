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

/// Run a command and return its first line of stdout, trimmed.  Empty on any
/// failure -- callers treat that as "unknown" rather than propagating an error,
/// because a missing stamp field must never take down a benchmark run.
std::string first_line_of(const char* cmd) {
    std::array<char, 512> buf{};
    FILE* pipe = ::popen(cmd, "r");
    if (!pipe) return {};
    std::string out;
    if (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
        out = buf.data();
    }
    const int rc = ::pclose(pipe);
    if (rc != 0) return {};
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
        out.pop_back();
    }
    return out;
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
    if (git_sha == "unknown")      return "git SHA could not be resolved";
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
    const std::string git_dir = " -C \"" VECCORE_SOURCE_DIR "\" ";
    const std::string sha = first_line_of(("git" + git_dir + "rev-parse --short HEAD 2>/dev/null").c_str());
    if (!sha.empty()) {
        s.git_sha = sha;
        const std::string status =
            first_line_of(("git" + git_dir + "status --porcelain 2>/dev/null | head -1").c_str());
        s.git_dirty = !status.empty();
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
