#pragma once

// Environment capture for benchmark records.  CONTEXT.md D10:
//
//   "Every JSON record is stamped with git SHA, CPU model, compiler version,
//    flags, thread count, dataset, RNG seed, and every index parameter.
//    A record missing those does not go in a plot."
//
// The `trusted` flag is the load-bearing part.  It is EdgeRAG's
// "no perf number from an untrusted device" rule, translated to a project where
// the CPU *is* the device: here the untrustworthy thing is not the machine but
// the *build*.  An ASan build runs several times slower than Release, and a
// latency copied out of one would be indistinguishable from a real regression.

#include <string>

namespace veccore {

struct EnvStamp {
    // Runtime
    std::string timestamp_utc;
    std::string hostname;
    std::string kernel;        ///< uname -r; "microsoft-standard-WSL2" is expected here
    std::string cpu_model;
    unsigned    hw_threads = 0;
    double      free_disk_gb = 0.0;  ///< L-05: a sweep that dies because the disk
                                     ///< filled is otherwise indistinguishable
                                     ///< from a code regression

    // Provenance
    std::string git_sha = "unknown";
    bool        git_dirty = true;

    // Compile-time (from the generated build_info.hpp)
    std::string build_type;
    std::string compiler;
    std::string cxx_flags;
    std::string sanitizers;

    /// True only when this build is fit to publish a performance number from:
    /// Release, no sanitizers, a known git SHA, and a clean working tree.
    /// bench refuses to write a timing record when this is false unless
    /// --allow-untrusted is passed, and then it stamps the reason.
    [[nodiscard]] bool trusted() const;

    /// Human-readable reason `trusted()` is false; empty when it is true.
    [[nodiscard]] std::string untrusted_reason() const;
};

/// Capture the current environment.  `disk_path` selects the filesystem whose
/// free space is recorded -- pass the directory the dataset lives in.
EnvStamp capture_env(const std::string& disk_path = ".");

/// Serialise to a JSON fragment (an object, no trailing comma).
std::string to_json(const EnvStamp& stamp);

}  // namespace veccore
