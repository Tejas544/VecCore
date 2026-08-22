#pragma once

// Recall, latency percentiles, and peak memory.
//
// 00_FOUNDATIONS.md section 4 is the spec for this file, and it is blunt about
// why: "Bad benchmarking is the fastest way to lose an interviewer's trust, and
// it is trivially detectable."

#include "veccore/flat_index.hpp"
#include "veccore/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace veccore {

/// Fraction of the true top-k that we returned.
///
/// Set intersection, not positional comparison: getting the right ten in the
/// wrong order is recall 1.0.  Ordering quality is a different metric (nDCG,
/// MRR) and conflating them is a way to report a number nobody else's number
/// can be compared to.
[[nodiscard]] inline double recall_at_k(const std::vector<Neighbor>& returned,
                                        const std::int32_t* truth,
                                        std::size_t k) {
    if (k == 0) return 1.0;
    std::size_t hits = 0;
    const std::size_t limit = std::min(k, returned.size());
    for (std::size_t i = 0; i < k; ++i) {
        const auto want = static_cast<vec_id_t>(truth[i]);
        for (std::size_t j = 0; j < limit; ++j) {
            if (returned[j].id == want) { ++hits; break; }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

/// Nearest-rank percentile on an already-sorted sample.
///
/// 00_FOUNDATIONS.md section 4 rule 3: report percentiles, never a mean alone.
/// A mean hides exactly the tail behaviour that matters in serving -- and it is
/// the tail an interviewer asks about.
[[nodiscard]] inline double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const double rank = p / 100.0 * static_cast<double>(sorted.size());
    auto idx = static_cast<std::size_t>(std::ceil(rank));
    if (idx == 0) idx = 1;
    if (idx > sorted.size()) idx = sorted.size();
    return sorted[idx - 1];
}

struct LatencyStats {
    double p50 = 0.0, p95 = 0.0, p99 = 0.0;
    double mean = 0.0, min = 0.0, max = 0.0;
    std::size_t count = 0;

    /// Sorts its input.  Takes by value to make that obvious at the call site
    /// rather than surprising the caller with a reordered sample.
    [[nodiscard]] static LatencyStats from(std::vector<double> samples) {
        LatencyStats s;
        if (samples.empty()) return s;
        std::sort(samples.begin(), samples.end());
        s.count = samples.size();
        s.min = samples.front();
        s.max = samples.back();
        s.p50 = percentile(samples, 50.0);
        s.p95 = percentile(samples, 95.0);
        s.p99 = percentile(samples, 99.0);
        double sum = 0.0;
        for (const double v : samples) sum += v;
        s.mean = sum / static_cast<double>(samples.size());
        return s;
    }
};

/// Sample standard deviation (n-1).  00_FOUNDATIONS.md section 4 rule 4:
/// "A single number is not a result."
[[nodiscard]] inline double stddev(const std::vector<double>& xs) {
    if (xs.size() < 2) return 0.0;
    double sum = 0.0;
    for (const double x : xs) sum += x;
    const double mean = sum / static_cast<double>(xs.size());
    double acc = 0.0;
    for (const double x : xs) acc += (x - mean) * (x - mean);
    return std::sqrt(acc / static_cast<double>(xs.size() - 1));
}

[[nodiscard]] inline double mean_of(const std::vector<double>& xs) {
    if (xs.empty()) return 0.0;
    double sum = 0.0;
    for (const double x : xs) sum += x;
    return sum / static_cast<double>(xs.size());
}

/// Peak resident set size in MiB, from /proc/self/status VmHWM.
///
/// P-13: this is the WHOLE PROCESS -- dataset, ground truth, allocator slack,
/// everything.  It is reported alongside, never instead of, the index's own
/// computed byte count.  Quoting RSS as "index memory" inflates the number and
/// makes the FAISS memory comparison meaningless.
///
/// VmHWM is a high-water mark the kernel does not let userspace reset, so
/// within one process it only ever rises.  Comparing two configurations by
/// peak RSS therefore requires two separate process invocations, which is why
/// bench takes one configuration per run.
[[nodiscard]] inline double peak_rss_mib() {
    std::ifstream f("/proc/self/status");
    std::string key;
    while (f >> key) {
        if (key == "VmHWM:") {
            double kb = 0.0;
            f >> kb;
            return kb / 1024.0;
        }
        f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0.0;
}

}  // namespace veccore
