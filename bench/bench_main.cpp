// VecCore benchmark harness.
//
// 00_FOUNDATIONS.md section 4, applied literally:
//   1. Warmup -- discard the first iterations.
//   2. Percentiles, never a mean alone.
//   3. >= 5 trials, report the spread.
//   4. Fix everything you are not measuring, and say what was held constant.
//   5. Always have a baseline.
//   6. Report peak memory.
//
// Plus CONTEXT.md D10 rule 2, which is specific to this machine: interleave
// compared configurations A/B/A/B rather than running them in blocks. This is a
// 6-core mobile CPU; run all of A then all of B and part of what you measured
// is how hot the laptop got.

#include "veccore/distance.hpp"
#include "veccore/flat_index.hpp"
#include "veccore/json.hpp"
#include "veccore/metrics.hpp"
#include "veccore/stamp.hpp"
#include "veccore/storage.hpp"
#include "veccore/xvecs.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace veccore;

namespace {

struct Args {
    std::string tag = "flat";
    std::string out = "results/bench.jsonl";
    std::string base;
    std::string query;
    std::string truth;
    std::size_t k = 10;
    std::size_t n_queries = 0;    // 0 = all
    std::size_t max_base = 0;     // 0 = all
    int trials = 5;
    int warmup = 20;
    bool layout_ab = false;
    bool allow_untrusted = false;
    bool help = false;
};

void usage() {
    std::cout <<
        "usage: bench --base F.fvecs --query F.fvecs [--truth F.ivecs] [options]\n"
        "  --tag NAME          label for the record (default: flat)\n"
        "  --out PATH          JSONL to append to (default: results/bench.jsonl)\n"
        "  --k N               neighbours to retrieve (default: 10)\n"
        "  --queries N         use only the first N queries (default: all)\n"
        "  --max-base N        index only the first N base vectors (default: all)\n"
        "  --trials N          repeat the full query set N times (default: 5)\n"
        "  --warmup N          queries discarded before timing starts (default: 20)\n"
        "  --layout-ab         also time the naive vector<vector<float>> layout (D5)\n"
        "  --allow-untrusted   write a record from a Debug/sanitized/dirty build\n";
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::cerr << "bench: " << arg << " needs " << what << "\n"; std::exit(2); }
            return argv[++i];
        };
        if      (arg == "--tag")             a.tag = next("a value");
        else if (arg == "--out")             a.out = next("a path");
        else if (arg == "--base")            a.base = next("a path");
        else if (arg == "--query")           a.query = next("a path");
        else if (arg == "--truth")           a.truth = next("a path");
        else if (arg == "--k")               a.k = std::stoul(next("a number"));
        else if (arg == "--queries")         a.n_queries = std::stoul(next("a number"));
        else if (arg == "--max-base")        a.max_base = std::stoul(next("a number"));
        else if (arg == "--trials")          a.trials = std::stoi(next("a number"));
        else if (arg == "--warmup")          a.warmup = std::stoi(next("a number"));
        else if (arg == "--layout-ab")       a.layout_ab = true;
        else if (arg == "--allow-untrusted") a.allow_untrusted = true;
        else if (arg == "-h" || arg == "--help") a.help = true;
        else { std::cerr << "bench: unknown argument '" << arg << "'\n"; std::exit(2); }
    }
    return a;
}

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    if (args.help || args.base.empty() || args.query.empty()) {
        usage();
        return args.help ? 0 : 2;
    }

    const EnvStamp env = capture_env(".");
    std::cout << "veccore bench\n"
              << "  cpu        : " << env.cpu_model << " (" << env.hw_threads << " threads)\n"
              << "  build      : " << env.build_type << ", " << env.compiler << "\n"
              << "  flags      : " << env.cxx_flags << "\n"
              << "  git        : " << env.git_sha << (env.git_dirty ? " (dirty)" : " (clean)") << "\n";

    if (!env.trusted()) {
        std::cout << "\n  NOT PUBLISHABLE: " << env.untrusted_reason() << "\n";
        if (!args.allow_untrusted) {
            std::cerr << "\nbench: refusing to write a record from an untrusted build.\n"
                      << "       Use a Release build with a clean tree, or pass --allow-untrusted\n"
                      << "       (which stamps trusted=false into the record).\n";
            return 1;
        }
        std::cout << "  --allow-untrusted given: continuing, stamped trusted=false.\n";
    }

    // ---- load -------------------------------------------------------------
    const auto t_load = Clock::now();
    VectorStore base = read_fvecs(args.base, args.max_base);
    VectorStore queries = read_fvecs(args.query, args.n_queries);
    const double load_ms = ms_since(t_load);

    if (base.dim() != queries.dim()) {
        std::cerr << "bench: dimension mismatch -- base " << base.dim()
                  << " vs query " << queries.dim() << "\n";
        return 1;
    }

    IvecsData truth;
    if (!args.truth.empty()) {
        // B-04. Ground truth is computed against a specific base set. Index a
        // subset of it and the published neighbours mostly are not in the
        // index, so recall collapses for a reason that has nothing to do with
        // the search -- and it reports a number that looks like a result.
        // 00_FOUNDATIONS.md section 4: a benchmark that measures nothing is
        // worse than no benchmark, because someone will quote it.
        const XvecsHeader full = xvecs_header(args.base);
        if (base.size() < full.count) {
            std::cerr << "bench: --max-base " << base.size() << " truncates a "
                      << full.count << "-vector base, but --truth is ground truth for the\n"
                      << "       FULL base. Recall against it would be meaningless.\n"
                      << "       Either drop --max-base, or drop --truth and measure speed only.\n";
            return 1;
        }
        truth = read_ivecs(args.truth, args.n_queries);
        if (truth.rows() < queries.size()) {
            std::cerr << "bench: ground truth has " << truth.rows() << " rows but there are "
                      << queries.size() << " queries\n";
            return 1;
        }
        if (truth.width < args.k) {
            std::cerr << "bench: ground truth is " << truth.width << " wide, k=" << args.k << "\n";
            return 1;
        }
    }

    std::cout << "\n  base       : " << base.size() << " x " << base.dim()
              << "  (" << base.bytes() / (1024.0 * 1024.0) << " MiB)\n"
              << "  queries    : " << queries.size() << "\n"
              << "  k          : " << args.k << "\n"
              << "  truth      : " << (args.truth.empty() ? "none" : args.truth) << "\n"
              << "  load       : " << load_ms << " ms\n\n";

    const FlatL2 index(base);

    // ---- warmup -----------------------------------------------------------
    // Rule 1. The first queries page in 512 MB of base vectors from the page
    // cache and prime the branch predictors; timing them measures the operating
    // system, not the index.
    const auto warm = static_cast<std::size_t>(args.warmup);
    for (std::size_t i = 0; i < warm && i < queries.size(); ++i) {
        volatile auto sink = index.search(queries.at(static_cast<vec_id_t>(i)), args.k).size();
        (void)sink;
    }

    // ---- measure ----------------------------------------------------------
    std::vector<double> all_latencies_ms;
    std::vector<double> qps_per_trial;
    std::vector<double> recall_per_trial;
    all_latencies_ms.reserve(queries.size() * static_cast<std::size_t>(args.trials));

    for (int trial = 0; trial < args.trials; ++trial) {
        double hits = 0.0;
        const auto t_trial = Clock::now();

        for (vec_id_t q = 0; q < queries.size(); ++q) {
            const auto t0 = Clock::now();
            const std::vector<Neighbor> got = index.search(queries.at(q), args.k);
            all_latencies_ms.push_back(ms_since(t0));

            if (!args.truth.empty()) {
                hits += recall_at_k(got, truth.row(q), args.k);
            }
        }

        const double trial_ms = ms_since(t_trial);
        qps_per_trial.push_back(static_cast<double>(queries.size()) / (trial_ms / 1000.0));
        if (!args.truth.empty()) {
            recall_per_trial.push_back(hits / static_cast<double>(queries.size()));
        }
        std::cout << "  trial " << (trial + 1) << "/" << args.trials
                  << "  " << qps_per_trial.back() << " QPS";
        if (!recall_per_trial.empty()) std::cout << "  recall@" << args.k << " " << recall_per_trial.back();
        std::cout << "\n";
    }

    const LatencyStats lat = LatencyStats::from(all_latencies_ms);
    const double qps_mean = mean_of(qps_per_trial);
    const double qps_sd = stddev(qps_per_trial);

    std::cout << std::fixed << std::setprecision(4)
              << "\n  QPS        : " << qps_mean << " +/- " << qps_sd
              << "  (" << (qps_sd / (qps_mean ? qps_mean : 1.0) * 100.0) << "% rsd, "
              << args.trials << " trials)\n"
              << "  latency ms : p50 " << lat.p50 << "  p95 " << lat.p95 << "  p99 " << lat.p99
              << "  max " << lat.max << "\n";
    if (!recall_per_trial.empty()) {
        std::cout << "  recall@" << args.k << "   : " << mean_of(recall_per_trial) << "\n";
    }
    std::cout << "  peak RSS   : " << peak_rss_mib() << " MiB"
              << "   (index data alone: " << base.bytes() / (1024.0 * 1024.0) << " MiB)\n";

    // ---- optional D5 A/B: flat vs pointer-chasing --------------------------
    // Interleaved, per D10 rule 2 -- alternating flat/naive/flat/naive so
    // thermal drift hits both arms equally instead of penalising whichever ran
    // second.
    double ab_flat_seq = 0.0, ab_naive_seq = 0.0, ab_flat_rand = 0.0, ab_naive_rand = 0.0;
    offset_t naive_bytes = 0;
    if (args.layout_ab) {
        // The naive store is built with junk allocations interleaved between
        // the rows. Without that, malloc hands out 200k identically-sized
        // blocks consecutively and the "pointer-chasing" layout is contiguous
        // in practice -- which is why the first version of this benchmark
        // measured no difference at all (B-05). Real heaps are not built by a
        // single uninterrupted loop.
        NaiveVectorStore naive(base.dim());
        std::vector<std::vector<float>> ballast;
        ballast.reserve(base.size());
        for (vec_id_t i = 0; i < base.size(); ++i) {
            naive.add(base.at(i));
            ballast.emplace_back(base.dim() / 2);  // fragment the arena
        }
        naive_bytes = naive.bytes();

        const std::size_t n_ab = std::min<std::size_t>(queries.size(), 100);
        const L2Sqr dist;

        // Sequential order vs a fixed shuffle. This is the comparison that
        // actually matters for Phase 2: a brute-force scan walks ids 0..n in
        // order, and the hardware prefetcher handles that well for BOTH
        // layouts. HNSW does not -- graph traversal visits neighbours in an
        // order the prefetcher cannot predict, and that is where a dependent
        // pointer load costs a full DRAM round trip.
        std::vector<vec_id_t> order(base.size());
        std::iota(order.begin(), order.end(), 0u);
        std::vector<vec_id_t> shuffled = order;
        std::mt19937 rng(42);  // seeded: D10, the sweep must be reproducible
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        auto scan_flat = [&](const std::vector<vec_id_t>& ord) {
            const auto t0 = Clock::now();
            for (std::size_t q = 0; q < n_ab; ++q) {
                TopK top(args.k);
                const float* qv = queries.at(static_cast<vec_id_t>(q));
                for (const vec_id_t i : ord) top.offer(dist(qv, base.at(i), base.dim()), i);
                volatile auto s = top.size(); (void)s;
            }
            return ms_since(t0);
        };
        auto scan_naive = [&](const std::vector<vec_id_t>& ord) {
            const auto t0 = Clock::now();
            for (std::size_t q = 0; q < n_ab; ++q) {
                TopK top(args.k);
                const float* qv = queries.at(static_cast<vec_id_t>(q));
                for (const vec_id_t i : ord) top.offer(dist(qv, naive.at(i), naive.dim()), i);
                volatile auto s = top.size(); (void)s;
            }
            return ms_since(t0);
        };

        std::vector<double> fs, ns, fr, nr;
        for (int round = 0; round < args.trials; ++round) {
            fs.push_back(scan_flat(order));       // interleaved A/B/C/D per round,
            ns.push_back(scan_naive(order));      // so thermal drift hits every arm
            fr.push_back(scan_flat(shuffled));    // equally rather than penalising
            nr.push_back(scan_naive(shuffled));   // whichever ran last (D10 rule 2)
        }

        const auto qps = [&](const std::vector<double>& ms) {
            return static_cast<double>(n_ab) / (mean_of(ms) / 1000.0);
        };
        ab_flat_seq   = qps(fs);
        ab_naive_seq  = qps(ns);
        ab_flat_rand  = qps(fr);
        ab_naive_rand = qps(nr);

        std::cout << "\n  layout A/B (D5), " << n_ab << " queries x " << args.trials
                  << " interleaved rounds, " << base.size() << " vectors:\n"
                  << "                      flat        naive(frag)   flat/naive\n"
                  << "    sequential : " << ab_flat_seq << "   " << ab_naive_seq
                  << "   " << (ab_naive_seq > 0 ? ab_flat_seq / ab_naive_seq : 0.0) << "x\n"
                  << "    random     : " << ab_flat_rand << "   " << ab_naive_rand
                  << "   " << (ab_naive_rand > 0 ? ab_flat_rand / ab_naive_rand : 0.0) << "x\n"
                  << "    memory     : " << base.bytes() / (1024.0 * 1024.0) << " MiB   "
                  << naive_bytes / (1024.0 * 1024.0) << " MiB\n";
    }

    // ---- record -----------------------------------------------------------
    json::Object m;
    m.num("k", static_cast<long long>(args.k))
     .num("n_base", static_cast<long long>(base.size()))
     .num("n_queries", static_cast<long long>(queries.size()))
     .num("dim", static_cast<long long>(base.dim()))
     .num("trials", static_cast<long long>(args.trials))
     .num("warmup", static_cast<long long>(args.warmup))
     .num("qps_mean", qps_mean)
     .num("qps_stddev", qps_sd)
     .array("qps_per_trial", qps_per_trial)
     .num("latency_p50_ms", lat.p50)
     .num("latency_p95_ms", lat.p95)
     .num("latency_p99_ms", lat.p99)
     .num("latency_mean_ms", lat.mean)
     .num("latency_max_ms", lat.max)
     .num("index_bytes", static_cast<long long>(base.bytes()))
     .num("peak_rss_mib", peak_rss_mib())
     .num("load_ms", load_ms);

    if (!recall_per_trial.empty()) {
        m.num("recall_at_k", mean_of(recall_per_trial));
    } else {
        m.raw("recall_at_k", "null");
    }
    if (args.layout_ab) {
        m.num("ab_flat_seq_qps", ab_flat_seq)
         .num("ab_naive_seq_qps", ab_naive_seq)
         .num("ab_flat_rand_qps", ab_flat_rand)
         .num("ab_naive_rand_qps", ab_naive_rand)
         .num("ab_naive_bytes", static_cast<long long>(naive_bytes));
    }

    json::Object record;
    record.str("tag", args.tag)
          .str("phase", "1")
          .str("index", "flat_l2_exact")
          .str("dataset_base", args.base)
          .str("dataset_query", args.query)
          .str("dataset_truth", args.truth)
          .raw("env", to_json(env))
          .raw("measurements", m.str());

    std::ofstream out(args.out, std::ios::app);
    if (!out) { std::cerr << "bench: cannot open " << args.out << "\n"; return 1; }
    out << record.str() << '\n';
    if (!out.good()) { std::cerr << "bench: write failed\n"; return 1; }

    std::cout << "\n  wrote 1 record to " << args.out << "\n";
    return 0;
}
