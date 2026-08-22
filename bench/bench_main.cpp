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
// Plus CONTEXT.md D10 rule 2, specific to this machine: interleave compared
// configurations rather than running them in blocks. This is a 6-core mobile
// CPU; run all of A then all of B and part of what you measured is how hot the
// laptop got.

#include "veccore/distance.hpp"
#include "veccore/flat_index.hpp"
#include "veccore/hnsw.hpp"
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
#include <sstream>
#include <string>
#include <vector>

using namespace veccore;

namespace {

struct Args {
    std::string tag = "flat";
    std::string index = "flat";          // flat | hnsw
    std::string out = "results/bench.jsonl";
    std::string base, query, truth;
    std::size_t k = 10;
    std::size_t n_queries = 0;
    std::size_t max_base = 0;
    int trials = 5;
    int warmup = 20;
    std::size_t M = 16;
    std::size_t ef_construction = 200;
    std::vector<std::size_t> ef_search{10, 20, 40, 80, 160, 320};
    std::uint64_t seed = 42;
    bool layout_ab = false;
    bool allow_untrusted = false;
    bool help = false;
};

void usage() {
    std::cout <<
        "usage: bench --base F.fvecs --query F.fvecs [--truth F.ivecs] [options]\n"
        "  --index flat|hnsw   which index (default: flat)\n"
        "  --tag NAME          label for the record\n"
        "  --out PATH          JSONL to append to\n"
        "  --k N               neighbours to retrieve (default: 10)\n"
        "  --queries N         use only the first N queries\n"
        "  --max-base N        index only the first N base vectors\n"
        "  --trials N          repeat the query set N times (default: 5)\n"
        "  --warmup N          queries discarded before timing (default: 20)\n"
        "  --M N               HNSW neighbours per node (default: 16)\n"
        "  --ef-construction N HNSW build beam width (default: 200)\n"
        "  --ef-search A,B,C   HNSW query beam widths to sweep (default: 10,20,40,80,160,320)\n"
        "  --seed N            RNG seed; stamped into every record (default: 42)\n"
        "  --layout-ab         also run the D5 flat-vs-pointer-chasing comparison\n"
        "  --allow-untrusted   write a record from a Debug/sanitized/dirty build\n";
}

std::vector<std::size_t> parse_list(const std::string& s) {
    std::vector<std::size_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoul(item));
    }
    return out;
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
        else if (arg == "--index")           a.index = next("flat|hnsw");
        else if (arg == "--out")             a.out = next("a path");
        else if (arg == "--base")            a.base = next("a path");
        else if (arg == "--query")           a.query = next("a path");
        else if (arg == "--truth")           a.truth = next("a path");
        else if (arg == "--k")               a.k = std::stoul(next("a number"));
        else if (arg == "--queries")         a.n_queries = std::stoul(next("a number"));
        else if (arg == "--max-base")        a.max_base = std::stoul(next("a number"));
        else if (arg == "--trials")          a.trials = std::stoi(next("a number"));
        else if (arg == "--warmup")          a.warmup = std::stoi(next("a number"));
        else if (arg == "--M")               a.M = std::stoul(next("a number"));
        else if (arg == "--ef-construction") a.ef_construction = std::stoul(next("a number"));
        else if (arg == "--ef-search")       a.ef_search = parse_list(next("a list"));
        else if (arg == "--seed")            a.seed = std::stoull(next("a number"));
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

struct RunResult {
    LatencyStats latency;
    double qps_mean = 0.0, qps_sd = 0.0, recall = 0.0;
    std::vector<double> qps_per_trial;
    bool has_recall = false;
};

/// One measured configuration. `search` returns the neighbours for a query.
template <typename SearchFn>
RunResult measure(SearchFn&& search, const VectorStore& queries, const IvecsData& truth,
                  std::size_t k, int trials, int warmup, bool has_truth) {
    // Rule 1. The first queries page in the base vectors and prime the branch
    // predictors; timing them measures the operating system, not the index.
    for (std::size_t i = 0; i < static_cast<std::size_t>(warmup) && i < queries.size(); ++i) {
        volatile auto sink = search(queries.at(static_cast<vec_id_t>(i))).size();
        (void)sink;
    }

    RunResult r;
    std::vector<double> latencies;
    latencies.reserve(queries.size() * static_cast<std::size_t>(trials));

    for (int t = 0; t < trials; ++t) {
        double hits = 0.0;
        const auto t_trial = Clock::now();
        for (vec_id_t q = 0; q < queries.size(); ++q) {
            const auto t0 = Clock::now();
            const std::vector<Neighbor> got = search(queries.at(q));
            latencies.push_back(ms_since(t0));
            if (has_truth) hits += recall_at_k(got, truth.row(q), k);
        }
        const double trial_ms = ms_since(t_trial);
        r.qps_per_trial.push_back(static_cast<double>(queries.size()) / (trial_ms / 1000.0));
        if (has_truth) r.recall += hits / static_cast<double>(queries.size());
    }

    r.latency = LatencyStats::from(latencies);
    r.qps_mean = mean_of(r.qps_per_trial);
    r.qps_sd = stddev(r.qps_per_trial);
    r.has_recall = has_truth;
    if (has_truth) r.recall /= static_cast<double>(trials);
    return r;
}

void print_run(const std::string& label, const RunResult& r, std::size_t k) {
    std::cout << std::fixed << std::setprecision(4)
              << "  " << std::setw(22) << std::left << label
              << " QPS " << std::setw(11) << std::right << r.qps_mean
              << " +/- " << std::setw(8) << r.qps_sd
              << "   p50 " << std::setw(8) << r.latency.p50
              << "  p99 " << std::setw(8) << r.latency.p99;
    if (r.has_recall) std::cout << "   recall@" << k << " " << r.recall;
    std::cout << "\n";
}

json::Object run_json(const RunResult& r) {
    json::Object m;
    m.num("qps_mean", r.qps_mean)
     .num("qps_stddev", r.qps_sd)
     .array("qps_per_trial", r.qps_per_trial)
     .num("latency_p50_ms", r.latency.p50)
     .num("latency_p95_ms", r.latency.p95)
     .num("latency_p99_ms", r.latency.p99)
     .num("latency_mean_ms", r.latency.mean)
     .num("latency_max_ms", r.latency.max);
    if (r.has_recall) m.num("recall_at_k", r.recall);
    else m.raw("recall_at_k", "null");
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    if (args.help || args.base.empty() || args.query.empty()) {
        usage();
        return args.help ? 0 : 2;
    }
    if (args.index != "flat" && args.index != "hnsw") {
        std::cerr << "bench: --index must be flat or hnsw\n";
        return 2;
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
                      << "       Use a Release build with a clean tree, or pass --allow-untrusted.\n";
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
    const bool has_truth = !args.truth.empty();
    if (has_truth) {
        // B-04. Ground truth is computed against a specific base set. Index a
        // subset and the published neighbours mostly are not in the index, so
        // recall collapses for a reason unrelated to the search -- while
        // reporting a number that looks like a result.
        const XvecsHeader full = xvecs_header(args.base);
        if (base.size() < full.count) {
            std::cerr << "bench: --max-base " << base.size() << " truncates a " << full.count
                      << "-vector base, but --truth is ground truth for the FULL base.\n"
                      << "       Recall against it would be meaningless. Drop one of them.\n";
            return 1;
        }
        truth = read_ivecs(args.truth, args.n_queries);
        if (truth.rows() < queries.size()) {
            std::cerr << "bench: ground truth has " << truth.rows() << " rows, "
                      << queries.size() << " queries\n";
            return 1;
        }
        if (truth.width < args.k) {
            std::cerr << "bench: ground truth is " << truth.width << " wide, k=" << args.k << "\n";
            return 1;
        }
    }

    std::cout << "\n  index      : " << args.index << "\n"
              << "  base       : " << base.size() << " x " << base.dim()
              << "  (" << base.bytes() / (1024.0 * 1024.0) << " MiB)\n"
              << "  queries    : " << queries.size() << "\n"
              << "  k          : " << args.k << "\n"
              << "  truth      : " << (has_truth ? args.truth : "none") << "\n"
              << "  load       : " << load_ms << " ms\n\n";

    std::ofstream out(args.out, std::ios::app);
    if (!out) { std::cerr << "bench: cannot open " << args.out << "\n"; return 1; }

    auto write_record = [&](const json::Object& measurements, const json::Object& index_params) {
        json::Object rec;
        rec.str("tag", args.tag)
           .str("index", args.index)
           .str("dataset_base", args.base)
           .str("dataset_query", args.query)
           .str("dataset_truth", args.truth)
           .num("n_base", static_cast<long long>(base.size()))
           .num("n_queries", static_cast<long long>(queries.size()))
           .num("dim", static_cast<long long>(base.dim()))
           .num("k", static_cast<long long>(args.k))
           .num("trials", static_cast<long long>(args.trials))
           .num("warmup", static_cast<long long>(args.warmup))
           .num("load_ms", load_ms)
           .num("peak_rss_mib", peak_rss_mib())
           .raw("index_params", index_params.str())
           .raw("env", to_json(env))
           .raw("measurements", measurements.str());
        out << rec.str() << '\n';
    };

    // ---- flat -------------------------------------------------------------
    if (args.index == "flat") {
        const FlatL2 index(base);
        const RunResult r = measure(
            [&](const float* q) { return index.search(q, args.k); },
            queries, truth, args.k, args.trials, args.warmup, has_truth);
        print_run("flat exact", r, args.k);
        std::cout << "  peak RSS   : " << peak_rss_mib() << " MiB  (vectors: "
                  << base.bytes() / (1024.0 * 1024.0) << " MiB)\n";

        json::Object p;
        p.str("kind", "flat_l2_exact").num("vector_bytes", static_cast<long long>(base.bytes()));
        json::Object m = run_json(r);
        m.num("index_bytes", static_cast<long long>(base.bytes()));
        write_record(m, p);
    }

    // ---- hnsw -------------------------------------------------------------
    if (args.index == "hnsw") {
        HnswParams hp;
        hp.M = args.M;
        hp.ef_construction = args.ef_construction;
        hp.seed = args.seed;

        HnswIndex index(base, hp);
        std::cout << "  building M=" << hp.M << " ef_construction=" << hp.ef_construction
                  << " seed=" << hp.seed << " ...\n";
        const auto t_build = Clock::now();
        index.build();
        const double build_ms = ms_since(t_build);

        const HnswStats s = index.stats();
        std::cout << "  built in   : " << build_ms / 1000.0 << " s\n"
                  << "  max level  : " << s.max_level << "\n"
                  << "  levels     : ";
        for (std::size_t l = 0; l < s.nodes_per_level.size(); ++l) {
            std::cout << s.nodes_per_level[l] << (l + 1 < s.nodes_per_level.size() ? " / " : "");
        }
        std::cout << "\n  mean deg L0: " << s.mean_degree_layer0 << "\n"
                  << "  graph bytes: " << index.graph_bytes() / (1024.0 * 1024.0) << " MiB"
                  << "  (vectors: " << base.bytes() / (1024.0 * 1024.0) << " MiB)\n\n";

        // The ef_search sweep. The index is built ONCE and swept -- rebuilding
        // between points would sweep two different graphs and produce a
        // non-monotonic curve that looks like a search bug (P-03).
        for (const std::size_t ef : args.ef_search) {
            index.reset_distance_calls();
            const RunResult r = measure(
                [&](const float* q) { return index.search(q, args.k, ef); },
                queries, truth, args.k, args.trials, args.warmup, has_truth);

            const double dist_per_query =
                static_cast<double>(index.distance_calls()) /
                static_cast<double>(queries.size() * static_cast<std::size_t>(args.trials) +
                                    static_cast<std::size_t>(args.warmup));

            print_run("ef_search=" + std::to_string(ef), r, args.k);

            json::Object p;
            p.str("kind", "hnsw")
             .num("M", static_cast<long long>(hp.M))
             .num("ef_construction", static_cast<long long>(hp.ef_construction))
             .num("ef_search", static_cast<long long>(ef))
             .num("seed", static_cast<long long>(hp.seed))
             .num("max_level", static_cast<long long>(s.max_level))
             .num("mean_degree_layer0", s.mean_degree_layer0)
             .num("graph_bytes", static_cast<long long>(index.graph_bytes()))
             .num("vector_bytes", static_cast<long long>(base.bytes()))
             .num("build_ms", build_ms);
            json::Object m = run_json(r);
            m.num("build_ms", build_ms)
             .num("distance_calls_per_query", dist_per_query)
             .num("index_bytes", static_cast<long long>(index.graph_bytes() + base.bytes()));
            write_record(m, p);
        }
        std::cout << "\n  peak RSS   : " << peak_rss_mib() << " MiB\n";
    }

    // ---- optional D5 layout A/B -------------------------------------------
    if (args.layout_ab) {
        NaiveVectorStore naive(base.dim());
        std::vector<std::vector<float>> ballast;
        ballast.reserve(base.size());
        for (vec_id_t i = 0; i < base.size(); ++i) {
            naive.add(base.at(i));
            ballast.emplace_back(base.dim() / 2);  // fragment the arena (B-05)
        }

        const std::size_t n_ab = std::min<std::size_t>(queries.size(), 100);
        const L2Sqr dist;
        std::vector<vec_id_t> order(base.size());
        std::iota(order.begin(), order.end(), 0u);
        std::vector<vec_id_t> shuffled = order;
        std::mt19937 rng(static_cast<std::mt19937::result_type>(args.seed));
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        auto scan = [&](bool flat, const std::vector<vec_id_t>& ord) {
            const auto t0 = Clock::now();
            for (std::size_t q = 0; q < n_ab; ++q) {
                TopK top(args.k);
                const float* qv = queries.at(static_cast<vec_id_t>(q));
                if (flat) for (const vec_id_t i : ord) top.offer(dist(qv, base.at(i), base.dim()), i);
                else      for (const vec_id_t i : ord) top.offer(dist(qv, naive.at(i), naive.dim()), i);
                volatile auto s = top.size(); (void)s;
            }
            return ms_since(t0);
        };

        std::vector<double> fs, ns, fr, nr;
        for (int round = 0; round < args.trials; ++round) {
            fs.push_back(scan(true, order));      // interleaved per round so
            ns.push_back(scan(false, order));     // thermal drift hits every arm
            fr.push_back(scan(true, shuffled));   // equally rather than
            nr.push_back(scan(false, shuffled));  // penalising whichever ran last
        }
        const auto qps = [&](const std::vector<double>& ms) {
            return static_cast<double>(n_ab) / (mean_of(ms) / 1000.0);
        };

        std::cout << "\n  layout A/B (D5), " << n_ab << " queries x " << args.trials
                  << " interleaved rounds, " << base.size() << " vectors:\n"
                  << "                      flat        naive(frag)   flat/naive\n"
                  << "    sequential : " << qps(fs) << "   " << qps(ns)
                  << "   " << qps(fs) / qps(ns) << "x\n"
                  << "    random     : " << qps(fr) << "   " << qps(nr)
                  << "   " << qps(fr) / qps(nr) << "x\n";

        json::Object p; p.str("kind", "layout_ab");
        json::Object m;
        m.num("ab_flat_seq_qps", qps(fs)).num("ab_naive_seq_qps", qps(ns))
         .num("ab_flat_rand_qps", qps(fr)).num("ab_naive_rand_qps", qps(nr))
         .num("ab_flat_bytes", static_cast<long long>(base.bytes()))
         .num("ab_naive_bytes", static_cast<long long>(naive.bytes()));
        write_record(m, p);
    }

    if (!out.good()) { std::cerr << "bench: write failed\n"; return 1; }
    std::cout << "\n  wrote records to " << args.out << "\n";
    return 0;
}
