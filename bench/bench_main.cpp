// VecCore benchmark harness.
//
// 00_FOUNDATIONS.md section 4: "Before any feature work on any project, write
// bench.py. [...] This ordering is not optional -- it's what turns a hobby
// project into an engineering result."
//
// Phase 0 ships the skeleton: argument parsing, the environment stamp, the
// trusted-build gate, and incremental append to results/.  There is nothing to
// time yet.  Phase 1 adds the timing loop (warmup, percentiles, trials) and the
// first real measurement.

#include "veccore/json.hpp"
#include "veccore/stamp.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string tag = "phase0_rails";
    std::string out = "results/bench.jsonl";
    std::string data_dir = ".";
    bool allow_untrusted = false;
    bool help = false;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "bench: " << arg << " needs " << what << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--tag")                   a.tag = next("a value");
        else if (arg == "--out")              a.out = next("a path");
        else if (arg == "--data-dir")         a.data_dir = next("a path");
        else if (arg == "--allow-untrusted")  a.allow_untrusted = true;
        else if (arg == "-h" || arg == "--help") a.help = true;
        else {
            std::cerr << "bench: unknown argument '" << arg << "'\n";
            std::exit(2);
        }
    }
    return a;
}

void usage() {
    std::cout <<
        "usage: bench [options]\n"
        "  --tag NAME           label for this record (default: phase0_rails)\n"
        "  --out PATH           JSONL file to append to (default: results/bench.jsonl)\n"
        "  --data-dir PATH      filesystem whose free space is stamped\n"
        "  --allow-untrusted    write a record even from a Debug/sanitized/dirty build,\n"
        "                       stamping trusted=false and the reason\n";
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    if (args.help) {
        usage();
        return 0;
    }

    const veccore::EnvStamp env = veccore::capture_env(args.data_dir);

    std::cout << "veccore bench\n"
              << "  cpu        : " << env.cpu_model << " (" << env.hw_threads << " threads)\n"
              << "  kernel     : " << env.kernel << "\n"
              << "  build      : " << env.build_type << ", " << env.compiler << "\n"
              << "  flags      : " << env.cxx_flags << "\n"
              << "  sanitizers : " << env.sanitizers << "\n"
              << "  git        : " << env.git_sha << (env.git_dirty ? " (dirty)" : " (clean)") << "\n"
              << "  free disk  : " << env.free_disk_gb << " GB\n";

    // D10 / EdgeRAG's rule, adapted: the untrustworthy thing in this project is
    // not the machine, it is the *build*.  An ASan build runs several times
    // slower than Release, and a latency copied out of one is indistinguishable
    // from a real regression once it reaches a plot.
    if (!env.trusted()) {
        std::cout << "\n  NOT PUBLISHABLE: " << env.untrusted_reason() << "\n";
        if (!args.allow_untrusted) {
            std::cerr << "\nbench: refusing to write a record from an untrusted build.\n"
                      << "       Use a Release build with a clean tree, or pass --allow-untrusted\n"
                      << "       (which stamps trusted=false so the record can never be mistaken\n"
                      << "       for a publishable number).\n";
            return 1;
        }
        std::cout << "  --allow-untrusted given: writing anyway, stamped trusted=false.\n";
    }

    veccore::json::Object record;
    record.str("tag", args.tag)
          .str("phase", "0")
          .raw("env", veccore::to_json(env));

    // Phase 1 replaces this with real measurements: recall@k, QPS, p50/p95/p99,
    // peak RSS, build time -- over >= 5 interleaved trials.
    record.raw("measurements", "null")
          .str("note", "Phase 0 rails: harness exists, no measurements yet");

    std::ofstream out(args.out, std::ios::app);
    if (!out) {
        std::cerr << "bench: cannot open " << args.out << " for append\n";
        return 1;
    }
    // JSONL, appended incrementally.  00_FOUNDATIONS.md section 3 rule 5: a
    // disconnect at hour 3 must not cost hours 1 and 2.
    out << record.str() << '\n';
    if (!out.good()) {
        std::cerr << "bench: write to " << args.out << " failed\n";
        return 1;
    }

    std::cout << "\n  wrote 1 record to " << args.out << "\n";
    return 0;
}
