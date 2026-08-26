// Phase 5 gate, and the direct counterpart of tools/asan_probe.cpp.
//
// Two jobs, both necessary:
//
//   1. Prove ThreadSanitizer is actually running. L-01's lesson was that a
//      sanitizer you never saw fire is a sanitizer you cannot rely on. A clean
//      TSan run over the concurrency tests means nothing unless TSan can be
//      shown to detect a race on this build.
//
//   2. Prove B-10 was a real race rather than a reasoned-about one. The
//      unsafe `search(query, k, ef)` overload writes to a shared member
//      (`scratch_`) despite being `const`. Calling it from several threads is
//      exactly the mistake the SearchScratch refactor exists to prevent, so
//      running it here converts "I argued this was a data race" into "TSan
//      says this is a data race, here are the two stacks."
//
// This program is EXPECTED to produce a TSan report. It is never linked into
// the library or the test suite.

#include "veccore/hnsw.hpp"
#include "veccore/storage.hpp"

#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using namespace veccore;

int main() {
    std::puts("tsan_probe: calling the NON-thread-safe search overload from 4 threads.");
    std::puts("EXPECTED: a ThreadSanitizer data-race report on HnswIndex::scratch_.");
    std::puts("If this exits silently, TSan is NOT active and the Phase 5 gate is meaningless.");
    std::fflush(stdout);

    std::mt19937_64 rng(1);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    VectorStore store(8);
    std::vector<float> v(8);
    for (int i = 0; i < 2000; ++i) {
        for (auto& x : v) x = u(rng);
        store.add(v.data());
    }

    HnswParams p;
    p.M = 8;
    p.ef_construction = 50;
    HnswIndex index(store, p);
    index.build();

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            for (int i = 0; i < 200; ++i) {
                // The deliberate mistake: the overload without a SearchScratch
                // argument, which writes to a member shared by every caller.
                const auto hits = index.search(store.at(static_cast<vec_id_t>(i)), 10, 32);
                (void)hits;
            }
        });
    }
    for (auto& th : threads) th.join();

    std::puts("probe finished -- if no TSan report appeared above, TSan IS NOT ACTIVE");
    return 0;
}
