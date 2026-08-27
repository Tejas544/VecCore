#include <doctest/doctest.h>

#include "veccore/concurrent.hpp"
#include "veccore/flat_index.hpp"
#include "veccore/hnsw.hpp"
#include "veccore/storage.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <thread>
#include <vector>

using namespace veccore;

namespace {

VectorStore random_store(std::size_t n, dim_t d, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    VectorStore s(d);
    s.reserve(n);
    std::vector<float> v(d);
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& x : v) x = u(rng);
        s.add(v.data());
    }
    return s;
}

}  // namespace

TEST_CASE("a scratch search returns exactly what the single-threaded one does") {
    // The refactor that removed shared mutable state (B-10) must not have
    // changed any answer. Same index, same query, both overloads.
    const VectorStore store = random_store(2000, 16, 101);
    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    HnswIndex index(store, p);
    index.build();

    SearchScratch scratch = index.make_scratch();
    const VectorStore queries = random_store(30, 16, 202);
    for (vec_id_t q = 0; q < queries.size(); ++q) {
        const auto a = index.search(queries.at(q), 10, 64);
        const auto b = index.search(queries.at(q), 10, 64, scratch);
        REQUIRE(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i].id == b[i].id);
            CHECK(a[i].dist == doctest::Approx(b[i].dist));
        }
    }
    CHECK(scratch.distance_calls > 0);
}

TEST_CASE("concurrent readers agree with the single-threaded answer") {
    // The correctness half of Phase 5. Not a timing test -- it asserts that
    // running the same queries from 8 threads produces bit-identical results to
    // running them one at a time. A shared visited list would fail this
    // nondeterministically, which is the worst possible failure mode.
    const VectorStore store = random_store(3000, 16, 303);
    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    ConcurrentHnsw index(store, p);
    index.build();

    const VectorStore queries = random_store(200, 16, 404);

    std::vector<std::vector<Neighbor>> expected(queries.size());
    {
        SearchScratch s = index.make_scratch();
        for (vec_id_t q = 0; q < queries.size(); ++q) {
            expected[q] = index.search(queries.at(q), 10, 64, s);
        }
    }

    constexpr int kThreads = 8;
    std::vector<std::vector<std::vector<Neighbor>>> got(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            SearchScratch scratch = index.make_scratch();  // per thread, always
            got[t].resize(queries.size());
            for (vec_id_t q = 0; q < queries.size(); ++q) {
                got[t][q] = index.search(queries.at(q), 10, 64, scratch);
            }
        });
    }
    for (auto& th : threads) th.join();

    for (int t = 0; t < kThreads; ++t) {
        for (std::size_t q = 0; q < queries.size(); ++q) {
            REQUIRE(got[t][q].size() == expected[q].size());
            for (std::size_t i = 0; i < expected[q].size(); ++i) {
                CHECK(got[t][q][i].id == expected[q][i].id);
            }
        }
    }
}

TEST_CASE("readers see a consistent index while an insert is in flight") {
    // What the lock actually buys. Readers run throughout; a writer inserts
    // concurrently. Every result must be a valid id, every distance finite, and
    // the run must terminate. Under TSan this is also the race detector's
    // workload.
    //
    // **WriterPriority is not optional here.** The first version of this test
    // used the default shared_mutex and hung for ten minutes at 400% CPU: four
    // continuous readers starve the writer completely on glibc's
    // reader-preferring rwlock (B-11). The test did not "run slowly" -- it was
    // never going to finish.
    VectorStore store(8);
    std::mt19937_64 rng(505);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(8);
    for (int i = 0; i < 4000; ++i) {
        for (auto& x : v) x = u(rng);
        store.add(v.data());
    }

    HnswParams p;
    p.M = 8;
    p.ef_construction = 50;
    ConcurrentHnsw index(store, p, LockMode::WriterPriority);

    // Seed with half the corpus, then insert the rest while readers run.
    for (vec_id_t i = 0; i < 2000; ++i) index.insert(i);

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> searches{0};
    std::atomic<bool> bad{false};

    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t] {
            SearchScratch scratch = index.make_scratch();
            std::mt19937_64 local(900 + static_cast<std::uint64_t>(t));
            std::uniform_int_distribution<std::size_t> pick(0, 1999);
            while (!stop.load(std::memory_order_relaxed)) {
                const auto hits = index.search(store.at(static_cast<vec_id_t>(pick(local))), 10, 32, scratch);
                for (const Neighbor& n : hits) {
                    if (n.id >= store.size() || !std::isfinite(n.dist)) bad.store(true);
                }
                searches.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (vec_id_t i = 2000; i < 3000; ++i) index.insert(i);
    stop.store(true);
    for (auto& th : readers) th.join();

    CHECK_FALSE(bad.load());
    CHECK(searches.load() > 0);
    CHECK(index.index().check_invariants() == "");
}

TEST_CASE("incremental insert reaches the same recall as a single build") {
    // Inserting one at a time must not produce a worse graph than build(),
    // because build() *is* a loop of inserts. If these diverge, insert() is
    // carrying state it should not.
    const VectorStore store = random_store(2000, 16, 606);
    const VectorStore queries = random_store(50, 16, 707);
    const FlatL2 exact(store);

    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    p.seed = 999;

    HnswIndex built(store, p);
    built.build();

    HnswIndex incremental(store, p);
    for (vec_id_t i = 0; i < store.size(); ++i) incremental.insert(i);

    auto recall = [&](const HnswIndex& idx) {
        double total = 0.0;
        for (vec_id_t q = 0; q < queries.size(); ++q) {
            const auto truth = exact.search(queries.at(q), 10);
            const auto got = idx.search(queries.at(q), 10, 64);
            std::set<vec_id_t> t;
            for (const auto& n : truth) t.insert(n.id);
            std::size_t hits = 0;
            for (const auto& n : got) if (t.count(n.id)) ++hits;
            total += hits / 10.0;
        }
        return total / static_cast<double>(queries.size());
    };

    CHECK(recall(incremental) == doctest::Approx(recall(built)));
}

TEST_CASE("writer priority bounds insert latency under continuous read load (B-11)") {
    // The regression test for the starvation. With the default shared_mutex,
    // 4 continuous readers prevent a writer from ever acquiring the lock on
    // glibc. With the turnstile, the writer gets in.
    //
    // The assertion is deliberately loose -- this is a timing test on a laptop,
    // and a tight bound would be flaky by construction. "Completes at all,
    // comfortably inside a generous budget" is the property that matters;
    // the *quantified* comparison lives in bench, not here.
    const VectorStore store = random_store(3000, 8, 1212);
    HnswParams p;
    p.M = 8;
    p.ef_construction = 50;
    ConcurrentHnsw index(store, p, LockMode::WriterPriority);
    for (vec_id_t i = 0; i < 1500; ++i) index.insert(i);

    std::atomic<bool> stop{false};
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&] {
            SearchScratch scratch = index.make_scratch();
            while (!stop.load(std::memory_order_relaxed)) {
                (void)index.search(store.at(0), 10, 32, scratch);
            }
        });
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (vec_id_t i = 1500; i < 1700; ++i) index.insert(i);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    stop.store(true);
    for (auto& th : readers) th.join();

    // Under SharedMutex this same loop does not finish in 30 seconds.
    CHECK(elapsed_ms < 20000.0);
}

TEST_CASE("the unsafe lock mode returns the same answers read-only") {
    // LockMode::None exists only for the P-30 control experiment. It must be a
    // pure lock elision -- identical results, no other behaviour change --
    // or the control measures something other than the lock.
    const VectorStore store = random_store(1500, 8, 808);
    HnswParams p;
    p.M = 8;
    ConcurrentHnsw locked(store, p, LockMode::SharedMutex);
    ConcurrentHnsw unlocked(store, p, LockMode::None);
    locked.build();
    unlocked.build();

    SearchScratch sa = locked.make_scratch();
    SearchScratch sb = unlocked.make_scratch();
    const VectorStore queries = random_store(20, 8, 909);
    for (vec_id_t q = 0; q < queries.size(); ++q) {
        const auto a = locked.search(queries.at(q), 10, 64, sa);
        const auto b = unlocked.search(queries.at(q), 10, 64, sb);
        REQUIRE(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i].id == b[i].id);
    }
}

TEST_CASE("each thread's scratch counts only its own distance calls") {
    // The reason the counter moved into the scratch: as a shared atomic it
    // would have been an RMW in the hottest loop plus false sharing across
    // every thread. Per-thread counters are independent by construction.
    const VectorStore store = random_store(1000, 8, 111);
    HnswParams p;
    p.M = 8;
    HnswIndex index(store, p);
    index.build();

    SearchScratch a = index.make_scratch();
    SearchScratch b = index.make_scratch();
    CHECK(a.distance_calls == 0);

    (void)index.search(store.at(0), 10, 64, a);
    CHECK(a.distance_calls > 0);
    CHECK(b.distance_calls == 0);   // untouched by the other scratch's work

    const std::size_t after_first = a.distance_calls;
    (void)index.search(store.at(1), 10, 64, a);
    CHECK(a.distance_calls > after_first);  // accumulates within one scratch

    a.reset_counters();
    CHECK(a.distance_calls == 0);
}

// ---------------------------------------------------------------------------
// Growth: adding vectors to an index that was already built.
//
// Every test above pre-sizes the store and inserts a prefix, which is what the
// benchmarks do and what made the reallocation hazard invisible for five
// phases. These tests grow the store *after* construction, which is the path a
// Python caller takes and the only path where `VectorStore::add` can move the
// buffer under a reader.
// ---------------------------------------------------------------------------

TEST_CASE("an index grows past the size it was constructed at") {
    // Without HnswIndex::grow_to, `levels_[id]` here is an out-of-bounds write
    // into a vector that usually has spare capacity -- so it would not crash,
    // it would corrupt a neighbouring allocation and still return k results.
    VectorStore store = random_store(500, 16, 4242);
    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    p.seed = 7;

    ConcurrentHnsw index(store, p);
    index.build();
    REQUIRE(index.growable());
    REQUIRE(index.index().capacity() == 500);

    std::mt19937_64 rng(99);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(16);

    SearchScratch scratch = index.make_scratch();
    for (std::size_t i = 0; i < 500; ++i) {
        for (auto& x : v) x = u(rng);
        const vec_id_t id = index.insert(v.data());
        CHECK(id == 500 + i);

        // The vector just added must be its own nearest neighbour at distance
        // zero. A graph that grew its arrays but never linked the node would
        // return k results happily and fail exactly here.
        const auto hits = index.search(v.data(), 1, 64, scratch);
        REQUIRE(hits.size() == 1);
        CHECK(hits[0].id == id);
        CHECK(hits[0].dist == doctest::Approx(0.0f));
    }

    CHECK(index.index().capacity() == 1000);
    CHECK(store.size() == 1000);
    CHECK(index.index().check_invariants() == "");
}

TEST_CASE("a scratch built before growth stays valid after it") {
    // VisitedList indexes without a bounds check -- it is the hottest loop in
    // the project. A long-lived per-thread scratch made when the index held n
    // nodes is therefore a latent out-of-bounds read the moment the index holds
    // n+1, unless search() tops it up.
    VectorStore store = random_store(300, 16, 31);
    HnswParams p;
    p.M = 8;
    p.ef_construction = 64;
    ConcurrentHnsw index(store, p);
    index.build();

    SearchScratch old_scratch = index.make_scratch();   // sized for 300
    REQUIRE(old_scratch.visited.size() == 300);

    std::mt19937_64 rng(32);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(16);
    for (std::size_t i = 0; i < 200; ++i) {
        for (auto& x : v) x = u(rng);
        index.insert(v.data());
    }

    // Reused as-is, deliberately: no caller re-made it.
    const auto hits = index.search(v.data(), 10, 64, old_scratch);
    CHECK(hits.size() == 10);
    CHECK(hits[0].id == 499);
    CHECK(old_scratch.visited.size() >= 500);
}

TEST_CASE("growing inserts do not corrupt concurrent readers") {
    // The reallocation race, exercised directly. Readers run continuously while
    // the store doubles in size; VectorStore::add reallocates several times over
    // this range. Under TSan this is the test that would report the race if the
    // append happened outside the exclusive section.
    VectorStore store = random_store(1000, 16, 555);
    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    ConcurrentHnsw index(store, p);
    index.build();

    // The final size, as a constant. The first draft of this test read
    // `index.index().capacity()` from inside the reader loop, which reaches past
    // the lock into arrays that grow_to reallocates -- TSan reported it, and it
    // was the test that was wrong, not the library (B-13). A reader that needs a
    // bound either takes the lock or uses a value it already knows.
    constexpr std::uint32_t kFinalN = 2000;

    std::atomic<bool> stop{false};
    std::atomic<std::size_t> searches{0};
    std::atomic<bool> bad_result{false};

    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t] {
            std::mt19937_64 rng(1000 + t);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            std::vector<float> q(16);
            SearchScratch scratch = index.make_scratch();
            while (!stop.load(std::memory_order_relaxed)) {
                for (auto& x : q) x = u(rng);
                const auto hits = index.search(q.data(), 10, 32, scratch);
                if (hits.size() != 10) bad_result.store(true);
                for (const auto& h : hits) {
                    // An id past the graph, or a NaN distance, is what a
                    // use-after-free on the vector buffer looks like.
                    if (h.id >= kFinalN || !std::isfinite(h.dist)) {
                        bad_result.store(true);
                    }
                }
                searches.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    std::mt19937_64 rng(2000);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(16);
    for (std::size_t i = 0; i < 1000; ++i) {
        for (auto& x : v) x = u(rng);
        index.insert(v.data());
    }

    stop.store(true);
    for (auto& th : readers) th.join();

    CHECK(searches.load() > 0);
    CHECK_FALSE(bad_result.load());
    CHECK(store.size() == kFinalN);
    // Safe unlocked here and only here: every thread is joined.
    CHECK(index.index().check_invariants() == "");
}

TEST_CASE("batch insert lands the same graph as one-at-a-time") {
    // insert_batch exists to take the lock once, not to take a shortcut. Same
    // seed, same vectors, same order => byte-identical outcome, or the batch
    // path is doing something the single path is not.
    const VectorStore extra = random_store(200, 16, 8080);

    auto build_with = [&](bool batched) {
        VectorStore store = random_store(400, 16, 909);
        HnswParams p;
        p.M = 16;
        p.ef_construction = 100;
        p.seed = 4;
        ConcurrentHnsw index(store, p);
        index.build();
        if (batched) {
            index.insert_batch(extra.at(0), extra.size(), extra.dim());
        } else {
            for (vec_id_t i = 0; i < extra.size(); ++i) index.insert(extra.at(i));
        }
        const VectorStore queries = random_store(40, 16, 1234);
        std::vector<vec_id_t> flat;
        SearchScratch s = index.make_scratch();
        for (vec_id_t q = 0; q < queries.size(); ++q) {
            for (const auto& n : index.search(queries.at(q), 10, 64, s)) flat.push_back(n.id);
        }
        return flat;
    };

    CHECK(build_with(true) == build_with(false));
}

TEST_CASE("an index over a const store refuses to grow, and says why") {
    // The const-store constructor is still the right default: bench and the
    // tests above size their stores up front and must not pay for a growth path
    // they never use. Asking it to grow is a programming error, not a runtime
    // condition, so it throws rather than returning a sentinel.
    const VectorStore store = random_store(100, 16, 77);
    HnswParams p;
    p.M = 8;
    ConcurrentHnsw index(store, p);
    index.build();

    CHECK_FALSE(index.growable());
    std::vector<float> v(16, 0.5f);
    CHECK_THROWS_AS(index.insert(v.data()), std::logic_error);
    CHECK_THROWS_AS(index.insert_batch(v.data(), 1, 16), std::logic_error);
}
