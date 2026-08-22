#include <doctest/doctest.h>

#include "veccore/flat_index.hpp"
#include "veccore/hnsw.hpp"
#include "veccore/metrics.hpp"
#include "veccore/storage.hpp"
#include "veccore/visited.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <set>
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

/// Clustered data, because uniform random points are the case where the
/// neighbour heuristic matters LEAST. P-05's failure mode is the graph
/// fragmenting into per-cluster islands, so a test that hopes to see it has to
/// supply clusters.
VectorStore clustered_store(std::size_t clusters, std::size_t per_cluster, dim_t d, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> centre(-50.0f, 50.0f);
    std::normal_distribution<float> jitter(0.0f, 0.3f);

    VectorStore s(d);
    s.reserve(clusters * per_cluster);
    std::vector<float> c(d), v(d);
    for (std::size_t k = 0; k < clusters; ++k) {
        for (auto& x : c) x = centre(rng);
        for (std::size_t i = 0; i < per_cluster; ++i) {
            for (dim_t j = 0; j < d; ++j) v[j] = c[j] + jitter(rng);
            s.add(v.data());
        }
    }
    return s;
}

double mean_recall(const HnswIndex& index, const FlatL2& exact, const VectorStore& queries,
                   std::size_t k, std::size_t ef) {
    double total = 0.0;
    for (vec_id_t q = 0; q < queries.size(); ++q) {
        const auto truth = exact.search(queries.at(q), k);
        const auto got = index.search(queries.at(q), k, ef);
        std::set<vec_id_t> t;
        for (const auto& n : truth) t.insert(n.id);
        std::size_t hits = 0;
        for (const auto& n : got) if (t.count(n.id)) ++hits;
        total += static_cast<double>(hits) / static_cast<double>(k);
    }
    return total / static_cast<double>(queries.size());
}

}  // namespace

// ---------------------------------------------------------------------------
// The visited list -- P-09
// ---------------------------------------------------------------------------

TEST_CASE("visited list marks, resets, and reports first-visit") {
    VisitedList v(10);
    v.reset();
    CHECK(v.test_and_set(3));
    CHECK_FALSE(v.test_and_set(3));
    CHECK_FALSE(v.is_visited(4));
    v.reset();
    CHECK_FALSE(v.is_visited(3));
    CHECK(v.test_and_set(3));
}

TEST_CASE("visited list survives the epoch wrap (P-09)") {
    // The bug this pins is nearly invisible: without wrap handling, one query
    // in every 65536 sees stale stamps as live and silently skips nodes. An
    // aggregate recall number over 10,000 queries would never show it.
    VisitedList v(8);
    v.reset();
    CHECK(v.test_and_set(5));         // stamped with epoch 1

    v.force_epoch_for_test(65535);    // jump to the last epoch before wrap
    v.reset();                        // must zero the array, not roll over
    CHECK(v.wraps() == 1);

    // Node 5 still carries the stale stamp from epoch 1. If reset() had merely
    // wrapped the counter back to 1, this would report already-visited.
    CHECK_FALSE(v.is_visited(5));
    CHECK(v.test_and_set(5));
}

// ---------------------------------------------------------------------------
// Heap polarity -- P-04, and B-01 is why this exists before the search does
// ---------------------------------------------------------------------------

TEST_CASE("the two heaps order in opposite directions") {
    std::priority_queue<Neighbor, std::vector<Neighbor>, WorseFirst> worst_on_top;
    std::priority_queue<Neighbor, std::vector<Neighbor>, CloserFirst> best_on_top;
    for (const auto& n : {Neighbor{5.0f, 0}, Neighbor{1.0f, 1}, Neighbor{9.0f, 2}}) {
        worst_on_top.push(n);
        best_on_top.push(n);
    }
    // W evicts the farthest, so its top must be the farthest.
    CHECK(worst_on_top.top().dist == doctest::Approx(9.0f));
    // C explores the closest next, so its top must be the closest.
    CHECK(best_on_top.top().dist == doctest::Approx(1.0f));
}

// ---------------------------------------------------------------------------
// Structure -- PLAN.md 2.7 step 1
// ---------------------------------------------------------------------------

TEST_CASE("a small graph is structurally sound") {
    const VectorStore store = random_store(200, 8, 1);
    HnswParams p;
    p.M = 4;
    p.ef_construction = 32;
    HnswIndex index(store, p);
    index.build();

    // No self-loops, no duplicate neighbours, no node over its degree cap,
    // no link to a node that does not reach that layer. P-06 and P-16.
    CHECK(index.check_invariants() == "");

    const HnswStats s = index.stats();
    CHECK(s.nodes == 200);
    CHECK(s.nodes_per_level[0] == 200);   // every node exists at layer 0
    CHECK(s.mean_degree_layer0 > 0.0);
    CHECK(s.mean_degree_layer0 <= static_cast<double>(p.m_max0()));
}

TEST_CASE("the level distribution is geometric with the right ratio (P-15)") {
    const VectorStore store = random_store(20000, 4, 7);
    HnswParams p;
    p.M = 16;
    p.ef_construction = 16;
    HnswIndex index(store, p);
    index.build();

    const HnswStats s = index.stats();
    REQUIRE(s.nodes_per_level.size() >= 2);

    // With mL = 1/ln(M), P(level >= l+1) / P(level >= l) = exp(-1/mL) = 1/M.
    // So each level should hold about 1/M of the one below it. Getting mL wrong
    // (1/M or ln(M) instead) breaks this ratio badly while erroring nowhere --
    // which is exactly why the histogram is asserted rather than eyeballed.
    const double ratio = static_cast<double>(s.nodes_per_level[1]) /
                         static_cast<double>(s.nodes_per_level[0]);
    CHECK(ratio == doctest::Approx(1.0 / static_cast<double>(p.M)).epsilon(0.25));

    // ~94% of nodes should exist only at layer 0 when M=16.
    const double only_level0 =
        static_cast<double>(s.nodes_per_level[0] - s.nodes_per_level[1]) / 20000.0;
    CHECK(only_level0 > 0.90);
    CHECK(only_level0 < 0.98);
}

TEST_CASE("the entry point tracks the maximum level (P-18)") {
    // If the entry point is not promoted when a node draws above the current
    // maximum, the upper layers are never entered and you get flat-NSW
    // behaviour while believing you built a hierarchy.
    const VectorStore store = random_store(5000, 8, 3);
    HnswParams p;
    p.M = 8;
    p.ef_construction = 32;
    HnswIndex index(store, p);
    index.build();

    const HnswStats s = index.stats();
    CHECK(s.max_level >= 1);
    // Exactly the top level must be non-empty -- an entry point stranded below
    // the top would leave the highest level unreachable.
    CHECK(s.nodes_per_level[s.max_level] >= 1);
}

// ---------------------------------------------------------------------------
// Recall -- PLAN.md 2.7 steps 2 and 3
// ---------------------------------------------------------------------------

TEST_CASE("recall against exact search is high on uniform data") {
    const VectorStore store = random_store(3000, 16, 11);
    const VectorStore queries = random_store(100, 16, 999);
    const FlatL2 exact(store);

    HnswParams p;
    p.M = 16;
    p.ef_construction = 200;
    HnswIndex index(store, p);
    index.build();

    CHECK(index.check_invariants() == "");
    CHECK(mean_recall(index, exact, queries, 10, 200) >= 0.95);
}

TEST_CASE("recall rises monotonically with ef_search") {
    // PLAN.md 2.7 step 3. A curve that does NOT rise means either the RNG is
    // unseeded and the graph changed underneath the sweep (P-03), or the graph
    // itself is broken -- flat in ef_search is the signature of P-05.
    const VectorStore store = random_store(3000, 16, 13);
    const VectorStore queries = random_store(100, 16, 555);
    const FlatL2 exact(store);

    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    HnswIndex index(store, p);
    index.build();

    double previous = 0.0;
    for (const std::size_t ef : {10u, 20u, 40u, 80u, 160u}) {
        const double r = mean_recall(index, exact, queries, 10, ef);
        CHECK(r >= previous - 1e-9);  // non-decreasing
        previous = r;
    }
    CHECK(previous >= 0.95);  // and it actually gets somewhere
}

TEST_CASE("clustered data does not trap the search (P-05)") {
    // The case naive top-M neighbour selection fails on. 40 tight clusters far
    // apart: with no long-range links, greedy descent enters one cluster and
    // cannot leave, and recall plateaus in the 0.6-0.8 range regardless of
    // ef_search.
    const VectorStore store = clustered_store(40, 100, 12, 21);
    const VectorStore queries = random_store(60, 12, 777);
    const FlatL2 exact(store);

    HnswParams p;
    p.M = 16;
    p.ef_construction = 200;
    HnswIndex index(store, p);
    index.build();

    CHECK(index.check_invariants() == "");
    CHECK(mean_recall(index, exact, queries, 10, 200) >= 0.95);
}

TEST_CASE("searching for an indexed vector finds it") {
    const VectorStore store = random_store(2000, 8, 31);
    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    HnswIndex index(store, p);
    index.build();

    for (const vec_id_t probe : {vec_id_t{0}, vec_id_t{1000}, vec_id_t{1999}}) {
        const auto got = index.search(store.at(probe), 1, 64);
        REQUIRE(got.size() == 1);
        CHECK(got[0].id == probe);
        CHECK(got[0].dist == doctest::Approx(0.0f));
    }
}

// ---------------------------------------------------------------------------
// Determinism and edge cases
// ---------------------------------------------------------------------------

TEST_CASE("the same seed builds the same graph (P-03)") {
    const VectorStore store = random_store(1000, 8, 5);
    HnswParams p;
    p.M = 8;
    p.ef_construction = 50;
    p.seed = 12345;

    HnswIndex a(store, p);
    a.build();
    HnswIndex b(store, p);
    b.build();

    const auto sa = a.stats();
    const auto sb = b.stats();
    CHECK(sa.max_level == sb.max_level);
    CHECK(sa.edges_layer0 == sb.edges_layer0);
    CHECK(sa.nodes_per_level == sb.nodes_per_level);

    const VectorStore q = random_store(20, 8, 6);
    for (vec_id_t i = 0; i < q.size(); ++i) {
        const auto ra = a.search(q.at(i), 10, 50);
        const auto rb = b.search(q.at(i), 10, 50);
        REQUIRE(ra.size() == rb.size());
        for (std::size_t j = 0; j < ra.size(); ++j) CHECK(ra[j].id == rb[j].id);
    }
}

TEST_CASE("a different seed builds a different graph, and that is expected") {
    const VectorStore store = random_store(1000, 8, 5);
    HnswParams a_p;  a_p.M = 8; a_p.ef_construction = 50; a_p.seed = 1;
    HnswParams b_p;  b_p.M = 8; b_p.ef_construction = 50; b_p.seed = 2;

    HnswIndex a(store, a_p); a.build();
    HnswIndex b(store, b_p); b.build();

    // Not a correctness property -- a demonstration of why the seed goes into
    // every JSON record. Two runs of "the same" build are not the same index.
    CHECK(a.stats().nodes_per_level != b.stats().nodes_per_level);
}

TEST_CASE("ef_search below k is clamped rather than returning short (P-17)") {
    const VectorStore store = random_store(500, 8, 17);
    HnswParams p;
    p.M = 8;
    p.ef_construction = 50;
    HnswIndex index(store, p);
    index.build();

    const auto got = index.search(store.at(0), 10, 1);
    CHECK(got.size() == 10);
}

TEST_CASE("an index with one vector answers rather than crashing") {
    VectorStore store(4);
    const std::vector<float> v{1, 2, 3, 4};
    store.add(v.data());
    HnswParams p;
    p.M = 4;
    HnswIndex index(store, p);
    index.build();

    const auto got = index.search(v.data(), 10, 32);
    REQUIRE(got.size() == 1);
    CHECK(got[0].id == 0);
}

TEST_CASE("results come back in ascending distance order") {
    const VectorStore store = random_store(1000, 8, 41);
    HnswParams p;
    p.M = 16;
    HnswIndex index(store, p);
    index.build();

    const VectorStore q = random_store(1, 8, 42);
    const auto got = index.search(q.at(0), 10, 64);
    REQUIRE(got.size() == 10);
    for (std::size_t i = 1; i < got.size(); ++i) CHECK(got[i - 1].dist <= got[i].dist);
}
