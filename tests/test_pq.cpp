#include <doctest/doctest.h>

#include "veccore/distance.hpp"
#include "veccore/flat_index.hpp"
#include "veccore/kmeans.hpp"
#include "veccore/pq.hpp"
#include "veccore/storage.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

using namespace veccore;

namespace {

VectorStore clustered(std::size_t clusters, std::size_t per, dim_t d, std::uint64_t seed,
                      float spread = 30.0f, float jitter_sd = 0.5f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> centre(-spread, spread);
    std::normal_distribution<float> jitter(0.0f, jitter_sd);
    VectorStore s(d);
    s.reserve(clusters * per);
    std::vector<float> c(d), v(d);
    for (std::size_t k = 0; k < clusters; ++k) {
        for (auto& x : c) x = centre(rng);
        for (std::size_t i = 0; i < per; ++i) {
            for (dim_t j = 0; j < d; ++j) v[j] = c[j] + jitter(rng);
            s.add(v.data());
        }
    }
    return s;
}

VectorStore uniform(std::size_t n, dim_t d, std::uint64_t seed) {
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

// ---------------------------------------------------------------------------
// k-means
// ---------------------------------------------------------------------------

TEST_CASE("k-means recovers obvious clusters") {
    // Four blobs on a fixed grid, deliberately NOT drawn at random. The first
    // version of this test used random centres and two of them landed 2.9 apart
    // with jitter 0.2 -- close enough that k-means++ could seed both initial
    // centroids inside one of them (B-07). Test data that is "well separated"
    // on average is not well separated.
    VectorStore s(2);
    std::mt19937_64 rng(5);
    std::normal_distribution<float> jitter(0.0f, 0.2f);
    const float centres[4][2] = {{-40, -40}, {-40, 40}, {40, -40}, {40, 40}};
    for (const auto& c : centres) {
        for (int i = 0; i < 200; ++i) {
            const std::vector<float> v{c[0] + jitter(rng), c[1] + jitter(rng)};
            s.add(v.data());
        }
    }

    KMeansParams p;
    p.k = 4;
    p.seed = 1;
    const KMeansResult r = kmeans(s.raw().data(), s.size(), 2, p);

    CHECK(r.k == 4);
    CHECK(r.empty_cluster_reseeds == 0);
    // Mean squared distance to the assigned centroid should sit at the jitter
    // variance (0.2^2 per dim x 2 dims = 0.08), not the inter-cluster scale.
    CHECK(r.inertia / static_cast<double>(s.size()) < 0.2);
}

TEST_CASE("restarts escape a local minimum that a single run gets stuck in (B-07)") {
    // The exact configuration that failed: two clusters 2.9 apart with jitter
    // 0.2, plus two far away. Seed 1 with n_init=1 converges to ~14x the
    // inertia of the good solution, because Lloyd's only moves downhill and
    // cannot un-merge two clusters sharing one centroid.
    VectorStore s(2);
    std::mt19937_64 rng(77);
    std::normal_distribution<float> jitter(0.0f, 0.2f);
    const float centres[4][2] = {{13.845f, -36.920f}, {1.404f, -27.891f},
                                 {-21.943f, 8.931f}, {-22.365f, 11.790f}};
    for (const auto& c : centres) {
        for (int i = 0; i < 200; ++i) {
            const std::vector<float> v{c[0] + jitter(rng), c[1] + jitter(rng)};
            s.add(v.data());
        }
    }

    KMeansParams single;
    single.k = 4; single.seed = 1; single.n_init = 1;
    const KMeansResult one = kmeans(s.raw().data(), s.size(), 2, single);

    KMeansParams many = single;
    many.n_init = 5;
    const KMeansResult best = kmeans(s.raw().data(), s.size(), 2, many);

    // Restarts can only help: we keep the minimum.
    CHECK(best.inertia <= one.inertia + 1e-6);
    CHECK(best.inertia_per_init.size() == 5);
    CHECK(std::is_sorted(best.inertia_per_init.begin(), best.inertia_per_init.end()));
    // And on this data they demonstrably do: the good solution sits at the
    // jitter variance, which a single unlucky seed does not reach.
    CHECK(best.inertia / static_cast<double>(s.size()) < 0.2);
}

TEST_CASE("inertia never increases across iterations") {
    const VectorStore s = uniform(500, 4, 11);
    KMeansParams p;
    p.k = 8;
    p.seed = 3;
    p.max_iters = 1;
    const KMeansResult one = kmeans(s.raw().data(), s.size(), 4, p);
    p.max_iters = 20;
    const KMeansResult many = kmeans(s.raw().data(), s.size(), 4, p);
    // Lloyd's only ever moves centroids downhill, so more iterations cannot be
    // worse. If this ever fails, the update step is wrong.
    CHECK(many.inertia <= one.inertia + 1e-6);
}

TEST_CASE("more centroids fit the data better") {
    const VectorStore s = uniform(600, 4, 13);
    double previous = std::numeric_limits<double>::max();
    for (const std::size_t k : {2u, 4u, 8u, 16u}) {
        KMeansParams p;
        p.k = k;
        p.seed = 7;
        const KMeansResult r = kmeans(s.raw().data(), s.size(), 4, p);
        CHECK(r.inertia <= previous + 1e-6);
        previous = r.inertia;
    }
}

TEST_CASE("asking for more clusters than distinct points does not produce NaN (P-07)") {
    // The direct route to an empty cluster: 10 identical points, 256 requested
    // centroids. Without the re-seed, centroids owning nothing get 0/0 = NaN,
    // the NaN propagates into every distance, every comparison against it is
    // false, and the index returns garbage with no error anywhere.
    VectorStore s(3);
    const std::vector<float> v{1.0f, 2.0f, 3.0f};
    for (int i = 0; i < 10; ++i) s.add(v.data());

    KMeansParams p;
    p.k = 256;
    p.seed = 9;
    KMeansResult r;
    REQUIRE_NOTHROW(r = kmeans(s.raw().data(), s.size(), 3, p));

    for (const float x : r.centroids) CHECK(std::isfinite(x));
    CHECK(r.inertia == doctest::Approx(0.0));
}

TEST_CASE("degenerate data with many duplicate points still yields finite centroids") {
    VectorStore s(2);
    const std::vector<float> a{0.0f, 0.0f}, b{5.0f, 5.0f};
    for (int i = 0; i < 50; ++i) { s.add(a.data()); s.add(b.data()); }

    KMeansParams p;
    p.k = 16;   // far more clusters than the 2 distinct points
    p.seed = 4;
    const KMeansResult r = kmeans(s.raw().data(), s.size(), 2, p);
    for (const float x : r.centroids) CHECK(std::isfinite(x));
    CHECK(r.empty_cluster_reseeds > 0);  // the guard actually fired
}

// ---------------------------------------------------------------------------
// PQ structure
// ---------------------------------------------------------------------------

TEST_CASE("m must divide d, and saying so beats truncating silently (P-21)") {
    CHECK_THROWS_AS(ProductQuantizer(128, PqParams{12, 1000, 5, 1}), std::invalid_argument);
    CHECK_NOTHROW(ProductQuantizer(128, PqParams{8, 1000, 5, 1}));
    CHECK_NOTHROW(ProductQuantizer(128, PqParams{16, 1000, 5, 1}));
}

TEST_CASE("codes are exactly one byte per subspace (P-24)") {
    const VectorStore s = uniform(500, 16, 21);
    PqParams p;
    p.m = 4;
    p.train_size = 500;
    p.kmeans_iters = 5;
    ProductQuantizer pq(16, p);
    pq.train(s);
    pq.encode(s);

    // The entire premise is "256 centroids => one byte". Measured from the real
    // buffer, not from arithmetic on paper: storing codes in an int vector
    // would be a 4x memory regression while the formula still said 16x.
    CHECK(pq.bytes_per_vector() == 4);
    CHECK(pq.code_bytes() == 500 * 4);
    CHECK(pq.code_bytes() == s.size() * p.m * sizeof(std::uint8_t));

    const std::size_t raw = s.size() * 16 * sizeof(float);
    CHECK(static_cast<double>(raw) / static_cast<double>(pq.code_bytes()) == doctest::Approx(16.0));
}

TEST_CASE("reconstruction error falls as m rises") {
    // More subspaces means fewer dimensions per codebook entry, so each
    // subvector is represented more precisely. If this is not monotone, the
    // subspace slicing is wrong.
    const VectorStore s = clustered(30, 40, 16, 31, 10.0f, 1.0f);
    double previous = std::numeric_limits<double>::max();
    for (const std::size_t m : {2u, 4u, 8u, 16u}) {
        PqParams p;
        p.m = m;
        p.train_size = 1200;
        p.kmeans_iters = 10;
        p.seed = 5;
        ProductQuantizer pq(16, p);
        pq.train(s);
        pq.encode(s);
        const double mse = pq.reconstruction_mse(s, 0);
        CHECK(mse <= previous + 1e-4);
        previous = mse;
    }
}

TEST_CASE("m == d makes PQ nearly exact, which pins the ADC table (P-22)") {
    // With one dimension per subspace and 256 centroids, each scalar is
    // represented almost exactly, so ADC distances must closely track true
    // squared L2. This is the assertion that catches an ADC table built with
    // unsquared distances -- which otherwise ranks almost correctly and looks
    // like ordinary quantization error.
    const VectorStore s = uniform(300, 8, 41);
    PqParams p;
    p.m = 8;        // == d, so dsub == 1
    p.train_size = 300;
    p.kmeans_iters = 25;
    p.seed = 2;
    ProductQuantizer pq(8, p);
    pq.train(s);
    pq.encode(s);

    const VectorStore q = uniform(5, 8, 42);
    const L2Sqr dist;
    for (vec_id_t i = 0; i < q.size(); ++i) {
        const AdcTable t = pq.adc_table(q.at(i));
        for (vec_id_t j = 0; j < 20; ++j) {
            const float adc = t.distance(pq.code(j));
            const float exact = dist(q.at(i), s.at(j), 8);
            CHECK(adc == doctest::Approx(exact).epsilon(0.05));
        }
    }
}

TEST_CASE("the ADC table is the size the L1-cache argument claims") {
    const VectorStore s = uniform(100, 32, 51);
    PqParams p;
    p.m = 8;
    p.train_size = 100;
    p.kmeans_iters = 3;
    ProductQuantizer pq(32, p);
    pq.train(s);
    const AdcTable t = pq.adc_table(s.at(0));
    // 8 subspaces x 256 centroids x 4 bytes = 8 KiB, comfortably inside a
    // 32-48 KiB L1d. That is the whole reason the scan is fast.
    CHECK(t.bytes() == 8 * 256 * sizeof(float));
    CHECK(t.bytes() <= 32 * 1024);
}

// ---------------------------------------------------------------------------
// PQ search quality
// ---------------------------------------------------------------------------

TEST_CASE("PQ search finds most of the true neighbours") {
    const VectorStore s = clustered(50, 60, 32, 61, 20.0f, 1.0f);
    const VectorStore q = uniform(40, 32, 62);
    const FlatL2 exact(s);

    PqParams p;
    p.m = 8;
    p.train_size = 3000;
    p.kmeans_iters = 20;
    p.seed = 3;
    ProductQuantizer pq(32, p);
    pq.train(s);
    pq.encode(s);

    double recall = 0.0;
    for (vec_id_t i = 0; i < q.size(); ++i) {
        const auto truth = exact.search(q.at(i), 10);
        const auto got = pq.search(q.at(i), 10);
        std::set<vec_id_t> t;
        for (const auto& n : truth) t.insert(n.id);
        std::size_t hits = 0;
        for (const auto& n : got) if (t.count(n.id)) ++hits;
        recall += hits / 10.0;
    }
    recall /= static_cast<double>(q.size());
    CHECK(recall > 0.5);  // deliberately loose: this is 16x compression
}

TEST_CASE("reranking a PQ shortlist recovers accuracy") {
    // The practical use of PQ: cheap approximate shortlist, exact rescore on a
    // few hundred. This test asserts the direction, which is the claim the
    // README will make.
    const VectorStore s = clustered(50, 60, 32, 71, 20.0f, 1.0f);
    const VectorStore q = uniform(40, 32, 72);
    const FlatL2 exact(s);

    PqParams p;
    p.m = 8;
    p.train_size = 3000;
    p.kmeans_iters = 20;
    p.seed = 3;
    ProductQuantizer pq(32, p);
    pq.train(s);
    pq.encode(s);

    auto measure = [&](bool rerank) {
        double recall = 0.0;
        for (vec_id_t i = 0; i < q.size(); ++i) {
            const auto truth = exact.search(q.at(i), 10);
            const auto got = rerank ? pq.search_rerank(q.at(i), 10, 100, s)
                                    : pq.search(q.at(i), 10);
            std::set<vec_id_t> t;
            for (const auto& n : truth) t.insert(n.id);
            std::size_t hits = 0;
            for (const auto& n : got) if (t.count(n.id)) ++hits;
            recall += hits / 10.0;
        }
        return recall / static_cast<double>(q.size());
    };

    CHECK(measure(true) >= measure(false));
}

TEST_CASE("encoding is deterministic for a given seed") {
    const VectorStore s = uniform(400, 16, 81);
    PqParams p;
    p.m = 4;
    p.train_size = 400;
    p.kmeans_iters = 10;
    p.seed = 1234;

    ProductQuantizer a(16, p), b(16, p);
    a.train(s); a.encode(s);
    b.train(s); b.encode(s);
    for (vec_id_t i = 0; i < s.size(); ++i) {
        for (std::size_t j = 0; j < p.m; ++j) CHECK(a.code(i)[j] == b.code(i)[j]);
    }
}

TEST_CASE("encoding before training is an error, not undefined behaviour") {
    const VectorStore s = uniform(50, 8, 91);
    ProductQuantizer pq(8, PqParams{4, 50, 5, 1});
    CHECK_THROWS_AS(pq.encode(s), std::logic_error);
}
