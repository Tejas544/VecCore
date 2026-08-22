#include <doctest/doctest.h>

#include "veccore/distance.hpp"
#include "veccore/flat_index.hpp"
#include "veccore/metrics.hpp"
#include "veccore/storage.hpp"
#include "veccore/xvecs.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace veccore;

namespace {

/// Write a temporary .fvecs/.ivecs file so the reader can be tested against
/// bytes we control, rather than only against SIFT -- which is not present on a
/// fresh clone and would make these tests silently skip.
template <typename T>
std::string write_xvecs_tmp(const std::string& name, const std::vector<std::vector<T>>& rows) {
    const std::string path = "/tmp/veccore_test_" + name;
    std::ofstream f(path, std::ios::binary);
    for (const auto& r : rows) {
        const auto d = static_cast<std::int32_t>(r.size());
        f.write(reinterpret_cast<const char*>(&d), 4);
        f.write(reinterpret_cast<const char*>(r.data()),
                static_cast<std::streamsize>(r.size() * sizeof(T)));
    }
    return path;
}

}  // namespace

// ---------------------------------------------------------------------------
// Distance
// ---------------------------------------------------------------------------

TEST_CASE("squared L2 is squared, and inner product is negated") {
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{4.0f, 6.0f, 3.0f};

    // (3^2 + 4^2 + 0^2) = 25.  Not 5 -- D4, no sqrt.
    CHECK(L2Sqr{}(a.data(), b.data(), 3) == doctest::Approx(25.0f));

    // 1*4 + 2*6 + 3*3 = 25, negated so that smaller stays better.
    CHECK(NegInnerProduct{}(a.data(), b.data(), 3) == doctest::Approx(-25.0f));
}

TEST_CASE("distance to self is zero and the metric is symmetric") {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> a(64), b(64);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    CHECK(L2Sqr{}(a.data(), a.data(), 64) == doctest::Approx(0.0f));
    CHECK(L2Sqr{}(a.data(), b.data(), 64) == doctest::Approx(L2Sqr{}(b.data(), a.data(), 64)));
}

// ---------------------------------------------------------------------------
// Storage -- D5 and P-14
// ---------------------------------------------------------------------------

TEST_CASE("flat storage lays vectors out contiguously with the right stride") {
    VectorStore store(4);
    const std::vector<float> v0{0, 1, 2, 3};
    const std::vector<float> v1{4, 5, 6, 7};
    CHECK(store.add(v0.data()) == 0);
    CHECK(store.add(v1.data()) == 1);

    CHECK(store.size() == 2);
    CHECK(store.bytes() == 8 * sizeof(float));

    // The whole point of D5: vector 1 begins exactly dim floats after vector 0,
    // in the same allocation. If this ever stops holding, the cache argument
    // the README makes is no longer true.
    CHECK(store.at(1) == store.at(0) + 4);
    CHECK(store.at(1)[0] == 4.0f);
}

TEST_CASE("both stores agree element for element") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    VectorStore flat(8);
    NaiveVectorStore naive(8);
    for (int i = 0; i < 50; ++i) {
        std::vector<float> v(8);
        for (auto& x : v) x = d(rng);
        flat.add(v.data());
        naive.add(v.data());
    }
    REQUIRE(flat.size() == naive.size());
    for (vec_id_t i = 0; i < flat.size(); ++i) {
        for (dim_t j = 0; j < 8; ++j) {
            CHECK(flat.at(i)[j] == naive.at(i)[j]);
        }
    }
    // The naive layout is strictly larger: 24 bytes of std::vector header per
    // row, and that is before allocator bookkeeping.
    CHECK(naive.bytes() > flat.bytes());
}

// ---------------------------------------------------------------------------
// TopK and exact search
// ---------------------------------------------------------------------------

TEST_CASE("TopK keeps the k best and drains in ascending distance") {
    TopK top(3);
    top.offer(5.0f, 0);
    top.offer(1.0f, 1);
    top.offer(9.0f, 2);
    top.offer(3.0f, 3);
    top.offer(7.0f, 4);

    const auto out = top.take();
    REQUIRE(out.size() == 3);
    CHECK(out[0].id == 1);
    CHECK(out[1].id == 3);
    CHECK(out[2].id == 0);
    CHECK(out[0].dist <= out[1].dist);
    CHECK(out[1].dist <= out[2].dist);
}

TEST_CASE("TopK breaks ties on the smaller id, deterministically (P-12)") {
    TopK top(2);
    top.offer(1.0f, 9);
    top.offer(1.0f, 3);
    top.offer(1.0f, 7);
    const auto out = top.take();
    REQUIRE(out.size() == 2);
    CHECK(out[0].id == 3);
    CHECK(out[1].id == 7);
}

TEST_CASE("TopK returns fewer than k when offered fewer than k") {
    TopK top(10);
    top.offer(1.0f, 0);
    top.offer(2.0f, 1);
    CHECK(top.take().size() == 2);
}

TEST_CASE("exact search on a hand-checkable set") {
    // Points on a line at 0,1,2,...,9 in one dimension. The answer is obvious
    // by inspection, which is the point: this test can be verified by a human
    // in five seconds, so it can be trusted to grade the machinery.
    VectorStore store(1);
    for (int i = 0; i < 10; ++i) {
        const float v = static_cast<float>(i);
        store.add(&v);
    }
    const FlatL2 index(store);

    const float q = 3.2f;
    const auto got = index.search(&q, 3);
    REQUIRE(got.size() == 3);
    CHECK(got[0].id == 3);   // distance 0.04
    CHECK(got[1].id == 4);   // distance 0.64
    CHECK(got[2].id == 2);   // distance 1.44
    CHECK(got[0].dist == doctest::Approx(0.04f));
}

TEST_CASE("exact search finds a query that is also in the index at distance 0") {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    VectorStore store(16);
    for (int i = 0; i < 200; ++i) {
        std::vector<float> v(16);
        for (auto& x : v) x = d(rng);
        store.add(v.data());
    }
    const FlatL2 index(store);
    const auto got = index.search(store.at(57), 1);
    REQUIRE(got.size() == 1);
    CHECK(got[0].id == 57);
    CHECK(got[0].dist == doctest::Approx(0.0f));
    // P-19 in miniature: this is why a recall of exactly 1.0 is a warning sign
    // rather than a triumph -- it usually means the queries came from the base.
}

// ---------------------------------------------------------------------------
// Recall and percentiles
// ---------------------------------------------------------------------------

TEST_CASE("recall is a set intersection, so order does not matter") {
    const std::vector<Neighbor> got{{0.1f, 5}, {0.2f, 3}, {0.3f, 9}};
    const std::vector<std::int32_t> truth{3, 5, 9};
    CHECK(recall_at_k(got, truth.data(), 3) == doctest::Approx(1.0));

    const std::vector<std::int32_t> partial{3, 5, 42};
    CHECK(recall_at_k(got, partial.data(), 3) == doctest::Approx(2.0 / 3.0));

    const std::vector<std::int32_t> none{1, 2, 4};
    CHECK(recall_at_k(got, none.data(), 3) == doctest::Approx(0.0));
}

TEST_CASE("percentiles use nearest-rank and do not interpolate") {
    std::vector<double> s{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CHECK(percentile(s, 50.0) == doctest::Approx(5.0));
    CHECK(percentile(s, 90.0) == doctest::Approx(9.0));
    CHECK(percentile(s, 99.0) == doctest::Approx(10.0));
    CHECK(percentile(s, 100.0) == doctest::Approx(10.0));

    const auto stats = LatencyStats::from(s);
    CHECK(stats.p50 == doctest::Approx(5.0));
    CHECK(stats.mean == doctest::Approx(5.5));
    CHECK(stats.min == doctest::Approx(1.0));
    CHECK(stats.max == doctest::Approx(10.0));
    // p99 > mean on this sample. The reason to report both is that on a real
    // latency distribution the gap is much larger, and it is the gap an
    // interviewer asks about.
    CHECK(stats.p99 > stats.mean);
}

TEST_CASE("stddev is the sample form and zero for a constant sample") {
    CHECK(stddev({5.0, 5.0, 5.0}) == doctest::Approx(0.0));
    CHECK(stddev({2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0}) == doctest::Approx(2.13809).epsilon(0.001));
    CHECK(stddev({1.0}) == doctest::Approx(0.0));
}

// ---------------------------------------------------------------------------
// The xvecs reader -- P-01
// ---------------------------------------------------------------------------

TEST_CASE("fvecs round-trips through the reader") {
    const std::vector<std::vector<float>> rows{
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f},
    };
    const std::string path = write_xvecs_tmp("ok.fvecs", rows);

    const XvecsHeader h = xvecs_header(path);
    CHECK(h.dim == 4);
    CHECK(h.count == 3);

    const VectorStore store = read_fvecs(path);
    REQUIRE(store.size() == 3);
    REQUIRE(store.dim() == 4);
    CHECK(store.at(2)[3] == doctest::Approx(12.0f));

    // max_count truncates rather than failing -- SIFT10K is built this way.
    const VectorStore two = read_fvecs(path, 2);
    CHECK(two.size() == 2);

    std::remove(path.c_str());
}

TEST_CASE("a file whose size is not a multiple of the record size is rejected (P-01)") {
    // The exact failure mode P-01 warns about: bytes that parse into plausible
    // floats but are not this format. Without the size check it would read as
    // valid data, and the resulting bad recall would be blamed on the algorithm.
    const std::string path = "/tmp/veccore_test_bad.fvecs";
    {
        std::ofstream f(path, std::ios::binary);
        const std::int32_t d = 4;
        f.write(reinterpret_cast<const char*>(&d), 4);
        const float vals[4] = {1, 2, 3, 4};
        f.write(reinterpret_cast<const char*>(vals), sizeof(vals));
        const char junk[7] = {1, 2, 3, 4, 5, 6, 7};  // trailing bytes
        f.write(junk, sizeof(junk));
    }
    CHECK_THROWS_AS(read_fvecs(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST_CASE("a record that disagrees about its own dimension is rejected (P-01)") {
    const std::string path = "/tmp/veccore_test_ragged.fvecs";
    {
        std::ofstream f(path, std::ios::binary);
        std::int32_t d = 2;
        const float a[2] = {1, 2};
        f.write(reinterpret_cast<const char*>(&d), 4);
        f.write(reinterpret_cast<const char*>(a), sizeof(a));
        d = 2;  // keeps the file size a valid multiple...
        f.write(reinterpret_cast<const char*>(&d), 4);
        f.write(reinterpret_cast<const char*>(a), sizeof(a));
        const std::int32_t wrong = 99;  // ...but this record lies about itself
        f.write(reinterpret_cast<const char*>(&wrong), 4);
        f.write(reinterpret_cast<const char*>(a), sizeof(a));
    }
    CHECK_THROWS_AS(read_fvecs(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST_CASE("ivecs reads ground-truth rows") {
    const std::vector<std::vector<std::int32_t>> rows{{7, 1, 4}, {2, 9, 0}};
    const std::string path = write_xvecs_tmp("gt.ivecs", rows);

    const IvecsData gt = read_ivecs(path);
    REQUIRE(gt.rows() == 2);
    CHECK(gt.width == 3);
    CHECK(gt.row(0)[0] == 7);
    CHECK(gt.row(1)[2] == 0);

    std::remove(path.c_str());
}

TEST_CASE("a missing file throws rather than returning something empty") {
    CHECK_THROWS_AS(read_fvecs("/tmp/veccore_definitely_not_here.fvecs"), std::runtime_error);
}
