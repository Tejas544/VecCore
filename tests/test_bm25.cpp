#include <doctest/doctest.h>

#include "veccore/bm25.hpp"
#include "veccore/fusion.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace veccore;

// ---------------------------------------------------------------------------
// Tokenizer -- P-25
// ---------------------------------------------------------------------------

TEST_CASE("tokenizer lowercases and splits on non-alphanumerics") {
    CHECK(tokenize("Hello World") == std::vector<std::string>{"hello", "world"});
    CHECK(tokenize("  spaced   out  ") == std::vector<std::string>{"spaced", "out"});
    CHECK(tokenize("error-code X_4417!") ==
          std::vector<std::string>{"error", "code", "x", "4417"});
    CHECK(tokenize("").empty());
    CHECK(tokenize("!!!").empty());
    CHECK(tokenize("trailing") == std::vector<std::string>{"trailing"});
}

TEST_CASE("index and query go through the same tokenizer (P-25)") {
    // The bug this pins: lowercase on one side only, and the term never
    // matches. Recall craters and BM25 is blameless.
    Bm25Index idx;
    idx.build({"The Quick Brown Fox"});
    const auto hits = idx.search("QUICK", 5);
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].id == 0);
}

// ---------------------------------------------------------------------------
// BM25 -- the Phase 4 gate
// ---------------------------------------------------------------------------

TEST_CASE("BM25 matches a hand-computed score to 6 decimal places") {
    // PLAN.md Phase 4 gate. The expected value is DERIVED here from the formula
    // and the corpus statistics, not pasted from a previous run -- a golden
    // number copied out of the implementation only proves the implementation
    // agrees with itself.
    const std::vector<std::string> docs = {
        "the cat sat on the mat",   // doc 0: 6 terms, tf(cat)=1
        "the cat cat cat",          // doc 1: 4 terms, tf(cat)=3
        "dogs are loud",            // doc 2: 3 terms, tf(cat)=0
    };
    Bm25Index idx;               // k1 = 1.2, b = 0.75
    idx.build(docs);

    const double N = 3.0;
    const double avgdl = (6.0 + 4.0 + 3.0) / 3.0;   // 4.333...
    REQUIRE(idx.doc_count() == 3);
    CHECK(idx.avgdl() == doctest::Approx(avgdl).epsilon(1e-12));
    CHECK(idx.doc_freq("cat") == 2);
    CHECK(idx.doc_freq("the") == 2);
    CHECK(idx.doc_freq("dogs") == 1);

    const double k1 = 1.2, b = 0.75;

    // idf("cat") = ln(1 + (N - df + 0.5)/(df + 0.5)), df = 2
    const double idf_cat = std::log(1.0 + (N - 2.0 + 0.5) / (2.0 + 0.5));
    CHECK(idx.idf("cat") == doctest::Approx(idf_cat).epsilon(1e-12));

    // Doc 1: tf = 3, len = 4.
    const double norm1 = k1 * (1.0 - b + b * (4.0 / avgdl));
    const double expected1 = idf_cat * (3.0 * (k1 + 1.0)) / (3.0 + norm1);
    CHECK(idx.score(1, {"cat"}) == doctest::Approx(expected1).epsilon(1e-9));

    // Doc 0: tf = 1, len = 6.
    const double norm0 = k1 * (1.0 - b + b * (6.0 / avgdl));
    const double expected0 = idf_cat * (1.0 * (k1 + 1.0)) / (1.0 + norm0);
    CHECK(idx.score(0, {"cat"}) == doctest::Approx(expected0).epsilon(1e-9));

    // Doc 2 never mentions the term.
    CHECK(idx.score(2, {"cat"}) == doctest::Approx(0.0));

    // And the ranking follows: more mentions in a shorter document wins.
    CHECK(expected1 > expected0);
    const auto hits = idx.search("cat", 3);
    REQUIRE(hits.size() == 2);
    CHECK(hits[0].id == 1);
    CHECK(hits[1].id == 0);
    // dist is the negated score, so search() and score() must agree exactly.
    CHECK(-hits[0].dist == doctest::Approx(expected1).epsilon(1e-6));
    CHECK(-hits[1].dist == doctest::Approx(expected0).epsilon(1e-6));
}

TEST_CASE("IDF is never negative, even for a term in most documents (P-08)") {
    // The textbook form, ln((N - df + 0.5)/(df + 0.5)), goes negative once a
    // term appears in more than half the corpus -- so matching a common term
    // would REDUCE a document's score. On a 362-document corpus like EdgeRAG's
    // that is a routine occurrence, not an edge case.
    Bm25Index idx;
    idx.build({"common term here", "common term there", "common term everywhere",
               "common term always", "rare"});

    CHECK(idx.doc_freq("common") == 4);   // 4 of 5 documents
    CHECK(idx.idf("common") > 0.0);
    CHECK(idx.idf("rare") > idx.idf("common"));

    // The textbook form on this data, for contrast: it is negative.
    const double textbook = std::log((5.0 - 4.0 + 0.5) / (4.0 + 0.5));
    CHECK(textbook < 0.0);
}

TEST_CASE("term frequency saturates -- that is what k1 is for") {
    Bm25Index idx;
    idx.build({"x", "x x", "x x x x", "x x x x x x x x", "other"});
    // Scores must rise with tf but sub-linearly. Doubling tf must never double
    // the contribution, which is the whole point of the saturation term.
    const double s1 = idx.score(0, {"x"});
    const double s2 = idx.score(1, {"x"});
    const double s4 = idx.score(2, {"x"});
    const double s8 = idx.score(3, {"x"});
    CHECK(s1 < s2);
    CHECK(s2 < s4);
    CHECK(s4 < s8);
    CHECK(s8 < 8.0 * s1);         // strongly sub-linear
    CHECK((s8 - s4) < (s2 - s1)); // and the increments shrink
}

TEST_CASE("k1 = 0 collapses to presence/absence") {
    Bm25Index idx(Bm25Params{0.0, 0.75});
    idx.build({"x", "x x x x x x x x", "other"});
    // With k1 = 0 the tf factor becomes tf*1/(tf+0) = 1 for any tf > 0.
    CHECK(idx.score(0, {"x"}) == doctest::Approx(idx.score(1, {"x"})));
}

TEST_CASE("b = 0 disables length normalisation") {
    Bm25Index no_norm(Bm25Params{1.2, 0.0});
    const std::vector<std::string> docs = {"x padding padding padding padding", "x"};
    no_norm.build(docs);
    // Same tf, wildly different lengths: with b = 0 they must score identically.
    CHECK(no_norm.score(0, {"x"}) == doctest::Approx(no_norm.score(1, {"x"})));

    Bm25Index with_norm(Bm25Params{1.2, 0.75});
    with_norm.build(docs);
    // With normalisation on, the short document wins.
    CHECK(with_norm.score(1, {"x"}) > with_norm.score(0, {"x"}));
}

TEST_CASE("avgdl and doc lengths come from the same pass (P-26)") {
    Bm25Index idx;
    idx.build({"a b c", "d e", "f"});
    CHECK(idx.doc_len(0) == 3);
    CHECK(idx.doc_len(1) == 2);
    CHECK(idx.doc_len(2) == 1);
    CHECK(idx.avgdl() == doctest::Approx(2.0));
}

TEST_CASE("querying an empty index, or with unknown terms, returns nothing") {
    Bm25Index empty;
    CHECK(empty.search("anything", 5).empty());

    Bm25Index idx;
    idx.build({"alpha", "beta"});
    CHECK(idx.search("gamma", 5).empty());
    CHECK(idx.search("", 5).empty());
    CHECK(idx.search("!!!", 5).empty());
}

TEST_CASE("multi-term queries sum per-term contributions") {
    Bm25Index idx;
    idx.build({"alpha beta", "alpha", "beta", "gamma"});
    const double both = idx.score(0, {"alpha", "beta"});
    CHECK(both == doctest::Approx(idx.score(0, {"alpha"}) + idx.score(0, {"beta"})));
    // And a document matching both should outrank one matching either.
    const auto hits = idx.search("alpha beta", 4);
    REQUIRE(!hits.empty());
    CHECK(hits[0].id == 0);
}

// ---------------------------------------------------------------------------
// RRF -- P-27
// ---------------------------------------------------------------------------

TEST_CASE("RRF matches hand arithmetic, with 1-based ranks (P-27)") {
    // Two retrievers. Doc 7 is ranked 1st by one and 2nd by the other; doc 3 is
    // 2nd and 1st. They should tie exactly.
    const std::vector<Neighbor> dense{{0.1f, 7}, {0.2f, 3}, {0.3f, 5}};
    const std::vector<Neighbor> sparse{{-9.0f, 3}, {-8.0f, 7}, {-7.0f, 9}};

    const auto fused = reciprocal_rank_fusion({dense, sparse}, 4);
    REQUIRE(fused.size() == 4);

    // 1-based: doc 7 gets 1/(60+1) + 1/(60+2).
    const double expect_7 = 1.0 / 61.0 + 1.0 / 62.0;
    const double expect_3 = 1.0 / 62.0 + 1.0 / 61.0;
    const double expect_5 = 1.0 / 63.0;
    const double expect_9 = 1.0 / 63.0;
    CHECK(expect_7 == doctest::Approx(expect_3));

    double got_7 = 0, got_5 = 0, got_9 = 0;
    for (const Neighbor& n : fused) {
        if (n.id == 7) got_7 = -static_cast<double>(n.dist);
        if (n.id == 5) got_5 = -static_cast<double>(n.dist);
        if (n.id == 9) got_9 = -static_cast<double>(n.dist);
    }
    CHECK(got_7 == doctest::Approx(expect_7).epsilon(1e-6));
    CHECK(got_5 == doctest::Approx(expect_5).epsilon(1e-6));
    CHECK(got_9 == doctest::Approx(expect_9).epsilon(1e-6));

    // Documents found by both retrievers beat documents found by only one.
    CHECK(got_7 > got_5);
}

TEST_CASE("RRF ignores score magnitudes entirely") {
    // The property that makes RRF work: the sparse list's scores are 1000x the
    // dense list's, and it changes nothing. This is the answer to "why not a
    // weighted score blend?" -- there is no scale to blend.
    const std::vector<Neighbor> dense{{0.01f, 1}, {0.02f, 2}};
    const std::vector<Neighbor> sparse_small{{-1.0f, 2}, {-0.5f, 1}};
    const std::vector<Neighbor> sparse_huge{{-1000.0f, 2}, {-500.0f, 1}};

    const auto a = reciprocal_rank_fusion({dense, sparse_small}, 2);
    const auto b = reciprocal_rank_fusion({dense, sparse_huge}, 2);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].id == b[i].id);
        CHECK(a[i].dist == doctest::Approx(b[i].dist));
    }
}

TEST_CASE("a document ranked well by both beats one ranked 1st by one only") {
    const std::vector<Neighbor> dense{{0.1f, 1}, {0.2f, 2}, {0.3f, 3}};
    const std::vector<Neighbor> sparse{{-1.0f, 4}, {-2.0f, 2}, {-3.0f, 3}};
    // Doc 2: ranks 2 and 2 -> 1/62 + 1/62 = 0.032258
    // Doc 1: rank 1 and absent -> 1/61        = 0.016393
    // Doc 4: rank 1 and absent -> 1/61        = 0.016393
    const auto fused = reciprocal_rank_fusion({dense, sparse}, 4);
    REQUIRE(!fused.empty());
    CHECK(fused[0].id == 2);
}

TEST_CASE("RRF with a single list preserves that list's order") {
    const std::vector<Neighbor> only{{0.1f, 5}, {0.2f, 6}, {0.3f, 7}};
    const auto fused = reciprocal_rank_fusion({only}, 3);
    REQUIRE(fused.size() == 3);
    CHECK(fused[0].id == 5);
    CHECK(fused[1].id == 6);
    CHECK(fused[2].id == 7);
}

TEST_CASE("RRF handles empty and missing lists") {
    CHECK(reciprocal_rank_fusion({}, 5).empty());
    CHECK(reciprocal_rank_fusion({{}, {}}, 5).empty());
    const std::vector<Neighbor> one{{0.1f, 1}};
    CHECK(reciprocal_rank_fusion({one, {}}, 5).size() == 1);
}

TEST_CASE("k damps the advantage of the top rank") {
    const std::vector<Neighbor> a{{0.1f, 1}, {0.2f, 2}};
    const std::vector<Neighbor> b{{0.1f, 2}, {0.2f, 1}};

    // Large k: rank 1 and rank 2 are nearly equivalent, so the two documents
    // (each 1st in one list, 2nd in the other) tie.
    const auto big = reciprocal_rank_fusion({a, b}, 2, RrfParams{60.0});
    CHECK(-big[0].dist == doctest::Approx(-big[1].dist).epsilon(1e-6));

    // Small k exaggerates the top rank -- still a tie here by symmetry, but the
    // absolute scores are much larger, which is the mechanism k controls.
    const auto small = reciprocal_rank_fusion({a, b}, 2, RrfParams{1.0});
    CHECK(-small[0].dist > -big[0].dist);
}
