#pragma once

// BM25 (Robertson & Zaragoza, "The Probabilistic Relevance Framework"), the
// sparse half of hybrid retrieval.
//
// Still the default baseline in serious IR after thirty years, and it is small
// enough to write from memory in an interview -- which you should be able to do:
//
//   score(D,Q) = SUM over query terms t of
//                    idf(t) * ( tf * (k1+1) ) / ( tf + k1*(1 - b + b*len/avglen) )
//
// Three ideas, and each parameter controls exactly one of them:
//
//   * a document mentioning the term more often is more relevant, BUT WITH
//     DIMINISHING RETURNS. `k1` sets how fast the returns saturate. k1 = 0
//     collapses the whole term-frequency component to presence/absence.
//   * rare terms matter more than common ones -- that is idf().
//   * long documents match more words by luck, so they are discounted. `b`
//     sets how hard. b = 0 disables length normalisation entirely.
//
// Why this exists alongside HNSW: dense embeddings have no concept of a token
// they never saw. Search for "error code X-4417" and an embedding model returns
// paragraphs about errors in general, while an inverted index nails it, because
// the token is either present or it is not.

#include "veccore/flat_index.hpp"
#include "veccore/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace veccore {

struct Bm25Params {
    double k1 = 1.2;   ///< term-frequency saturation
    double b = 0.75;   ///< document-length normalisation strength
};

/// THE tokenizer. Singular deliberately.
///
/// P-25: the classic BM25 bug is two tokenizers that agree today -- one for
/// indexing, one for querying. Lowercase on one side only, or split on a
/// slightly different character class, and terms simply never match. Recall
/// craters and the formula is blameless. There is exactly one of these, and
/// both paths call it.
///
/// Lowercase, split on anything not alphanumeric. No stemming: it is out of
/// scope, and it would change the numbers underneath the measurements.
std::vector<std::string> tokenize(std::string_view text);

class Bm25Index {
public:
    explicit Bm25Index(Bm25Params params = {}) : params_(params) {}

    /// Build from a corpus. Document ids are positions in `docs`.
    void build(const std::vector<std::string>& docs);

    /// Top-k by BM25.
    ///
    /// Returns `Neighbor` with **dist = -score**, so that "smaller is better"
    /// holds everywhere in this codebase. Same precedent as NegInnerProduct in
    /// distance.hpp: the sign flip lives in exactly one place (D4), because a
    /// comparison sense that varies by call site is how P-04 happens.
    [[nodiscard]] std::vector<Neighbor> search(std::string_view query, std::size_t k) const;

    /// Score one document against one query. Exposed for the hand-computed
    /// test that is Phase 4's gate.
    [[nodiscard]] double score(std::uint32_t doc_id, const std::vector<std::string>& terms) const;

    /// Lucene's IDF variant, and the choice is load-bearing (P-08):
    ///
    ///     ln( 1 + (N - df + 0.5) / (df + 0.5) )
    ///
    /// The textbook form omits the `1 +` and goes **negative** for any term
    /// appearing in more than half the corpus. On a small corpus -- like
    /// EdgeRAG's 362 documents -- that means matching a common term actively
    /// *lowers* a document's score, which is both wrong and extremely confusing
    /// to debug. This form is strictly positive.
    [[nodiscard]] double idf(const std::string& term) const;

    [[nodiscard]] std::size_t doc_count() const noexcept { return doc_len_.size(); }
    [[nodiscard]] std::size_t vocab_size() const noexcept { return postings_.size(); }
    [[nodiscard]] double avgdl() const noexcept { return avgdl_; }
    [[nodiscard]] std::uint32_t doc_freq(const std::string& term) const;
    [[nodiscard]] std::size_t doc_len(std::uint32_t doc_id) const { return doc_len_[doc_id]; }
    [[nodiscard]] std::uint32_t term_freq(const std::string& term, std::uint32_t doc_id) const;

    /// Postings plus lengths. Excludes the std::string keys, which dominate for
    /// a small corpus -- stated rather than quietly omitted.
    [[nodiscard]] offset_t index_bytes() const noexcept;

private:
    struct Posting {
        std::uint32_t doc_id;
        std::uint32_t tf;
    };

    Bm25Params params_;
    std::unordered_map<std::string, std::vector<Posting>> postings_;
    std::vector<std::uint32_t> doc_len_;
    double avgdl_ = 0.0;
};

}  // namespace veccore
