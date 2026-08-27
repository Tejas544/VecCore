#pragma once

// Product Quantization (Jégou, Douze & Schmid, TPAMI 2011) with Asymmetric
// Distance Computation.
//
// The idea in one paragraph: chop each d-dimensional vector into `m` subvectors
// of d/m dimensions. For each subvector position independently, learn 256
// representative subvectors by k-means. Store, instead of d floats, m bytes --
// one codebook index per subspace. 256 is not arbitrary: it is exactly what
// fits in a byte.
//
// SIFT: 128 dims x 4 bytes = 512 B/vector. With m=8 that is 8 B/vector, 64x
// smaller. Because the subspaces are independent, 8 codebooks of 256 entries
// span 256^8 ~= 1.8e19 distinct vectors -- the combinatorial "product" the name
// refers to.
//
// ADC is the half that gets asked about. See `AdcTable`.

#include "veccore/flat_index.hpp"
#include "veccore/kmeans.hpp"
#include "veccore/storage.hpp"
#include "veccore/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace veccore {

struct PqParams {
    std::size_t m = 8;              ///< subspaces; must divide the dimensionality
    std::size_t train_size = 100000;///< subsample for codebook training (PLAN 0.3 cut 6)
    std::size_t kmeans_iters = 25;
    std::uint64_t seed = 42;
    /// k-means restarts per subspace (B-07). Costs n_init x training time.
    std::size_t kmeans_n_init = 3;

    /// Centroids per subquantizer. Fixed at 256 deliberately: one byte per
    /// code. Exposed as a constant rather than a parameter so the "why 256?"
    /// answer stays in the code.
    static constexpr std::size_t kCentroids = 256;
};

/// Per-query lookup table: distance from this query's subvector to each of the
/// 256 centroids, for every subspace.
///
/// This is the asymmetric part, and the reason it is both faster AND more
/// accurate than the symmetric alternative:
///
///   symmetric  -- quantize the query too, compare code to code. Error is
///                 introduced on BOTH sides for no speed gain.
///   asymmetric -- keep the query exact, precompute this table once per query,
///                 then a distance is m table lookups and m-1 adds. Error is
///                 introduced on one side only.
///
/// For m=8 the table is 8 * 256 * 4 = 8 KiB, which fits in L1. That is why the
/// inner loop is fast: it never touches DRAM for centroids, and it contains no
/// multiplications at all.
class AdcTable {
public:
    AdcTable(std::size_t m, std::size_t centroids) : m_(m), c_(centroids), t_(m * centroids) {}

    [[nodiscard]] float* row(std::size_t sub) noexcept { return t_.data() + sub * c_; }
    [[nodiscard]] const float* row(std::size_t sub) const noexcept { return t_.data() + sub * c_; }

    /// Sum the per-subspace distances for one code.
    [[nodiscard]] float distance(const std::uint8_t* code) const noexcept {
        float sum = 0.0f;
        for (std::size_t s = 0; s < m_; ++s) sum += t_[s * c_ + code[s]];
        return sum;
    }

    [[nodiscard]] std::size_t bytes() const noexcept { return t_.size() * sizeof(float); }

private:
    std::size_t m_, c_;
    std::vector<float> t_;
};

class ProductQuantizer {
public:
    ProductQuantizer(dim_t d, PqParams params);

    /// Learn the codebooks. Trains on a deterministic subsample of `store`.
    void train(const VectorStore& store);

    /// Encode every vector in `store` into m bytes each.
    void encode(const VectorStore& store);

    /// Build the per-query lookup table.
    [[nodiscard]] AdcTable adc_table(const float* query) const;

    /// Exhaustive ADC scan over the encoded vectors.
    [[nodiscard]] std::vector<Neighbor> search(const float* query, std::size_t k) const;

    /// Over-fetch by ADC, then rescore the shortlist with exact distances.
    /// This is how PQ is actually used in practice: the cheap approximate pass
    /// picks candidates, and an exact pass on a few hundred of them recovers
    /// most of the lost accuracy for very little time.
    [[nodiscard]] std::vector<Neighbor> search_rerank(const float* query, std::size_t k,
                                                      std::size_t candidates,
                                                      const VectorStore& full) const;

    /// Write codebooks, codes and parameters to `path`.
    ///
    /// This is the persistence case with the clearest payoff in the repo:
    /// training codebooks at SIFT1M takes **86.5 s** and loading them takes
    /// milliseconds. The codes are the compressed representation, so the file is
    /// small by construction -- 30.6 MiB at m=32 against 488 MiB of vectors.
    ///
    /// The **vectors are not included**, which is the opposite choice from
    /// `HnswIndex::save` and for a reason worth being able to state: writing
    /// them would multiply a 30 MiB artifact by 16x to carry data that only
    /// `search_rerank` needs. So a loaded quantizer can `search` immediately and
    /// can `search_rerank` only if the caller supplies the store -- which is
    /// already how that method's signature works. `CONTEXT.md` D15.
    void save(const std::string& path) const;

    /// Read a quantizer written by `save`. The result can `search` at once;
    /// `search_rerank` still needs a `VectorStore` from the caller.
    [[nodiscard]] static ProductQuantizer load(const std::string& path);

    /// Decode a code back to an approximate vector -- used only to measure
    /// reconstruction error, never in the search path.
    void decode(const std::uint8_t* code, float* out) const;

    [[nodiscard]] double reconstruction_mse(const VectorStore& store, std::size_t sample) const;

    [[nodiscard]] const std::uint8_t* code(vec_id_t id) const noexcept {
        return codes_.data() + static_cast<offset_t>(id) * params_.m;
    }
    [[nodiscard]] std::size_t n_codes() const noexcept {
        return params_.m ? codes_.size() / params_.m : 0;
    }

    /// Bytes actually occupied by the codes. P-24: this is measured with
    /// sizeof on the real buffer, never computed on paper -- storing codes in
    /// an int vector would be a 4x memory regression while the arithmetic
    /// still reported 64x compression.
    [[nodiscard]] offset_t code_bytes() const noexcept { return codes_.size() * sizeof(std::uint8_t); }
    [[nodiscard]] offset_t codebook_bytes() const noexcept {
        return codebooks_.size() * sizeof(float);
    }
    [[nodiscard]] std::size_t bytes_per_vector() const noexcept { return params_.m; }
    [[nodiscard]] std::size_t empty_cluster_reseeds() const noexcept { return reseeds_; }
    [[nodiscard]] const PqParams& params() const noexcept { return params_; }
    [[nodiscard]] dim_t dsub() const noexcept { return dsub_; }

private:
    [[nodiscard]] const float* centroid(std::size_t sub, std::size_t idx) const noexcept {
        return codebooks_.data() + (sub * PqParams::kCentroids + idx) * static_cast<offset_t>(dsub_);
    }

    dim_t d_ = 0;
    dim_t dsub_ = 0;
    PqParams params_;
    std::vector<float> codebooks_;      ///< m * 256 * dsub
    std::vector<std::uint8_t> codes_;   ///< n * m, one byte per subspace
    std::size_t reseeds_ = 0;
    bool trained_ = false;
};

}  // namespace veccore
