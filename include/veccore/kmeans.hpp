#pragma once

// Lloyd's algorithm with k-means++ initialisation, used to learn PQ codebooks.
//
// Small, well-understood, and with exactly one way to destroy everything
// downstream silently -- see P-07 and `empty_cluster_reseeds` below.

#include "veccore/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace veccore {

struct KMeansParams {
    std::size_t k = 256;         ///< 256 because that is what fits in one byte
    std::size_t max_iters = 25;  ///< convergence is not the goal; a good-enough codebook is
    double tol = 1e-4;           ///< stop when inertia improves by less than this fraction
    std::uint64_t seed = 42;
    /// Independent restarts, keeping the lowest-inertia run.
    ///
    /// B-07: k-means++ reduces bad initialisations, it does not eliminate them.
    /// Lloyd's only moves centroids downhill, so two centroids seeded inside
    /// one cluster while a neighbouring cluster gets none is a state it can
    /// never escape. Measured on 4 blobs where two sat 2.9 apart with jitter
    /// 0.2: one seed in four converged to 14x the inertia of the other three.
    ///
    /// Costs n_init x the training time and buys codebooks that do not depend
    /// on a lucky seed -- which matters because a bad codebook in ONE PQ
    /// subspace quietly degrades recall for every query.
    std::size_t n_init = 3;
};

struct KMeansResult {
    std::vector<float> centroids;  ///< k * d, row-major
    std::size_t k = 0;
    dim_t d = 0;
    double inertia = 0.0;          ///< sum of squared distances to assigned centroid
    std::size_t iters = 0;
    /// How many times a centroid ended an iteration owning zero points and had
    /// to be re-seeded. Reported rather than hidden: it is a signal that k is
    /// too large for the data, and it is the counter that proves P-07's guard
    /// is doing something.
    std::size_t empty_cluster_reseeds = 0;
    /// Inertia of every restart, best first. A wide spread here means the data
    /// has bad local minima and n_init is earning its keep.
    std::vector<double> inertia_per_init;

    [[nodiscard]] const float* centroid(std::size_t i) const noexcept {
        return centroids.data() + i * static_cast<offset_t>(d);
    }
};

/// `data` is n * d row-major.
KMeansResult kmeans(const float* data, std::size_t n, dim_t d, const KMeansParams& params);

}  // namespace veccore
