#pragma once

// Distance kernels.  Header-only and templated on nothing but the metric, so
// they inline into the search loop.
//
// D6, and it will be probed: these are *functors*, not std::function and not
// virtual methods.  A virtual call or a std::function indirection per candidate
// would sit in the hottest loop in the project -- HNSW at ef_search=64 on
// SIFT1M evaluates thousands of distances per query -- and it would also block
// the compiler from vectorising, because it cannot inline through a function
// pointer.  Templates cost compile time and buy an inlined, vectorisable loop.

#include "veccore/types.hpp"

#include <cstddef>

namespace veccore {

/// Squared Euclidean distance.  Smaller is closer.
///
/// D4: no sqrt.  sqrt is monotonic, so taking it cannot change a ranking -- and
/// it is an expensive instruction to execute per candidate.  SIFT's published
/// ground truth is ordered by this same quantity, which is what makes our recall
/// numbers directly comparable to the field's.
///
/// Written as a plain loop on purpose.  D11: -O3 -march=native auto-vectorises
/// this into AVX2 (verify with -fopt-info-vec-optimized before believing it),
/// and hand-written intrinsics only get written if a measurement says the
/// compiler left something behind.
struct L2Sqr {
    [[nodiscard]] float operator()(const float* __restrict a,
                                   const float* __restrict b,
                                   dim_t d) const noexcept {
        float sum = 0.0f;
        for (dim_t i = 0; i < d; ++i) {
            const float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    /// Ordering for "best first": for L2, smaller distance is better.
    [[nodiscard]] static constexpr bool better(float lhs, float rhs) noexcept {
        return lhs < rhs;
    }
};

/// Negated inner product.  Larger dot product means more similar, but the whole
/// search machinery is written around "smaller is better", so we negate here.
///
/// D4 warns about exactly this sign flip: it belongs in ONE place. Handling it
/// by branching inside the search would give P-04-shaped symptoms (plausible
/// results, quietly bad recall) from a completely different cause.
///
/// Cosine similarity is this, on L2-normalised vectors.  It is deliberately not
/// a third implementation.
struct NegInnerProduct {
    [[nodiscard]] float operator()(const float* __restrict a,
                                   const float* __restrict b,
                                   dim_t d) const noexcept {
        float sum = 0.0f;
        for (dim_t i = 0; i < d; ++i) {
            sum += a[i] * b[i];
        }
        return -sum;
    }

    [[nodiscard]] static constexpr bool better(float lhs, float rhs) noexcept {
        return lhs < rhs;
    }
};

/// Runtime dispatch, for the boundaries where the metric is a config value
/// (the CLI, the Python bindings).  Never call this inside a search loop --
/// that is what the functors above are for.
[[nodiscard]] inline float distance(Metric metric, const float* a, const float* b, dim_t d) noexcept {
    switch (metric) {
        case Metric::L2Squared:    return L2Sqr{}(a, b, d);
        case Metric::InnerProduct: return NegInnerProduct{}(a, b, d);
    }
    return 0.0f;  // unreachable; silences -Wreturn-type on some compilers
}

}  // namespace veccore
