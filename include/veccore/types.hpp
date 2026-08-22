#pragma once

#include <cstddef>
#include <cstdint>

namespace veccore {

/// Dense internal identifier for a stored vector.  uint32_t caps the index at
/// ~4.29e9 vectors, which is far beyond anything that fits in this machine's
/// RAM; going to 64 bits would inflate every adjacency list by 2x for no reach.
/// Note this is the *internal* id -- the mapping to an external key (a document
/// string, say) lives at the boundary, and getting that mapping wrong is P-20.
using vec_id_t = std::uint32_t;

/// Vector dimensionality.  128 for SIFT.
using dim_t = std::uint32_t;

/// Offsets into the flat vector store are size_t, never int.  P-14: i * d
/// overflows int32 at ~2.1e9 elements, and the failure mode is silent memory
/// corruption at exactly the scale worth bragging about.
using offset_t = std::size_t;

enum class Metric {
    /// Squared L2.  D4: sqrt is monotonic so it cannot change a ranking, and
    /// skipping it removes an expensive instruction from the hottest loop.
    /// SIFT's published ground truth is defined against this.
    L2Squared,
    /// Larger is better.  Cosine is this, on L2-normalised vectors -- not a
    /// third code path (D4).
    InnerProduct,
};

}  // namespace veccore
