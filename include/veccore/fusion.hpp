#pragma once

// Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).
//
// Four lines of code, and it is asked about because that combination of trivial
// and effective is genuinely interesting.
//
//     score(d) = SUM over retrievers of  1 / (k + rank of d in that retriever)
//
// **Why not just add the dense and sparse scores?** Because they live on
// incomparable scales. Cosine similarity is bounded in [-1, 1]. BM25 is
// unbounded and depends on corpus statistics -- N, df, avgdl -- so its typical
// magnitude changes when the corpus changes. Adding them means one silently
// dominates, and any weight you pick is fitted to today's data and quietly
// wrong on tomorrow's.
//
// RRF throws the magnitudes away and keeps only the ordering. A rank is a rank.
// No normalisation, no tuning, nothing to re-fit when the corpus grows.
//
// **Why k = 60?** It is the original paper's value and the method is famously
// insensitive to it -- which is the point rather than a caveat. k dampens the
// influence of the very top ranks: with k=60, rank 1 contributes 1/61 and rank
// 2 contributes 1/62, a 1.6% difference, so no single retriever's confident
// first place can steamroll the other's opinion. A small k would make the
// fusion behave like "whoever ranked something first wins".

#include "veccore/flat_index.hpp"
#include "veccore/types.hpp"

#include <cstddef>
#include <vector>

namespace veccore {

struct RrfParams {
    double k = 60.0;
};

/// Fuse several ranked lists into one.
///
/// Each input list must be ordered best-first. **Ranks are 1-based** -- P-27:
/// 0-based turns 1/(60+1) into 1/(60+0), which barely moves results, and that
/// is exactly the problem. Two runs disagree slightly and neither is
/// reproducible from the README. The convention is fixed here, tested, and
/// written down.
///
/// Returns `Neighbor` with **dist = -rrf_score**, keeping the codebase-wide
/// "smaller is better" convention (D4).
[[nodiscard]] std::vector<Neighbor> reciprocal_rank_fusion(
    const std::vector<std::vector<Neighbor>>& lists,
    std::size_t k_out,
    RrfParams params = {});

}  // namespace veccore
