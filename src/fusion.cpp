#include "veccore/fusion.hpp"

#include <unordered_map>

namespace veccore {

std::vector<Neighbor> reciprocal_rank_fusion(const std::vector<std::vector<Neighbor>>& lists,
                                             std::size_t k_out,
                                             RrfParams params) {
    std::unordered_map<vec_id_t, double> fused;

    for (const std::vector<Neighbor>& list : lists) {
        for (std::size_t i = 0; i < list.size(); ++i) {
            // 1-based rank (P-27). The first element is rank 1, contributing
            // 1/(k+1), not 1/(k+0).
            const double rank = static_cast<double>(i + 1);
            fused[list[i].id] += 1.0 / (params.k + rank);
        }
    }

    // Note what is NOT here: no reference to any input score. Only positions.
    // That is the whole mechanism -- a document ranked 3rd by BM25 contributes
    // exactly 1/63 whether its BM25 score was 0.4 or 400.
    TopK top(k_out);
    for (const auto& [id, score] : fused) {
        top.offer(static_cast<float>(-score), id);
    }
    return top.take();
}

}  // namespace veccore
