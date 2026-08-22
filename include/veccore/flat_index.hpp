#pragma once

// Exact brute-force search.  This is not a contender -- it is the GROUND TRUTH
// that every approximate method in this project gets graded against.
//
// PLAN.md Phase 1: "You cannot claim your fast method is 95% accurate unless you
// have the exact answer to compare it to."  Phase 1's gate is this matching
// SIFT's published ground truth exactly, and nothing after Phase 1 is
// trustworthy if that gate was waved through.

#include "veccore/distance.hpp"
#include "veccore/storage.hpp"
#include "veccore/types.hpp"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <vector>

namespace veccore {

struct Neighbor {
    float    dist = 0.0f;
    vec_id_t id   = 0;
};

/// Strict weak ordering for "worse than", used to build a MAX-heap whose top is
/// the worst result currently held -- the one to evict when something better
/// arrives.
///
/// Ties break on the smaller id.  This matters more than it looks: P-12 says a
/// tie at rank k makes recall land at something like 0.997 and look like a bug.
/// Deterministic tie-breaking will not make our answer agree with a ground
/// truth that broke ties differently, but it does make OUR answer reproducible
/// across runs, which is what turns "is this a bug?" into a question with an
/// answer.
/// Reads as "a is better than b".  std::priority_queue's top() is the GREATEST
/// element under the comparator, so making "greater" mean "worse" puts the
/// eviction candidate on top.
///
/// Both clauses point the same way and it is easy to get the second one
/// backwards -- it was, on the first attempt (B-01). Larger distance is worse;
/// among equal distances, larger id is worse, because smaller id wins ties.
struct WorseFirst {
    bool operator()(const Neighbor& a, const Neighbor& b) const noexcept {
        if (a.dist != b.dist) return a.dist < b.dist;
        return a.id < b.id;
    }
};

/// Bounded top-k collector.
///
/// A max-heap of size k, not a full sort of n: O(n log k) instead of
/// O(n log n).  At k=10 against n=1e6 that is not a rounding error -- log2(10)
/// is 3.3 against log2(1e6) = 20, and the heap also stays resident in L1 while
/// the full sort would touch 1M elements twice.
class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) {}

    void offer(float dist, vec_id_t id) {
        if (heap_.size() < k_) {
            heap_.push(Neighbor{dist, id});
        } else if (WorseFirst{}(Neighbor{dist, id}, heap_.top())) {
            heap_.pop();
            heap_.push(Neighbor{dist, id});
        }
    }

    /// Drain into ascending-distance order.  Destroys the collector.
    [[nodiscard]] std::vector<Neighbor> take() {
        std::vector<Neighbor> out;
        out.reserve(heap_.size());
        while (!heap_.empty()) {
            out.push_back(heap_.top());
            heap_.pop();
        }
        std::reverse(out.begin(), out.end());  // heap drains worst-first
        return out;
    }

    [[nodiscard]] std::size_t size() const noexcept { return heap_.size(); }

private:
    std::size_t k_;
    std::priority_queue<Neighbor, std::vector<Neighbor>, WorseFirst> heap_;
};

/// Exhaustive search over a VectorStore.  Templated on the distance functor so
/// the metric inlines into the scan (see distance.hpp).
template <typename Dist>
class FlatIndex {
public:
    explicit FlatIndex(const VectorStore& store) : store_(store) {}

    [[nodiscard]] std::vector<Neighbor> search(const float* query, std::size_t k) const {
        TopK top(k);
        const vec_id_t n = store_.size();
        const dim_t d = store_.dim();
        for (vec_id_t i = 0; i < n; ++i) {
            top.offer(dist_(query, store_.at(i), d), i);
        }
        return top.take();
    }

    [[nodiscard]] const VectorStore& store() const noexcept { return store_; }

private:
    const VectorStore& store_;
    Dist dist_{};
};

using FlatL2 = FlatIndex<L2Sqr>;
using FlatIP = FlatIndex<NegInnerProduct>;

}  // namespace veccore
