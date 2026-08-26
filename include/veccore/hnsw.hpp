#pragma once

// Hierarchical Navigable Small World index.
//
// Transcribed from Malkov & Yashunin, arXiv:1603.09320, Algorithms 1-5.
// Method names below deliberately match the paper's, so the code can be read
// side by side with it -- which is the point, because you will be asked
// line-level questions about this file.
//
// The three knobs, and the distinction people get wrong under pressure:
//   M               build time  -- neighbours per node at layers >= 1
//   ef_construction build time  -- beam width while inserting
//   ef_search       QUERY time  -- beam width while searching. This is the
//                                 recall/speed dial that draws the curve.

#include "veccore/distance.hpp"
#include "veccore/flat_index.hpp"
#include "veccore/storage.hpp"
#include "veccore/types.hpp"
#include "veccore/visited.hpp"

#include <cstddef>
#include <cstdint>
#include <queue>
#include <random>
#include <string>
#include <vector>

namespace veccore {

/// Candidate ordering with top() = CLOSEST. The partner of WorseFirst, which
/// gives top() = farthest.
///
/// B-01 is why both comparators are written out explicitly with their intent in
/// a comment rather than inferred at the call site. P-04 is the same mistake one
/// level up: swap these two and the search still runs, still returns k results,
/// and recall is quietly terrible.
struct CloserFirst {
    bool operator()(const Neighbor& a, const Neighbor& b) const noexcept {
        if (a.dist != b.dist) return a.dist > b.dist;  // min-heap on distance
        return a.id > b.id;                            // smaller id wins ties
    }
};

struct HnswParams {
    std::size_t M = 16;                 ///< neighbours per node, layers >= 1
    std::size_t ef_construction = 200;  ///< beam width during insert
    std::uint64_t seed = 42;            ///< P-03: recorded in every JSON record
    /// Algorithm 4's extendCandidates. Off by default -- the paper says it
    /// helps mainly for clustered data, and it costs build time.
    bool extend_candidates = false;
    /// Algorithm 4's keepPrunedConnections. On: backfill up to M from the
    /// rejected set so degree does not collapse in sparse regions.
    bool keep_pruned = true;

    [[nodiscard]] std::size_t m_max0() const noexcept { return M * 2; }  ///< paper's M_0
    [[nodiscard]] double mL() const noexcept;
};

/// Per-thread scratch space for a search.
///
/// This exists because of a race that Phase 2 shipped and Phase 5 exposed
/// (B-10). `search` needs a visited set and a distance counter. Holding them as
/// members of the index looks harmless single-threaded, and it is -- but under a
/// `shared_mutex` **multiple readers run concurrently by design**, and a shared
/// lock grants shared ownership of the index, not permission to write to it.
/// Two concurrent searches would corrupt each other's visited stamps and race
/// on the counter.
///
/// The counter matters separately. Making it a `std::atomic` would have been the
/// obvious fix and a bad one: it is incremented once per distance computation --
/// thousands of times per query -- so it would put an atomic RMW in the hottest
/// loop in the project AND put every thread's increments on the same cache line.
/// That is textbook false sharing, and it would have shown up as "concurrency
/// does not scale" with the lock taking the blame. Per-thread state has neither
/// problem.
struct SearchScratch {
    VisitedList visited;
    std::size_t distance_calls = 0;

    void reset_counters() noexcept { distance_calls = 0; }
};

struct HnswStats {
    std::size_t nodes = 0;
    std::size_t max_level = 0;
    std::vector<std::size_t> nodes_per_level;
    std::size_t edges_layer0 = 0;
    double mean_degree_layer0 = 0.0;
    std::size_t distance_calls = 0;  ///< reset per search; the real cost metric
};

class HnswIndex {
public:
    HnswIndex(const VectorStore& store, HnswParams params);

    /// Insert every vector in the store, in id order.
    void build();

    /// Insert one vector that is already in the store.
    void insert(vec_id_t id);

    /// Algorithm 5. `ef_search` is clamped up to k (P-17): with ef < k the
    /// result set physically cannot hold k items, and returning fewer reads as
    /// catastrophic recall.
    ///
    /// **This overload is the thread-safe one.** `scratch` must be owned by the
    /// calling thread and never shared.
    [[nodiscard]] std::vector<Neighbor> search(const float* query,
                                               std::size_t k,
                                               std::size_t ef_search,
                                               SearchScratch& scratch) const;

    /// Convenience overload for single-threaded callers, using an internal
    /// scratch buffer.
    ///
    /// **NOT safe to call concurrently**, even though it is `const` and even
    /// under a shared lock -- it writes to a member. Concurrent callers must use
    /// the scratch overload above. `const` describes the logical operation, not
    /// the absence of writes (B-10).
    [[nodiscard]] std::vector<Neighbor> search(const float* query,
                                               std::size_t k,
                                               std::size_t ef_search) const;

    /// A scratch buffer correctly sized for this index.
    [[nodiscard]] SearchScratch make_scratch() const;

    [[nodiscard]] HnswStats stats() const;

    /// Bytes owned by the graph itself, excluding the vectors. Reported
    /// separately from peak RSS (P-13) and from the vector data, because
    /// "HNSW's memory overhead" is a specific number people ask for.
    [[nodiscard]] offset_t graph_bytes() const noexcept;

    /// Debug-only invariant check. Verifies no self-loops, no duplicate
    /// neighbours, and no node over its degree cap (P-16, P-06). Returns an
    /// empty string when the graph is sound, else the first violation found.
    [[nodiscard]] std::string check_invariants() const;

    [[nodiscard]] const HnswParams& params() const noexcept { return params_; }
    /// Distance calls made through the *internal* scratch only -- i.e. by
    /// insert() and by the single-threaded search overload. Concurrent callers
    /// read their own `SearchScratch::distance_calls`.
    [[nodiscard]] std::size_t distance_calls() const noexcept { return scratch_.distance_calls; }
    void reset_distance_calls() const noexcept { scratch_.distance_calls = 0; }

private:
    /// Algorithm 2. Returns the ef closest nodes found on `layer`, as a
    /// max-heap ordered so top() is the farthest.
    using ResultHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, WorseFirst>;
    using CandidateHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, CloserFirst>;

    ResultHeap search_layer(const float* query, vec_id_t entry, std::size_t ef,
                            std::size_t layer, SearchScratch& scratch) const;

    /// Algorithm 4. Consumes `candidates`, returns at most `m` chosen ids.
    void select_neighbors_heuristic(const float* base_vec,
                                    CandidateHeap& candidates,
                                    std::size_t m,
                                    std::vector<vec_id_t>& out,
                                    SearchScratch& scratch) const;

    [[nodiscard]] std::size_t random_level();

    // --- flat adjacency, D5 -------------------------------------------------
    // Layer 0: one contiguous buffer, stride (m_max0 + 1). Slot 0 holds the
    // neighbour count, slots 1.. hold ids. Layer 0 holds every node and is
    // where essentially all the traversal happens, so it gets the good layout.
    [[nodiscard]] std::uint32_t* links0(vec_id_t id) noexcept {
        return links0_.data() + static_cast<offset_t>(id) * stride0_;
    }
    [[nodiscard]] const std::uint32_t* links0(vec_id_t id) const noexcept {
        return links0_.data() + static_cast<offset_t>(id) * stride0_;
    }

    // Upper layers: also one contiguous buffer. A node's level never changes
    // after insert, so its block is appended once and never moves -- which lets
    // upper layers be flat too, rather than the per-node malloc that the
    // reference implementation uses.
    [[nodiscard]] std::uint32_t* links_upper(vec_id_t id, std::size_t layer) noexcept {
        return upper_.data() + upper_offset_[id] + (layer - 1) * strideU_;
    }
    [[nodiscard]] const std::uint32_t* links_upper(vec_id_t id, std::size_t layer) const noexcept {
        return upper_.data() + upper_offset_[id] + (layer - 1) * strideU_;
    }

    [[nodiscard]] std::uint32_t* links(vec_id_t id, std::size_t layer) noexcept {
        return layer == 0 ? links0(id) : links_upper(id, layer);
    }
    [[nodiscard]] const std::uint32_t* links(vec_id_t id, std::size_t layer) const noexcept {
        return layer == 0 ? links0(id) : links_upper(id, layer);
    }

    void set_links(vec_id_t id, std::size_t layer, const std::vector<vec_id_t>& ids);

    [[nodiscard]] float dist(const float* a, const float* b, SearchScratch& scratch) const noexcept {
        ++scratch.distance_calls;
        return metric_(a, b, store_.dim());
    }

    const VectorStore& store_;
    HnswParams params_;
    L2Sqr metric_{};

    std::vector<std::uint32_t> links0_;
    std::vector<std::uint32_t> upper_;
    std::vector<offset_t> upper_offset_;
    std::vector<std::uint8_t> levels_;

    std::size_t stride0_ = 0;
    std::size_t strideU_ = 0;

    vec_id_t entry_point_ = 0;
    std::size_t max_level_ = 0;
    bool empty_ = true;

    std::mt19937_64 rng_;
    /// Used by insert() (single writer, exclusive by construction) and by the
    /// single-threaded search overload. Never touched by the scratch overload.
    mutable SearchScratch scratch_;
};

}  // namespace veccore
