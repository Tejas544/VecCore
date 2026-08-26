#include "veccore/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace veccore {

double HnswParams::mL() const noexcept {
    // P-15. The paper's normalisation factor is 1/ln(M). Using 1/M or ln(M)
    // instead gives a badly shaped layer distribution -- too many layers (wasted
    // descent) or too few (no long-range structure) -- and neither errors.
    // The level histogram in stats() is what catches it.
    return 1.0 / std::log(static_cast<double>(M));
}

HnswIndex::HnswIndex(const VectorStore& store, HnswParams params)
    : store_(store), params_(params), rng_(params.seed) {
    if (params_.M < 2) throw std::invalid_argument("HnswIndex: M must be >= 2");
    stride0_ = params_.m_max0() + 1;
    strideU_ = params_.M + 1;

    const std::size_t n = store_.size();
    links0_.assign(n * stride0_, 0);
    upper_offset_.assign(n, 0);
    levels_.assign(n, 0);
    scratch_.visited.resize(n);
}

std::size_t HnswIndex::random_level() {
    // level = floor(-ln(U(0,1)) * mL), an exponentially decaying draw: most
    // nodes land at level 0, a few reach higher, one or two reach the top.
    // Nobody designs the hierarchy; it falls out of this line.
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng_);
    if (r <= 0.0) r = std::numeric_limits<double>::min();  // -ln(0) is infinite
    const double lvl = -std::log(r) * params_.mL();
    // Cap so a freak draw cannot allocate an absurd number of layers; uint8_t
    // levels_ makes the cap explicit rather than accidental.
    return static_cast<std::size_t>(std::min(lvl, 30.0));
}

void HnswIndex::set_links(vec_id_t id, std::size_t layer, const std::vector<vec_id_t>& ids) {
    std::uint32_t* l = links(id, layer);
    const std::size_t cap = (layer == 0) ? params_.m_max0() : params_.M;
    const std::size_t n = std::min(ids.size(), cap);
    l[0] = static_cast<std::uint32_t>(n);
    for (std::size_t i = 0; i < n; ++i) l[i + 1] = ids[i];
}

SearchScratch HnswIndex::make_scratch() const {
    SearchScratch s;
    s.visited.resize(store_.size());
    return s;
}

HnswIndex::ResultHeap HnswIndex::search_layer(const float* query,
                                              vec_id_t entry,
                                              std::size_t ef,
                                              std::size_t layer,
                                              SearchScratch& scratch) const {
    scratch.visited.reset();

    ResultHeap W;       // top() = FARTHEST kept result; the eviction candidate
    CandidateHeap C;    // top() = CLOSEST unexplored candidate

    const float d_entry = dist(query, store_.at(entry), scratch);
    (void)scratch.visited.test_and_set(entry);
    W.push({d_entry, entry});
    C.push({d_entry, entry});

    while (!C.empty()) {
        const Neighbor c = C.top();
        // Termination: the closest thing left to explore is farther than the
        // worst thing we already hold, so nothing reachable from here can
        // improve the result. This comparison is why W must be a max-heap.
        if (c.dist > W.top().dist) break;
        C.pop();

        const std::uint32_t* l = links(c.id, layer);
        const std::uint32_t degree = l[0];
        for (std::uint32_t i = 1; i <= degree; ++i) {
            const auto e = static_cast<vec_id_t>(l[i]);
            if (!scratch.visited.test_and_set(e)) continue;

            const float d_e = dist(query, store_.at(e), scratch);
            if (W.size() < ef || d_e < W.top().dist) {
                C.push({d_e, e});
                W.push({d_e, e});
                if (W.size() > ef) W.pop();
            }
        }
    }
    return W;
}

void HnswIndex::select_neighbors_heuristic(const float* base_vec,
                                           CandidateHeap& candidates,
                                           std::size_t m,
                                           std::vector<vec_id_t>& out,
                                           SearchScratch& scratch) const {
    out.clear();

    // Algorithm 4, and P-05 is the whole reason this function is not three
    // lines of "take the m closest".
    //
    // Naive top-m links every node in a dense cluster only to others in the
    // same cluster. The graph fragments into tight islands with no bridges;
    // greedy search walks into one and cannot leave, because every neighbour
    // looks worse while the true answer sits elsewhere. The signature symptom
    // is recall plateauing around 0.6-0.8 and NOT improving when ef_search
    // rises -- flat in ef_search means the graph is wrong, not the search.
    //
    // The heuristic: keep a candidate only if it is closer to us than to any
    // neighbour we have already kept. That deliberately spends link budget on
    // diverse directions instead of a redundant huddle, and those long-range
    // links are what make the graph navigable rather than merely connected.
    if (params_.extend_candidates) {
        // Optional: pull in the candidates' own neighbours first. Helps on
        // clustered data, costs build time. Off by default.
        std::vector<Neighbor> seen;
        seen.reserve(candidates.size());
        CandidateHeap copy = candidates;
        std::unordered_set<vec_id_t> present;
        while (!copy.empty()) { present.insert(copy.top().id); seen.push_back(copy.top()); copy.pop(); }
        for (const Neighbor& c : seen) {
            const std::uint32_t* l = links(c.id, 0);
            for (std::uint32_t i = 1; i <= l[0]; ++i) {
                const auto e = static_cast<vec_id_t>(l[i]);
                if (present.insert(e).second) {
                    candidates.push({dist(base_vec, store_.at(e), scratch), e});
                }
            }
        }
    }

    std::vector<Neighbor> discarded;
    while (!candidates.empty() && out.size() < m) {
        const Neighbor e = candidates.top();
        candidates.pop();

        bool keep = true;
        for (const vec_id_t r : out) {
            // Closer to an already-kept neighbour than to us => redundant.
            if (dist(store_.at(e.id), store_.at(r), scratch) < e.dist) { keep = false; break; }
        }
        if (keep) out.push_back(e.id);
        else if (params_.keep_pruned) discarded.push_back(e);
    }

    // keepPrunedConnections: backfill from the rejected set rather than let
    // degree collapse in sparse regions. discarded is already in increasing
    // distance order, because it was filled from a min-heap.
    for (std::size_t i = 0; i < discarded.size() && out.size() < m; ++i) {
        out.push_back(discarded[i].id);
    }
}

void HnswIndex::insert(vec_id_t id) {
    const float* q = store_.at(id);
    const std::size_t level = random_level();
    levels_[id] = static_cast<std::uint8_t>(level);

    if (level > 0) {
        upper_offset_[id] = upper_.size();
        upper_.resize(upper_.size() + level * strideU_, 0);
    }

    if (empty_) {
        entry_point_ = id;
        max_level_ = level;
        empty_ = false;
        return;
    }

    vec_id_t ep = entry_point_;

    // Phase 1: greedy descent through layers above ours, ef = 1. This is the
    // motorway: cross the space cheaply before doing detailed work.
    for (std::size_t lc = max_level_; lc > level; --lc) {
        ResultHeap w = search_layer(q, ep, 1, lc, scratch_);
        ep = w.top().id;
    }

    // Phase 2: from our own level down to 0, beam search and link.
    std::vector<vec_id_t> chosen;
    const std::size_t start = std::min(level, max_level_);
    for (std::size_t lc = start + 1; lc-- > 0;) {
        ResultHeap w = search_layer(q, ep, params_.ef_construction, lc, scratch_);

        CandidateHeap cands;
        {
            ResultHeap tmp = w;
            while (!tmp.empty()) { cands.push(tmp.top()); tmp.pop(); }
        }
        // Descend from the CLOSEST member of the beam, not the farthest.
        // `w` is a max-heap so w.top() is the worst result -- using it would
        // still "work" (any node on the layer is a legal entry point) while
        // starting every subsequent layer from the worst place we found, which
        // costs recall for no reason and produces no error.
        ep = cands.top().id;

        const std::size_t m_target = (lc == 0) ? params_.m_max0() : params_.M;
        select_neighbors_heuristic(q, cands, params_.M, chosen, scratch_);
        set_links(id, lc, chosen);

        // Bidirectional linking, then the shrink that P-06 says gets skipped.
        for (const vec_id_t nb : chosen) {
            std::uint32_t* nl = links(nb, lc);
            const std::uint32_t deg = nl[0];

            if (deg < m_target) {
                nl[deg + 1] = id;
                nl[0] = deg + 1;
                continue;
            }

            // Over cap: re-run the heuristic on (existing neighbours + id).
            // Skipping this is silent -- degree grows without bound, memory and
            // latency inflate, and nothing ever errors.
            CandidateHeap nb_cands;
            const float* nb_vec = store_.at(nb);
            for (std::uint32_t i = 1; i <= deg; ++i) {
                const auto e = static_cast<vec_id_t>(nl[i]);
                nb_cands.push({dist(nb_vec, store_.at(e), scratch_), e});
            }
            nb_cands.push({dist(nb_vec, q, scratch_), id});

            std::vector<vec_id_t> reselected;
            select_neighbors_heuristic(nb_vec, nb_cands, m_target, reselected, scratch_);
            set_links(nb, lc, reselected);
        }
    }

    // P-18: a node drawing above the current maximum becomes the new entry
    // point. Miss this and the upper layers are never entered -- you get
    // flat-NSW behaviour while believing you built a hierarchy.
    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = id;
    }
}

void HnswIndex::build() {
    for (vec_id_t i = 0; i < store_.size(); ++i) insert(i);
}

std::vector<Neighbor> HnswIndex::search(const float* query,
                                        std::size_t k,
                                        std::size_t ef_search) const {
    return search(query, k, ef_search, scratch_);
}

std::vector<Neighbor> HnswIndex::search(const float* query,
                                        std::size_t k,
                                        std::size_t ef_search,
                                        SearchScratch& scratch) const {
    if (empty_) return {};
    const std::size_t ef = std::max(ef_search, k);  // P-17

    vec_id_t ep = entry_point_;
    for (std::size_t lc = max_level_; lc > 0; --lc) {
        ResultHeap w = search_layer(query, ep, 1, lc, scratch);
        ep = w.top().id;
    }

    ResultHeap w = search_layer(query, ep, ef, 0, scratch);
    std::vector<Neighbor> out;
    out.reserve(w.size());
    while (!w.empty()) { out.push_back(w.top()); w.pop(); }
    std::reverse(out.begin(), out.end());  // heap drains farthest-first
    if (out.size() > k) out.resize(k);
    return out;
}

HnswStats HnswIndex::stats() const {
    HnswStats s;
    s.nodes = store_.size();
    s.max_level = max_level_;
    s.nodes_per_level.assign(max_level_ + 1, 0);
    for (vec_id_t i = 0; i < store_.size(); ++i) {
        for (std::size_t l = 0; l <= levels_[i]; ++l) ++s.nodes_per_level[l];
        s.edges_layer0 += links0(i)[0];
    }
    if (s.nodes) s.mean_degree_layer0 = static_cast<double>(s.edges_layer0) / static_cast<double>(s.nodes);
    return s;
}

offset_t HnswIndex::graph_bytes() const noexcept {
    return links0_.size() * sizeof(std::uint32_t) + upper_.size() * sizeof(std::uint32_t) +
           upper_offset_.size() * sizeof(offset_t) + levels_.size() * sizeof(std::uint8_t) +
           scratch_.visited.bytes();
}

std::string HnswIndex::check_invariants() const {
    std::ostringstream err;
    for (vec_id_t i = 0; i < store_.size(); ++i) {
        for (std::size_t lc = 0; lc <= levels_[i]; ++lc) {
            const std::uint32_t* l = links(i, lc);
            const std::size_t cap = (lc == 0) ? params_.m_max0() : params_.M;
            if (l[0] > cap) {
                err << "node " << i << " layer " << lc << " has degree " << l[0]
                    << " over cap " << cap << " (P-06: the backward shrink was skipped)";
                return err.str();
            }
            std::unordered_set<std::uint32_t> seen;
            for (std::uint32_t j = 1; j <= l[0]; ++j) {
                if (l[j] == i) {
                    err << "node " << i << " layer " << lc << " links to itself (P-16)";
                    return err.str();
                }
                if (!seen.insert(l[j]).second) {
                    err << "node " << i << " layer " << lc << " has duplicate neighbour " << l[j]
                        << " (P-16)";
                    return err.str();
                }
                if (levels_[l[j]] < lc) {
                    err << "node " << i << " layer " << lc << " links to " << l[j]
                        << " which only reaches level " << static_cast<int>(levels_[l[j]]);
                    return err.str();
                }
            }
        }
    }
    return {};
}

}  // namespace veccore
