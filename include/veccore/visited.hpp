#pragma once

// Epoch-stamped visited set.
//
// BUGS.md P-09. The obvious implementation is a std::unordered_set per query,
// and it is correct and slow enough to dominate the search: hashing, plus an
// allocation per query, plus pointer-chasing through buckets -- all to answer a
// question about dense integer ids in [0, n).
//
// The fix is to spend n * 2 bytes once: an array of epoch stamps and a counter.
// A node is visited iff stamp[id] == current_epoch. Starting a new query is
// ++epoch, which is O(1) and touches nothing. No clearing, no allocation, no
// hashing.
//
// The subtle part -- and P-09 names it because it is nearly invisible -- is the
// wrap. uint16_t epochs run out after 65535 queries. If the wrap is not
// handled, epoch returns to a value some stamps still hold, and those nodes
// look pre-visited: the search silently skips them and returns fewer, worse
// results for ONE query in every 65536. That is far more dangerous than a bug
// that fires every time, because an aggregate recall number over 10,000 queries
// will not move enough to notice.

#include "veccore/types.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace veccore {

class VisitedList {
public:
    VisitedList() = default;

    explicit VisitedList(std::size_t n) { resize(n); }

    void resize(std::size_t n) {
        stamps_.assign(n, 0);
        epoch_ = 0;
    }

    /// Begin a new query. O(1) in the common case.
    void reset() {
        if (epoch_ == kMaxEpoch) {
            // The wrap. Zero the array and start over, so no stale stamp can
            // ever equal a live epoch. Happens once per 65535 queries and costs
            // one memset of n bytes -- amortised to nothing, and it is the
            // difference between correct and subtly-wrong-once-in-a-while.
            std::memset(stamps_.data(), 0, stamps_.size() * sizeof(Stamp));
            epoch_ = 0;
            ++wraps_;
        }
        ++epoch_;
    }

    /// Mark visited; returns true if this call was the first to visit it.
    [[nodiscard]] bool test_and_set(vec_id_t id) noexcept {
        if (stamps_[id] == epoch_) return false;
        stamps_[id] = epoch_;
        return true;
    }

    [[nodiscard]] bool is_visited(vec_id_t id) const noexcept { return stamps_[id] == epoch_; }

    [[nodiscard]] std::size_t size() const noexcept { return stamps_.size(); }
    [[nodiscard]] std::size_t bytes() const noexcept { return stamps_.size() * sizeof(Stamp); }
    /// Exposed only so a test can force the wrap without issuing 65535 queries.
    [[nodiscard]] std::uint64_t wraps() const noexcept { return wraps_; }
    void force_epoch_for_test(std::uint16_t e) noexcept { epoch_ = e; }

private:
    using Stamp = std::uint16_t;
    static constexpr Stamp kMaxEpoch = 65535;

    std::vector<Stamp> stamps_;
    Stamp epoch_ = 0;
    std::uint64_t wraps_ = 0;
};

}  // namespace veccore
