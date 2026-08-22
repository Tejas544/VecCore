#pragma once

// Flat, row-major vector storage.
//
// CONTEXT.md D5, and 02_VECCORE.md asks about this directly (question 13).
// This is ONE std::vector<float> of n*d, not a std::vector<std::vector<float>>.
//
// The argument, with numbers, because "it's faster" is not an answer:
//
//   vector<vector<float>> is n separate heap allocations at unpredictable
//   addresses.  Reaching vector i costs a dependent load (read the pointer,
//   then follow it) that the hardware prefetcher cannot anticipate, because the
//   address it needs is only known after the previous load retires.  Every
//   vector access is a likely cache miss -- ~200 cycles to DRAM against ~4 to
//   L1.  It also pays 24 bytes of std::vector header per row plus allocator
//   bookkeeping, so 1M SIFT vectors carry ~24-48 MB of pure overhead.
//
//   Flat storage streams linearly.  The prefetcher sees the access pattern
//   immediately.  A 128-dim float vector is exactly 512 bytes = 8 cache lines
//   with nothing wasted, and scanning n of them is the ideal case for hardware
//   prefetch.
//
// The A/B is in bench (--layout naive), because the spec calls this "the best
// performance story in the project" and an unmeasured claim is not a story.

#include "veccore/types.hpp"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace veccore {

class VectorStore {
public:
    VectorStore() = default;

    explicit VectorStore(dim_t dim) : dim_(dim) {}

    /// Take ownership of an already-laid-out n*d buffer.  Moves, never copies:
    /// SIFT1M is 512 MB and a silent copy here would be both a latency spike
    /// and 512 MB of peak RSS that lands in the memory column of a results
    /// table without explanation.
    VectorStore(std::vector<float>&& data, dim_t dim)
        : data_(std::move(data)), dim_(dim) {
        if (dim_ == 0) throw std::invalid_argument("VectorStore: dim must be non-zero");
        if (data_.size() % dim_ != 0) {
            throw std::invalid_argument("VectorStore: buffer size is not a multiple of dim");
        }
    }

    void reserve(std::size_t n) { data_.reserve(n * static_cast<offset_t>(dim_)); }

    /// Append one vector.  Returns its id.
    vec_id_t add(const float* v) {
        const vec_id_t id = size();
        data_.insert(data_.end(), v, v + dim_);
        return id;
    }

    /// Pointer to vector `id`.  offset_t (size_t) arithmetic, never int -- P-14:
    /// i*d wraps int32 around 2.1e9 elements and the failure mode is silent
    /// memory corruption at exactly the scale worth bragging about.
    [[nodiscard]] const float* at(vec_id_t id) const noexcept {
        assert(id < size());
        return data_.data() + static_cast<offset_t>(id) * static_cast<offset_t>(dim_);
    }

    [[nodiscard]] vec_id_t size() const noexcept {
        return static_cast<vec_id_t>(data_.size() / (dim_ ? dim_ : 1));
    }

    [[nodiscard]] dim_t dim() const noexcept { return dim_; }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    /// Bytes actually occupied by vector data.  Reported separately from peak
    /// RSS (P-13): RSS includes the dataset, the ground truth, and anything
    /// else resident, and quoting it as "index memory" inflates the number and
    /// makes the FAISS comparison meaningless.
    [[nodiscard]] offset_t bytes() const noexcept { return data_.size() * sizeof(float); }

    [[nodiscard]] const std::vector<float>& raw() const noexcept { return data_; }

private:
    std::vector<float> data_;
    dim_t dim_ = 0;
};

/// The deliberately bad layout, kept only so the comparison can be measured
/// rather than asserted.  Never used by the index -- bench constructs it
/// directly for the A/B.  See D5.
class NaiveVectorStore {
public:
    explicit NaiveVectorStore(dim_t dim) : dim_(dim) {}

    vec_id_t add(const float* v) {
        rows_.emplace_back(v, v + dim_);
        return static_cast<vec_id_t>(rows_.size() - 1);
    }

    [[nodiscard]] const float* at(vec_id_t id) const noexcept { return rows_[id].data(); }
    [[nodiscard]] vec_id_t size() const noexcept { return static_cast<vec_id_t>(rows_.size()); }
    [[nodiscard]] dim_t dim() const noexcept { return dim_; }

    /// Payload plus the per-row std::vector header.  Excludes allocator
    /// bookkeeping, so it *understates* the real overhead -- which is the
    /// honest direction to err in when the number argues for your own design.
    [[nodiscard]] offset_t bytes() const noexcept {
        return rows_.size() * (static_cast<offset_t>(dim_) * sizeof(float) + sizeof(std::vector<float>));
    }

private:
    std::vector<std::vector<float>> rows_;
    dim_t dim_ = 0;
};

}  // namespace veccore
