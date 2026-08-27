#pragma once

// Concurrent access to an HNSW index.
//
// CONTEXT.md D8: one std::shared_mutex for the whole index. Readers take a
// shared_lock, the inserter takes a unique_lock. Coarse, correct, and measured
// -- which beats fine-grained and racy, because a subtle race in a graph
// mutation path produces a corrupted index that still returns k results, and
// there is no assertion anywhere that would catch it.
//
// ---------------------------------------------------------------------------
// The pathology this file exists to handle: WRITER STARVATION (B-11)
// ---------------------------------------------------------------------------
// `std::shared_mutex` on libstdc++ is a `pthread_rwlock_t`, and glibc's default
// is **reader-preferring**. A steady stream of readers therefore blocks a writer
// indefinitely -- not "slows it down", blocks it. Measured on this machine,
// 200 inserts against a continuous reader load:
//
//     readers=0   4.0 ms total   mean 0.02 ms   worst 0.05 ms
//     readers=1  15.2 ms total   mean 0.08 ms   worst 0.31 ms
//     readers=2 278.6 ms total   mean 1.39 ms   worst 7.83 ms
//     readers=4  DID NOT COMPLETE in 30 s
//     readers=8  DID NOT COMPLETE in 30 s
//
// A vector index is exactly the workload that triggers this: many concurrent
// searches, occasional inserts. The default is the wrong default here.
//
// `WriterPriority` fixes it with a turnstile -- a plain mutex that everyone
// touches briefly on the way in, and that a writer *holds* while it waits for
// the unique_lock. New readers queue behind the writer instead of overtaking
// it; readers already inside drain; the writer gets in. Roughly ten lines, and
// the standard construction.
//
// The cost is real and is measured rather than assumed (P-30): every reader now
// takes an extra mutex, so all readers serialise briefly on the turnstile even
// when no writer exists. Whether that hurts read throughput is an empirical
// question, and `bench --lock-mode` answers it.

#include "veccore/hnsw.hpp"
#include "veccore/storage.hpp"
#include "veccore/types.hpp"

#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace veccore {

enum class LockMode {
    /// shared_mutex alone. Reader-preferring on glibc: fastest reads, and it
    /// starves writers under sustained read load (B-11).
    SharedMutex,
    /// shared_mutex plus a turnstile. Bounded writer latency; costs readers one
    /// uncontended mutex each.
    WriterPriority,
    /// UNSAFE with a writer. Read-only control for P-30: "is the lock actually
    /// what limits read scaling?" answered by measurement, not assertion.
    None,
};

[[nodiscard]] inline const char* to_string(LockMode m) noexcept {
    switch (m) {
        case LockMode::SharedMutex:    return "shared_mutex";
        case LockMode::WriterPriority: return "writer_priority";
        case LockMode::None:           return "none";
    }
    return "unknown";
}

class ConcurrentHnsw {
public:
    /// Read-only store: search and `insert(id)` work, `insert(const float*)`
    /// does not, because there is nothing it is allowed to append to.
    ConcurrentHnsw(const VectorStore& store, HnswParams params,
                   LockMode mode = LockMode::WriterPriority)
        : index_(store, params), mode_(mode) {}

    /// Mutable store: additionally enables `insert(const float*)`, which grows
    /// the store and the graph together.
    ConcurrentHnsw(VectorStore& store, HnswParams params,
                   LockMode mode = LockMode::WriterPriority)
        : index_(store, params), mode_(mode), growable_(&store) {}

    /// Adopt an index that came off disk.
    ///
    /// `store` must be the very store `HnswIndex::load` filled -- the moved-in
    /// index already holds a reference to it, and it must outlive this object.
    /// Taking it as a parameter anyway is not redundant: it is what re-enables
    /// growth, since the loaded index borrows the store as const and only this
    /// handle can append to it.
    ConcurrentHnsw(VectorStore& store, HnswIndex&& loaded,
                   LockMode mode = LockMode::WriterPriority)
        : index_(std::move(loaded)), mode_(mode), growable_(&store) {}

    /// Build single-threaded. Building under the lock would be pointless -- the
    /// writer is exclusive anyway, so there is nothing to overlap with.
    void build() { index_.build(); }

    /// Caller-owned scratch is the whole point -- see `SearchScratch`. A shared
    /// scratch would be a data race between readers even though they hold only
    /// shared locks: "shared" means shared ownership of the index, not
    /// permission to write to it (B-10).
    [[nodiscard]] std::vector<Neighbor> search(const float* query, std::size_t k,
                                               std::size_t ef, SearchScratch& scratch) const {
        switch (mode_) {
            case LockMode::None:
                return index_.search(query, k, ef, scratch);

            case LockMode::WriterPriority: {
                // Touch the turnstile and let go immediately. If a writer holds
                // it, this blocks here rather than overtaking the writer at the
                // rwlock -- which is the entire fix.
                { const std::lock_guard<std::mutex> gate(turnstile_); }
                const std::shared_lock<std::shared_mutex> lock(mu_);
                return index_.search(query, k, ef, scratch);
            }

            case LockMode::SharedMutex:
            default: {
                const std::shared_lock<std::shared_mutex> lock(mu_);
                return index_.search(query, k, ef, scratch);
            }
        }
    }

    void insert(vec_id_t id) {
        if (mode_ == LockMode::WriterPriority) {
            // Hold the turnstile across the wait AND the write. Readers arriving
            // from now on block at the turnstile; readers already holding the
            // shared lock drain; then the unique_lock is granted.
            const std::lock_guard<std::mutex> gate(turnstile_);
            const std::unique_lock<std::shared_mutex> lock(mu_);
            index_.insert(id);
            return;
        }
        const std::unique_lock<std::shared_mutex> lock(mu_);
        index_.insert(id);
    }

    /// Append a vector to the store and link it into the graph, both inside one
    /// exclusive section. Returns the new id.
    ///
    /// **Why the append cannot happen outside the lock**, which is the whole
    /// reason this overload exists rather than leaving callers to do the two
    /// steps themselves: `VectorStore::add` appends to a `std::vector<float>`,
    /// so it **may reallocate**. Every concurrent reader is at that moment
    /// holding `const float*` pointers into the old buffer -- `search_layer`
    /// keeps one for the query's neighbours across the whole beam search. A
    /// reallocation under them is a use-after-free that does not crash: it reads
    /// freed-but-still-mapped memory and returns k plausible, wrong neighbours.
    /// Taking the unique_lock first means every reader has drained before the
    /// buffer moves.
    ///
    /// `insert(id)` on a pre-populated store has no such hazard, which is why
    /// the C++ benchmarks and tests could use it without this machinery -- they
    /// size the store up front. A Python caller cannot.
    vec_id_t insert(const float* vec) {
        if (growable_ == nullptr) {
            throw std::logic_error(
                "ConcurrentHnsw::insert(const float*): this index was constructed over a const "
                "VectorStore, so it cannot grow. Construct it with a mutable store to add vectors.");
        }
        const auto do_insert = [&] {
            const vec_id_t id = growable_->add(vec);
            index_.insert(id);
            return id;
        };
        if (mode_ == LockMode::WriterPriority) {
            const std::lock_guard<std::mutex> gate(turnstile_);
            const std::unique_lock<std::shared_mutex> lock(mu_);
            return do_insert();
        }
        const std::unique_lock<std::shared_mutex> lock(mu_);
        return do_insert();
    }

    /// Append and link `n` vectors under **one** exclusive section.
    ///
    /// Not just a convenience: inserting one at a time reacquires the lock per
    /// vector, and under `WriterPriority` each acquisition stalls every reader
    /// at the turnstile. One batch is one stall. It also lets the store and the
    /// graph reserve once instead of growing geometrically n times.
    std::vector<vec_id_t> insert_batch(const float* vecs, std::size_t n, dim_t dim) {
        if (growable_ == nullptr) {
            throw std::logic_error(
                "ConcurrentHnsw::insert_batch: this index was constructed over a const "
                "VectorStore, so it cannot grow. Construct it with a mutable store to add vectors.");
        }
        std::vector<vec_id_t> ids;
        ids.reserve(n);
        const auto do_batch = [&] {
            growable_->reserve(growable_->size() + n);
            index_.grow_to(growable_->size() + n);
            for (std::size_t i = 0; i < n; ++i) {
                const vec_id_t id = growable_->add(vecs + i * static_cast<offset_t>(dim));
                index_.insert(id);
                ids.push_back(id);
            }
        };
        if (mode_ == LockMode::WriterPriority) {
            const std::lock_guard<std::mutex> gate(turnstile_);
            const std::unique_lock<std::shared_mutex> lock(mu_);
            do_batch();
            return ids;
        }
        const std::unique_lock<std::shared_mutex> lock(mu_);
        do_batch();
        return ids;
    }

    /// Persist the index, under the shared lock.
    ///
    /// The lock is not optional. `save` walks `links0_`, `levels_` and the store
    /// while writing them out; an insert landing halfway through produces a file
    /// that is internally inconsistent but still passes its own checksum,
    /// because the checksum is computed over whatever was written. That file
    /// then loads cleanly and returns wrong neighbours -- the same silent-corruption
    /// class as B-13, one layer out.
    void save(const std::string& path) const {
        const std::shared_lock<std::shared_mutex> lock(mu_);
        index_.save(path);
    }

    [[nodiscard]] SearchScratch make_scratch() const {
        const std::shared_lock<std::shared_mutex> lock(mu_);
        return index_.make_scratch();
    }

    /// The bare index, **unsynchronised**.
    ///
    /// Safe only when no writer can run: after joining every thread, or on an
    /// index that never grows. It became sharper than it looks once `insert`
    /// could grow the graph -- `grow_to` reallocates `levels_`, `links0_` and
    /// `upper_offset_`, so even reading `capacity()` through this handle races
    /// with a concurrent insert. TSan caught exactly that in the first draft of
    /// the growth test (B-13). Use the locked accessors below from anything
    /// that runs alongside a writer.
    [[nodiscard]] const HnswIndex& index() const noexcept { return index_; }

    /// Locked accessors. These exist because the interesting reads -- how big is
    /// the graph, is it still sound -- are exactly the ones a monitoring thread
    /// wants to make *while* inserts are happening.
    [[nodiscard]] std::size_t capacity() const {
        const std::shared_lock<std::shared_mutex> lock(mu_);
        return index_.capacity();
    }
    [[nodiscard]] HnswStats stats() const {
        const std::shared_lock<std::shared_mutex> lock(mu_);
        return index_.stats();
    }
    [[nodiscard]] offset_t graph_bytes() const {
        const std::shared_lock<std::shared_mutex> lock(mu_);
        return index_.graph_bytes();
    }
    [[nodiscard]] std::string check_invariants() const {
        const std::shared_lock<std::shared_mutex> lock(mu_);
        return index_.check_invariants();
    }

    [[nodiscard]] LockMode mode() const noexcept { return mode_; }
    [[nodiscard]] bool growable() const noexcept { return growable_ != nullptr; }

private:
    mutable std::shared_mutex mu_;
    mutable std::mutex turnstile_;
    HnswIndex index_;
    LockMode mode_;
    /// Non-null only when constructed over a mutable store. The index itself
    /// borrows the store as const, so this is the only handle able to grow it.
    VectorStore* growable_ = nullptr;
};

}  // namespace veccore
