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
#include "veccore/types.hpp"

#include <cstddef>
#include <mutex>
#include <shared_mutex>
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
    ConcurrentHnsw(const VectorStore& store, HnswParams params,
                   LockMode mode = LockMode::WriterPriority)
        : index_(store, params), mode_(mode) {}

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

    [[nodiscard]] SearchScratch make_scratch() const { return index_.make_scratch(); }
    [[nodiscard]] const HnswIndex& index() const noexcept { return index_; }
    [[nodiscard]] LockMode mode() const noexcept { return mode_; }

private:
    mutable std::shared_mutex mu_;
    mutable std::mutex turnstile_;
    HnswIndex index_;
    LockMode mode_;
};

}  // namespace veccore
