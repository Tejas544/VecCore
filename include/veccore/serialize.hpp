#pragma once

// On-disk format for VecCore indexes.
//
// CONTEXT.md D15. The short version of why this file is shaped the way it is:
// building an HNSW over SIFT1M takes 1,139 s and training PQ codebooks takes
// 86.5 s, so an index that only exists inside the process that built it is one
// you pay for again on every restart. Persistence is not a feature here so much
// as the difference between a benchmark and something usable.
//
// ---------------------------------------------------------------------------
// The format, and the four ways a binary dump silently corrupts your data
// ---------------------------------------------------------------------------
// Writing `reinterpret_cast<const char*>(vec.data())` to a file is three lines
// and works perfectly until it does not. Every field in the header below exists
// because of a specific way that shortcut fails *without an error*:
//
//   1. **Wrong file entirely.** Load a JPEG as an index and you get a graph of
//      garbage ids that still returns k results. -> MAGIC, checked first.
//   2. **Format drift.** v1 wrote `levels_` as uint8, v2 as uint16; a v1 file
//      read by v2 misparses every field after it and the recall just gets
//      worse. -> VERSION, exact-match only. There is no forward compatibility
//      and this file does not pretend otherwise.
//   3. **Different machine.** Endianness and type widths change what the same
//      bytes mean. Nothing in a float array announces its byte order.
//      -> ENDIAN_PROBE and the sizeof fields, which turn "wrong answers" into
//      "refuses to load".
//   4. **Truncated or corrupted file.** A write interrupted at 90% leaves a
//      file whose header parses cleanly. -> a checksum over the payload, and a
//      length field that must match what was actually read.
//
// Only the first of those four produces an obvious symptom. The other three
// produce *plausible* indexes, which on this project means quietly wrong recall
// -- the exact failure class B-01, B-07 and P-04 are all instances of.
//
// **What this format deliberately does not claim:** portability across
// architectures. It is native-endian and native-width by design, because the
// alternative -- byte-swapping every float on read -- costs real time on a
// 488 MB store to serve a use case (moving an index from x86 to a big-endian
// machine) that does not exist here. The claim is that a mismatch is *detected*,
// not that it works. Saying "I detect it and refuse" is a defensible engineering
// position; silently producing wrong distances is not.

#include "veccore/types.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace veccore {

/// 8 bytes, no terminator. Chosen to be recognisable in `xxd` output.
inline constexpr char kMagic[8] = {'V', 'E', 'C', 'C', 'O', 'R', 'E', '\0'};

/// Bumped on **any** layout change, including one that looks additive.
/// Exact-match only -- see the header comment.
inline constexpr std::uint32_t kFormatVersion = 1;

/// Reads back as 0x01020304 only on a machine with the same byte order.
inline constexpr std::uint32_t kEndianProbe = 0x01020304u;

enum class IndexKind : std::uint32_t {
    Hnsw = 1,
    ProductQuantizer = 2,
};

/// Raised for every rejected file. One exception type, with a message that says
/// which of the four checks failed and what was expected -- because "failed to
/// load index" sends you to the wrong place, and "format version 2, this build
/// reads 1" does not.
class SerializationError : public std::runtime_error {
public:
    explicit SerializationError(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {

/// FNV-1a, 64-bit. Not cryptographic and not meant to be: the threat model is a
/// truncated write or a corrupted disk, not an adversary. It is ~1 cycle/byte,
/// which matters when the payload is 629 MB.
inline std::uint64_t fnv1a(const void* data, std::size_t n, std::uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

}  // namespace detail

/// Append-only binary writer that checksums everything it writes.
class BinaryWriter {
public:
    explicit BinaryWriter(const std::string& path)
        : path_(path), out_(path, std::ios::binary | std::ios::trunc) {
        if (!out_) throw SerializationError("cannot open for writing: " + path);
    }

    template <typename T>
    void pod(const T& value) {
        static_assert(std::is_trivially_copyable<T>::value, "pod() needs a trivially copyable type");
        raw(&value, sizeof(T));
    }

    /// Length-prefixed vector of trivially copyable elements. The length is what
    /// lets the reader detect truncation before it allocates.
    template <typename T>
    void vec(const std::vector<T>& v) {
        static_assert(std::is_trivially_copyable<T>::value, "vec() needs a trivially copyable type");
        pod(static_cast<std::uint64_t>(v.size()));
        if (!v.empty()) raw(v.data(), v.size() * sizeof(T));
    }

    void str(const std::string& s) {
        pod(static_cast<std::uint64_t>(s.size()));
        if (!s.empty()) raw(s.data(), s.size());
    }

    /// Writes the running checksum and closes. Must be called, or the file is
    /// rejected on load -- a half-written file failing loudly is the point.
    void finish() {
        const std::uint64_t sum = checksum_;
        out_.write(reinterpret_cast<const char*>(&sum), sizeof(sum));
        out_.flush();
        if (!out_) throw SerializationError("write failed (disk full?): " + path_);
        out_.close();
    }

    [[nodiscard]] std::uint64_t checksum() const noexcept { return checksum_; }

private:
    void raw(const void* data, std::size_t n) {
        out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
        if (!out_) throw SerializationError("write failed (disk full?): " + path_);
        checksum_ = detail::fnv1a(data, n, checksum_);
    }

    std::string path_;
    std::ofstream out_;
    std::uint64_t checksum_ = 1469598103934665603ull;
};

/// Reader that recomputes the checksum as it goes and verifies it at the end.
class BinaryReader {
public:
    explicit BinaryReader(const std::string& path)
        : path_(path), in_(path, std::ios::binary) {
        if (!in_) throw SerializationError("cannot open for reading: " + path);
    }

    template <typename T>
    T pod() {
        static_assert(std::is_trivially_copyable<T>::value, "pod() needs a trivially copyable type");
        T value{};
        raw(&value, sizeof(T));
        return value;
    }

    template <typename T>
    std::vector<T> vec() {
        static_assert(std::is_trivially_copyable<T>::value, "vec() needs a trivially copyable type");
        const auto n = pod<std::uint64_t>();
        // A corrupted length field would otherwise become a multi-terabyte
        // allocation and an OOM kill, which reads as "the loader hung" rather
        // than "the file is bad". Bound it by what is left in the file.
        guard_length(n, sizeof(T));
        std::vector<T> v(static_cast<std::size_t>(n));
        if (n) raw(v.data(), static_cast<std::size_t>(n) * sizeof(T));
        return v;
    }

    std::string str() {
        const auto n = pod<std::uint64_t>();
        guard_length(n, 1);
        std::string s(static_cast<std::size_t>(n), '\0');
        if (n) raw(s.data(), static_cast<std::size_t>(n));
        return s;
    }

    /// Verifies the trailing checksum against what was actually read, and that
    /// nothing is left over.
    void finish() {
        const std::uint64_t computed = checksum_;
        std::uint64_t stored = 0;
        in_.read(reinterpret_cast<char*>(&stored), sizeof(stored));
        if (in_.gcount() != static_cast<std::streamsize>(sizeof(stored))) {
            throw SerializationError("truncated file, no checksum present: " + path_);
        }
        if (computed != stored) {
            throw SerializationError("checksum mismatch in " + path_ +
                                     " -- the file is corrupt or was written by a different build");
        }
        in_.peek();
        if (!in_.eof()) {
            throw SerializationError("trailing bytes after the checksum in " + path_ +
                                     " -- this is not the file it claims to be");
        }
    }

private:
    void raw(void* data, std::size_t n) {
        in_.read(static_cast<char*>(data), static_cast<std::streamsize>(n));
        if (in_.gcount() != static_cast<std::streamsize>(n)) {
            throw SerializationError("truncated file: " + path_);
        }
        checksum_ = detail::fnv1a(data, n, checksum_);
    }

    void guard_length(std::uint64_t n, std::size_t elem) {
        const auto here = in_.tellg();
        in_.seekg(0, std::ios::end);
        const auto end = in_.tellg();
        in_.seekg(here);
        if (here < 0 || end < 0) return;  // non-seekable: skip the guard rather than fail
        const auto remaining = static_cast<std::uint64_t>(end - here);
        if (elem && n > remaining / elem) {
            throw SerializationError("declared length " + std::to_string(n) +
                                     " exceeds the bytes remaining in " + path_ +
                                     " -- the file is truncated or corrupt");
        }
    }

    std::string path_;
    std::ifstream in_;
    std::uint64_t checksum_ = 1469598103934665603ull;
};

/// Writes MAGIC / VERSION / ENDIAN_PROBE / widths / kind.
void write_header(BinaryWriter& w, IndexKind kind);

/// Verifies all of the above and returns the kind. Throws `SerializationError`
/// with a specific message for each distinct failure.
IndexKind read_header(BinaryReader& r, IndexKind expected);

}  // namespace veccore
