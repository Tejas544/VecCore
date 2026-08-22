#pragma once

// Reader for the .fvecs / .ivecs format used by the TEXMEX SIFT corpus.
//
// BUGS.md P-01, and this is the single most important comment in the file:
//
//   Each record is a little-endian int32 DIMENSION COUNT, followed by that many
//   values.  The dimension is repeated for EVERY vector -- it is not a one-time
//   header.  A reader that assumes a flat array of floats produces data that is
//   wrong but entirely plausible: no crash, no NaN, no warning.  Recall then
//   comes out low and looks like an algorithm bug, and you spend an evening in
//   the wrong file.
//
// So this reader validates rather than trusts: it checks that the file size is
// an exact multiple of the record size, and that every single record repeats
// the same dimension.  Both checks are cheap and both fire loudly.

#include "veccore/storage.hpp"
#include "veccore/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace veccore {

struct XvecsHeader {
    dim_t dim = 0;
    std::size_t count = 0;
};

/// Peek at a file's shape without reading the payload.
XvecsHeader xvecs_header(const std::string& path);

/// Read a .fvecs file into flat row-major storage.
/// `max_count` = 0 reads everything; otherwise reads at most that many vectors.
VectorStore read_fvecs(const std::string& path, std::size_t max_count = 0);

/// Read an .ivecs file (ground truth: one row of neighbour ids per query).
/// Returns row-major ids plus the row width.
struct IvecsData {
    std::vector<std::int32_t> ids;
    dim_t width = 0;
    [[nodiscard]] std::size_t rows() const noexcept { return width ? ids.size() / width : 0; }
    [[nodiscard]] const std::int32_t* row(std::size_t i) const noexcept {
        return ids.data() + i * static_cast<std::size_t>(width);
    }
};

IvecsData read_ivecs(const std::string& path, std::size_t max_count = 0);

}  // namespace veccore
