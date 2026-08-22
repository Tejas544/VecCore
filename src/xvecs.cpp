#include "veccore/xvecs.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace veccore {
namespace {

std::ifstream open_or_throw(const std::string& path, std::uintmax_t& size_out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("xvecs: cannot open " + path);
    const std::streamoff end = f.tellg();
    if (end < 4) throw std::runtime_error("xvecs: " + path + " is too small to hold one record");
    size_out = static_cast<std::uintmax_t>(end);
    f.seekg(0);
    return f;
}

/// Read the leading int32 and derive the record geometry.  Throws if the file
/// size is not an exact multiple of the record size -- P-01's cheap guard.
XvecsHeader geometry(std::ifstream& f, const std::string& path, std::uintmax_t file_size) {
    std::int32_t dim_raw = 0;
    f.read(reinterpret_cast<char*>(&dim_raw), 4);
    if (!f) throw std::runtime_error("xvecs: " + path + " truncated reading first dimension");
    if (dim_raw <= 0 || dim_raw > 100000) {
        throw std::runtime_error("xvecs: " + path + " first record claims dim=" +
                                 std::to_string(dim_raw) +
                                 " -- this is not an xvecs file, or the endianness is wrong");
    }

    const std::uintmax_t record = 4u + 4u * static_cast<std::uintmax_t>(dim_raw);
    if (file_size % record != 0) {
        throw std::runtime_error(
            "xvecs: " + path + " is " + std::to_string(file_size) + " bytes, not a multiple of " +
            std::to_string(record) + " (4 + 4*" + std::to_string(dim_raw) + "). " +
            "The layout is not what this reader assumes -- do NOT build an index on it (P-01).");
    }

    f.seekg(0);
    return XvecsHeader{static_cast<dim_t>(dim_raw), static_cast<std::size_t>(file_size / record)};
}

}  // namespace

XvecsHeader xvecs_header(const std::string& path) {
    std::uintmax_t file_size = 0;
    std::ifstream f = open_or_throw(path, file_size);
    return geometry(f, path, file_size);
}

VectorStore read_fvecs(const std::string& path, std::size_t max_count) {
    std::uintmax_t file_size = 0;
    std::ifstream f = open_or_throw(path, file_size);
    const XvecsHeader h = geometry(f, path, file_size);

    const std::size_t n = (max_count && max_count < h.count) ? max_count : h.count;

    std::vector<float> data;
    data.resize(n * static_cast<offset_t>(h.dim));

    std::int32_t dim_check = 0;
    for (std::size_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(&dim_check), 4);
        // The per-record check.  This is what actually catches a misread file:
        // the header check above passes on plenty of files that are not xvecs.
        if (!f || static_cast<dim_t>(dim_check) != h.dim) {
            throw std::runtime_error("xvecs: " + path + " record " + std::to_string(i) +
                                     " claims dim=" + std::to_string(dim_check) + ", expected " +
                                     std::to_string(h.dim) + " (P-01)");
        }
        f.read(reinterpret_cast<char*>(data.data() + i * static_cast<offset_t>(h.dim)),
               static_cast<std::streamsize>(4u * h.dim));
        if (!f) throw std::runtime_error("xvecs: " + path + " truncated at record " + std::to_string(i));
    }

    return VectorStore(std::move(data), h.dim);
}

IvecsData read_ivecs(const std::string& path, std::size_t max_count) {
    std::uintmax_t file_size = 0;
    std::ifstream f = open_or_throw(path, file_size);
    const XvecsHeader h = geometry(f, path, file_size);

    const std::size_t n = (max_count && max_count < h.count) ? max_count : h.count;

    IvecsData out;
    out.width = h.dim;
    out.ids.resize(n * static_cast<std::size_t>(h.dim));

    std::int32_t dim_check = 0;
    for (std::size_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(&dim_check), 4);
        if (!f || static_cast<dim_t>(dim_check) != h.dim) {
            throw std::runtime_error("ivecs: " + path + " record " + std::to_string(i) +
                                     " claims dim=" + std::to_string(dim_check) + ", expected " +
                                     std::to_string(h.dim) + " (P-01)");
        }
        f.read(reinterpret_cast<char*>(out.ids.data() + i * static_cast<std::size_t>(h.dim)),
               static_cast<std::streamsize>(4u * h.dim));
        if (!f) throw std::runtime_error("ivecs: " + path + " truncated at record " + std::to_string(i));
    }

    return out;
}

}  // namespace veccore
