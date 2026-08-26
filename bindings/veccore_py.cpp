// Python bindings.
//
// PLAN.md §0.3 cut the FastAPI service but never this: the module is what
// EdgeRAG actually calls, and it is the artifact that proves the C++/Python
// seam works. An HTTP wrapper on top adds one interview answer you can give
// from a whiteboard.
//
// Two things here are load-bearing and both are easy to get wrong silently.
//
// **1. `py::array::c_style | py::array::forcecast` on every array argument.**
// P-34: a sliced or transposed numpy array has strides the C++ loop knows
// nothing about, so it would read the wrong memory -- no crash, plausible
// garbage, wrong recall. These flags force a contiguous float32 copy when the
// caller hands over anything else. The copy is the point, not a cost to
// optimise away.
//
// **2. `py::gil_scoped_release` around every search.** P-35: without it, every
// Phase 5 thread-scaling claim becomes false the moment the caller is Python,
// because the GIL serialises the threads that the C++ side was carefully made
// safe for. It costs nothing to add and it is invisible when missing -- the
// C++ benchmark is unaffected, so only a Python-side threaded test catches it.
// There is one, in tests/test_bindings.py.

#include "veccore/bm25.hpp"
#include "veccore/concurrent.hpp"
#include "veccore/flat_index.hpp"
#include "veccore/fusion.hpp"
#include "veccore/hnsw.hpp"
#include "veccore/pq.hpp"
#include "veccore/storage.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace veccore;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

VectorStore store_from_array(const FloatArray& arr) {
    const auto info = arr.request();
    if (info.ndim != 2) {
        throw std::invalid_argument("expected a 2-D array of shape (n, dim), got ndim=" +
                                    std::to_string(info.ndim));
    }
    const auto n = static_cast<std::size_t>(info.shape[0]);
    const auto d = static_cast<dim_t>(info.shape[1]);
    if (d == 0) throw std::invalid_argument("dim must be non-zero");

    std::vector<float> data(n * static_cast<offset_t>(d));
    std::copy_n(static_cast<const float*>(info.ptr), data.size(), data.begin());
    return VectorStore(std::move(data), d);
}

const float* query_ptr(const FloatArray& q, dim_t expected_dim) {
    const auto info = q.request();
    if (info.ndim != 1) {
        throw std::invalid_argument("query must be 1-D, got ndim=" + std::to_string(info.ndim));
    }
    if (static_cast<dim_t>(info.shape[0]) != expected_dim) {
        throw std::invalid_argument("query has dim " + std::to_string(info.shape[0]) +
                                    ", index has dim " + std::to_string(expected_dim));
    }
    return static_cast<const float*>(info.ptr);
}

/// Owns its vectors, unlike the C++ classes which borrow a VectorStore.
///
/// The C++ side deliberately takes `const VectorStore&` -- the index does not
/// own the data, which keeps ownership explicit and avoids a copy. Python has
/// no way to express that lifetime, and a caller who let the array go out of
/// scope would get a dangling reference and a segfault from an import. So the
/// binding owns a copy. That is the right trade at a language boundary and it
/// is worth being able to say why.
class PyHnsw {
public:
    PyHnsw(const FloatArray& vectors, std::size_t M, std::size_t ef_construction,
           std::uint64_t seed, const std::string& lock_mode)
        : store_(store_from_array(vectors)) {
        HnswParams p;
        p.M = M;
        p.ef_construction = ef_construction;
        p.seed = seed;

        LockMode mode = LockMode::WriterPriority;
        if (lock_mode == "shared_mutex") mode = LockMode::SharedMutex;
        else if (lock_mode == "none") mode = LockMode::None;
        else if (lock_mode != "writer_priority") {
            throw std::invalid_argument("lock_mode must be writer_priority, shared_mutex or none");
        }

        index_ = std::make_unique<ConcurrentHnsw>(store_, p, mode);
        index_->build();
    }

    [[nodiscard]] py::tuple search(const FloatArray& query, std::size_t k, std::size_t ef_search) {
        const float* q = query_ptr(query, store_.dim());
        std::vector<Neighbor> hits;
        {
            // P-35. Every Phase 5 scaling number is a lie without this line.
            py::gil_scoped_release release;
            SearchScratch scratch = index_->make_scratch();
            hits = index_->search(q, k, ef_search, scratch);
        }
        return pack(hits);
    }

    /// Batch search, releasing the GIL once for the whole batch. This is what
    /// EdgeRAG should call in a loop-heavy path: acquiring and releasing the
    /// GIL per query would dominate a 0.5 ms search.
    [[nodiscard]] py::tuple search_batch(const FloatArray& queries, std::size_t k,
                                         std::size_t ef_search) {
        const auto info = queries.request();
        if (info.ndim != 2) throw std::invalid_argument("queries must be 2-D (n, dim)");
        if (static_cast<dim_t>(info.shape[1]) != store_.dim()) {
            throw std::invalid_argument("query dim does not match index dim");
        }
        const auto n = static_cast<std::size_t>(info.shape[0]);
        const auto* data = static_cast<const float*>(info.ptr);

        std::vector<std::uint32_t> ids(n * k, 0);
        std::vector<float> dists(n * k, 0.0f);
        {
            py::gil_scoped_release release;
            SearchScratch scratch = index_->make_scratch();
            for (std::size_t i = 0; i < n; ++i) {
                const auto hits = index_->search(data + i * store_.dim(), k, ef_search, scratch);
                for (std::size_t j = 0; j < hits.size() && j < k; ++j) {
                    ids[i * k + j] = hits[j].id;
                    dists[i * k + j] = hits[j].dist;
                }
            }
        }
        return py::make_tuple(
            py::array_t<std::uint32_t>({n, k}, ids.data()),
            py::array_t<float>({n, k}, dists.data()));
    }

    [[nodiscard]] std::size_t size() const noexcept { return store_.size(); }
    [[nodiscard]] dim_t dim() const noexcept { return store_.dim(); }
    [[nodiscard]] std::size_t graph_bytes() const noexcept { return index_->index().graph_bytes(); }
    [[nodiscard]] std::size_t vector_bytes() const noexcept { return store_.bytes(); }
    [[nodiscard]] std::string lock_mode() const { return to_string(index_->mode()); }

    [[nodiscard]] py::dict stats() const {
        const HnswStats s = index_->index().stats();
        py::dict d;
        d["nodes"] = s.nodes;
        d["max_level"] = s.max_level;
        d["nodes_per_level"] = s.nodes_per_level;
        d["mean_degree_layer0"] = s.mean_degree_layer0;
        d["graph_bytes"] = index_->index().graph_bytes();
        d["vector_bytes"] = store_.bytes();
        return d;
    }

private:
    static py::tuple pack(const std::vector<Neighbor>& hits) {
        std::vector<std::uint32_t> ids;
        std::vector<float> dists;
        ids.reserve(hits.size());
        dists.reserve(hits.size());
        for (const Neighbor& n : hits) { ids.push_back(n.id); dists.push_back(n.dist); }
        return py::make_tuple(py::array_t<std::uint32_t>(ids.size(), ids.data()),
                              py::array_t<float>(dists.size(), dists.data()));
    }

    VectorStore store_;
    std::unique_ptr<ConcurrentHnsw> index_;
};

class PyFlat {
public:
    explicit PyFlat(const FloatArray& vectors) : store_(store_from_array(vectors)), index_(store_) {}

    [[nodiscard]] py::tuple search(const FloatArray& query, std::size_t k) {
        const float* q = query_ptr(query, store_.dim());
        std::vector<Neighbor> hits;
        {
            py::gil_scoped_release release;
            hits = index_.search(q, k);
        }
        std::vector<std::uint32_t> ids;
        std::vector<float> dists;
        for (const Neighbor& n : hits) { ids.push_back(n.id); dists.push_back(n.dist); }
        return py::make_tuple(py::array_t<std::uint32_t>(ids.size(), ids.data()),
                              py::array_t<float>(dists.size(), dists.data()));
    }

    [[nodiscard]] std::size_t size() const noexcept { return store_.size(); }

private:
    VectorStore store_;
    FlatL2 index_;
};

class PyBm25 {
public:
    PyBm25(const std::vector<std::string>& docs, double k1, double b) {
        Bm25Params p;
        p.k1 = k1;
        p.b = b;
        index_ = std::make_unique<Bm25Index>(p);
        index_->build(docs);
    }

    [[nodiscard]] std::vector<std::pair<std::uint32_t, double>> search(const std::string& query,
                                                                       std::size_t k) const {
        std::vector<std::pair<std::uint32_t, double>> out;
        for (const Neighbor& n : index_->search(query, k)) {
            // Un-negate: the C++ side keeps "smaller is better" everywhere (D4),
            // but a Python caller asking for a BM25 score expects the score.
            out.emplace_back(n.id, -static_cast<double>(n.dist));
        }
        return out;
    }

    [[nodiscard]] double idf(const std::string& term) const { return index_->idf(term); }
    [[nodiscard]] std::size_t vocab_size() const noexcept { return index_->vocab_size(); }
    [[nodiscard]] double avgdl() const noexcept { return index_->avgdl(); }

private:
    std::unique_ptr<Bm25Index> index_;
};

}  // namespace

PYBIND11_MODULE(_veccore, m) {
    m.doc() = "VecCore -- HNSW, Product Quantization and BM25, from scratch in C++17.";

    py::class_<PyHnsw>(m, "HnswIndex")
        .def(py::init<const FloatArray&, std::size_t, std::size_t, std::uint64_t, const std::string&>(),
             py::arg("vectors"), py::arg("M") = 16, py::arg("ef_construction") = 200,
             py::arg("seed") = 42, py::arg("lock_mode") = "writer_priority",
             "Build an HNSW index over an (n, dim) float32 array. The array is copied.")
        .def("search", &PyHnsw::search, py::arg("query"), py::arg("k") = 10,
             py::arg("ef_search") = 64, "Return (ids, squared_distances) for one query.")
        .def("search_batch", &PyHnsw::search_batch, py::arg("queries"), py::arg("k") = 10,
             py::arg("ef_search") = 64,
             "Return (ids, squared_distances) of shape (n, k). Releases the GIL once for the batch.")
        .def("stats", &PyHnsw::stats)
        .def_property_readonly("size", &PyHnsw::size)
        .def_property_readonly("dim", &PyHnsw::dim)
        .def_property_readonly("graph_bytes", &PyHnsw::graph_bytes)
        .def_property_readonly("vector_bytes", &PyHnsw::vector_bytes)
        .def_property_readonly("lock_mode", &PyHnsw::lock_mode);

    py::class_<PyFlat>(m, "FlatIndex")
        .def(py::init<const FloatArray&>(), py::arg("vectors"),
             "Exact brute-force search. The ground truth every approximate result is graded against.")
        .def("search", &PyFlat::search, py::arg("query"), py::arg("k") = 10)
        .def_property_readonly("size", &PyFlat::size);

    py::class_<PyBm25>(m, "Bm25Index")
        .def(py::init<const std::vector<std::string>&, double, double>(),
             py::arg("documents"), py::arg("k1") = 1.2, py::arg("b") = 0.75)
        .def("search", &PyBm25::search, py::arg("query"), py::arg("k") = 10,
             "Return [(doc_id, score)] ordered best-first.")
        .def("idf", &PyBm25::idf, py::arg("term"))
        .def_property_readonly("vocab_size", &PyBm25::vocab_size)
        .def_property_readonly("avgdl", &PyBm25::avgdl);

    m.def("reciprocal_rank_fusion",
          [](const std::vector<std::vector<std::uint32_t>>& lists, std::size_t k, double rrf_k) {
              std::vector<std::vector<Neighbor>> as_neighbors;
              as_neighbors.reserve(lists.size());
              for (const auto& l : lists) {
                  std::vector<Neighbor> nl;
                  nl.reserve(l.size());
                  // Only the ORDER matters -- that is the whole mechanism. The
                  // distances here are placeholders and are never read.
                  for (std::size_t i = 0; i < l.size(); ++i) {
                      nl.push_back(Neighbor{static_cast<float>(i), l[i]});
                  }
                  as_neighbors.push_back(std::move(nl));
              }
              RrfParams p;
              p.k = rrf_k;
              std::vector<std::uint32_t> out;
              for (const Neighbor& n : reciprocal_rank_fusion(as_neighbors, k, p)) {
                  out.push_back(n.id);
              }
              return out;
          },
          py::arg("ranked_lists"), py::arg("k") = 10, py::arg("rrf_k") = 60.0,
          "Fuse ranked id lists by rank alone. Scores are never consulted.");
}
