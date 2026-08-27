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
#include <optional>
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

    /// Reopen a saved index. Private tag ctor rather than a second public
    /// overload: `HnswIndex(path)` and `HnswIndex(vectors)` would be two very
    /// different operations behind one name.
    struct LoadTag {};
    PyHnsw(LoadTag, const std::string& path, const std::string& lock_mode) {
        LockMode mode = LockMode::WriterPriority;
        if (lock_mode == "shared_mutex") mode = LockMode::SharedMutex;
        else if (lock_mode == "none") mode = LockMode::None;
        else if (lock_mode != "writer_priority") {
            throw std::invalid_argument("lock_mode must be writer_priority, shared_mutex or none");
        }
        HnswIndex loaded = HnswIndex::load(path, store_);
        index_ = std::make_unique<ConcurrentHnsw>(store_, std::move(loaded), mode);
    }

    /// Write the index to `path`, vectors included, under the shared lock.
    void save(const std::string& path) const {
        py::gil_scoped_release release;
        index_->save(path);
    }

    /// Returns a `unique_ptr`, and that is load-bearing rather than a style
    /// choice.
    ///
    /// `PyHnsw` owns `store_` by value, and the `ConcurrentHnsw` inside it holds
    /// a `const VectorStore&` pointing at that member. Returning `PyHnsw` **by
    /// value** therefore moves the store to a new address while leaving the
    /// reference aimed at the old one -- a dangling reference that reads freed
    /// stack memory. The first symptom was a test getting the wrong neighbours
    /// back; the second was a segfault (B-15).
    ///
    /// The irony is documented in `HnswIndex::load`'s own comment, which
    /// explains why a self-contained `{store; index;}` struct is unsafe -- and
    /// then this binding built exactly that. Heap-allocating means the store's
    /// address is fixed no matter what happens to the handle.
    [[nodiscard]] static std::unique_ptr<PyHnsw> load(const std::string& path,
                                                      const std::string& lock_mode) {
        py::gil_scoped_release release;
        return std::unique_ptr<PyHnsw>(new PyHnsw(LoadTag{}, path, lock_mode));
    }

    // Non-movable and non-copyable, so the bug above cannot come back by
    // accident. The compiler now rejects the shape that produced it.
    PyHnsw(const PyHnsw&) = delete;
    PyHnsw& operator=(const PyHnsw&) = delete;
    PyHnsw(PyHnsw&&) = delete;
    PyHnsw& operator=(PyHnsw&&) = delete;

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

    /// Add one vector to a live index. Returns its id.
    ///
    /// This is `PLAN.md` §5.4's incremental insert, reachable from Python. It is
    /// worth being able to say why it is not simply "append then insert": the
    /// append can reallocate the vector store underneath a concurrent reader, so
    /// both halves happen inside one exclusive section. `ConcurrentHnsw::insert`
    /// carries the full argument.
    ///
    /// The GIL is released around it (P-35), which means an insert from one
    /// Python thread genuinely overlaps searches on others -- and it is also
    /// what makes the writer-starvation fix (B-11) observable from Python at
    /// all. With the GIL held, readers could never be concurrent enough to
    /// starve anything.
    std::uint32_t add(const FloatArray& vector) {
        const float* v = query_ptr(vector, store_.dim());
        py::gil_scoped_release release;
        return index_->insert(v);
    }

    /// Add many vectors under one lock acquisition.
    ///
    /// Under the default `writer_priority` mode each acquisition stalls every
    /// reader at the turnstile, so N separate `add` calls are N stalls. Prefer
    /// this whenever you have more than one vector.
    [[nodiscard]] py::array_t<std::uint32_t> add_batch(const FloatArray& vectors) {
        const auto info = vectors.request();
        if (info.ndim != 2) {
            throw std::invalid_argument("add_batch expects a 2-D array of shape (n, dim), got ndim=" +
                                        std::to_string(info.ndim));
        }
        if (static_cast<dim_t>(info.shape[1]) != store_.dim()) {
            throw std::invalid_argument("vectors have dim " + std::to_string(info.shape[1]) +
                                        ", index has dim " + std::to_string(store_.dim()));
        }
        const auto n = static_cast<std::size_t>(info.shape[0]);
        const auto* data = static_cast<const float*>(info.ptr);
        std::vector<vec_id_t> ids;
        {
            py::gil_scoped_release release;
            ids = index_->insert_batch(data, n, store_.dim());
        }
        return py::array_t<std::uint32_t>(ids.size(), ids.data());
    }

    [[nodiscard]] std::size_t size() const noexcept { return store_.size(); }
    [[nodiscard]] dim_t dim() const noexcept { return store_.dim(); }
    /// Locked, not `index()`: once `add` can grow the graph, reading these
    /// arrays while another thread inserts is a genuine race (B-13).
    [[nodiscard]] std::size_t graph_bytes() const { return index_->graph_bytes(); }
    [[nodiscard]] std::size_t vector_bytes() const noexcept { return store_.bytes(); }
    [[nodiscard]] std::string lock_mode() const { return to_string(index_->mode()); }

    [[nodiscard]] py::dict stats() const {
        HnswStats s;
        offset_t gbytes = 0;
        {
            py::gil_scoped_release release;
            s = index_->stats();
            gbytes = index_->graph_bytes();
        }
        py::dict d;
        d["nodes"] = s.nodes;
        d["max_level"] = s.max_level;
        d["nodes_per_level"] = s.nodes_per_level;
        d["mean_degree_layer0"] = s.mean_degree_layer0;
        d["graph_bytes"] = gbytes;
        d["vector_bytes"] = store_.bytes();
        return d;
    }

    /// Debug invariant check, exposed so a Python test can assert the graph is
    /// still sound after a batch of concurrent adds.
    [[nodiscard]] std::string check_invariants() const {
        py::gil_scoped_release release;
        return index_->check_invariants();
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

/// Product Quantization, exposed because an entire phase of this project was
/// otherwise unreachable from Python.
///
/// Note what the properties deliberately keep separate (D12, and the honesty
/// item in the README): `code_bytes` is the compressed footprint,
/// `codebook_bytes` is the fixed cost of the codebooks, and `rerank` needs the
/// **full vectors resident** -- so a reranking configuration's real memory is
/// all of them. Quoting 64x compression for a configuration that also keeps the
/// uncompressed vectors in RAM would be exactly the overclaim this repo
/// criticises elsewhere, so the numbers are never summed for you.
class PyPq {
public:
    PyPq(const FloatArray& vectors, std::size_t m, std::size_t train_size,
         std::size_t kmeans_iters, std::uint64_t seed, std::size_t n_init)
        : store_(store_from_array(vectors)) {
        PqParams p;
        p.m = m;
        p.train_size = train_size;
        p.kmeans_iters = kmeans_iters;
        p.seed = seed;
        p.kmeans_n_init = n_init;
        if (store_.dim() % m != 0) {
            // P-21: integer division would silently drop dimensions and the only
            // symptom would be recall that is merely disappointing.
            throw std::invalid_argument("m=" + std::to_string(m) + " does not divide dim=" +
                                        std::to_string(store_.dim()));
        }
        pq_ = std::make_unique<ProductQuantizer>(store_.dim(), p);
        {
            // Training is the slow half -- tens of seconds at SIFT scale (see the
            // README limitation about BLAS). Holding the GIL across it would
            // freeze the whole interpreter.
            py::gil_scoped_release release;
            pq_->train(store_);
            pq_->encode(store_);
        }
    }

    /// Reopen saved codebooks and codes. `vectors` is optional and only
    /// `search_rerank` needs it -- see `save`.
    struct LoadTag {};
    PyPq(LoadTag, const std::string& path, const std::optional<FloatArray>& vectors) {
        ProductQuantizer loaded = ProductQuantizer::load(path);
        if (vectors.has_value()) {
            store_ = store_from_array(*vectors);
            if (store_.dim() != loaded.params().m * loaded.dsub()) {
                throw std::invalid_argument(
                    "vectors have dim " + std::to_string(store_.dim()) +
                    " but the saved quantizer was trained on dim " +
                    std::to_string(loaded.params().m * loaded.dsub()));
            }
            if (store_.size() != loaded.n_codes()) {
                // The codes index *these* vectors by position. A store of a
                // different length is a different corpus, and rerank would score
                // the wrong documents while returning a full result set.
                throw std::invalid_argument(
                    "the saved quantizer holds " + std::to_string(loaded.n_codes()) +
                    " codes but " + std::to_string(store_.size()) +
                    " vectors were supplied -- these do not describe the same corpus");
            }
        }
        pq_ = std::make_unique<ProductQuantizer>(std::move(loaded));
    }

    /// Write codebooks and codes. **Vectors are not included** -- they are 16x
    /// larger and only `search_rerank` needs them.
    void save(const std::string& path) const {
        py::gil_scoped_release release;
        pq_->save(path);
    }

    [[nodiscard]] static PyPq load(const std::string& path,
                                   const std::optional<FloatArray>& vectors) {
        return PyPq(LoadTag{}, path, vectors);
    }

    [[nodiscard]] py::tuple search(const FloatArray& query, std::size_t k) const {
        const float* q = query_ptr(query, dim());
        std::vector<Neighbor> hits;
        {
            py::gil_scoped_release release;
            hits = pq_->search(q, k);
        }
        return pack(hits);
    }

    /// ADC shortlist, then exact rescoring of `candidates` of them.
    [[nodiscard]] py::tuple search_rerank(const FloatArray& query, std::size_t k,
                                          std::size_t candidates) const {
        require_vectors("search_rerank");
        const float* q = query_ptr(query, dim());
        std::vector<Neighbor> hits;
        {
            py::gil_scoped_release release;
            hits = pq_->search_rerank(q, k, candidates, store_);
        }
        return pack(hits);
    }

    [[nodiscard]] double reconstruction_mse(std::size_t sample) const {
        require_vectors("reconstruction_mse");
        py::gil_scoped_release release;
        return pq_->reconstruction_mse(store_, sample);
    }

    /// Both come from the quantizer, not the store: a loaded PqIndex may have no
    /// vectors at all, and reporting dim=0 for a perfectly usable index would be
    /// a lie of convenience.
    [[nodiscard]] std::size_t size() const noexcept { return pq_->n_codes(); }
    [[nodiscard]] dim_t dim() const noexcept {
        return static_cast<dim_t>(pq_->params().m * pq_->dsub());
    }
    [[nodiscard]] std::size_t m() const noexcept { return pq_->params().m; }
    /// False after `load` without vectors; `search_rerank` needs this.
    [[nodiscard]] bool has_vectors() const noexcept { return store_.size() > 0; }
    [[nodiscard]] std::size_t bytes_per_vector() const noexcept { return pq_->bytes_per_vector(); }
    [[nodiscard]] std::size_t code_bytes() const noexcept { return pq_->code_bytes(); }
    [[nodiscard]] std::size_t codebook_bytes() const noexcept { return pq_->codebook_bytes(); }
    [[nodiscard]] std::size_t vector_bytes() const noexcept { return store_.bytes(); }
    [[nodiscard]] std::size_t empty_cluster_reseeds() const noexcept {
        return pq_->empty_cluster_reseeds();
    }
    /// Codes only, against raw fp32. Quote it with the recall it lands at.
    [[nodiscard]] double compression_ratio() const noexcept {
        const auto per_vec = static_cast<double>(pq_->bytes_per_vector());
        return per_vec > 0.0 ? (static_cast<double>(store_.dim()) * sizeof(float)) / per_vec : 0.0;
    }

private:
    /// A loaded quantizer without vectors can `search` but not rerank. Saying so
    /// beats returning a silently ADC-only result from a method whose entire
    /// purpose is the exact rescoring pass.
    void require_vectors(const char* what) const {
        if (!has_vectors()) {
            throw std::invalid_argument(
                std::string(what) + " needs the full vectors, and this PqIndex was loaded without "
                "them. PqIndex.save() stores codes and codebooks only (they are 16x smaller); pass "
                "vectors= to PqIndex.load() to enable exact reranking.");
        }
    }

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
    std::unique_ptr<ProductQuantizer> pq_;
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
        .def("add", &PyHnsw::add, py::arg("vector"),
             "Add one vector to the live index and return its id. Releases the GIL; "
             "safe against concurrent search().")
        .def("add_batch", &PyHnsw::add_batch, py::arg("vectors"),
             "Add (n, dim) vectors under one lock acquisition. Returns their ids.")
        .def("stats", &PyHnsw::stats)
        .def("check_invariants", &PyHnsw::check_invariants,
             "Empty string when the graph is sound, else the first violation found.")
        .def("save", &PyHnsw::save, py::arg("path"),
             "Write the index and its vectors to a self-contained file, under the shared lock.")
        .def_static("load", &PyHnsw::load, py::arg("path"),
                    py::arg("lock_mode") = "writer_priority",
                    "Reopen an index written by save(). Raises RuntimeError on a corrupt, "
                    "truncated, wrong-version or foreign-endian file rather than loading it.")
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

    py::class_<PyPq>(m, "PqIndex")
        .def(py::init<const FloatArray&, std::size_t, std::size_t, std::size_t,
                      std::uint64_t, std::size_t>(),
             py::arg("vectors"), py::arg("m") = 8, py::arg("train_size") = 100000,
             py::arg("kmeans_iters") = 25, py::arg("seed") = 42, py::arg("n_init") = 3,
             "Train PQ codebooks on a subsample of `vectors`, then encode all of them. "
             "`m` must divide the dimensionality.")
        .def("search", &PyPq::search, py::arg("query"), py::arg("k") = 10,
             "Exhaustive ADC scan. Squared L2, approximate.")
        .def("search_rerank", &PyPq::search_rerank, py::arg("query"), py::arg("k") = 10,
             py::arg("candidates") = 100,
             "ADC shortlist of `candidates`, rescored exactly. Needs the full vectors resident.")
        .def("reconstruction_mse", &PyPq::reconstruction_mse, py::arg("sample") = 1000)
        .def_property_readonly("size", &PyPq::size)
        .def_property_readonly("dim", &PyPq::dim)
        .def_property_readonly("m", &PyPq::m)
        .def_property_readonly("bytes_per_vector", &PyPq::bytes_per_vector)
        .def_property_readonly("code_bytes", &PyPq::code_bytes)
        .def_property_readonly("codebook_bytes", &PyPq::codebook_bytes)
        .def_property_readonly("vector_bytes", &PyPq::vector_bytes)
        .def_property_readonly("compression_ratio", &PyPq::compression_ratio)
        .def_property_readonly("empty_cluster_reseeds", &PyPq::empty_cluster_reseeds)
        .def_property_readonly("has_vectors", &PyPq::has_vectors)
        .def("save", &PyPq::save, py::arg("path"),
             "Write codebooks and codes. Vectors are NOT included -- they are 16x larger and only "
             "search_rerank needs them.")
        .def_static("load", &PyPq::load, py::arg("path"), py::arg("vectors") = py::none(),
                    "Reopen codebooks and codes. Pass vectors= to re-enable search_rerank; "
                    "without them the index can still search().");

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
