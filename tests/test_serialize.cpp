#include <doctest/doctest.h>

#include "veccore/concurrent.hpp"
#include "veccore/flat_index.hpp"
#include "veccore/hnsw.hpp"
#include "veccore/pq.hpp"
#include "veccore/serialize.hpp"
#include "veccore/storage.hpp"

#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace veccore;

namespace {

VectorStore random_store(std::size_t n, dim_t d, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    VectorStore s(d);
    s.reserve(n);
    std::vector<float> v(d);
    for (std::size_t i = 0; i < n; ++i) {
        for (auto& x : v) x = u(rng);
        s.add(v.data());
    }
    return s;
}

/// A path under /tmp that removes itself, so a failing assertion cannot leave
/// a stale file that makes the *next* run pass for the wrong reason.
struct TempFile {
    explicit TempFile(const char* name) : path("/tmp/veccore_test_" + std::string(name)) {}
    ~TempFile() { std::remove(path.c_str()); }
    std::string path;
};

std::vector<char> read_all(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_all(const std::string& path, const std::vector<char>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

/// Flatten an index's answers over a query set into one comparable vector.
/// Comparing whole result sets rather than a recall average is deliberate: a
/// recall number can match while the *rankings* differ, and a save/load cycle
/// that reorders results is still a save/load cycle that is wrong.
std::vector<std::uint32_t> fingerprint(const HnswIndex& index, const VectorStore& queries,
                                       std::size_t k = 10, std::size_t ef = 64) {
    std::vector<std::uint32_t> out;
    SearchScratch scratch = index.make_scratch();
    for (vec_id_t q = 0; q < queries.size(); ++q) {
        for (const auto& n : index.search(queries.at(q), k, ef, scratch)) out.push_back(n.id);
    }
    return out;
}

/// `HnswIndex::load` is `[[nodiscard]]`, and doctest's CHECK_THROWS_AS expands
/// its argument as a bare statement -- which trips -Wunused-result on every
/// rejection test below. Wrapping the call keeps both the attribute (which is
/// right: silently discarding a loaded index is a bug) and the warning-clean
/// build the repo treats as a hard requirement.
void load_discarding(const std::string& path, VectorStore& store) {
    HnswIndex index = HnswIndex::load(path, store);
    (void)index;
}

}  // namespace

TEST_CASE("a saved HNSW index loads back and answers identically") {
    const VectorStore store = random_store(1500, 16, 4001);
    const VectorStore queries = random_store(60, 16, 4002);

    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    p.seed = 11;
    HnswIndex original(store, p);
    original.build();

    TempFile f("roundtrip.vci");
    original.save(f.path);

    VectorStore loaded_store;
    HnswIndex loaded = HnswIndex::load(f.path, loaded_store);

    CHECK(loaded_store.size() == store.size());
    CHECK(loaded_store.dim() == store.dim());
    CHECK(loaded.params().M == p.M);
    CHECK(loaded.params().ef_construction == p.ef_construction);
    CHECK(loaded.params().seed == p.seed);
    CHECK(loaded.check_invariants() == "");

    // Not "recall is close" -- byte-for-byte the same answers, in the same order.
    CHECK(fingerprint(loaded, queries) == fingerprint(original, queries));

    const HnswStats a = original.stats();
    const HnswStats b = loaded.stats();
    CHECK(b.nodes == a.nodes);
    CHECK(b.max_level == a.max_level);
    CHECK(b.nodes_per_level == a.nodes_per_level);
    CHECK(b.mean_degree_layer0 == doctest::Approx(a.mean_degree_layer0));
}

TEST_CASE("the vectors survive the round trip, not just the graph") {
    // The graph is a list of ids. If the store were dropped or misread, every
    // id would still be in range and every search would still return k results
    // -- against the wrong vectors. So compare the data, not the shape.
    const VectorStore store = random_store(400, 8, 4010);
    HnswParams p;
    p.M = 8;
    HnswIndex index(store, p);
    index.build();

    TempFile f("vectors.vci");
    index.save(f.path);

    VectorStore loaded_store;
    HnswIndex loaded = HnswIndex::load(f.path, loaded_store);

    REQUIRE(loaded_store.size() == store.size());
    for (vec_id_t i = 0; i < store.size(); ++i) {
        for (dim_t d = 0; d < store.dim(); ++d) {
            REQUIRE(loaded_store.at(i)[d] == store.at(i)[d]);
        }
    }
}

TEST_CASE("a loaded index continues the same graph the original would have built") {
    // The reason the RNG state is in the file. Level assignment is a random
    // draw, so an index whose generator restarts after a load diverges from one
    // that never stopped -- silently, and only for nodes inserted afterwards.
    // P-03 is the same hazard one level up.
    VectorStore store_a = random_store(600, 16, 4020);
    const VectorStore extra = random_store(200, 16, 4021);
    const VectorStore queries = random_store(40, 16, 4022);

    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    p.seed = 7;

    // A: build, insert 200 more, never touch the disk.
    VectorStore store_uninterrupted = store_a;
    HnswIndex uninterrupted(store_uninterrupted, p);
    uninterrupted.build();
    for (vec_id_t i = 0; i < extra.size(); ++i) {
        store_uninterrupted.add(extra.at(i));
        uninterrupted.insert(store_uninterrupted.size() - 1);
    }

    // B: build, save, load, then insert the same 200.
    VectorStore store_saved = store_a;
    HnswIndex to_save(store_saved, p);
    to_save.build();
    TempFile f("continuity.vci");
    to_save.save(f.path);

    VectorStore store_loaded;
    HnswIndex resumed = HnswIndex::load(f.path, store_loaded);
    for (vec_id_t i = 0; i < extra.size(); ++i) {
        store_loaded.add(extra.at(i));
        resumed.insert(store_loaded.size() - 1);
    }

    CHECK(resumed.stats().nodes_per_level == uninterrupted.stats().nodes_per_level);
    CHECK(fingerprint(resumed, queries) == fingerprint(uninterrupted, queries));
}

TEST_CASE("a saved index can be reopened through ConcurrentHnsw and grown") {
    const VectorStore seed_store = random_store(500, 16, 4030);
    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    HnswIndex built(seed_store, p);
    built.build();

    TempFile f("concurrent.vci");
    built.save(f.path);

    VectorStore store;
    HnswIndex loaded = HnswIndex::load(f.path, store);
    CHECK(loaded.capacity() == 500);

    // The point of persisting an index that supports incremental insert: pick
    // up where the last process left off rather than rebuilding.
    std::mt19937_64 rng(4031);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(16);
    for (int i = 0; i < 100; ++i) {
        for (auto& x : v) x = u(rng);
        store.add(v.data());
        loaded.insert(store.size() - 1);
    }
    CHECK(loaded.capacity() == 600);
    CHECK(loaded.check_invariants() == "");

    const auto hits = loaded.search(v.data(), 1, 64);
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].id == 599);
}

// ---------------------------------------------------------------------------
// The rejection paths. Each of these is a way a naive binary dump corrupts data
// without erroring, so each one asserts that this format *refuses* instead.
// ---------------------------------------------------------------------------

TEST_CASE("a file that is not an index is rejected on the magic, not later") {
    TempFile f("notanindex.vci");
    write_all(f.path, std::vector<char>(4096, '\x7f'));
    VectorStore store;
    CHECK_THROWS_AS(load_discarding(f.path, store), SerializationError);
}

TEST_CASE("a missing file says so rather than returning an empty index") {
    VectorStore store;
    CHECK_THROWS_AS(load_discarding("/tmp/veccore_definitely_absent.vci", store),
                    SerializationError);
}

TEST_CASE("a future format version is refused, not guessed at") {
    const VectorStore store = random_store(200, 8, 4040);
    HnswParams p;
    p.M = 8;
    HnswIndex index(store, p);
    index.build();

    TempFile f("version.vci");
    index.save(f.path);

    auto bytes = read_all(f.path);
    REQUIRE(bytes.size() > 12);
    bytes[8] = static_cast<char>(kFormatVersion + 1);  // first byte of the version field
    write_all(f.path, bytes);

    VectorStore out;
    CHECK_THROWS_AS(load_discarding(f.path, out), SerializationError);
}

TEST_CASE("a byte-order mismatch is detected instead of producing wrong distances") {
    const VectorStore store = random_store(200, 8, 4050);
    HnswParams p;
    p.M = 8;
    HnswIndex index(store, p);
    index.build();

    TempFile f("endian.vci");
    index.save(f.path);

    auto bytes = read_all(f.path);
    REQUIRE(bytes.size() > 16);
    // Reverse the four probe bytes, which is exactly what a big-endian writer
    // would have produced.
    std::swap(bytes[12], bytes[15]);
    std::swap(bytes[13], bytes[14]);
    write_all(f.path, bytes);

    VectorStore out;
    CHECK_THROWS_AS(load_discarding(f.path, out), SerializationError);
}

TEST_CASE("a single flipped byte in the payload is caught by the checksum") {
    // The case with no other defence. Header, lengths and cross-field checks all
    // still pass; one adjacency entry now points somewhere else, and the only
    // symptom would be slightly worse recall.
    const VectorStore store = random_store(500, 8, 4060);
    HnswParams p;
    p.M = 8;
    HnswIndex index(store, p);
    index.build();

    TempFile f("bitflip.vci");
    index.save(f.path);

    auto bytes = read_all(f.path);
    REQUIRE(bytes.size() > 2000);
    bytes[bytes.size() / 2] = static_cast<char>(bytes[bytes.size() / 2] ^ 0x01);
    write_all(f.path, bytes);

    VectorStore out;
    CHECK_THROWS_AS(load_discarding(f.path, out), SerializationError);
}

TEST_CASE("a truncated file is refused rather than half-loaded") {
    const VectorStore store = random_store(500, 8, 4070);
    HnswParams p;
    p.M = 8;
    HnswIndex index(store, p);
    index.build();

    TempFile f("truncated.vci");
    index.save(f.path);

    auto bytes = read_all(f.path);
    bytes.resize(bytes.size() * 3 / 4);  // an interrupted write
    write_all(f.path, bytes);

    VectorStore out;
    CHECK_THROWS_AS(load_discarding(f.path, out), SerializationError);
}

TEST_CASE("trailing bytes after the checksum are refused") {
    const VectorStore store = random_store(200, 8, 4080);
    HnswParams p;
    p.M = 8;
    HnswIndex index(store, p);
    index.build();

    TempFile f("trailing.vci");
    index.save(f.path);

    auto bytes = read_all(f.path);
    bytes.push_back('\x00');
    write_all(f.path, bytes);

    VectorStore out;
    CHECK_THROWS_AS(load_discarding(f.path, out), SerializationError);
}

TEST_CASE("loading a PQ file as an HNSW index is refused") {
    const VectorStore store = random_store(400, 8, 4090);
    PqParams pp;
    pp.m = 4;
    pp.train_size = 400;
    pp.kmeans_iters = 5;
    pp.kmeans_n_init = 1;
    ProductQuantizer pq(store.dim(), pp);
    pq.train(store);
    pq.encode(store);

    TempFile f("kind.vci");
    pq.save(f.path);

    VectorStore out;
    CHECK_THROWS_AS(load_discarding(f.path, out), SerializationError);
}

// ---------------------------------------------------------------------------
// ProductQuantizer
// ---------------------------------------------------------------------------

TEST_CASE("a saved quantizer loads back and scores identically") {
    // The strongest argument for persistence in this repo: training costs 86.5 s
    // at SIFT1M and loading costs milliseconds.
    const VectorStore store = random_store(2000, 16, 4100);
    const VectorStore queries = random_store(40, 16, 4101);

    PqParams pp;
    pp.m = 4;
    pp.train_size = 2000;
    pp.kmeans_iters = 10;
    pp.kmeans_n_init = 1;
    pp.seed = 5;
    ProductQuantizer original(store.dim(), pp);
    original.train(store);
    original.encode(store);

    TempFile f("pq.vci");
    original.save(f.path);

    ProductQuantizer loaded = ProductQuantizer::load(f.path);

    CHECK(loaded.params().m == pp.m);
    CHECK(loaded.dsub() == original.dsub());
    CHECK(loaded.n_codes() == original.n_codes());
    CHECK(loaded.code_bytes() == original.code_bytes());
    CHECK(loaded.codebook_bytes() == original.codebook_bytes());

    for (vec_id_t q = 0; q < queries.size(); ++q) {
        const auto a = original.search(queries.at(q), 10);
        const auto b = loaded.search(queries.at(q), 10);
        REQUIRE(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i].id == b[i].id);
            CHECK(a[i].dist == doctest::Approx(b[i].dist));
        }
    }

    // The vectors are deliberately NOT in the file, so rerank still needs the
    // caller's store -- and works when given it.
    const auto reranked = loaded.search_rerank(queries.at(0), 10, 100, store);
    CHECK(reranked.size() == 10);
}

TEST_CASE("a PQ file is small because it holds codes, not vectors") {
    const VectorStore store = random_store(2000, 16, 4110);
    PqParams pp;
    pp.m = 4;
    pp.train_size = 2000;
    pp.kmeans_iters = 5;
    pp.kmeans_n_init = 1;
    ProductQuantizer pq(store.dim(), pp);
    pq.train(store);
    pq.encode(store);

    TempFile f("pqsize.vci");
    pq.save(f.path);

    const auto file_bytes = read_all(f.path).size();
    // 2000 x 16 floats = 128,000 B of vectors; codes are 2000 x 4 = 8,000 B plus
    // 4 x 256 x 4 floats of codebooks. The file must be far under the raw data.
    CHECK(file_bytes < store.bytes());
    CHECK(file_bytes > pq.code_bytes());
}

TEST_CASE("an untrained quantizer refuses to save") {
    // It would write a valid file full of zero codebooks, load without error,
    // and return garbage neighbours. Refusing is the only honest option.
    PqParams pp;
    pp.m = 4;
    ProductQuantizer pq(16, pp);
    TempFile f("untrained.vci");
    CHECK_THROWS_AS(pq.save(f.path), SerializationError);
}
