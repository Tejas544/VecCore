#include "veccore/serialize.hpp"

#include "veccore/hnsw.hpp"
#include "veccore/pq.hpp"
#include "veccore/storage.hpp"

#include <sstream>

namespace veccore {

namespace {

/// The header is fixed-size and comes first, so every rejection below happens
/// before a single byte of payload is allocated. Ordering matters: check MAGIC
/// before VERSION, because a non-VecCore file will produce a nonsense version
/// number and "format version 1919249509" is a worse error message than "this
/// is not a VecCore index file".
constexpr const char* kKindName(IndexKind k) {
    return k == IndexKind::Hnsw ? "hnsw" : "product_quantizer";
}

}  // namespace

void write_header(BinaryWriter& w, IndexKind kind) {
    for (char c : kMagic) w.pod(c);
    w.pod(kFormatVersion);
    w.pod(kEndianProbe);
    w.pod(static_cast<std::uint8_t>(sizeof(std::size_t)));
    w.pod(static_cast<std::uint8_t>(sizeof(float)));
    w.pod(static_cast<std::uint8_t>(sizeof(vec_id_t)));
    w.pod(static_cast<std::uint8_t>(0));  // reserved, keeps the header 8-aligned
    w.pod(static_cast<std::uint32_t>(kind));
}

IndexKind read_header(BinaryReader& r, IndexKind expected) {
    char magic[8];
    for (char& c : magic) c = r.pod<char>();
    if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw SerializationError(
            "not a VecCore index file (magic mismatch). If this is a real index, it was written "
            "by something other than this library.");
    }

    const auto version = r.pod<std::uint32_t>();
    if (version != kFormatVersion) {
        throw SerializationError(
            "index format version " + std::to_string(version) + ", but this build reads version " +
            std::to_string(kFormatVersion) +
            ". There is no forward or backward compatibility by design -- rebuild the index.");
    }

    const auto probe = r.pod<std::uint32_t>();
    if (probe != kEndianProbe) {
        throw SerializationError(
            "byte-order mismatch: this file was written on a machine with the opposite endianness. "
            "The format is native-endian by design (CONTEXT.md D15); it detects this rather than "
            "silently producing wrong distances.");
    }

    const auto size_t_width = r.pod<std::uint8_t>();
    const auto float_width = r.pod<std::uint8_t>();
    const auto id_width = r.pod<std::uint8_t>();
    (void)r.pod<std::uint8_t>();  // reserved
    if (size_t_width != sizeof(std::size_t) || float_width != sizeof(float) ||
        id_width != sizeof(vec_id_t)) {
        throw SerializationError(
            "type-width mismatch: file has size_t=" + std::to_string(size_t_width) + " float=" +
            std::to_string(float_width) + " id=" + std::to_string(id_width) + ", this build has " +
            std::to_string(sizeof(std::size_t)) + "/" + std::to_string(sizeof(float)) + "/" +
            std::to_string(sizeof(vec_id_t)) + ". Rebuild the index on this platform.");
    }

    const auto kind = static_cast<IndexKind>(r.pod<std::uint32_t>());
    if (kind != expected) {
        throw SerializationError(std::string("this file holds a ") + kKindName(kind) +
                                 " index, but a " + kKindName(expected) + " was requested");
    }
    return kind;
}

// ---------------------------------------------------------------------------
// HnswIndex
// ---------------------------------------------------------------------------

void HnswIndex::save(const std::string& path) const {
    BinaryWriter w(path);
    write_header(w, IndexKind::Hnsw);

    // Parameters first, so a reader can construct the index before it has read
    // the (much larger) arrays.
    w.pod(static_cast<std::uint64_t>(params_.M));
    w.pod(static_cast<std::uint64_t>(params_.ef_construction));
    w.pod(params_.seed);
    w.pod(static_cast<std::uint8_t>(params_.extend_candidates));
    w.pod(static_cast<std::uint8_t>(params_.keep_pruned));

    // The vectors. See the declaration for why they are in the file at all.
    w.pod(static_cast<std::uint32_t>(store_.dim()));
    w.vec(store_.raw());

    w.pod(static_cast<std::uint64_t>(stride0_));
    w.pod(static_cast<std::uint64_t>(strideU_));
    w.pod(static_cast<std::uint32_t>(entry_point_));
    w.pod(static_cast<std::uint64_t>(max_level_));
    w.pod(static_cast<std::uint8_t>(empty_));

    w.vec(levels_);
    w.vec(upper_offset_);
    w.vec(links0_);
    w.vec(upper_);

    // P-03 is about an unseeded RNG making a benchmark unreproducible. Dropping
    // the RNG state here would reintroduce exactly that, one level down: two
    // indexes with identical contents would diverge on their next insert
    // depending on whether one of them had been through a save/load cycle.
    // mt19937_64 has a standard stream representation, so this is portable
    // across libstdc++ versions in a way that memcpy-ing the state would not be.
    std::ostringstream rng_state;
    rng_state << rng_;
    w.str(rng_state.str());

    w.finish();
}

HnswIndex HnswIndex::load(const std::string& path, VectorStore& store_out) {
    BinaryReader r(path);
    read_header(r, IndexKind::Hnsw);

    HnswParams params;
    params.M = static_cast<std::size_t>(r.pod<std::uint64_t>());
    params.ef_construction = static_cast<std::size_t>(r.pod<std::uint64_t>());
    params.seed = r.pod<std::uint64_t>();
    params.extend_candidates = r.pod<std::uint8_t>() != 0;
    params.keep_pruned = r.pod<std::uint8_t>() != 0;

    const auto dim = static_cast<dim_t>(r.pod<std::uint32_t>());
    if (dim == 0) throw SerializationError("index file declares dim=0: " + path);
    auto data = r.vec<float>();
    if (data.size() % dim != 0) {
        throw SerializationError("vector buffer is not a multiple of dim in " + path);
    }
    store_out = VectorStore(std::move(data), dim);

    // Constructing here sizes the per-node arrays to the store, then every one
    // of them is overwritten below. The constructor is still the right way in:
    // it validates M and derives the strides, and re-deriving those by hand here
    // would be a second source of truth for the layout.
    HnswIndex index(store_out, params);

    index.stride0_ = static_cast<std::size_t>(r.pod<std::uint64_t>());
    index.strideU_ = static_cast<std::size_t>(r.pod<std::uint64_t>());
    index.entry_point_ = static_cast<vec_id_t>(r.pod<std::uint32_t>());
    index.max_level_ = static_cast<std::size_t>(r.pod<std::uint64_t>());
    index.empty_ = r.pod<std::uint8_t>() != 0;

    index.levels_ = r.vec<std::uint8_t>();
    index.upper_offset_ = r.vec<offset_t>();
    index.links0_ = r.vec<std::uint32_t>();
    index.upper_ = r.vec<std::uint32_t>();

    const std::string rng_state = r.str();
    std::istringstream rng_in(rng_state);
    rng_in >> index.rng_;
    if (!rng_in) throw SerializationError("could not restore the RNG state from " + path);

    r.finish();

    // Cross-field consistency. The checksum proves the bytes are the bytes that
    // were written; it says nothing about whether they are *coherent*, and a
    // file written by a future version with a different layout could pass the
    // checksum and still describe an impossible graph. These are cheap and they
    // turn "wrong answers later" into "refused to load now".
    const std::size_t n = store_out.size();
    if (index.levels_.size() != n) {
        throw SerializationError("levels array covers " + std::to_string(index.levels_.size()) +
                                 " nodes but the store holds " + std::to_string(n));
    }
    if (index.upper_offset_.size() != n) {
        throw SerializationError("upper-offset array does not cover every node in " + path);
    }
    if (index.links0_.size() != n * index.stride0_) {
        throw SerializationError("layer-0 adjacency is " + std::to_string(index.links0_.size()) +
                                 " entries, expected " + std::to_string(n * index.stride0_));
    }
    if (!index.empty_ && index.entry_point_ >= n) {
        throw SerializationError("entry point " + std::to_string(index.entry_point_) +
                                 " is out of range for " + std::to_string(n) + " nodes");
    }

    index.scratch_.visited.resize(index.capacity());
    return index;
}

// ---------------------------------------------------------------------------
// ProductQuantizer
// ---------------------------------------------------------------------------

void ProductQuantizer::save(const std::string& path) const {
    if (!trained_) {
        throw SerializationError(
            "refusing to save an untrained quantizer: it holds no codebooks, so the file would "
            "load successfully and return garbage");
    }
    BinaryWriter w(path);
    write_header(w, IndexKind::ProductQuantizer);

    w.pod(static_cast<std::uint32_t>(d_));
    w.pod(static_cast<std::uint32_t>(dsub_));
    w.pod(static_cast<std::uint64_t>(params_.m));
    w.pod(static_cast<std::uint64_t>(params_.train_size));
    w.pod(static_cast<std::uint64_t>(params_.kmeans_iters));
    w.pod(params_.seed);
    w.pod(static_cast<std::uint64_t>(params_.kmeans_n_init));
    w.pod(static_cast<std::uint64_t>(reseeds_));

    w.vec(codebooks_);
    w.vec(codes_);

    w.finish();
}

ProductQuantizer ProductQuantizer::load(const std::string& path) {
    BinaryReader r(path);
    read_header(r, IndexKind::ProductQuantizer);

    const auto d = static_cast<dim_t>(r.pod<std::uint32_t>());
    const auto dsub = static_cast<dim_t>(r.pod<std::uint32_t>());

    PqParams params;
    params.m = static_cast<std::size_t>(r.pod<std::uint64_t>());
    params.train_size = static_cast<std::size_t>(r.pod<std::uint64_t>());
    params.kmeans_iters = static_cast<std::size_t>(r.pod<std::uint64_t>());
    params.seed = r.pod<std::uint64_t>();
    params.kmeans_n_init = static_cast<std::size_t>(r.pod<std::uint64_t>());
    const auto reseeds = static_cast<std::size_t>(r.pod<std::uint64_t>());

    ProductQuantizer pq(d, params);
    pq.dsub_ = dsub;
    pq.reseeds_ = reseeds;
    pq.codebooks_ = r.vec<float>();
    pq.codes_ = r.vec<std::uint8_t>();
    pq.trained_ = true;

    r.finish();

    const std::size_t expect_codebooks = params.m * PqParams::kCentroids * dsub;
    if (pq.codebooks_.size() != expect_codebooks) {
        throw SerializationError("codebooks are " + std::to_string(pq.codebooks_.size()) +
                                 " floats, expected " + std::to_string(expect_codebooks) +
                                 " for m=" + std::to_string(params.m));
    }
    if (params.m == 0 || pq.codes_.size() % params.m != 0) {
        throw SerializationError("code buffer is not a whole number of m-byte codes in " + path);
    }
    if (dsub == 0 || static_cast<std::size_t>(dsub) * params.m != static_cast<std::size_t>(d)) {
        // P-21 one layer down: a file whose m does not divide its d describes an
        // encoding that cannot be decoded, and every distance after it is wrong.
        throw SerializationError("m=" + std::to_string(params.m) + " and dsub=" +
                                 std::to_string(dsub) + " do not reconstruct dim=" +
                                 std::to_string(d));
    }
    return pq;
}

}  // namespace veccore
