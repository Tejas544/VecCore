#include "veccore/pq.hpp"

#include "veccore/distance.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace veccore {

ProductQuantizer::ProductQuantizer(dim_t d, PqParams params) : d_(d), params_(params) {
    if (params_.m == 0) throw std::invalid_argument("PQ: m must be non-zero");
    // P-21. Integer division would silently drop the trailing dimensions from
    // every code, and the recall loss would look like ordinary quantization
    // error rather than a bug. 128/8 = 16 is fine; 128/12 is not.
    if (d_ % params_.m != 0) {
        throw std::invalid_argument("PQ: m=" + std::to_string(params_.m) +
                                    " does not divide d=" + std::to_string(d_) +
                                    " (P-21: integer division would silently truncate each code)");
    }
    dsub_ = static_cast<dim_t>(d_ / params_.m);
}

void ProductQuantizer::train(const VectorStore& store) {
    if (store.dim() != d_) throw std::invalid_argument("PQ: dimension mismatch");

    // Deterministic subsample of the BASE vectors. P-23: never train on the
    // query set, and never report recall on the same vectors the codebooks
    // were fitted to. The sample size goes into the results record.
    const std::size_t n = store.size();
    const std::size_t take = std::min(params_.train_size, n);
    std::vector<vec_id_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0u);
    if (take < n) {
        std::mt19937_64 rng(params_.seed);
        std::shuffle(ids.begin(), ids.end(), rng);
        ids.resize(take);
    }

    codebooks_.assign(params_.m * PqParams::kCentroids * static_cast<offset_t>(dsub_), 0.0f);
    reseeds_ = 0;

    // One independent k-means per subspace. They are independent by
    // construction -- that independence is what makes the codebook product
    // combinatorially large while each codebook stays small enough to fit in
    // cache.
    std::vector<float> sub(take * static_cast<offset_t>(dsub_));
    for (std::size_t s = 0; s < params_.m; ++s) {
        for (std::size_t i = 0; i < take; ++i) {
            const float* v = store.at(ids[i]) + s * static_cast<offset_t>(dsub_);
            std::copy_n(v, dsub_, sub.begin() + i * static_cast<offset_t>(dsub_));
        }

        KMeansParams kp;
        kp.k = PqParams::kCentroids;
        kp.max_iters = params_.kmeans_iters;
        kp.n_init = params_.kmeans_n_init;
        // Different seed per subspace: identical seeds would give every
        // subspace the same k-means++ *draw order*, which is a subtle
        // correlation between codebooks that should not exist.
        kp.seed = params_.seed + s;

        const KMeansResult r = kmeans(sub.data(), take, dsub_, kp);
        reseeds_ += r.empty_cluster_reseeds;
        std::copy(r.centroids.begin(), r.centroids.end(),
                  codebooks_.begin() + s * PqParams::kCentroids * static_cast<offset_t>(dsub_));
    }
    trained_ = true;
}

void ProductQuantizer::encode(const VectorStore& store) {
    if (!trained_) throw std::logic_error("PQ: encode before train");
    if (store.dim() != d_) throw std::invalid_argument("PQ: dimension mismatch");

    const std::size_t n = store.size();
    codes_.assign(n * params_.m, 0);
    const L2Sqr dist;

    for (vec_id_t i = 0; i < n; ++i) {
        const float* v = store.at(i);
        std::uint8_t* out = codes_.data() + static_cast<offset_t>(i) * params_.m;
        for (std::size_t s = 0; s < params_.m; ++s) {
            const float* vs = v + s * static_cast<offset_t>(dsub_);
            float best = std::numeric_limits<float>::max();
            std::size_t best_c = 0;
            for (std::size_t c = 0; c < PqParams::kCentroids; ++c) {
                const float dd = dist(vs, centroid(s, c), dsub_);
                if (dd < best) { best = dd; best_c = c; }
            }
            out[s] = static_cast<std::uint8_t>(best_c);
        }
    }
}

AdcTable ProductQuantizer::adc_table(const float* query) const {
    AdcTable t(params_.m, PqParams::kCentroids);
    const L2Sqr dist;
    for (std::size_t s = 0; s < params_.m; ++s) {
        const float* qs = query + s * static_cast<offset_t>(dsub_);
        float* row = t.row(s);
        for (std::size_t c = 0; c < PqParams::kCentroids; ++c) {
            // P-22: squared distances, matching D4 and the ground truth. A
            // table built with unsquared distances still ranks *almost*
            // correctly, so the recall loss looks like normal PQ error.
            row[c] = dist(qs, centroid(s, c), dsub_);
        }
    }
    return t;
}

std::vector<Neighbor> ProductQuantizer::search(const float* query, std::size_t k) const {
    const AdcTable t = adc_table(query);
    const std::size_t n = n_codes();
    TopK top(k);
    for (vec_id_t i = 0; i < n; ++i) {
        top.offer(t.distance(code(i)), i);
    }
    return top.take();
}

std::vector<Neighbor> ProductQuantizer::search_rerank(const float* query, std::size_t k,
                                                      std::size_t candidates,
                                                      const VectorStore& full) const {
    const std::size_t shortlist = std::max(candidates, k);
    const std::vector<Neighbor> approx = search(query, shortlist);

    const L2Sqr dist;
    TopK top(k);
    for (const Neighbor& c : approx) {
        top.offer(dist(query, full.at(c.id), d_), c.id);
    }
    return top.take();
}

void ProductQuantizer::decode(const std::uint8_t* code_in, float* out) const {
    for (std::size_t s = 0; s < params_.m; ++s) {
        std::copy_n(centroid(s, code_in[s]), dsub_, out + s * static_cast<offset_t>(dsub_));
    }
}

double ProductQuantizer::reconstruction_mse(const VectorStore& store, std::size_t sample) const {
    const std::size_t n = std::min(sample ? sample : store.size(), n_codes());
    if (n == 0) return 0.0;
    std::vector<float> rec(d_);
    const L2Sqr dist;
    double total = 0.0;
    for (vec_id_t i = 0; i < n; ++i) {
        decode(code(i), rec.data());
        total += static_cast<double>(dist(store.at(i), rec.data(), d_));
    }
    return total / static_cast<double>(n);
}

}  // namespace veccore
