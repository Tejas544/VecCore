#include "veccore/kmeans.hpp"

#include "veccore/distance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace veccore {
namespace {

/// k-means++ seeding: first centroid uniform, each subsequent one drawn with
/// probability proportional to squared distance from the nearest already-chosen
/// centroid.
///
/// Worth the ~20 lines. Uniform random init regularly places two centroids
/// inside the same dense region and leaves another region unrepresented, and
/// Lloyd's cannot recover from that -- it only ever moves centroids downhill.
/// The result is a visibly worse codebook, which shows up as PQ recall that is
/// mysteriously below what the compression ratio should buy.
std::vector<float> kmeanspp_init(const float* data, std::size_t n, dim_t d, std::size_t k,
                                 std::mt19937_64& rng) {
    const L2Sqr dist;
    std::vector<float> centroids(k * static_cast<offset_t>(d));
    std::vector<double> best(n, std::numeric_limits<double>::max());

    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    std::size_t chosen = pick(rng);
    std::copy_n(data + chosen * static_cast<offset_t>(d), d, centroids.begin());

    for (std::size_t c = 1; c < k; ++c) {
        double total = 0.0;
        const float* prev = centroids.data() + (c - 1) * static_cast<offset_t>(d);
        for (std::size_t i = 0; i < n; ++i) {
            const double dd = dist(data + i * static_cast<offset_t>(d), prev, d);
            if (dd < best[i]) best[i] = dd;
            total += best[i];
        }

        if (total <= 0.0) {
            // Every remaining point coincides with a centroid -- there is no
            // meaningful k-th cluster. Duplicate rather than divide by zero;
            // the empty-cluster handler below will deal with the consequence.
            std::copy_n(data + pick(rng) * static_cast<offset_t>(d), d,
                        centroids.begin() + c * static_cast<offset_t>(d));
            continue;
        }

        std::uniform_real_distribution<double> u(0.0, total);
        double target = u(rng);
        std::size_t idx = n - 1;
        for (std::size_t i = 0; i < n; ++i) {
            target -= best[i];
            if (target <= 0.0) { idx = i; break; }
        }
        std::copy_n(data + idx * static_cast<offset_t>(d), d,
                    centroids.begin() + c * static_cast<offset_t>(d));
    }
    return centroids;
}

}  // namespace

namespace {

KMeansResult kmeans_once(const float* data, std::size_t n, dim_t d, const KMeansParams& params) {
    const std::size_t k = std::min(params.k, n);

    std::mt19937_64 rng(params.seed);
    const L2Sqr dist;

    KMeansResult r;
    r.k = k;
    r.d = d;
    r.centroids = kmeanspp_init(data, n, d, k, rng);

    std::vector<std::size_t> assign(n, 0);
    std::vector<double> sums(k * static_cast<offset_t>(d), 0.0);
    std::vector<std::size_t> counts(k, 0);
    std::vector<double> point_dist(n, 0.0);

    double previous = std::numeric_limits<double>::max();

    for (std::size_t iter = 0; iter < params.max_iters; ++iter) {
        // --- assign ---------------------------------------------------------
        double inertia = 0.0;
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(sums.begin(), sums.end(), 0.0);

        for (std::size_t i = 0; i < n; ++i) {
            const float* p = data + i * static_cast<offset_t>(d);
            // float throughout the inner comparison: promoting each distance to
            // double buys no accuracy (the inputs are float) and costs a
            // conversion in the hottest loop in codebook training.
            float best = std::numeric_limits<float>::max();
            std::size_t best_c = 0;
            for (std::size_t c = 0; c < k; ++c) {
                const float dd = dist(p, r.centroid(c), d);
                if (dd < best) { best = dd; best_c = c; }
            }
            assign[i] = best_c;
            point_dist[i] = static_cast<double>(best);
            inertia += static_cast<double>(best);
            ++counts[best_c];
            double* acc = sums.data() + best_c * static_cast<offset_t>(d);
            // Accumulate in double explicitly: summing 100k floats into a float
            // loses low-order bits, and the centroid mean is what the whole
            // codebook rests on.
            for (dim_t j = 0; j < d; ++j) acc[j] += static_cast<double>(p[j]);
        }

        // --- update ---------------------------------------------------------
        for (std::size_t c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                float* cen = r.centroids.data() + c * static_cast<offset_t>(d);
                const double* acc = sums.data() + c * static_cast<offset_t>(d);
                const auto inv = 1.0 / static_cast<double>(counts[c]);
                for (dim_t j = 0; j < d; ++j) cen[j] = static_cast<float>(acc[j] * inv);
                continue;
            }

            // P-07, the one that destroys everything downstream in silence.
            //
            // A centroid owning zero points would get 0/0 = NaN on the mean
            // update. The NaN then propagates into every distance computed
            // against it, and because every comparison with NaN is false, the
            // "nearest centroid" logic quietly stops working -- no crash, no
            // warning, garbage results.
            //
            // Re-seed it onto the point that is currently worst-served: the
            // farthest point from its own centroid. That both removes the NaN
            // and does something useful, since splitting the worst-covered
            // region is exactly where an extra centroid belongs.
            const auto worst = static_cast<std::size_t>(
                std::max_element(point_dist.begin(), point_dist.end()) - point_dist.begin());
            std::copy_n(data + worst * static_cast<offset_t>(d), d,
                        r.centroids.begin() + c * static_cast<offset_t>(d));
            point_dist[worst] = 0.0;  // do not re-seed two centroids onto the same point
            ++r.empty_cluster_reseeds;
        }

        r.iters = iter + 1;
        r.inertia = inertia;

        // Relative improvement, not absolute -- inertia scales with n and with
        // the data's units, so an absolute threshold would mean something
        // different for every subspace.
        if (previous != std::numeric_limits<double>::max()) {
            const double improvement = (previous - inertia) / (previous > 0.0 ? previous : 1.0);
            if (improvement >= 0.0 && improvement < params.tol) break;
        }
        previous = inertia;
    }

    // The guard that makes P-07 impossible to reintroduce silently. Cheap
    // (k*d floats, once) and it fires before any codebook reaches the encoder.
    for (const float v : r.centroids) {
        if (!std::isfinite(v)) {
            throw std::runtime_error(
                "kmeans: produced a non-finite centroid. This is P-07 -- an empty cluster "
                "reached the mean update. Every distance against it would be NaN, every "
                "comparison false, and the index would return garbage without erroring.");
        }
    }
    return r;
}

}  // namespace

KMeansResult kmeans(const float* data, std::size_t n, dim_t d, const KMeansParams& params) {
    if (n == 0 || d == 0) throw std::invalid_argument("kmeans: empty input");

    const std::size_t restarts = std::max<std::size_t>(params.n_init, 1);
    KMeansResult best;
    std::vector<double> inertias;
    inertias.reserve(restarts);

    for (std::size_t attempt = 0; attempt < restarts; ++attempt) {
        KMeansParams p = params;
        p.seed = params.seed + attempt * 7919;  // a prime, so seeds do not collide
                                                // across nearby subspaces
        KMeansResult r = kmeans_once(data, n, d, p);
        inertias.push_back(r.inertia);
        if (attempt == 0 || r.inertia < best.inertia) best = std::move(r);
    }

    std::sort(inertias.begin(), inertias.end());
    best.inertia_per_init = std::move(inertias);
    return best;
}

}  // namespace veccore
