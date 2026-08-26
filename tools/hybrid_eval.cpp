// Hybrid retrieval evaluation on EdgeRAG's real corpus.
//
// CONTEXT.md D7 and BUGS.md L-03. The tempting claim -- "VecCore made EdgeRAG's
// retrieval faster with HNSW" -- is false at 362 documents, where HNSW is
// slower than a brute-force scan. The defensible claims are:
//
//   1. a no-regression drop-in (Phase 6),
//   2. **TF-IDF -> BM25, measured on 650 real held-out queries** (this file),
//   3. a measured crossover curve (Phase 6).
//
// The TF-IDF baseline below replicates EdgeRAG's own implementation
// (edgerag/retrieval/embed.py, TfidfVectorizer) exactly: sklearn-smoothed IDF
// log((1+N)/(1+df)) + 1, raw term counts, L2-normalised, cosine by dot product.
// It lives here in tools/ rather than in the library, for the same reason FAISS
// does: it is a baseline, not a component.
//
// Its tokenizer is `_TOKEN_RE.findall(text.lower())` -- lowercased alphanumeric
// runs, which is character-for-character what veccore::tokenize does. That is
// deliberate and load-bearing: it means this comparison isolates the three
// things that actually differ (term-frequency saturation, explicit length
// normalisation, and the IDF form) rather than confounding them with
// tokenisation.

#include "veccore/bm25.hpp"
#include "veccore/flat_index.hpp"
#include "veccore/fusion.hpp"
#include "veccore/json.hpp"
#include "veccore/stamp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace veccore;

namespace {

struct Corpus {
    std::vector<std::string> keys;
    std::vector<std::string> texts;
};

struct Queries {
    std::vector<std::string> ids;
    std::vector<std::uint32_t> gold;   ///< index into Corpus
    std::vector<std::string> text;
};

std::vector<std::string> split_tabs(const std::string& line, std::size_t expected) {
    std::vector<std::string> parts;
    parts.reserve(expected);
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == '\t') {
            parts.emplace_back(line, start, i - start);
            start = i + 1;
        }
    }
    return parts;
}

Corpus read_corpus(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    Corpus c;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const auto parts = split_tabs(line, 2);
        c.keys.push_back(parts[0]);
        c.texts.push_back(parts.size() > 1 ? parts[1] : "");
    }
    return c;
}

Queries read_queries(const std::string& path, const Corpus& corpus) {
    std::unordered_map<std::string, std::uint32_t> key_to_id;
    for (std::size_t i = 0; i < corpus.keys.size(); ++i) {
        key_to_id[corpus.keys[i]] = static_cast<std::uint32_t>(i);
    }

    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    Queries q;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const auto parts = split_tabs(line, 3);
        if (parts.size() < 3) continue;
        const auto it = key_to_id.find(parts[1]);
        if (it == key_to_id.end()) continue;  // exported script already drops these
        q.ids.push_back(parts[0]);
        q.gold.push_back(it->second);
        q.text.push_back(parts[2]);
    }
    return q;
}

/// EdgeRAG's TF-IDF, replicated. See the file header.
class TfidfBaseline {
public:
    void build(const std::vector<std::string>& docs) {
        std::unordered_map<std::string, std::uint32_t> df;
        std::vector<std::unordered_map<std::string, std::uint32_t>> tfs(docs.size());

        for (std::size_t d = 0; d < docs.size(); ++d) {
            for (const std::string& t : tokenize(docs[d])) ++tfs[d][t];
            for (const auto& [term, _] : tfs[d]) ++df[term];
        }

        const auto N = static_cast<double>(docs.size());
        for (const auto& [term, count] : df) {
            // sklearn's smoothed IDF, plus one. A term in every document keeps
            // a small positive weight rather than vanishing to zero.
            idf_[term] = std::log((1.0 + N) / (1.0 + static_cast<double>(count))) + 1.0;
        }

        vectors_.resize(docs.size());
        for (std::size_t d = 0; d < docs.size(); ++d) {
            double norm2 = 0.0;
            for (const auto& [term, count] : tfs[d]) {
                const double w = static_cast<double>(count) * idf_[term];
                vectors_[d][term] = w;
                norm2 += w * w;
            }
            const double norm = std::sqrt(norm2);
            if (norm > 0.0) {
                for (auto& [term, w] : vectors_[d]) w /= norm;
            }
        }
    }

    [[nodiscard]] std::vector<Neighbor> search(const std::string& query, std::size_t k) const {
        std::unordered_map<std::string, double> qv;
        double norm2 = 0.0;
        {
            std::unordered_map<std::string, std::uint32_t> tf;
            for (const std::string& t : tokenize(query)) ++tf[t];
            for (const auto& [term, count] : tf) {
                const auto it = idf_.find(term);
                if (it == idf_.end()) continue;  // out of vocabulary
                const double w = static_cast<double>(count) * it->second;
                qv[term] = w;
                norm2 += w * w;
            }
        }
        const double norm = std::sqrt(norm2);
        if (norm > 0.0) for (auto& [term, w] : qv) w /= norm;

        TopK top(k);
        for (std::size_t d = 0; d < vectors_.size(); ++d) {
            double dot = 0.0;
            // Iterate the shorter side -- queries are a handful of terms while
            // documents run to hundreds.
            for (const auto& [term, qw] : qv) {
                const auto it = vectors_[d].find(term);
                if (it != vectors_[d].end()) dot += qw * it->second;
            }
            if (dot > 0.0) top.offer(static_cast<float>(-dot), static_cast<vec_id_t>(d));
        }
        return top.take();
    }

private:
    std::unordered_map<std::string, double> idf_;
    std::vector<std::unordered_map<std::string, double>> vectors_;
};

struct Scores {
    double r1 = 0, r5 = 0, r10 = 0;
    double p50_ms = 0, mean_ms = 0;
};

template <typename SearchFn>
Scores evaluate(SearchFn&& search, const Queries& q) {
    using Clock = std::chrono::steady_clock;
    Scores s;
    std::vector<double> lat;
    lat.reserve(q.ids.size());

    for (std::size_t i = 0; i < q.ids.size(); ++i) {
        const auto t0 = Clock::now();
        const std::vector<Neighbor> hits = search(q.text[i]);
        lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());

        for (std::size_t rank = 0; rank < hits.size(); ++rank) {
            if (hits[rank].id != q.gold[i]) continue;
            if (rank < 1) s.r1 += 1.0;
            if (rank < 5) s.r5 += 1.0;
            if (rank < 10) s.r10 += 1.0;
            break;
        }
    }

    const auto n = static_cast<double>(q.ids.size());
    s.r1 /= n; s.r5 /= n; s.r10 /= n;
    std::sort(lat.begin(), lat.end());
    s.p50_ms = lat.empty() ? 0.0 : lat[lat.size() / 2];
    double sum = 0.0;
    for (const double v : lat) sum += v;
    s.mean_ms = lat.empty() ? 0.0 : sum / n;
    return s;
}

void print(const std::string& label, const Scores& s, double ceiling) {
    std::cout << std::fixed << std::setprecision(4)
              << "  " << std::setw(28) << std::left << label
              << " R@1 " << s.r1 << "   R@5 " << s.r5 << "   R@10 " << s.r10
              << "   |  R@5 vs ceiling " << std::setw(7) << (ceiling > 0 ? s.r5 / ceiling : 0.0)
              << "   p50 " << s.p50_ms << " ms\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "/home/tej/veccore-data/edgerag";
    std::string out_path = "results/bench.jsonl";
    bool allow_untrusted = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--allow-untrusted") allow_untrusted = true;
        else { std::cerr << "usage: hybrid_eval [--dir PATH] [--out PATH] [--allow-untrusted]\n"; return 2; }
    }

    const EnvStamp env = capture_env(".");
    if (!env.trusted() && !allow_untrusted) {
        std::cerr << "hybrid_eval: untrusted build (" << env.untrusted_reason() << ")\n";
        return 1;
    }

    const Corpus corpus = read_corpus(dir + "/corpus.tsv");
    const Queries queries = read_queries(dir + "/queries.tsv", corpus);

    std::size_t with_text = 0, reachable = 0;
    for (const std::string& t : corpus.texts) if (!t.empty()) ++with_text;
    for (const std::uint32_t g : queries.gold) if (!corpus.texts[g].empty()) ++reachable;
    const double ceiling = queries.gold.empty()
                               ? 0.0
                               : static_cast<double>(reachable) / static_cast<double>(queries.gold.size());

    std::cout << "EdgeRAG hybrid retrieval evaluation\n"
              << "  documents        : " << corpus.keys.size()
              << "  (" << with_text << " with OCR text, "
              << corpus.keys.size() - with_text << " with none)\n"
              << "  queries          : " << queries.ids.size() << "\n"
              << "  STRUCTURAL CEILING: " << std::fixed << std::setprecision(4) << ceiling
              << "  -- " << queries.gold.size() - reachable
              << " queries point at a document with no text at all and are\n"
              << "                      unreachable by ANY text retriever. Quote against this,\n"
              << "                      never against 1.0.\n\n";

    TfidfBaseline tfidf;
    tfidf.build(corpus.texts);
    Bm25Index bm25;
    bm25.build(corpus.texts);

    std::cout << "  bm25 vocab       : " << bm25.vocab_size()
              << "   avgdl " << bm25.avgdl() << "\n\n";

    const Scores s_tfidf = evaluate([&](const std::string& q) { return tfidf.search(q, 10); }, queries);
    const Scores s_bm25  = evaluate([&](const std::string& q) { return bm25.search(q, 10); }, queries);
    const Scores s_rrf   = evaluate(
        [&](const std::string& q) {
            return reciprocal_rank_fusion({bm25.search(q, 10), tfidf.search(q, 10)}, 10);
        }, queries);

    print("TF-IDF (EdgeRAG today)", s_tfidf, ceiling);
    print("BM25 (VecCore)", s_bm25, ceiling);
    print("RRF(BM25, TF-IDF)", s_rrf, ceiling);

    // --- why did RRF not help? -------------------------------------------
    //
    // PLAN.md 4.5: "If hybrid does not beat both, say so AND INVESTIGATE."
    // The hypothesis is that RRF needs the retrievers to be *complementary*,
    // and these two are merely correlated with one dominating. That is
    // testable, so it gets tested rather than asserted.
    std::size_t overlap_sum = 0, bm25_only = 0, tfidf_only = 0, both = 0, neither = 0;
    for (std::size_t i = 0; i < queries.ids.size(); ++i) {
        const auto b = bm25.search(queries.text[i], 10);
        const auto t = tfidf.search(queries.text[i], 10);

        for (const Neighbor& x : b) {
            for (const Neighbor& y : t) {
                if (x.id == y.id) { ++overlap_sum; break; }
            }
        }
        auto found_at5 = [&](const std::vector<Neighbor>& v) {
            for (std::size_t r = 0; r < v.size() && r < 5; ++r) {
                if (v[r].id == queries.gold[i]) return true;
            }
            return false;
        };
        const bool fb = found_at5(b), ft = found_at5(t);
        if (fb && ft) ++both;
        else if (fb) ++bm25_only;
        else if (ft) ++tfidf_only;
        else ++neither;
    }
    const double mean_overlap =
        queries.ids.empty() ? 0.0
                            : static_cast<double>(overlap_sum) / (10.0 * static_cast<double>(queries.ids.size()));

    const double lift5 = s_tfidf.r5 > 0 ? (s_bm25.r5 - s_tfidf.r5) / s_tfidf.r5 : 0.0;
    std::cout << "\n  BM25 vs TF-IDF, recall@5 : "
              << (s_bm25.r5 - s_tfidf.r5 >= 0 ? "+" : "") << (s_bm25.r5 - s_tfidf.r5)
              << " absolute  (" << (lift5 >= 0 ? "+" : "") << lift5 * 100.0 << "% relative)\n";

    // The oracle: recall a *perfect* fuser would reach, one that always picked
    // whichever retriever was right. It is the difference between "there is no
    // complementary signal to exploit" and "there is, and RRF does not exploit
    // it" -- two very different findings that the recall numbers alone cannot
    // tell apart.
    const double oracle_r5 =
        queries.ids.empty() ? 0.0
                            : static_cast<double>(both + bm25_only + tfidf_only) /
                                  static_cast<double>(queries.ids.size());

    std::cout << "\n  Why RRF did not help here -- measured, not asserted:\n"
              << "    mean top-10 overlap between BM25 and TF-IDF : " << mean_overlap << "\n"
              << "    queries where BOTH find the gold doc @5     : " << both << "\n"
              << "    BM25 finds it, TF-IDF does not              : " << bm25_only << "\n"
              << "    TF-IDF finds it, BM25 does not              : " << tfidf_only << "\n"
              << "    neither                                     : " << neither << "\n"
              << "\n    oracle fusion recall@5 (always pick the right one) : " << oracle_r5 << "\n"
              << "    BM25 alone                                         : " << s_bm25.r5 << "\n"
              << "    RRF                                                : " << s_rrf.r5 << "\n";

    std::cout << "\n  RRF fuses by RANK ALONE, which is exactly what makes it tuning-free -- and it\n"
              << "  is also why it cannot help here. It has no way to know one list is better; it\n"
              << "  treats every input as equally authoritative. That is the right bet when the\n"
              << "  retrievers are COMPLEMENTARY (each right about different queries), and the\n"
              << "  wrong one when they are CORRELATED with one dominating: averaging a strong\n"
              << "  ranker with a weaker one that mostly agrees can only drag it toward the\n"
              << "  weaker. The asymmetry above is the evidence.\n"
              << "\n  This is a property of the pairing, not a defect in RRF. EdgeRAG's dense\n"
              << "  signal was measured to be noise (DEFAULT_ALPHA = 0.0, hybrid == text_only\n"
              << "  bit-for-bit), so the only second retriever available on this corpus is another\n"
              << "  lexical one over the same tokens. A genuine dense+sparse hybrid needs a real\n"
              << "  text embedding model -- out of scope here, and named as a limitation rather\n"
              << "  than papered over with a number that flatters.\n";

    json::Object p;
    p.str("kind", "edgerag_hybrid")
     .num("n_docs", static_cast<long long>(corpus.keys.size()))
     .num("n_docs_with_text", static_cast<long long>(with_text))
     .num("n_queries", static_cast<long long>(queries.ids.size()))
     .num("structural_ceiling", ceiling)
     .num("bm25_k1", 1.2).num("bm25_b", 0.75).num("rrf_k", 60.0);

    json::Object m;
    m.num("tfidf_recall_at_1", s_tfidf.r1).num("tfidf_recall_at_5", s_tfidf.r5)
     .num("tfidf_recall_at_10", s_tfidf.r10).num("tfidf_p50_ms", s_tfidf.p50_ms)
     .num("bm25_recall_at_1", s_bm25.r1).num("bm25_recall_at_5", s_bm25.r5)
     .num("bm25_recall_at_10", s_bm25.r10).num("bm25_p50_ms", s_bm25.p50_ms)
     .num("rrf_recall_at_1", s_rrf.r1).num("rrf_recall_at_5", s_rrf.r5)
     .num("rrf_recall_at_10", s_rrf.r10).num("rrf_p50_ms", s_rrf.p50_ms)
     .num("bm25_lift_at_5_absolute", s_bm25.r5 - s_tfidf.r5)
     .num("bm25_lift_at_5_relative", lift5)
     .num("bm25_recall_at_5_vs_ceiling", ceiling > 0 ? s_bm25.r5 / ceiling : 0.0)
     .num("mean_top10_overlap", mean_overlap)
     .num("gold_found_by_both", static_cast<long long>(both))
     .num("gold_found_by_bm25_only", static_cast<long long>(bm25_only))
     .num("gold_found_by_tfidf_only", static_cast<long long>(tfidf_only))
     .num("gold_found_by_neither", static_cast<long long>(neither))
     .num("oracle_fusion_recall_at_5", oracle_r5)
     .num("index_bytes", static_cast<long long>(bm25.index_bytes()));

    json::Object rec;
    rec.str("tag", "edgerag_hybrid").str("index", "bm25")
       .num("n_base", static_cast<long long>(corpus.keys.size()))
       .num("n_queries", static_cast<long long>(queries.ids.size()))
       .raw("index_params", p.str()).raw("env", to_json(env)).raw("measurements", m.str());

    std::ofstream out(out_path, std::ios::app);
    if (!out) { std::cerr << "cannot open " << out_path << "\n"; return 1; }
    out << rec.str() << '\n';
    std::cout << "\n  wrote 1 record to " << out_path << "\n";
    return 0;
}
