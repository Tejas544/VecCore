#include "veccore/bm25.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>

namespace veccore {

std::vector<std::string> tokenize(std::string_view text) {
    std::vector<std::string> out;
    std::string current;
    current.reserve(24);
    for (const char raw : text) {
        const auto c = static_cast<unsigned char>(raw);
        if (std::isalnum(c)) {
            current += static_cast<char>(std::tolower(c));
        } else if (!current.empty()) {
            out.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) out.push_back(current);
    return out;
}

void Bm25Index::build(const std::vector<std::string>& docs) {
    postings_.clear();
    doc_len_.assign(docs.size(), 0);

    // P-26: document lengths and avgdl are computed in the SAME pass over the
    // SAME tokenization. Computing one before filtering and the other after --
    // or over different document sets -- makes the length-normalisation term
    // subtly wrong for every document, with no error anywhere.
    std::uint64_t total_len = 0;

    for (std::size_t d = 0; d < docs.size(); ++d) {
        const std::vector<std::string> terms = tokenize(docs[d]);
        doc_len_[d] = static_cast<std::uint32_t>(terms.size());
        total_len += terms.size();

        std::unordered_map<std::string, std::uint32_t> tf;
        tf.reserve(terms.size());
        for (const std::string& t : terms) ++tf[t];

        for (const auto& [term, count] : tf) {
            postings_[term].push_back(Posting{static_cast<std::uint32_t>(d), count});
        }
    }

    avgdl_ = docs.empty() ? 0.0
                          : static_cast<double>(total_len) / static_cast<double>(docs.size());

    // Postings sorted by doc id: cheap here, and it makes any future
    // intersection or skip-list work possible without a re-sort.
    for (auto& [term, list] : postings_) {
        std::sort(list.begin(), list.end(),
                  [](const Posting& a, const Posting& b) { return a.doc_id < b.doc_id; });
    }
}

std::uint32_t Bm25Index::doc_freq(const std::string& term) const {
    const auto it = postings_.find(term);
    return it == postings_.end() ? 0u : static_cast<std::uint32_t>(it->second.size());
}

std::uint32_t Bm25Index::term_freq(const std::string& term, std::uint32_t doc_id) const {
    const auto it = postings_.find(term);
    if (it == postings_.end()) return 0;
    for (const Posting& p : it->second) {
        if (p.doc_id == doc_id) return p.tf;
    }
    return 0;
}

double Bm25Index::idf(const std::string& term) const {
    const auto N = static_cast<double>(doc_len_.size());
    const auto df = static_cast<double>(doc_freq(term));
    // P-08: the `1 +` is what keeps this positive for terms in more than half
    // the corpus. Without it, a match can reduce a score.
    return std::log(1.0 + (N - df + 0.5) / (df + 0.5));
}

double Bm25Index::score(std::uint32_t doc_id, const std::vector<std::string>& terms) const {
    if (doc_id >= doc_len_.size()) return 0.0;
    const double len = static_cast<double>(doc_len_[doc_id]);
    const double norm = params_.k1 * (1.0 - params_.b + params_.b * (avgdl_ > 0.0 ? len / avgdl_ : 0.0));

    double total = 0.0;
    for (const std::string& t : terms) {
        const double tf = static_cast<double>(term_freq(t, doc_id));
        if (tf == 0.0) continue;
        total += idf(t) * (tf * (params_.k1 + 1.0)) / (tf + norm);
    }
    return total;
}

std::vector<Neighbor> Bm25Index::search(std::string_view query, std::size_t k) const {
    const std::vector<std::string> terms = tokenize(query);
    if (terms.empty() || doc_len_.empty()) return {};

    // Dense accumulator over documents. For a corpus of this size that is
    // cheaper and far more cache-friendly than a hash map keyed on doc id, and
    // it makes the scan over postings a linear write pattern.
    std::vector<double> acc(doc_len_.size(), 0.0);

    for (const std::string& t : terms) {
        const auto it = postings_.find(t);
        if (it == postings_.end()) continue;

        // idf and the query-term constant are hoisted out of the posting loop:
        // they depend on the term, not the document.
        const double term_idf = idf(t);
        for (const Posting& p : it->second) {
            const double len = static_cast<double>(doc_len_[p.doc_id]);
            const double norm =
                params_.k1 * (1.0 - params_.b + params_.b * (avgdl_ > 0.0 ? len / avgdl_ : 0.0));
            const double tf = static_cast<double>(p.tf);
            acc[p.doc_id] += term_idf * (tf * (params_.k1 + 1.0)) / (tf + norm);
        }
    }

    TopK top(k);
    for (std::size_t d = 0; d < acc.size(); ++d) {
        if (acc[d] == 0.0) continue;  // never matched a query term
        // Negated: "smaller is better" holds across the whole codebase (D4).
        top.offer(static_cast<float>(-acc[d]), static_cast<vec_id_t>(d));
    }
    return top.take();
}

offset_t Bm25Index::index_bytes() const noexcept {
    offset_t bytes = doc_len_.size() * sizeof(std::uint32_t);
    for (const auto& [term, list] : postings_) {
        bytes += list.size() * sizeof(Posting);
        bytes += term.size();  // the key's characters; the map's node overhead is not counted
    }
    return bytes;
}

}  // namespace veccore
