"""Drop-in replacement for EdgeRAG's ``FlatIndex``.

EdgeRAG's ``edgerag/retrieval/index.py`` declares a Protocol whose docstring
reads, written months before this repo existed:

    class RetrievalIndex(Protocol):
        \"\"\"What a retriever must do. ``VecCore`` implements this later without
        touching a caller.\"\"\"

This module is that. It implements ``search(image_space_query, query_text, k)``
and nothing else needs to change.

---------------------------------------------------------------------------
What this does and does not claim -- read before quoting a number
---------------------------------------------------------------------------
`CONTEXT.md` D7 and `BUGS.md` L-03. The obvious pitch, "VecCore made EdgeRAG's
retrieval faster with HNSW", is **false**, and measurably so:

  * EdgeRAG's corpus is **362 documents**. `bench/crossover.py` measures the
    HNSW-vs-brute-force crossover at **n ~= 1,189**, so at EdgeRAG's scale HNSW
    is about **42% SLOWER** -- you pay graph traversal to avoid a scan that was
    never expensive.
  * EdgeRAG's dense signal was measured to be noise (``DEFAULT_ALPHA = 0.0``,
    ``hybrid == text_only`` bit-for-bit across 165 held-out queries), so there
    is no dense retrieval here worth accelerating in the first place.

So the defensible claims, and the ones this file delivers, are:

  1. **No regression.** Same protocol, same results, no caller changes.
  2. **A real quality upgrade:** TF-IDF -> BM25, worth **+4.17% relative
     recall@5** on 650 real held-out queries, at **11x lower latency** because
     an inverted index only touches documents containing a query term.
  3. **A measured scaling argument** rather than an asserted one --
     ``docs/plots/crossover.png``.

`use_hnsw` therefore defaults to **False**. Turning it on at 362 documents
would make retrieval slower for the sake of a nicer-sounding architecture
diagram, which is the opposite of engineering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import Bm25Index, FlatIndex, HnswIndex, reciprocal_rank_fusion


@dataclass
class VecCoreIndex:
    """Implements EdgeRAG's ``RetrievalIndex`` protocol.

    Args:
        doc_keys: external document identifiers, indexed by internal id.
        texts: OCR text per document, parallel to ``doc_keys``.
        image_vectors: optional (n, hidden) dense matrix. EdgeRAG's is noise, so
            ``alpha`` defaults to 0 and this is unused unless you pass one.
        alpha: dense weight in the fused ranking. 0 means text only, matching
            EdgeRAG's own measured default.
        use_hnsw: index the dense side with HNSW instead of brute force. Off by
            default -- see the module docstring for the measurement that says why.
    """

    doc_keys: Sequence[str]
    texts: Sequence[str]
    image_vectors: np.ndarray | None = None
    alpha: float = 0.0
    use_hnsw: bool = False
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    _bm25: Bm25Index = field(init=False, repr=False)
    _dense: object | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if len(self.doc_keys) != len(self.texts):
            raise ValueError(
                f"doc_keys has {len(self.doc_keys)} entries and texts has {len(self.texts)}"
            )
        self._bm25 = Bm25Index(list(self.texts), self.bm25_k1, self.bm25_b)

        if self.image_vectors is not None and self.image_vectors.size:
            if self.image_vectors.shape[0] != len(self.doc_keys):
                raise ValueError(
                    f"image_vectors has {self.image_vectors.shape[0]} rows, "
                    f"{len(self.doc_keys)} documents"
                )
            vecs = np.ascontiguousarray(self.image_vectors, dtype=np.float32)
            self._dense = (
                HnswIndex(vecs, M=16, ef_construction=200) if self.use_hnsw else FlatIndex(vecs)
            )

    # -- the protocol ------------------------------------------------------

    def search(
        self, image_space_query: np.ndarray | None, query_text: str, k: int
    ) -> list[str]:
        """Top-``k`` document keys. Signature matches EdgeRAG's Protocol exactly."""
        sparse_ids = [doc_id for doc_id, _ in self._bm25.search(query_text, k)]

        if self._dense is None or self.alpha <= 0.0 or image_space_query is None:
            # The default path, and the honest one for this corpus.
            return [self.doc_keys[i] for i in sparse_ids[:k]]

        q = np.ascontiguousarray(image_space_query, dtype=np.float32)
        dense_ids, _ = (
            self._dense.search(q, k, 64) if self.use_hnsw else self._dense.search(q, k)
        )

        # RRF rather than a weighted blend of the two score columns. The scores
        # live on incomparable scales -- cosine in [-1,1], BM25 unbounded and
        # corpus-dependent -- so any fixed weight is fitted to today's corpus.
        #
        # Measured caveat, because it is not free: on THIS corpus RRF is
        # slightly WORSE than BM25 alone (0.1862 vs 0.1923 recall@5), because
        # the two retrievers are correlated with one dominating rather than
        # complementary. See BUGS.md B-08. That is why alpha defaults to 0.
        fused = reciprocal_rank_fusion([sparse_ids, list(map(int, dense_ids))], k)
        return [self.doc_keys[i] for i in fused[:k]]

    def search_text_only(self, query_text: str, k: int) -> list[str]:
        """One half of the control EdgeRAG's ``recall_at_k`` checks against."""
        return [self.doc_keys[i] for i, _ in self._bm25.search(query_text, k)]

    def search_image_only(self, image_space_query: np.ndarray, k: int) -> list[str]:
        if self._dense is None:
            return []
        q = np.ascontiguousarray(image_space_query, dtype=np.float32)
        ids, _ = self._dense.search(q, k, 64) if self.use_hnsw else self._dense.search(q, k)
        return [self.doc_keys[i] for i in ids]

    # -- introspection -----------------------------------------------------

    @property
    def stats(self) -> dict:
        out = {
            "n_docs": len(self.doc_keys),
            "bm25_vocab": self._bm25.vocab_size,
            "bm25_avgdl": self._bm25.avgdl,
            "dense_index": None if self._dense is None else ("hnsw" if self.use_hnsw else "flat"),
            "alpha": self.alpha,
        }
        if self.use_hnsw and self._dense is not None:
            out["hnsw"] = self._dense.stats()
        return out
