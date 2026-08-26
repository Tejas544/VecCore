"""The EdgeRAG drop-in, tested against the protocol it has to satisfy.

CONTEXT.md D7 claim 1 is "no regression": same protocol, same call sites, and a
result at least as good as the TF-IDF index it replaces. This file is what makes
that a checked statement rather than an intention.

Run:
    ~/veccore-venv/bin/python -m pytest tests/test_edgerag_adapter.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    import veccore
except ImportError:  # pragma: no cover
    pytest.skip("_veccore not built", allow_module_level=True)

DATA = Path(os.environ.get("VECCORE_DATA", Path.home() / "veccore-data")) / "edgerag"


@pytest.fixture(scope="module")
def corpus():
    if not (DATA / "corpus.tsv").exists():
        pytest.skip(f"{DATA}/corpus.tsv missing -- run scripts/export_edgerag.py")
    keys, texts = [], []
    for line in (DATA / "corpus.tsv").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        keys.append(parts[0])
        texts.append(parts[1] if len(parts) > 1 else "")
    return keys, texts


@pytest.fixture(scope="module")
def queries():
    if not (DATA / "queries.tsv").exists():
        pytest.skip("queries.tsv missing")
    out = []
    for line in (DATA / "queries.tsv").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 3:
            out.append((parts[0], parts[1], parts[2]))
    return out


def test_satisfies_the_protocol_signature(corpus):
    """EdgeRAG calls exactly this. If the signature drifts, the drop-in isn't one."""
    keys, texts = corpus
    index = veccore.VecCoreIndex(doc_keys=keys, texts=texts)
    result = index.search(None, "social media platform", 5)
    assert isinstance(result, list)
    assert len(result) <= 5
    assert all(isinstance(x, str) for x in result)
    assert all(x in set(keys) for x in result)


def test_defaults_to_text_only_because_the_dense_signal_is_noise(corpus):
    keys, texts = corpus
    index = veccore.VecCoreIndex(doc_keys=keys, texts=texts)
    assert index.alpha == 0.0
    assert index.use_hnsw is False   # 362 docs is below the measured crossover
    assert index.stats["dense_index"] is None
    assert index.search(None, "anything", 3) == index.search_text_only("anything", 3)


def test_no_regression_against_edgerags_tfidf(corpus, queries):
    """Claim 1 of D7, checked on the real corpus and the real queries.

    Recall is quoted against the STRUCTURAL CEILING, not against 1.0: 112 of 362
    documents have no OCR text, so most queries are unreachable by any text
    retriever. Measuring against 1.0 would make a 50%-of-achievable result look
    like a 19% failure.
    """
    keys, texts = corpus
    key_to_id = {k: i for i, k in enumerate(keys)}
    index = veccore.VecCoreIndex(doc_keys=keys, texts=texts)

    reachable = sum(1 for _, gold, _ in queries if texts[key_to_id[gold]].strip())
    ceiling = reachable / len(queries)

    hits = sum(1 for _, gold, q in queries if gold in index.search(None, q, 5))
    recall5 = hits / len(queries)

    # The TF-IDF baseline this replaces measured 0.1846 (tools/hybrid_eval).
    # BM25 measured 0.1923. Assert we are at least as good, with a little slack
    # so the test is not brittle to tokenizer-level changes.
    assert recall5 >= 0.1846, f"regression against EdgeRAG's TF-IDF: {recall5:.4f} < 0.1846"
    assert recall5 / ceiling > 0.45, f"only {recall5 / ceiling:.1%} of the achievable ceiling"


def test_hnsw_path_returns_valid_keys_even_though_it_is_off_by_default(corpus):
    """The dense path must work when asked for, even though D7 says not to use it here."""
    keys, texts = corpus
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((len(keys), 32)).astype(np.float32)

    index = veccore.VecCoreIndex(
        doc_keys=keys, texts=texts, image_vectors=vecs, alpha=0.5, use_hnsw=True
    )
    assert index.stats["dense_index"] == "hnsw"
    assert index.stats["hnsw"]["nodes"] == len(keys)

    out = index.search(vecs[7], "social media", 5)
    assert len(out) <= 5
    assert all(x in set(keys) for x in out)


def test_image_only_search_round_trips_its_own_vector(corpus):
    keys, texts = corpus
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((len(keys), 16)).astype(np.float32)
    index = veccore.VecCoreIndex(doc_keys=keys, texts=texts, image_vectors=vecs, alpha=1.0)
    assert index.search_image_only(vecs[42], 1) == [keys[42]]


def test_mismatched_lengths_raise_rather_than_silently_misalign(corpus):
    """P-20: an off-by-one between ids and keys gives perfect internal recall and
    nonsense answers. It must be impossible to construct, not merely unlikely."""
    keys, texts = corpus
    with pytest.raises(ValueError):
        veccore.VecCoreIndex(doc_keys=keys[:10], texts=texts)
    with pytest.raises(ValueError):
        veccore.VecCoreIndex(
            doc_keys=keys, texts=texts, image_vectors=np.zeros((5, 8), dtype=np.float32)
        )


def test_empty_and_unknown_queries_do_not_raise(corpus):
    keys, texts = corpus
    index = veccore.VecCoreIndex(doc_keys=keys, texts=texts)
    assert index.search(None, "", 5) == []
    assert index.search(None, "zzzzqqqxxx", 5) == []
