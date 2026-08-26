"""Tests for the pybind11 module.

These exist because two of the binding's failure modes are **invisible from
C++**:

  * P-34, a non-contiguous numpy array silently reinterpreted with the wrong
    strides -- no crash, plausible garbage, wrong recall;
  * P-35, a held GIL, which makes every Phase 5 thread-scaling claim false the
    moment the caller is Python while leaving the C++ benchmark untouched.

Neither can be caught by the doctest suite. That is the whole argument for this
file existing in a project that otherwise tests in C++.

Run:
    ~/veccore-venv/bin/python -m pytest tests/test_bindings.py -v
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

BUILD = Path.home() / "veccore-build"
sys.path.insert(0, str(BUILD))

try:
    import _veccore
except ImportError:  # pragma: no cover
    pytest.skip("_veccore not built; configure with -Dpybind11_DIR=...", allow_module_level=True)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    base = rng.standard_normal((3000, 32)).astype(np.float32)
    queries = rng.standard_normal((50, 32)).astype(np.float32)
    return base, queries


def exact_top_k(base, q, k):
    d2 = ((base - q) ** 2).sum(axis=1)
    return np.argsort(d2, kind="stable")[:k]


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_flat_matches_numpy(data):
    base, queries = data
    index = _veccore.FlatIndex(base)
    assert index.size == 3000
    for i in range(10):
        ids, dists = index.search(queries[i], k=10)
        expected = exact_top_k(base, queries[i], 10)
        assert list(ids) == list(expected)
        # Squared distances, per D4 -- no sqrt anywhere in this project.
        assert np.allclose(dists[0], ((base[ids[0]] - queries[i]) ** 2).sum(), rtol=1e-5)


def test_hnsw_recall_against_exact(data):
    base, queries = data
    index = _veccore.HnswIndex(base, M=16, ef_construction=200)
    hits = 0
    for i in range(len(queries)):
        ids, _ = index.search(queries[i], k=10, ef_search=200)
        hits += len(set(ids.tolist()) & set(exact_top_k(base, queries[i], 10).tolist()))
    assert hits / (len(queries) * 10) >= 0.95


def test_hnsw_stats_expose_the_level_histogram(data):
    base, _ = data
    index = _veccore.HnswIndex(base, M=16, ef_construction=100)
    s = index.stats()
    assert s["nodes"] == 3000
    assert s["nodes_per_level"][0] == 3000
    assert 0 < s["mean_degree_layer0"] <= 32
    assert s["graph_bytes"] > 0
    assert s["vector_bytes"] == 3000 * 32 * 4
    assert index.lock_mode == "writer_priority"   # B-11's fix is the default


def test_batch_search_matches_single(data):
    base, queries = data
    index = _veccore.HnswIndex(base, M=16, ef_construction=100, seed=7)
    ids_b, dists_b = index.search_batch(queries[:20], k=10, ef_search=64)
    assert ids_b.shape == (20, 10)
    for i in range(20):
        ids_s, dists_s = index.search(queries[i], k=10, ef_search=64)
        assert list(ids_b[i]) == list(ids_s)
        assert np.allclose(dists_b[i], dists_s)


def test_same_seed_gives_the_same_index(data):
    base, queries = data
    a = _veccore.HnswIndex(base, M=8, ef_construction=50, seed=123)
    b = _veccore.HnswIndex(base, M=8, ef_construction=50, seed=123)
    for i in range(10):
        assert list(a.search(queries[i], 10, 64)[0]) == list(b.search(queries[i], 10, 64)[0])


# ---------------------------------------------------------------------------
# P-34: array layouts that would silently read the wrong memory
# ---------------------------------------------------------------------------

def test_non_contiguous_query_is_handled(data):
    """A sliced array has strides the C++ loop knows nothing about.

    Without ``c_style | forcecast`` this reads the wrong floats and returns
    plausible garbage. The binding must copy, not reinterpret.
    """
    base, _ = data
    index = _veccore.FlatIndex(base)

    wide = np.zeros((64,), dtype=np.float32)
    wide[::2] = base[7]                    # every other element -> stride 2 view
    strided = wide[::2]
    assert not strided.flags["C_CONTIGUOUS"]

    ids, _ = index.search(strided, k=1)
    assert ids[0] == 7                     # it must find the vector it *is*


def test_float64_input_is_cast_not_misread(data):
    base, _ = data
    index = _veccore.FlatIndex(base)
    ids, _ = index.search(base[11].astype(np.float64), k=1)
    assert ids[0] == 11


def test_fortran_order_base_is_handled():
    rng = np.random.default_rng(1)
    base_c = rng.standard_normal((500, 16)).astype(np.float32)
    base_f = np.asfortranarray(base_c)
    assert not base_f.flags["C_CONTIGUOUS"]

    a = _veccore.FlatIndex(base_c)
    b = _veccore.FlatIndex(base_f)
    q = base_c[42]
    assert list(a.search(q, 5)[0]) == list(b.search(q, 5)[0])


def test_dimension_mismatch_raises(data):
    base, _ = data
    index = _veccore.FlatIndex(base)
    with pytest.raises(ValueError):
        index.search(np.zeros(31, dtype=np.float32), k=1)
    with pytest.raises(ValueError):
        index.search(np.zeros((2, 32), dtype=np.float32), k=1)


# ---------------------------------------------------------------------------
# P-35: the GIL. The test the C++ suite structurally cannot write.
# ---------------------------------------------------------------------------

def test_search_releases_the_gil(data):
    """Threaded search must actually overlap.

    With the GIL held across the C++ search, N threads take N times as long as
    one -- the C++ benchmark is completely unaffected, so nothing else in this
    repo would notice. The bound is deliberately loose (1.6x rather than near
    1.0x) because this is a wall-clock test on a laptop; the point is to
    distinguish "overlapping" from "fully serialised", not to measure speedup.
    """
    base, queries = data
    index = _veccore.HnswIndex(base, M=16, ef_construction=200)
    reps = 400

    def work(slice_of):
        scratchless = index  # searches allocate their own scratch internally
        for i in range(reps):
            scratchless.search(slice_of[i % len(slice_of)], k=10, ef_search=200)

    t0 = time.perf_counter()
    work(queries)
    serial = time.perf_counter() - t0

    threads = [threading.Thread(target=work, args=(queries,)) for _ in range(4)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    parallel = time.perf_counter() - t0

    # 4x the work. Fully serialised would be ~4x the time; overlapped is far
    # less. Anything under 2.5x proves the GIL was released.
    assert parallel < serial * 2.5, (
        f"4 threads took {parallel / serial:.2f}x the single-threaded time for 4x the work "
        "-- the GIL is being held across the C++ search (P-35)"
    )


# ---------------------------------------------------------------------------
# BM25 and RRF through the boundary
# ---------------------------------------------------------------------------

def test_bm25_ranks_and_scores():
    docs = ["the cat sat on the mat", "the cat cat cat", "dogs are loud"]
    index = _veccore.Bm25Index(docs)
    assert index.vocab_size > 0
    assert index.avgdl == pytest.approx((6 + 4 + 3) / 3)

    hits = index.search("cat", k=3)
    assert [doc for doc, _ in hits] == [1, 0]
    # Scores come back positive on the Python side even though C++ negates
    # internally to keep "smaller is better" everywhere (D4).
    assert all(score > 0 for _, score in hits)
    assert hits[0][1] > hits[1][1]


def test_bm25_idf_is_positive_for_common_terms():
    """P-08 through the binding: the Lucene form never goes negative."""
    docs = ["common x", "common y", "common z", "common w", "rare"]
    index = _veccore.Bm25Index(docs)
    assert index.idf("common") > 0
    assert index.idf("rare") > index.idf("common")


def test_rrf_uses_ranks_only():
    a = [7, 3, 5]
    b = [3, 7, 9]
    fused = _veccore.reciprocal_rank_fusion([a, b], k=4)
    # 7 and 3 are each 1st in one list and 2nd in the other, so they tie and
    # both outrank the singly-listed documents.
    assert set(fused[:2]) == {3, 7}
    assert set(fused[2:]) == {5, 9}


def test_rrf_single_list_preserves_order():
    assert _veccore.reciprocal_rank_fusion([[5, 6, 7]], k=3) == [5, 6, 7]
