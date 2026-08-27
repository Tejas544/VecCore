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

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# Where the extension might be, in the order it is worth looking.
#
#   1. $VECCORE_BUILD, for anyone building somewhere else entirely.
#   2. ~/veccore-build -- CONTEXT.md D2, the out-of-tree build every benchmark
#      in this repo uses, on ext4 rather than across the /mnt/d 9p bridge.
#   3. ./build next to the source, which is what CI does and what a reader
#      following the README's default would get.
#
# The third entry is not decoration. With only the second, this whole module
# skipped on CI -- and a skipped test suite is indistinguishable from a passing
# one in a green check mark, which is the specific way a badge comes to mean
# nothing.
_CANDIDATES = [
    Path(os.environ["VECCORE_BUILD"]) if os.environ.get("VECCORE_BUILD") else None,
    Path.home() / "veccore-build",
    Path(__file__).resolve().parent.parent / "build",
]
for _d in _CANDIDATES:
    if _d is not None and _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

try:
    import _veccore
except ImportError:  # pragma: no cover
    searched = ", ".join(str(d) for d in _CANDIDATES if d is not None)
    # CI sets this so that "not built" is a failure rather than a quiet skip.
    if os.environ.get("VECCORE_REQUIRE_EXTENSION"):
        raise
    pytest.skip(
        f"_veccore not built; configure with -Dpybind11_DIR=... (searched: {searched})",
        allow_module_level=True,
    )


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


# ---------------------------------------------------------------------------
# Incremental insert (PLAN.md 5.4), reachable from Python.
#
# Phase 5 built concurrent insert, found the writer-starvation cliff (B-11) and
# fixed it -- and until these bindings existed none of it could be called from
# the language the index is actually consumed from. These tests are the reason
# `add` is not simply "append then insert": the append reallocates the vector
# store, and doing that outside the exclusive section is a use-after-free that
# returns plausible neighbours rather than crashing.
# ---------------------------------------------------------------------------


def test_add_returns_sequential_ids_and_grows_the_index(data):
    base, _ = data
    index = _veccore.HnswIndex(base[:1000], M=16, ef_construction=100)
    assert index.size == 1000

    for i in range(1000, 1050):
        assert index.add(base[i]) == i
    assert index.size == 1050
    assert index.check_invariants() == ""


def test_an_added_vector_is_its_own_nearest_neighbour(data):
    """The check that separates 'grew the arrays' from 'linked the node'.

    An index that resized its buffers but never connected the new node still
    returns k results for every query -- it just never returns the new one.
    """
    base, _ = data
    index = _veccore.HnswIndex(base[:1000], M=16, ef_construction=100)
    for i in range(1000, 1100):
        new_id = index.add(base[i])
        ids, dists = index.search(base[i], k=1, ef_search=64)
        assert ids[0] == new_id
        assert dists[0] == pytest.approx(0.0, abs=1e-5)


def test_recall_after_growth_matches_recall_after_a_full_build(data):
    """Growing to n must not produce a worse graph than building at n.

    build() is a loop of inserts, so if these diverge, insert() is carrying
    state across calls that it should not.
    """
    base, queries = data
    grown = _veccore.HnswIndex(base[:1500], M=16, ef_construction=100, seed=7)
    grown.add_batch(base[1500:])

    built = _veccore.HnswIndex(base, M=16, ef_construction=100, seed=7)

    def recall(index):
        total = 0.0
        for q in queries:
            truth = set(exact_top_k(base, q, 10).tolist())
            got = set(index.search(q, k=10, ef_search=128)[0].tolist())
            total += len(truth & got) / 10
        return total / len(queries)

    grown_recall, built_recall = recall(grown), recall(built)
    assert grown_recall == pytest.approx(built_recall, abs=0.02)
    assert grown_recall > 0.90


def test_add_batch_matches_repeated_add(data):
    base, queries = data
    one_at_a_time = _veccore.HnswIndex(base[:1000], M=16, ef_construction=100, seed=3)
    for i in range(1000, 1200):
        one_at_a_time.add(base[i])

    batched = _veccore.HnswIndex(base[:1000], M=16, ef_construction=100, seed=3)
    ids = batched.add_batch(base[1000:1200])
    assert list(ids) == list(range(1000, 1200))
    assert batched.size == one_at_a_time.size

    for q in queries[:10]:
        a = batched.search(q, k=10, ef_search=64)[0]
        b = one_at_a_time.search(q, k=10, ef_search=64)[0]
        assert list(a) == list(b)


def test_add_rejects_a_wrong_dimension_vector(data):
    base, _ = data
    index = _veccore.HnswIndex(base[:500], M=8, ef_construction=50)
    with pytest.raises(ValueError):
        index.add(np.zeros(31, dtype=np.float32))
    with pytest.raises(ValueError):
        index.add_batch(np.zeros((4, 31), dtype=np.float32))


def test_add_survives_a_non_contiguous_source_array(data):
    """P-34 on the write path.

    Every existing P-34 test covers queries. A transposed or strided array
    reaching `add` would store a *wrong vector* -- permanently, and with no
    symptom until someone searches for it.
    """
    base, _ = data
    index = _veccore.HnswIndex(base[:500], M=8, ef_construction=50)
    strided = np.asfortranarray(base[500:520])[::2]     # non-contiguous on purpose
    ids = index.add_batch(strided)
    for offset, new_id in enumerate(ids):
        ids_back, dists = index.search(strided[offset], k=1, ef_search=64)
        assert ids_back[0] == new_id
        assert dists[0] == pytest.approx(0.0, abs=1e-5)


def test_concurrent_search_and_add_from_python_threads(data):
    """The Phase 5 result, finally exercised from the language that consumes it.

    Both `search` and `add` release the GIL, so these threads genuinely overlap
    and the writer-priority turnstile (B-11) is what keeps the adds from being
    starved by the readers. A test that never actually ran its readers would
    prove nothing, so it asserts that too.
    """
    base, queries = data
    index = _veccore.HnswIndex(base[:1500], M=16, ef_construction=100)

    stop = threading.Event()
    errors: list[BaseException] = []
    searches = [0]

    def reader():
        try:
            n = 0
            while not stop.is_set():
                ids, dists = index.search(queries[n % len(queries)], k=10, ef_search=32)
                assert len(ids) == 10
                assert np.all(np.isfinite(dists))
                n += 1
            searches[0] += n
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()

    added = [index.add(base[i]) for i in range(1500, 1700)]

    stop.set()
    for t in threads:
        t.join()

    assert not errors, errors[0]
    assert searches[0] > 0, "readers never ran; the test proved nothing"
    assert added == list(range(1500, 1700))
    assert index.size == 1700
    assert index.check_invariants() == ""


# ---------------------------------------------------------------------------
# Product Quantization through the binding.
# ---------------------------------------------------------------------------


def test_pq_compression_is_reported_against_the_raw_baseline(data):
    base, _ = data
    pq = _veccore.PqIndex(base, m=8, train_size=1000, seed=42)
    assert pq.m == 8
    assert pq.bytes_per_vector == 8
    # 32 dims x 4 bytes = 128 B raw, 8 B coded.
    assert pq.compression_ratio == pytest.approx(16.0)
    assert pq.code_bytes == 8 * len(base)
    # The three footprints stay separate on purpose (D12): reranking needs the
    # full vectors resident, so they are never summed into one flattering number.
    assert pq.vector_bytes == base.nbytes
    assert pq.codebook_bytes > 0


def test_pq_rerank_beats_plain_adc(data):
    """The whole reason search_rerank exists, asserted rather than assumed."""
    base, queries = data
    pq = _veccore.PqIndex(base, m=8, train_size=2000, seed=42)

    adc = rerank = 0.0
    for q in queries:
        truth = set(exact_top_k(base, q, 10).tolist())
        adc += len(truth & set(pq.search(q, k=10)[0].tolist())) / 10
        rerank += len(truth & set(pq.search_rerank(q, k=10, candidates=200)[0].tolist())) / 10
    adc /= len(queries)
    rerank /= len(queries)

    assert rerank > adc
    assert rerank > 0.90


def test_pq_reconstruction_error_falls_as_m_rises(data):
    """P-21's neighbourhood: the check that the subspace slicing is right.

    More subspaces means finer quantization, so reconstruction error must fall
    monotonically. If it does not, the vector is being sliced wrongly and every
    recall number downstream is measuring the wrong thing.
    """
    base, _ = data
    errs = [
        _veccore.PqIndex(base, m=m, train_size=1000, seed=42, n_init=1).reconstruction_mse(500)
        for m in (2, 4, 8, 16)
    ]
    assert errs == sorted(errs, reverse=True), errs


def test_pq_rejects_an_m_that_does_not_divide_the_dimension(data):
    """P-21. Integer division would silently drop dimensions."""
    base, _ = data
    with pytest.raises(ValueError):
        _veccore.PqIndex(base, m=7, train_size=500)


def test_pq_releases_the_gil_during_search():
    """P-35 for the PQ path, and a lesson about what the first two drafts measured.

    Draft 1 used 20 searches over the 3,000-vector fixture and failed at 5.85x.
    Draft 2 raised it to 300 searches and failed *worse*, at 4.25x -- which looks
    exactly like a held GIL and is not one. Measured across index sizes:

        n=  3,000    22.6 us/search   4-thread ratio 4.90x
        n= 50,000   291.9 us/search   4-thread ratio 1.33x
        n=200,000  1205.7 us/search   4-thread ratio 1.27x

    The GIL was being released the whole time. At n=3,000 an ADC scan is ~20 us
    while the per-call Python work around it -- forcecast on the query, building
    two numpy arrays for the result -- is comparable and runs *with* the GIL
    held. So the fixed overhead dominated the thing under test, and four threads
    serialised on the part that was never released.

    That is B-05 in a different phase: a benchmark that did not reproduce the
    effect it was built to demonstrate. The fix is to measure where the C++ work
    is actually the work, so this test builds its own larger index rather than
    reusing the module fixture.
    """
    rng = np.random.default_rng(11)
    base = rng.standard_normal((50000, 32)).astype(np.float32)
    queries = rng.standard_normal((50, 32)).astype(np.float32)
    pq = _veccore.PqIndex(base, m=8, train_size=2000, seed=42, n_init=1, kmeans_iters=10)
    reps = 100

    def work():
        for i in range(reps):
            pq.search(queries[i % len(queries)], k=10)

    t0 = time.perf_counter()
    work()
    serial = time.perf_counter() - t0

    threads = [threading.Thread(target=work) for _ in range(4)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    parallel = time.perf_counter() - t0

    assert parallel < serial * 2.5, (
        f"4 threads took {parallel / serial:.2f}x the single-threaded time for 4x the work "
        "-- the GIL is being held across the PQ scan (P-35)"
    )


# ---------------------------------------------------------------------------
# Persistence.
#
# The argument for this existing at all is arithmetic: an HNSW over SIFT1M takes
# 1,139 s to build and PQ codebooks take 86.5 s to train, so an index that only
# lives inside the process that built it is one you pay for on every restart.
# ---------------------------------------------------------------------------


def test_saved_index_reloads_with_identical_answers(data, tmp_path):
    base, queries = data
    original = _veccore.HnswIndex(base, M=16, ef_construction=100, seed=13)
    path = str(tmp_path / "index.vci")
    original.save(path)

    loaded = _veccore.HnswIndex.load(path)
    assert loaded.size == original.size
    assert loaded.dim == original.dim
    assert loaded.check_invariants() == ""

    # Identical answers in identical order, not "recall is close". A round trip
    # that reorders results is still a round trip that is wrong.
    for q in queries:
        a_ids, a_d = original.search(q, k=10, ef_search=64)
        b_ids, b_d = loaded.search(q, k=10, ef_search=64)
        assert list(a_ids) == list(b_ids)
        assert np.allclose(a_d, b_d)


def test_saved_index_carries_its_vectors(data, tmp_path):
    """The file is self-contained, so a loaded index needs nothing else.

    If the vectors were dropped, every id in the graph would still be in range
    and every search would still return k results -- against whatever happened
    to be in memory. Searching for a known vector and expecting distance zero is
    what distinguishes those two worlds.
    """
    base, _ = data
    path = str(tmp_path / "selfcontained.vci")
    _veccore.HnswIndex(base, M=16, ef_construction=100).save(path)

    loaded = _veccore.HnswIndex.load(path)
    for i in (0, 1, 500, len(base) - 1):
        ids, dists = loaded.search(base[i], k=1, ef_search=64)
        assert ids[0] == i
        assert dists[0] == pytest.approx(0.0, abs=1e-5)


def test_a_loaded_index_can_still_grow(data, tmp_path):
    """Persistence and incremental insert have to compose, or neither is useful.

    The whole point of saving an index that supports `add` is picking up where
    the last process stopped instead of rebuilding.
    """
    base, _ = data
    path = str(tmp_path / "grow.vci")
    _veccore.HnswIndex(base[:1000], M=16, ef_construction=100).save(path)

    loaded = _veccore.HnswIndex.load(path)
    assert loaded.size == 1000

    ids = loaded.add_batch(base[1000:1200])
    assert list(ids) == list(range(1000, 1200))
    assert loaded.size == 1200
    assert loaded.check_invariants() == ""

    back, dists = loaded.search(base[1150], k=1, ef_search=64)
    assert back[0] == 1150
    assert dists[0] == pytest.approx(0.0, abs=1e-5)


def test_save_load_preserves_the_lock_mode_choice(data, tmp_path):
    base, _ = data
    path = str(tmp_path / "lock.vci")
    _veccore.HnswIndex(base[:500], M=8, ef_construction=50).save(path)

    assert _veccore.HnswIndex.load(path).lock_mode == "writer_priority"
    assert _veccore.HnswIndex.load(path, lock_mode="shared_mutex").lock_mode == "shared_mutex"


def test_a_corrupt_file_raises_instead_of_loading(data, tmp_path):
    """The four rejection paths, from Python.

    Each is a way a plain binary dump corrupts data *without* erroring, which on
    this project means quietly wrong recall. C++ covers them individually in
    tests/test_serialize.cpp; this checks the exception survives the binding
    rather than becoming a segfault or a silently empty index.
    """
    base, _ = data
    good = tmp_path / "good.vci"
    _veccore.HnswIndex(base[:500], M=8, ef_construction=50).save(str(good))
    payload = good.read_bytes()

    cases = {
        "not an index": b"\x7f" * 4096,
        "wrong version": bytes([*payload[:8], payload[8] + 1, *payload[9:]]),
        "flipped byte": bytes([*payload[:-40], payload[-40] ^ 0x01, *payload[-39:]]),
        "truncated": payload[: len(payload) * 3 // 4],
        "trailing bytes": payload + b"\x00",
    }
    for name, blob in cases.items():
        bad = tmp_path / f"bad_{name.replace(' ', '_')}.vci"
        bad.write_bytes(blob)
        with pytest.raises(RuntimeError):
            _veccore.HnswIndex.load(str(bad))

    # The uncorrupted original must still load, or the test proves nothing about
    # the corruption and everything about the loader being broken.
    assert _veccore.HnswIndex.load(str(good)).size == 500


def test_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError):
        _veccore.HnswIndex.load(str(tmp_path / "does_not_exist.vci"))


def test_pq_save_reloads_with_identical_scores(data, tmp_path):
    base, queries = data
    original = _veccore.PqIndex(base, m=8, train_size=1000, seed=42, n_init=1, kmeans_iters=10)
    path = str(tmp_path / "pq.vci")
    original.save(path)

    loaded = _veccore.PqIndex.load(path)
    assert loaded.m == original.m
    assert loaded.size == original.size
    assert loaded.dim == original.dim
    assert loaded.code_bytes == original.code_bytes
    assert loaded.codebook_bytes == original.codebook_bytes

    for q in queries:
        a_ids, a_d = original.search(q, k=10)
        b_ids, b_d = loaded.search(q, k=10)
        assert list(a_ids) == list(b_ids)
        assert np.allclose(a_d, b_d)


def test_pq_file_is_small_because_it_holds_codes_not_vectors(data, tmp_path):
    """The asymmetry with HnswIndex.save, made concrete.

    HNSW saves its vectors because a graph without them indexes nothing. PQ does
    not, because codes *are* the compressed form and writing the vectors beside
    them would inflate a small artifact by 16x to carry data only reranking uses.
    """
    base, _ = data
    path = tmp_path / "pqsize.vci"
    pq = _veccore.PqIndex(base, m=8, train_size=1000, seed=42, n_init=1, kmeans_iters=10)
    pq.save(str(path))

    assert path.stat().st_size < base.nbytes
    assert path.stat().st_size > pq.code_bytes


def test_pq_loaded_without_vectors_can_search_but_not_rerank(data, tmp_path):
    base, queries = data
    path = str(tmp_path / "pqnovec.vci")
    _veccore.PqIndex(base, m=8, train_size=1000, seed=42, n_init=1, kmeans_iters=10).save(path)

    bare = _veccore.PqIndex.load(path)
    assert not bare.has_vectors
    assert bare.vector_bytes == 0
    ids, _ = bare.search(queries[0], k=10)
    assert len(ids) == 10

    # A silently ADC-only "rerank" would be the wrong answer with a right-looking
    # shape, so the method refuses.
    with pytest.raises(ValueError, match="needs the full vectors"):
        bare.search_rerank(queries[0], k=10, candidates=100)

    with_vecs = _veccore.PqIndex.load(path, vectors=base)
    assert with_vecs.has_vectors
    reranked, _ = with_vecs.search_rerank(queries[0], k=10, candidates=200)
    assert len(reranked) == 10


def test_pq_load_rejects_vectors_that_are_not_the_corpus_it_encoded(data, tmp_path):
    """Codes index vectors *by position*, so a different corpus scores the wrong
    documents while returning a perfectly well-formed result set."""
    base, _ = data
    path = str(tmp_path / "pqmismatch.vci")
    _veccore.PqIndex(base, m=8, train_size=1000, seed=42, n_init=1, kmeans_iters=10).save(path)

    with pytest.raises(ValueError, match="do not describe the same corpus"):
        _veccore.PqIndex.load(path, vectors=base[:100])

    with pytest.raises(ValueError):
        _veccore.PqIndex.load(path, vectors=np.zeros((len(base), 16), dtype=np.float32))


def test_loading_a_pq_file_as_hnsw_is_refused(data, tmp_path):
    base, _ = data
    path = str(tmp_path / "kind.vci")
    _veccore.PqIndex(base[:500], m=8, train_size=500, seed=1, n_init=1, kmeans_iters=5).save(path)
    with pytest.raises(RuntimeError):
        _veccore.HnswIndex.load(path)


def test_saving_is_safe_while_other_threads_search(data, tmp_path):
    """save() takes the shared lock, so it cannot capture a half-written graph.

    Without the lock an insert landing mid-write produces a file that is
    internally inconsistent and *still passes its own checksum*, because the
    checksum is computed over whatever was written -- then loads cleanly and
    returns wrong neighbours.
    """
    base, queries = data
    index = _veccore.HnswIndex(base[:1500], M=16, ef_construction=100)

    stop = threading.Event()
    errors: list[BaseException] = []

    def reader():
        try:
            n = 0
            while not stop.is_set():
                ids, _ = index.search(queries[n % len(queries)], k=10, ef_search=32)
                assert len(ids) == 10
                n += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()

    paths = []
    for i in range(3):
        p = str(tmp_path / f"concurrent_{i}.vci")
        index.save(p)
        paths.append(p)

    stop.set()
    for t in threads:
        t.join()

    assert not errors, errors[0]
    for p in paths:
        reloaded = _veccore.HnswIndex.load(p)
        assert reloaded.size == 1500
        assert reloaded.check_invariants() == ""
