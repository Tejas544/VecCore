#!/usr/bin/env python3
"""Build the SIFT10K inner-loop fixture and its exact ground truth.

CONTEXT.md D3: every unit test and correctness check runs on 10,000 vectors.
SIFT1M appears only in ``bench``.  A full 1M HNSW build takes minutes, and a
test suite that takes minutes stops being run -- which is EdgeRAG's D1 lesson,
already learned once on this laptop.

This script deliberately uses numpy rather than the C++ we are about to write.
The fixture's ground truth must not depend on code under test: if the brute
force implementation and the ground truth generator share a bug, Phase 1's gate
passes while being wrong, and every phase after it inherits the error.

Outputs, into ``$VECCORE_DATA/sift10k``:
    sift10k_base.fvecs        first 10,000 base vectors
    sift10k_query.fvecs       the first 1,000 held-out SIFT queries
    sift10k_groundtruth.ivecs exact top-100 neighbour ids, squared-L2 order
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

DIM = 128
N_BASE = 10_000
N_QUERY = 1_000
K = 100
CHUNK = 256


def read_xvecs(path: Path, dtype: np.dtype, expect_dim: int | None = None) -> np.ndarray:
    """Read a .fvecs/.ivecs file.

    P-01, stated in code because the comment is the bug guard: each record is a
    little-endian int32 dimension count followed by that many values.  The
    dimension is repeated for EVERY vector.  Reading this as a flat array
    produces numerically plausible garbage -- no crash, no NaN, just recall that
    looks like an algorithm bug.
    """
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise SystemExit(f"{path}: empty")

    dim = int(raw[0])
    if expect_dim is not None and dim != expect_dim:
        raise SystemExit(f"{path}: first record claims dim={dim}, expected {expect_dim}")

    stride = dim + 1
    if raw.size % stride != 0:
        raise SystemExit(
            f"{path}: size {raw.size} is not a multiple of (1 + {dim}). "
            "The file layout is not what the reader assumes -- stop here (P-01)."
        )

    records = raw.reshape(-1, stride)

    # Every record must repeat the same dimension.  This is the assertion that
    # actually catches a misread file, as opposed to the header check above.
    dims = records[:, 0]
    if not np.all(dims == dim):
        bad = int(np.argmax(dims != dim))
        raise SystemExit(f"{path}: record {bad} claims dim={int(dims[bad])}, expected {dim}")

    # ascontiguousarray before .view: a strided slice cannot always be
    # reinterpreted in place, and the numpy version that can do it differs
    # between the Windows and WSL environments.  The copy costs one pass and
    # removes a portability failure that would only show up on one machine.
    return np.ascontiguousarray(records[:, 1:]).view(dtype)


def write_xvecs(path: Path, data: np.ndarray, dtype: np.dtype) -> None:
    n, dim = data.shape
    out = np.empty((n, dim + 1), dtype=np.int32)
    out[:, 0] = dim
    out[:, 1:] = data.astype(dtype, copy=False).view(np.int32)
    out.tofile(path)


def exact_top_k(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact squared-L2 top-k.

    D4: squared distances, no sqrt -- sqrt is monotonic so it cannot change a
    ranking.  SIFT's own ground truth is defined this way, which is what makes
    the C++ side directly comparable to it.

    Computed as ||q||^2 - 2*q.b + ||b||^2 in float64.  That expansion is
    normally the *inaccurate* way to do this, because subtracting two large
    similar numbers loses precision -- but here it is bit-exact, and the reason
    is worth knowing:

        SIFT descriptors are integers in [0, 255] stored as float32.
        With d=128, every intermediate is bounded by 128 * 255^2 = 8,323,200,
        far below float64's exact-integer limit of 2^53.

    So every product and every sum is represented exactly, and the expansion
    gives literally the same answer as the naive difference-of-vectors -- while
    running through BLAS instead of materialising an (n_q x n_base x d) array,
    which at these sizes would be gigabytes.

    The integrality assumption is *checked*, not assumed, because it is the only
    thing holding the exactness argument up.
    """
    integral = (
        np.all(base == np.round(base))
        and np.all(queries == np.round(queries))
        and float(np.max(np.abs(base))) <= 255.0
        and float(np.max(np.abs(queries))) <= 255.0
    )
    if not integral:
        raise SystemExit(
            "base/query values are not integers in [0,255] as SIFT should be. "
            "The exactness argument above no longer holds -- do not trust this "
            "ground truth without revisiting the distance computation."
        )

    b2 = np.einsum("ij,ij->i", base, base)          # (n_base,)
    q2 = np.einsum("ij,ij->i", queries, queries)    # (n_query,)

    n_q = queries.shape[0]
    out = np.empty((n_q, k), dtype=np.int32)
    for start in range(0, n_q, CHUNK):
        stop = min(start + CHUNK, n_q)
        d2 = q2[start:stop, None] - 2.0 * (queries[start:stop] @ base.T) + b2[None, :]
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d2, idx, axis=1), axis=1)
        out[start:stop] = np.take_along_axis(idx, order, axis=1).astype(np.int32)
        print(f"  ground truth {stop}/{n_q}", end="\r", flush=True)
    print()
    return out


def main() -> int:
    data_dir = Path(os.environ.get("VECCORE_DATA", Path.home() / "veccore-data"))
    sift = data_dir / "sift"
    out_dir = data_dir / "sift10k"

    if not sift.is_dir():
        raise SystemExit(f"{sift} not found -- run scripts/fetch_sift.sh first")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {sift}/sift_base.fvecs")
    base_all = read_xvecs(sift / "sift_base.fvecs", np.float32, expect_dim=DIM)
    print(f"  {base_all.shape[0]} x {base_all.shape[1]}")

    print(f"reading {sift}/sift_query.fvecs")
    query_all = read_xvecs(sift / "sift_query.fvecs", np.float32, expect_dim=DIM)
    print(f"  {query_all.shape[0]} x {query_all.shape[1]}")

    base = np.ascontiguousarray(base_all[:N_BASE], dtype=np.float32)
    # P-19: the queries are the HELD-OUT set, never a slice of the base
    # vectors.  Query with base vectors and every query finds itself at distance
    # zero, recall@1 is trivially 1.0, and the number means nothing.
    query = np.ascontiguousarray(query_all[:N_QUERY], dtype=np.float32)

    print(f"computing exact top-{K} for {N_QUERY} queries over {N_BASE} base vectors")
    gt = exact_top_k(base.astype(np.float64), query.astype(np.float64), K)

    write_xvecs(out_dir / "sift10k_base.fvecs", base, np.float32)
    write_xvecs(out_dir / "sift10k_query.fvecs", query, np.float32)
    write_xvecs(out_dir / "sift10k_groundtruth.ivecs", gt, np.int32)

    # Round-trip the fixture through the same reader the tests will use.  A
    # writer bug here would poison every recall number in the project.
    back = read_xvecs(out_dir / "sift10k_base.fvecs", np.float32, expect_dim=DIM)
    assert np.array_equal(back, base), "fvecs round-trip failed"
    back_gt = read_xvecs(out_dir / "sift10k_groundtruth.ivecs", np.int32, expect_dim=K)
    assert np.array_equal(back_gt, gt), "ivecs round-trip failed"

    print(f"\nfixture written to {out_dir}")
    print(f"  base  {base.shape}  query {query.shape}  gt {gt.shape}")
    print("  round-trip verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
