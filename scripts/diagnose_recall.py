#!/usr/bin/env python3
"""Explain a recall shortfall: genuine miss, or a distance tie at rank k?

BUGS.md P-12 says exactly this, and it is why this script exists rather than an
afternoon of staring at the search loop:

    "Distance ties at rank k look like a recall bug. Multiple vectors at
     identical distance from the query; you return one, ground truth lists
     another. Recall 0.997 and hours of hunting. Guard: when investigating any
     recall shortfall, print the *distances* of your results and the truth's,
     not just the IDs. If the distances match, you are done -- it is a tie, not
     a bug."

The method matters more than this particular answer. We recompute the top-k in
numpy -- an implementation that shares no code with the C++ under test -- and
compare it to the published ground truth. That splits the question in two:

    numpy agrees with published    -> the C++ is wrong. A real bug.
    numpy disagrees the same way   -> the C++ is fine, and the disagreement is
                                      a property of the data (a tie), not of
                                      our code.

Usage:
    ~/veccore-venv/bin/python scripts/diagnose_recall.py [--queries N] [--k K]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

CHUNK = 20


def read_xvecs(path: Path, dtype, expect_dim: int | None = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    if expect_dim is not None and dim != expect_dim:
        raise SystemExit(f"{path}: dim {dim}, expected {expect_dim}")
    stride = dim + 1
    if raw.size % stride:
        raise SystemExit(f"{path}: size not a multiple of {stride} (P-01)")
    records = raw.reshape(-1, stride)
    if not np.all(records[:, 0] == dim):
        raise SystemExit(f"{path}: ragged dimensions (P-01)")
    return np.ascontiguousarray(records[:, 1:]).view(dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--data", default=os.environ.get("VECCORE_DATA", str(Path.home() / "veccore-data")))
    args = ap.parse_args()

    sift = Path(args.data) / "sift"
    k = args.k

    print(f"reading {sift}")
    base = read_xvecs(sift / "sift_base.fvecs", np.float32, 128).astype(np.float64)
    query = read_xvecs(sift / "sift_query.fvecs", np.float32, 128).astype(np.float64)[: args.queries]
    truth = read_xvecs(sift / "sift_groundtruth.ivecs", np.int32)[: args.queries]
    print(f"  base {base.shape}  query {query.shape}  truth {truth.shape}")

    # SIFT values are integers in [0,255] and d=128, so every intermediate is
    # bounded by 128*255^2 = 8,323,200 -- far below float64's exact-integer
    # limit of 2^53. This expansion is therefore bit-exact, not merely close.
    b2 = np.einsum("ij,ij->i", base, base)

    total_missing = 0
    ties = 0
    real = 0

    for start in range(0, query.shape[0], CHUNK):
        stop = min(start + CHUNK, query.shape[0])
        q = query[start:stop]
        d2 = np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ base.T) + b2[None, :]

        for row in range(stop - start):
            qi = start + row
            dist = d2[row]
            mine = np.argsort(dist, kind="stable")[:k]
            theirs = truth[qi][:k]

            missing = set(theirs.tolist()) - set(mine.tolist())
            if not missing:
                continue
            total_missing += len(missing)

            print(f"\nquery {qi}: {len(missing)} of the published top-{k} not in ours")
            kth = dist[mine[k - 1]]
            for doc in sorted(missing):
                d_theirs = dist[doc]
                # The whole question, in one comparison.
                verdict = "TIE with our k-th" if d_theirs == kth else "GENUINELY FARTHER"
                if d_theirs == kth:
                    ties += 1
                else:
                    real += 1
                print(f"  published id {doc:>7}  d2={d_theirs:.1f}")
                print(f"  our k-th  id {mine[k-1]:>7}  d2={kth:.1f}   -> {verdict}")
                if d_theirs != kth:
                    print(f"     difference: {d_theirs - kth:+.6f}")

    print("\n" + "=" * 70)
    print(f"queries checked        : {query.shape[0]}")
    print(f"published ids we missed: {total_missing}")
    print(f"  exact distance ties  : {ties}")
    print(f"  genuinely farther    : {real}")
    if total_missing and real == 0:
        print("\nVERDICT: every miss is an exact distance tie at rank k.")
        print("An independent numpy implementation makes the same 'mistake', so this is a")
        print("property of the dataset, not of the search. Not a bug (P-12).")
    elif real:
        print("\nVERDICT: at least one published neighbour is genuinely closer than our k-th.")
        print("That is a real bug in the search. Do not proceed to Phase 2.")
    else:
        print("\nVERDICT: exact match with the published ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
