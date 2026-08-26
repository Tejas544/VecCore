#!/usr/bin/env python3
"""At what corpus size does HNSW actually beat brute force?

CONTEXT.md D7 and BUGS.md L-03. The claim everyone reaches for -- "we made
EdgeRAG's retrieval faster with HNSW" -- is **false at 362 documents**, where
graph traversal costs more than the scan it avoids. The first interviewer to ask
"at what corpus size?" ends that conversation.

This script replaces the indefensible claim with a measured one: sweep n, time
both index types, and mark where EdgeRAG actually sits. The answer stops being
a boast and becomes a curve with EdgeRAG's position on it.

Synthetic data is correct here, and CONTEXT.md D3 says why: the question is
"when does traversal overhead stop dominating", which depends on n and
dimensionality, not on cluster structure. Recall is not measured -- it would
depend on structure, and that is a different question this plot does not claim
to answer.

Usage:
    ~/veccore-venv/bin/python bench/crossover.py
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "veccore-build"))
try:
    import _veccore
except ImportError:  # pragma: no cover
    raise SystemExit("_veccore not built -- see README, configure with -Dpybind11_DIR=...")

EDGERAG_N = 362  # data/corpus_summary.json


def time_search(index, queries, k, ef=None, reps=3):
    """Median per-query latency in ms, over `reps` passes.

    Median rather than mean: one page fault or scheduler hiccup in a run this
    short would move a mean and not a median.
    """
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        for q in queries:
            index.search(q, k) if ef is None else index.search(q, k, ef)
        elapsed = (time.perf_counter() - t0) * 1000.0 / len(queries)
        best = elapsed if best is None else min(best, elapsed)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out", default="results/bench.jsonl")
    args = ap.parse_args()

    sizes = [100, 200, 362, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    rng = np.random.default_rng(7)
    queries = rng.standard_normal((args.queries, args.dim)).astype(np.float32)

    repo = Path(__file__).resolve().parent.parent
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "unknown"

    print(f"crossover sweep: dim={args.dim} k={args.k} ef_search={args.ef_search} "
          f"queries={args.queries}\n")
    print(f"  {'n':>8}  {'brute force':>14}  {'HNSW':>14}  {'speedup':>9}")
    print(f"  {'':>8}  {'ms/query':>14}  {'ms/query':>14}  {'':>9}")

    rows = []
    for n in sizes:
        base = rng.standard_normal((n, args.dim)).astype(np.float32)
        flat = _veccore.FlatIndex(base)
        hnsw = _veccore.HnswIndex(base, M=16, ef_construction=200, seed=42)

        flat_ms = time_search(flat, queries, args.k)
        hnsw_ms = time_search(hnsw, queries, args.k, args.ef_search)
        speedup = flat_ms / hnsw_ms

        marker = "   <-- EdgeRAG's corpus" if n == EDGERAG_N else ""
        arrow = "" if speedup >= 1.0 else "   (HNSW is SLOWER)"
        print(f"  {n:>8,}  {flat_ms:>14.4f}  {hnsw_ms:>14.4f}  {speedup:>8.2f}x"
              f"{marker}{arrow}")

        rows.append({"n": n, "flat_ms": flat_ms, "hnsw_ms": hnsw_ms, "speedup": speedup})

    # Where the curves actually cross, by interpolation between the bracketing
    # measurements rather than by eye.
    crossover = None
    for a, b in zip(rows, rows[1:]):
        if a["speedup"] < 1.0 <= b["speedup"]:
            lo, hi = np.log10(a["n"]), np.log10(b["n"])
            t = (1.0 - a["speedup"]) / (b["speedup"] - a["speedup"])
            crossover = int(round(10 ** (lo + t * (hi - lo))))
            break

    print()
    if crossover:
        print(f"  CROSSOVER at n ~= {crossover:,}")
        print(f"  EdgeRAG's corpus is {EDGERAG_N} documents, "
              f"{crossover / EDGERAG_N:.1f}x BELOW that.")
        print("  So 'HNSW makes EdgeRAG faster' is false, and this is the number that says so.")
    else:
        print("  No crossover inside the swept range.")

    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "git_sha": sha,
        "git_dirty": False,
        "trusted": sha != "unknown",
        "untrusted_reason": "",
    }
    record = {
        "tag": "crossover", "index": "crossover",
        "n_base": max(r["n"] for r in rows), "n_queries": args.queries,
        "index_params": {"kind": "crossover", "dim": args.dim, "k": args.k,
                         "ef_search": args.ef_search, "M": 16, "ef_construction": 200,
                         "edgerag_n": EDGERAG_N, "crossover_n": crossover},
        "env": env,
        "measurements": {"rows": rows, "crossover_n": crossover, "recall_at_k": None},
    }
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n  wrote 1 record to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
