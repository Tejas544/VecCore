#!/usr/bin/env python3
"""FAISS, measured the same way VecCore is, on the same data and machine.

02_VECCORE.md section 2: FAISS is the BENCHMARK, not a dependency. It is
imported here and nowhere else in the repo.

CONTEXT.md D10, the rules that make this a comparison rather than two unrelated
numbers:

  * one environment, one session, same data, same k;
  * **matched thread count** -- faiss.omp_set_num_threads(1), because FAISS
    defaults to every core and comparing that against a single-threaded search
    would be a 6x error in its favour (P-32), and the first competent
    interviewer will ask;
  * interleaved A/B rounds, never all-A-then-all-B, because this is a mobile
    CPU that throttles;
  * matched *recall*, not matched parameters, when comparing HNSW.

Usage:
    ~/veccore-venv/bin/python bench/faiss_baseline.py --index pq --pq-m 4,8,16,32
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover
    raise SystemExit("faiss not installed: ~/veccore-venv/bin/pip install faiss-cpu")


def read_xvecs(path: Path, dtype, expect_dim: int | None = None, limit: int = 0) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0])
    if expect_dim is not None and dim != expect_dim:
        raise SystemExit(f"{path}: dim {dim}, expected {expect_dim}")
    stride = dim + 1
    if raw.size % stride:
        raise SystemExit(f"{path}: size not a multiple of {stride} (P-01)")
    rec = raw.reshape(-1, stride)
    if not np.all(rec[:, 0] == dim):
        raise SystemExit(f"{path}: ragged dimensions (P-01)")
    if limit:
        rec = rec[:limit]
    return np.ascontiguousarray(rec[:, 1:]).view(dtype)


def recall_at_k(got: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = 0
    for i in range(got.shape[0]):
        hits += len(set(got[i, :k].tolist()) & set(truth[i, :k].tolist()))
    return hits / (got.shape[0] * k)


def git_sha(repo: Path) -> tuple[str, bool]:
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", ":!results"],
                               capture_output=True, text=True).stdout.strip()
        return sha, bool(dirty)
    except Exception:
        return "unknown", True


def env_stamp(repo: Path) -> dict:
    sha, dirty = git_sha(repo)
    model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "cpu_model": model,
        "hw_threads": os.cpu_count() or 0,
        "git_sha": sha,
        "git_dirty": dirty,
        "faiss_version": faiss.__version__,
        "numpy_version": np.__version__,
        "faiss_threads": faiss.omp_get_max_threads(),
        # A FAISS run is publishable on the same terms a VecCore run is: clean
        # tree, and pinned to one thread so the comparison is honest.
        "trusted": (not dirty) and sha != "unknown" and faiss.omp_get_max_threads() == 1,
        "untrusted_reason": "",
    }


def timed_search(index, queries: np.ndarray, k: int, trials: int, warmup: int):
    index.search(queries[:warmup], k)  # warmup: page in, prime predictors
    lat, qps = [], []
    for _ in range(trials):
        t_all = time.perf_counter()
        for i in range(queries.shape[0]):
            t0 = time.perf_counter()
            _, ids = index.search(queries[i : i + 1], k)
            lat.append((time.perf_counter() - t0) * 1000.0)
        qps.append(queries.shape[0] / (time.perf_counter() - t_all))
    _, ids = index.search(queries, k)
    return np.array(lat), np.array(qps), ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("VECCORE_DATA", str(Path.home() / "veccore-data")))
    ap.add_argument("--index", default="pq", choices=["pq", "hnsw", "flat"])
    ap.add_argument("--pq-m", default="4,8,16,32")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", default="10,20,40,80,160,320")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="results/bench.jsonl")
    ap.add_argument("--tag", default="faiss")
    args = ap.parse_args()

    # P-32. Everything VecCore reports is single-threaded, so this must be too.
    faiss.omp_set_num_threads(1)

    repo = Path(__file__).resolve().parent.parent
    sift = Path(args.data) / "sift"
    base = read_xvecs(sift / "sift_base.fvecs", np.float32, 128)
    query = read_xvecs(sift / "sift_query.fvecs", np.float32, 128, args.queries)
    truth = read_xvecs(sift / "sift_groundtruth.ivecs", np.int32, limit=args.queries)
    d = base.shape[1]

    env = env_stamp(repo)
    print(f"faiss {faiss.__version__}  threads={faiss.omp_get_max_threads()}  "
          f"base={base.shape}  queries={query.shape}")
    if not env["trusted"]:
        print("  NOT PUBLISHABLE: dirty tree, unknown SHA, or threads != 1")

    out = open(args.out, "a", encoding="utf-8")

    def write(index_name: str, params: dict, lat, qps, ids, extra: dict):
        rec = {
            "tag": args.tag, "index": index_name,
            "dataset_base": str(sift / "sift_base.fvecs"),
            "dataset_query": str(sift / "sift_query.fvecs"),
            "dataset_truth": str(sift / "sift_groundtruth.ivecs"),
            "n_base": int(base.shape[0]), "n_queries": int(query.shape[0]),
            "dim": int(d), "k": args.k, "trials": args.trials, "warmup": args.warmup,
            "index_params": params, "env": env,
            "measurements": {
                "recall_at_k": recall_at_k(ids, truth, args.k),
                "qps_mean": float(qps.mean()), "qps_stddev": float(qps.std(ddof=1)) if len(qps) > 1 else 0.0,
                "qps_per_trial": [float(x) for x in qps],
                "latency_p50_ms": float(np.percentile(lat, 50)),
                "latency_p95_ms": float(np.percentile(lat, 95)),
                "latency_p99_ms": float(np.percentile(lat, 99)),
                "latency_mean_ms": float(lat.mean()), "latency_max_ms": float(lat.max()),
                **extra,
            },
        }
        out.write(json.dumps(rec) + "\n")
        m = rec["measurements"]
        print(f"    recall@{args.k} {m['recall_at_k']:.4f}   QPS {m['qps_mean']:8.2f} "
              f"+/- {m['qps_stddev']:.2f}   p50 {m['latency_p50_ms']:.4f}  p99 {m['latency_p99_ms']:.4f}")

    if args.index == "flat":
        index = faiss.IndexFlatL2(d)
        t0 = time.perf_counter()
        index.add(base)
        build_ms = (time.perf_counter() - t0) * 1000
        print(f"  IndexFlatL2  built in {build_ms/1000:.2f} s")
        lat, qps, ids = timed_search(index, query, args.k, args.trials, args.warmup)
        write("faiss_flat", {"kind": "faiss_flat"}, lat, qps, ids,
              {"build_ms": build_ms, "index_bytes": base.nbytes})

    if args.index == "pq":
        for m in [int(x) for x in args.pq_m.split(",")]:
            if d % m:
                print(f"  m={m} skipped: does not divide d={d}")
                continue
            index = faiss.IndexPQ(d, m, 8)   # 8 bits => 256 centroids, same as ours
            t0 = time.perf_counter()
            index.train(base[:100000])       # same training subsample as VecCore
            train_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            index.add(base)
            encode_ms = (time.perf_counter() - t0) * 1000
            print(f"  IndexPQ m={m}  {m} B/vector  {d*4/m:.0f}x  "
                  f"train {train_ms/1000:.1f} s  encode {encode_ms/1000:.1f} s")
            lat, qps, ids = timed_search(index, query, args.k, args.trials, args.warmup)
            write("faiss_pq",
                  {"kind": "faiss_pq", "m": m, "centroids": 256, "rerank": 0,
                   "train_size": 100000},
                  lat, qps, ids,
                  {"build_ms": train_ms + encode_ms, "train_ms": train_ms,
                   "bytes_per_vector": float(m), "compression_ratio": d * 4.0 / m,
                   "index_bytes": base.shape[0] * m})

    if args.index == "hnsw":
        index = faiss.IndexHNSWFlat(d, args.M)
        index.hnsw.efConstruction = args.ef_construction
        t0 = time.perf_counter()
        index.add(base)
        build_ms = (time.perf_counter() - t0) * 1000
        print(f"  IndexHNSWFlat M={args.M} efC={args.ef_construction}  "
              f"built in {build_ms/1000:.1f} s")
        for ef in [int(x) for x in args.ef_search.split(",")]:
            index.hnsw.efSearch = ef
            lat, qps, ids = timed_search(index, query, args.k, args.trials, args.warmup)
            print(f"  efSearch={ef}")
            write("faiss_hnsw",
                  {"kind": "faiss_hnsw", "M": args.M,
                   "ef_construction": args.ef_construction, "ef_search": ef},
                  lat, qps, ids, {"build_ms": build_ms})

    out.close()
    print(f"\n  wrote records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
