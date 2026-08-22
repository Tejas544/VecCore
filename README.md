# VecCore

**An HNSW vector index with Product-Quantization compression and hybrid dense+sparse retrieval,
written from scratch in C++17 and benchmarked against FAISS.**

> **Status: Phase 2 complete — HNSW verified against ground truth on SIFT1M.**
> Next: Product Quantization (Phase 3).
> This README is deliberately empty of numbers. Every claim below the line will arrive with a JSON
> record in [`results/`](results/) and the git SHA that produced it — see `CONTEXT.md` D10. If you
> are reading this and a number has no record behind it, that is a bug in the README.

New to vector search? **[`WHAT_IS_THIS.md`](WHAT_IS_THIS.md)** explains the whole problem from zero
background — what a vector is, why brute force fails, and what HNSW and PQ actually do.

| Document | What it holds |
|---|---|
| [`WHAT_IS_THIS.md`](WHAT_IS_THIS.md) | The no-background explanation, and the honest answer to "isn't this solved?" |
| [`PLAN.md`](PLAN.md) | Phase-by-phase build plan, schedule, cut order, gates |
| [`CONTEXT.md`](CONTEXT.md) | Every design decision with the alternative that was rejected |
| [`BUGS.md`](BUGS.md) | Bug log, defused landmines, and ~35 pre-registered failure modes |
| [`02_VECCORE.md`](02_VECCORE.md) | The original specification |

---

## Build

Linux only, and deliberately so — `CONTEXT.md` D2 records why (sanitizers, TSan, valgrind, and a
clean pybind11 story; `BUGS.md` L-01 is the finding that forced it).

```bash
sudo apt install -y build-essential cmake ninja-build valgrind python3-venv python3-pip pkg-config
```

Release — the only configuration any published number may come from:

```bash
cmake -S . -B ~/veccore-build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build ~/veccore-build
```

Debug, with AddressSanitizer and UBSan on by default:

```bash
cmake -S . -B ~/veccore-build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug && cmake --build ~/veccore-build-debug
```

**The Phase 0 gate.** This must abort with a heap-buffer-overflow report. If it prints
`probe survived`, the sanitizers are not active and the plan's main defence against memory bugs is
imaginary:

```bash
~/veccore-build-debug/bin/asan_probe
```

Python side — Ubuntu 24.04 enforces PEP 668, so this needs a venv rather than a bare
`pip install`:

```bash
python3 -m venv ~/veccore-venv && ~/veccore-venv/bin/pip install numpy matplotlib faiss-cpu pybind11
```

FAISS is a **benchmark, not a dependency** (`02_VECCORE.md` §2). It is imported only by
`bench/faiss_baseline.py`, never by anything in `include/` or `src/`.

## Data

```bash
bash scripts/fetch_sift.sh && ~/veccore-venv/bin/python scripts/make_fixture.py
```

SIFT1M (1M × 128-dim, published ground truth) for benchmarks; a 10,000-vector fixture with exact
ground truth for tests. `CONTEXT.md` D3 explains why those are two different things.

## Test and measure

```bash
ctest --test-dir ~/veccore-build --output-on-failure
```

```bash
~/veccore-build/bin/bench --tag phase0_rails --data-dir ~/veccore-data
```

`bench` refuses to write a record from a Debug, sanitized, or dirty-tree build unless you pass
`--allow-untrusted`, which stamps `trusted: false` and the reason into the record. An ASan build
runs several times slower than Release, and a latency copied out of one is indistinguishable from a
real regression once it reaches a plot.

---

## Architecture

*Diagram arrives at Phase 6. Placeholder rather than a lie: nothing but the rails exists yet.*

## Benchmarks

i5-11400H, **single thread**, Release `-O3 -march=native`, WSL2 Ubuntu 24.04, GCC 13.3.
Every row traces to a record in [`results/bench.jsonl`](results/bench.jsonl) carrying the git SHA,
CPU, compiler, flags, RNG seed and parameters it was produced under. Regenerate with
`bench/plots.py`; nothing here is typed by hand.

![recall vs QPS](docs/plots/recall_qps.png)

**SIFT1M — 1,000,000 × 128, published ground truth, k=10**

| index | params | recall@10 | QPS | p50 | p99 | vs brute force |
|---|---|---|---|---|---|---|
| brute force | exact | 0.9995 † | 12.9 ± 0.7 | 76.0 ms | 103.6 ms | 1× |
| HNSW | ef=40 | 0.9290 | 3,288 ± 138 | 0.30 ms | 0.55 ms | 254× |
| **HNSW** | **ef=80** | **0.9755** | **1,934 ± 15** | **0.53 ms** | **0.83 ms** | **150×** |
| HNSW | ef=160 | 0.9935 | 1,079 ± 10 | 0.95 ms | 1.51 ms | 83× |
| HNSW | ef=320 | 0.9987 | 595 ± 15 | 1.71 ms | 2.79 ms | 46× |

† Brute force is exact by construction. The 0.9995 is **one distance tie in 2000**, not an error:
query 93's published neighbour 274922 and our 196106 sit at d² = 42192.0 exactly, and an independent
numpy implementation makes the same choice. `scripts/diagnose_recall.py` proves it rather than
asserting it — that script is the answer to "why isn't it 1.0?".

**Build, memory, and the graph itself** (M=16, ef_construction=200, seed=42):

| | |
|---|---|
| build time | **1,139 s** single-threaded — see limitations, this is the weakest number here |
| level histogram | 1,000,000 / 62,312 / 3,911 / 250 / 21 / 1 |
| successive ratios | 0.0623, 0.0628, 0.0639 — against the predicted 1/M = 0.0625 |
| mean degree, layer 0 | 25.7 (cap 32) |
| graph | 140.7 MiB on top of 488.3 MiB of vectors — **28.8% overhead** |

The level histogram is not decoration. `mL = 1/ln(M)` is easy to get wrong (`1/M` and `ln(M)` are
both plausible), nothing errors when you do, and the only symptom is a badly shaped hierarchy. The
ratios matching 1/M to three decimal places is the evidence that it is right.

**Memory layout (D5)** — 200,000 vectors, 98 MiB, past this CPU's 12 MB L3:

![layout](docs/plots/layout_ab.png)

| access pattern | flat | naive `vector<vector<float>>` | flat advantage |
|---|---|---|---|
| sequential | 63.3 QPS | 55.3 QPS | 1.14× |
| random | 25.4 QPS | 16.2 QPS | **1.56×** |

Layout barely matters for a sequential scan — the prefetcher handles both. It matters under random
access, **which is what HNSW does**, since graph traversal visits neighbours in an order no
prefetcher can predict. The first version of this benchmark reported the naive layout as *faster*;
`BUGS.md` B-05 explains why, and that half is more interesting than the result.

![latency](docs/plots/latency_percentiles.png)

The FAISS head-to-head arrives in Phase 6, measured in the same session on the same machine with
matched thread counts, interleaved A/B/A/B.

## Design decisions

See [`CONTEXT.md`](CONTEXT.md). The ones most worth reading: D4 (squared L2, and why cosine is not
a third code path), D5 (flat arrays with fixed stride, and what that costs), D6 (the from-scratch
boundary), D10 (what makes a number publishable).

## Known limitations

Written before the code, because the list is already knowable and a limitations section added at
the end is a limitations section that flatters:

- **No deletes.** Tombstones + periodic rebuild is the design; removing a node in place severs the
  links that pass *through* it and quietly damages navigability for every other query.
- **Single node, memory resident.** 1M vectors fit in RAM. Nothing here addresses what changes at
  1B — sharding, disk-based indexes, DiskANN.
- **No filtered search.** "Nearest neighbours, but only documents from 2024" has no clean answer in
  this design, and no clean answer in the field either.
- **PQ codebooks trained on a subsample**, which is standard practice and will be stated with the
  sample size rather than left implicit.
- **Approximate with no bound.** Recall is measured on a test set. Nothing guarantees it holds on
  data that drifts.
- **Build time is bad: 1,139 s for SIFT1M, single-threaded.** FAISS does this in a small number of
  minutes. Three known reasons, none of them mysterious: insertion is single-threaded; the backward
  link shrink recomputes a neighbour's whole distance list every time it overflows, where hnswlib
  caches them; and `select_neighbors_heuristic` is O(M²) distance calls per insert with no reuse.
  This is measured, reported, and not excused — see the Phase 6 comparison when it lands.
- Plus whatever `PLAN.md` §0.3's cut order takes, which will be listed here by name.

## Reproducing

Every plot regenerates from the JSONL records in `results/`. Every record carries the git SHA, CPU
model, compiler version, exact flags, thread count, dataset, and RNG seed it was produced under.
