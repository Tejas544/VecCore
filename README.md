# VecCore

**An HNSW vector index with Product-Quantization compression and hybrid dense+sparse retrieval,
written from scratch in C++17 and benchmarked against FAISS.**

> **Status: Phase 1 complete — exact search verified. HNSW (Phase 2) not started.**
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

**Exact search only so far.** i5-11400H, single thread, Release `-O3 -march=native`, WSL2.
Every row traces to a record in [`results/bench.jsonl`](results/bench.jsonl).

| dataset | n | recall@10 | QPS | p50 | p99 | peak RSS |
|---|---|---|---|---|---|---|
| SIFT10K fixture | 10,000 | **1.0000** | 1298 ± 12 | 0.74 ms | 1.06 ms | 9.6 MiB |
| SIFT1M (200 queries) | 1,000,000 | **0.9995** | 12.30 ± 0.04 | 80.2 ms | 94.4 ms | 492 MiB |

The 0.9995 is not a shortfall — it is **one exact distance tie in 2000**. Query 93's published
neighbour 274922 and our 196106 are both at d² = 42192.0; an independent numpy implementation makes
the identical choice. `scripts/diagnose_recall.py` proves this rather than asserting it.

**Memory layout (D5), 200k vectors, 98 MiB — past this CPU's 12 MB L3:**

| access pattern | flat | naive `vector<vector<float>>` | flat advantage |
|---|---|---|---|
| sequential | 62.7 QPS | 54.8 QPS | 1.14× |
| random | 25.3 QPS | 15.1 QPS | **1.67×** |

Layout barely matters for a sequential scan; it matters under random access — which is what HNSW
does, since graph traversal visits neighbours in an order no prefetcher can predict. The first
version of this benchmark reported the naive layout as *faster*; see `BUGS.md` B-05 for why, which
is the more interesting half.

The FAISS head-to-head table arrives in Phase 6, measured **in the same session on the same machine
with matched thread counts, interleaved A/B/A/B**. Build time is expected to lose to FAISS, and that
row will be reported rather than omitted.

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
- Plus whatever `PLAN.md` §0.3's cut order takes, which will be listed here by name.

## Reproducing

Every plot regenerates from the JSONL records in `results/`. Every record carries the git SHA, CPU
model, compiler version, exact flags, thread count, dataset, and RNG seed it was produced under.
