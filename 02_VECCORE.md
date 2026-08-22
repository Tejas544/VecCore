# 02 — VecCore
### Multimodal Vector Index & Retrieval Service, from scratch (C++)

**Aug 19–24 (6 days, ~28 hrs).**
Category: mentor's #1 (Scalable & Efficient RAG) — and quietly #2, since Product Quantization
*is* quantization applied to vectors.
Risk: **Low-medium.** Prerequisite: file `00_FOUNDATIONS.md`, especially §5 (C++ ramp).

---

## 1. Thesis

> Build the retrieval layer under EdgeRAG from first principles: an HNSW graph index with
> Product-Quantization compression, hybrid dense+sparse retrieval, and a concurrent serving
> layer — benchmarked against FAISS.

**Why this project matters most for your SDE interviews:** it is graph algorithms, memory layout,
and concurrency with zero LLM hand-waving. An Amazon or Microsoft interviewer can grill you on it
for forty minutes without mentioning AI once. Both other projects are ML-flavoured; **this is the
one that proves you can do classical systems engineering.** It's your hedge, and it's the reason
the portfolio doesn't read as narrow.

It also completes the story: EdgeRAG ships with a flat index, then you replace it with VecCore
and measure the improvement. Two projects that compose are worth more than three that don't.

---

## 2. Hard Rules

**No FAISS. No hnswlib. No Annoy. No ScaNN.**

FAISS is your **benchmark**, not your dependency. You may read hnswlib's source to learn C++
idiom — then close it and write your own. You will be asked line-level questions about your
implementation, and "I used a library" ends the conversation.

Landing *within* FAISS's numbers is a strong result for a solo 6-day build. Say so plainly in the
README; don't overclaim.

---

## 3. Technology Stack & Required Depth

| Technology | Depth | Notes |
|---|---|---|
| **C++17/20** — RAII, smart pointers, move semantics, `const` | **L3** | See Foundations §5 |
| **Memory layout** — cache lines, padding, contiguous vs pointer-chasing | **L3** | The biggest perf lever in the whole project |
| **`std::thread`, `std::shared_mutex`, `std::atomic`** | **L2** | Concurrent reads during insert |
| Templates (basic), functors | **L2** | Distance functions, float/int8 vector types |
| **HNSW algorithm** | **L3** | You're implementing it |
| **Product Quantization / IVF-PQ** | **L3** | You're implementing it |
| **k-means** (for PQ codebooks) | **L3** | Simple, but know Lloyd's algorithm and initialisation |
| **BM25** | **L3** | Small formula, know every term |
| **Reciprocal Rank Fusion** | **L3** | Trivial to implement, know why it beats score blending |
| Cross-encoder reranking | **L2** | You use a pretrained model, don't build it |
| SIMD intrinsics (AVX2) | **L2** *(optional, high value)* | 4–8× on the distance loop |
| CMake | **L1–L2** | Copy a good template, understand the parts |
| pybind11 | **L1** | Python bindings for EdgeRAG to call |
| gRPC **or** FastAPI wrapper | **L1–L2** | Serving layer; FastAPI over pybind11 is the fast path |
| Sanitizers (ASan/UBSan), `perf` | **L2** | Debugging + a maturity signal |
| ann-benchmarks methodology | **L2** | How the field measures recall/latency honestly |

---

## 4. Learning Plan

### Before you start: Aug 15–18, ~2 hrs/day (runs alongside EdgeRAG)
The modern-C++ ramp in **Foundations §5**. Do not skip it. Competitive-programming C++ will get
you a working index; it will not get you a *fast* one, and the performance story is the project.

### Just-in-time, in build order

| When | Topic | Resource | Hours |
|---|---|---|---|
| Day 1 | **HNSW** | Malkov & Yashunin, *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"* ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)). Read **Algorithms 1–5 line by line** — they're pseudocode you can transcribe. Then Pinecone's HNSW explainer for intuition | 4 |
| Day 3 | **Product Quantization** | Jégou, Douze & Schmid, *"Product Quantization for Nearest Neighbor Search"*, TPAMI 2011. Then the **FAISS wiki** ("Guidelines to choose an index", "Faiss indexes") — genuinely excellent, better than most papers | 3 |
| Day 3 | **IVF** | FAISS wiki, same pages. Coarse quantizer + inverted lists — simple once PQ clicks | 1 |
| Day 4 | **BM25 + RRF** | Robertson & Zaragoza, *"The Probabilistic Relevance Framework: BM25 and Beyond"* (§3 only). RRF: Cormack et al., SIGIR 2009 — 2 pages | 1.5 |
| Day 5 | **Concurrency** | cppreference on `std::shared_mutex`; Foundations §5 refs | 1.5 |
| Throughout | **Benchmark methodology** | [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) — read how they measure. Recall/QPS curves are the field standard, match their format | 1 |
| Optional | **SIMD** | Agner Fog's manuals; Intel Intrinsics Guide. Only after correctness | 2 |

---

## 5. Build Plan — Day by Day

### Day 1 · Foundation + brute force
- CMake project skeleton, pybind11 wired, `-Wall -Wextra -fsanitize=address,undefined` in debug.
- **Flat contiguous vector storage.** `std::vector<float>` of size `n × d`, row-major.
  **Not** `vector<vector<float>>` — that's a pointer-chasing cache disaster, and knowing why is
  itself an interview answer.
- Exact brute-force search (L2 and inner product). This is your **ground truth for recall**.
- Benchmark harness: recall@k, QPS, p50/p95/p99 latency, memory, build time → JSON.
- Dataset: **SIFT1M** (standard, 128-dim, 1M vectors, has ground truth) plus your EdgeRAG embeddings.

### Day 2 · HNSW — the core
Transcribe Algorithms 1–5 from the paper, then make them fast.
- Multi-layer graph, exponentially decaying layer assignment (`level = ⌊-ln(unif(0,1)) · mL⌋`).
- Greedy search at upper layers, `ef_search` beam search at layer 0.
- **The neighbour-selection heuristic (Algorithm 4)** — this is what makes HNSW work rather than
  merely function. Implement the heuristic version, not the naive top-M. Know the difference.
- **Memory layout matters here:** store the graph as a flat `vector<uint32_t>` with fixed
  max-degree stride, not as per-node `vector`s. Measure both if you have time — the gap is the
  best performance story in the project.
- Tune `M`, `ef_construction`, `ef_search`. Produce a **recall/QPS curve**, the field-standard plot.

### Day 3 · Product Quantization
- k-means codebooks: split the `d`-dim vector into `m` subvectors, 256 centroids each (1 byte per
  subvector) → compression from `4d` bytes to `m` bytes.
- **Asymmetric Distance Computation (ADC)**: precompute a query-to-centroid distance lookup table,
  then distance is `m` table lookups and adds. Know why asymmetric beats symmetric.
- Optionally IVF on top: coarse quantizer + inverted lists + `nprobe`.
- **Plot the Pareto curve:** recall@10 vs. memory vs. latency, sweeping `m` and `nbits`.
  *(Expect 8–32× compression at a few points of recall.)*

### Day 4 · Hybrid retrieval
- **BM25** sparse index: inverted index, tf-idf with the `k1`/`b` saturation terms. Know what
  each parameter does.
- **Reciprocal Rank Fusion** to combine dense + sparse: `score = Σ 1/(k + rank_i)`, `k = 60`.
- Be ready for: *"why RRF and not a weighted score blend?"* → dense and sparse scores live on
  incomparable scales; RRF is rank-based, so it needs no normalisation or tuning.
- Measure the recall lift from hybrid over dense alone.

### Day 5 · Reranking + concurrency
- Cross-encoder reranking on the top-50 (quantized reranker — ties back to your theme).
  Measure recall lift *and* the latency cost. Both numbers, honestly.
- **Concurrency:** `std::shared_mutex` for concurrent reads during inserts. Benchmark QPS at
  1/2/4/8 threads and show the scaling curve. Note where it stops scaling and why.
- Incremental insert support (no full rebuild).

### Day 6 · Serve, benchmark, document
- FastAPI over pybind11 (fast path) or gRPC (stronger signal if time allows).
- **Head-to-head vs FAISS** on identical data: recall@10, QPS, p99, memory, build time.
- Swap VecCore into EdgeRAG; report the end-to-end delta. **This is the money shot** — two
  projects composing into one system.
- README: architecture diagram, recall/QPS curves, PQ Pareto plot, thread-scaling plot, limitations.

---

## 6. Metrics You Must Report

| Metric | Baseline | Target |
|---|---|---|
| recall@10 | brute force = 1.0 | ≥ 0.95 at competitive QPS |
| QPS at recall 0.95 | FAISS HNSW | within 2–3× is a good solo result |
| p99 query latency | FAISS | report honestly |
| Memory with PQ | raw fp32 | 8–32× compression |
| Index build time | FAISS | report, expect to lose — explain why |
| Thread scaling | 1 thread | near-linear to 4, then note the bottleneck |
| Hybrid recall lift | dense only | +3–10% typical |
| Rerank recall lift | no rerank | measure, with latency cost |

---

## 7. Interview Defense — Answer These Cold

**HNSW**
1. Why does the layered structure give logarithmic-ish search? What's the small-world property?
2. Explain the neighbour-selection heuristic. What goes wrong with naive top-M? *(Clustering —
   the graph loses long-range links and search gets trapped in local minima.)*
3. How is layer assignment decided, and why exponential decay?
4. What do `M`, `ef_construction`, `ef_search` each control? Which is query-time vs build-time?
5. How did you lay out the graph in memory, and why? *(Flat array + stride vs pointer-chasing.)*
6. Where does HNSW lose to IVF-PQ? *(Memory. HNSW stores full vectors plus graph edges.)*

**Product Quantization**
7. What does PQ do to distance computation? Where does the error come from?
8. Asymmetric vs symmetric distance computation — why is ADC more accurate?
9. Why 256 centroids per subquantizer? *(Exactly one byte per code.)*
10. How do you choose `m`? What's the recall/memory tradeoff shape?

**Systems**
11. How do you handle concurrent reads during an insert? What's your locking granularity?
12. Where does thread scaling break down, and why?
13. Why `vector<float>` flat rather than `vector<vector<float>>`? Quantify the cache impact.
14. How would you support deletes? *(Tombstones + periodic rebuild. Know why in-place delete
    breaks graph connectivity.)*

**Retrieval**
15. Write BM25 from memory. What do `k1` and `b` do?
16. Why RRF over weighted score fusion?
17. When does reranking help most? When is it wasted latency?

**Scale**
18. 1M vectors fits in RAM. What changes at 1B? *(Sharding, disk-based indexes, DiskANN.)*
19. How would you shard this across machines, and how do you merge results?

---

## 8. Target CV Bullet

> Implemented an HNSW vector index and IVF-PQ compression from scratch in **C++17** with
> concurrent reads and Python bindings; achieved **recall@10 of __** at **__ QPS / __ ms p99**
> with **__× memory compression**, within **__%** of FAISS on SIFT1M. Added hybrid dense+BM25
> retrieval with RRF and cross-encoder reranking for **+__%** recall.

---

## 9. Deliverables Checklist

- [ ] C++17 repo, CMake, warning-clean, sanitizers in debug, daily commits, `BUGS.md`
- [ ] Correctness test: HNSW recall vs. brute-force ground truth
- [ ] recall/QPS curve (ann-benchmarks format)
- [ ] PQ Pareto plot: recall vs. memory vs. latency
- [ ] Thread-scaling plot
- [ ] Head-to-head FAISS comparison table
- [ ] pybind11 bindings + FastAPI service
- [ ] **Integrated into EdgeRAG with a measured end-to-end delta**
- [ ] README: architecture diagram, design decisions, known limitations

---

## 10. Risks & Cut Lines

| Risk | Mitigation |
|---|---|
| C++ ramp takes longer than 20 hrs | Start Aug 15 alongside EdgeRAG, not Aug 19. It double-counts with your DSA prep |
| HNSW recall is silently poor | Test against brute force from Day 1. Recall < 0.9 usually means the neighbour heuristic is wrong — check Algorithm 4 first |
| Memory bugs eat a day | ASan/UBSan on from the first commit, not after the bug appears |
| PQ k-means won't converge | k-means++ init; cap iterations at 25; subsample the training set |
| Behind at Day 5 | **Cut:** reranking, then gRPC (use FastAPI), then IVF (keep plain PQ). **Never cut:** HNSW correctness testing or the FAISS comparison — without a baseline you have no result |
