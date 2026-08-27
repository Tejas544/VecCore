# VecCore — Decision Log

Every design decision, with the alternative that was rejected and the reason. Written as decisions
are made, not reconstructed afterwards.

**Why this file exists.** `00_FOUNDATIONS.md` §7 point 3: *"justify every design choice against the
alternative you rejected, with a reason that isn't 'it was easier'."* In December, "I used a flat
array for the adjacency lists" and "I used a flat array because pointer-chasing a graph is a cache
disaster, here is the A/B I ran, and here is the case where I'd choose otherwise" are different
answers from different candidates.

**Status values:** `PROPOSED` (needs your sign-off) · `ACCEPTED` · `REVISED` (with the date and
what changed) · `REJECTED`.

**Same convention as EdgeRAG and QuantKit.** When a decision changes during the build, amend the
entry — do not delete it. The revision history *is* the interview material.

---

## D1 · Schedule: Aug 21 evening → Aug 25, hard stop · **ACCEPTED** · 2026-08-21

**Decision.** VecCore runs from tonight to end of Aug 25, then stops regardless of state. QuantKit's
Phase 0 (scaffolding — `pyproject.toml`, repo layout, the bench harness skeleton; no algorithms, no
GPU, no thinking-under-load) moves into evening slack during that window, so QuantKit opens on
Aug 26 with its rails already laid.

**Why not the plan of record (Aug 19–24).** Two of its six days are gone. Holding the Aug 24 end
date means ~24 h against a scope `PLAN.md` §0.2 costs at 29.5–40 h, which forces cutting Phase 5
(concurrency) — the single highest-value-per-hour item for an SDE panel and the one thing in this
project that is neither ML nor an algorithm transcription.

**Why not slip further than Aug 25.** QuantKit is already at or over budget on its own estimate,
and it depends on Colab GPU availability, which VecCore does not. Spending QuantKit's slack twice
is how both projects end up unfinished.

**Why not abandon VecCore entirely and roll the days into QuantKit.** The portfolio currently has
two ML-flavoured projects and no from-scratch systems project. VecCore is the one an interviewer
who does not do ML can spend forty minutes on. That is the whole reason it is in the plan.

**Rejected alternative worth naming: running both part-time in parallel.** QuantKit's own D10 calls
this out as historically the worst option, and it is right — the C++ ramp and the GPTQ maths
compete for the same scarce resource, which is uninterrupted thinking time, not hours.

**Accepted 2026-08-21.** VecCore runs to end of Aug 25; QuantKit opens Aug 26 with its Phase 0
already laid. `PLAN.md` is costed against ~30 h on this basis.

**Sub-question resolved the same day: the Aug 15–18 modern-C++ ramp did happen.** So `PLAN.md`
§0.5's inlined-ramp fallback does **not** apply, and §0.2's estimates hold as written. RAII, move
semantics, memory layout, and `shared_mutex` are assumed available when the phases reach for them.
If that turns out optimistic in practice — most likely at Phase 5, where memory ordering is the
least-used corner of the ramp — amend this entry rather than silently absorbing the overrun.

---

## D2 · Build and measure in WSL2 Ubuntu, not MSYS2 or MSVC · **ACCEPTED** · 2026-08-21

**Decision.** Source of truth stays at `D:\Placement Projects\VecCore` (one working copy, edited on
Windows). All building, testing, and benchmarking happens inside WSL2 Ubuntu, with the **build
directory and the datasets on the Linux filesystem**, not on `/mnt/d`.

**Why.** Verified on this machine today, not assumed:

| | WSL2 Ubuntu | MSYS2 UCRT64 | MSVC |
|---|---|---|---|
| ASan / UBSan | ✅ | ❌ **verified: fails to link** | ✅ |
| TSan (Phase 5) | ✅ | ❌ | ❌ not supported on Windows at all |
| valgrind / cachegrind | ✅ | ❌ | ❌ |
| `pip install faiss-cpu` | ✅ | ⚠️ | ⚠️ |
| pybind11 against CPython | ✅ | ❌ MinGW-built ext vs MSVC-built CPython is unsupported | ✅ |

The deciding fact is **L-01** in `BUGS.md`: `-fsanitize=address,undefined` does not link under the
local MSYS2 toolchain, and `02_VECCORE.md` §10 names sanitizers-from-commit-one as the *only*
mitigation for "memory bugs eat a day." A mitigation that does not exist is worse than no
mitigation, because the plan was built assuming it.

**Why the build directory goes on ext4 and not next to the source.** `/mnt/d` crosses the 9p
protocol bridge. Compilation and linking are dominated by many small file operations, and index
build reads 512 MB of SIFT — putting either on the bridge means your build-time benchmark is partly
a measurement of the filesystem bridge.

**Why not two working copies (Windows for editing, WSL clone for building).** Hand-synced copies
diverge, and a stale copy produces a bug that looks like a code bug. One copy, one truth.

**Rejected: MSVC Build Tools.** It would give ASan and clean pybind11, but no TSan and no
valgrind, and it costs ~6 GB and an hour of setup for a strictly smaller capability set.

**Accepted 2026-08-21.** WSL2 Ubuntu it is. **Fallback if the install turns out to be blocked:**
MSYS2 for the day-to-day build plus Colab for sanitizer runs and the FAISS baseline — which moves
the FAISS comparison onto different hardware, weakening it materially. If it comes to that, the
README must label the table as cross-machine. `PLAN.md` §0 timeboxes the WSL setup at 45 minutes;
past 90 minutes with nothing compiling, take the fallback and log it here.

---

## D3 · SIFT1M for headline numbers, SIFT10K as the inner-loop fixture · **ACCEPTED** · 2026-08-21 · **CONFIRMED at Phase 0, 2026-08-22**

**Decision.** Every unit test and every correctness check runs on the first 10,000 SIFT base
vectors. SIFT1M appears only in `bench`, and only after a gate is green on 10K.

**Why.** `02_VECCORE.md` names SIFT1M and is right to — it is the standard, it has published ground
truth, and it is what the field's numbers are quoted on. What it does not say is that a full 1M
HNSW build takes minutes. **A test suite that takes minutes stops being run**, which is EdgeRAG's
D1 lesson, already learned once on this laptop. Test-cycle time is a first-class design concern,
not a nicety.

**Why SIFT and not something more modern.** Published ground truth is the entire point — it is what
makes recall *measurable* rather than *asserted*, and it makes the FAISS comparison a comparison
rather than two unconnected numbers. GIST1M (960-dim) is a reasonable stretch goal if Phase 2 lands
early, because higher dimensionality changes the cache story; it is not worth a day.

**Rejected: generating synthetic Gaussian vectors.** Faster to set up, and worthless — random
vectors have no cluster structure, which is exactly the structure HNSW's neighbour heuristic
exists to handle. You would get recall numbers that do not transfer to any real dataset.

**Note:** synthetic data *is* used for one thing — the corpus-size crossover curve in Phase 6.3,
where the question is "at what n does HNSW overtake brute force" and the answer only needs
realistic dimensionality, not realistic structure. Say which is which in the README.

### CONFIRMED 2026-08-22 at the Phase 0 gate — and it paid for itself twice

The 10,000-vector fixture shipped as proposed (`scripts/make_fixture.py`), and the split held for the
whole build: **140 tests — 95 C++ and 45 Python — run on small fixtures and finish in seconds, while
SIFT1M appears only in `bench`.** Almost all of them use 3,000 vectors or fewer; the largest is a
50,000-vector index built by one GIL test that needs the C++ work to dominate its own call overhead
(B-14). Two things the split bought that were not the stated reason:

- The Phase 1 gate on 10K is **exactly 1.0000**, while the same code on SIFT1M is 0.9995 against the
  *published* ground truth. Having both numbers is what made it obvious the 0.9995 was a property of
  the data rather than of the code (B-01's neighbourhood, resolved by `scripts/diagnose_recall.py`).
  A 1M-only test suite would have presented that as an unexplained defect.
- It is why 37/37 tests could run **under ASan+UBSan** at the Phase 2 gate. A sanitized build is
  several times slower; a sanitized 1M test suite is a suite nobody runs, which is the same failure
  D3 was written to avoid, one layer down.

**GIST1M was not attempted.** Phase 2 did not land early, and §0.3's cut order outranks a stretch
goal. It stays a stretch goal rather than becoming a limitation, because nothing in the repo claims
a result at 960 dimensions.

---

## D4 · Squared L2 as the primary metric; inner product supported, cosine by normalisation · **ACCEPTED** · 2026-08-21

**Decision.** `l2_sqr` is the default and the one SIFT ground truth is defined against. Inner
product is a second distance functor. Cosine is *not* a third implementation — it is inner product
on L2-normalised vectors, normalised once at insert and once per query.

**Why squared.** `sqrt` is monotonic, so it cannot change a ranking, and it is a genuinely
expensive instruction to execute per candidate. Every serious implementation drops it. Say
"squared L2" out loud rather than "L2" so it reads as deliberate.

**Why cosine is not its own code path.** A third distance function is a third thing to keep correct
and a third thing to test. Normalising at the boundary is one line and makes the equivalence
explicit — which is itself the interview answer.

**Consequence to watch:** with inner product, "larger is better" while for L2 "smaller is better."
That sign flip has to be handled in exactly one place (the comparator), not sprinkled through the
search. Getting it wrong produces P-04-shaped symptoms with a different cause.

---

## D5 · Flat arrays with fixed stride for both vectors and graph adjacency · **ACCEPTED** · 2026-08-21

**Decision.** Vectors: one `std::vector<float>` of `n × d`, row-major. Adjacency: one
`std::vector<uint32_t>` per level with stride `M_max + 1`, slot 0 holding the neighbour count.
Never `vector<vector<...>>`.

**Why.** This is the biggest single performance lever in the project and `02_VECCORE.md` says so
twice. A vector-of-vectors is `n` separate heap allocations at unpredictable addresses: every
access is a dependent load the prefetcher cannot anticipate, plus 24 bytes of header per row and
the allocator's per-block overhead. The flat version streams, and a 128-dim float vector is exactly
512 bytes — 8 cache lines, nothing wasted.

**What we give up.** Fixed stride wastes memory for nodes below max degree (a node with 4
neighbours still occupies `M_max + 1` slots), and growing the index means reallocating one large
buffer rather than appending to small ones. Both are the right trade at this scale, and being able
to state the cost is the difference between having made a decision and having copied one.

**Commitment: measure it, do not just assert it.** Keep the naive variant behind a compile flag and
A/B the brute-force scan once. `02_VECCORE.md` calls this "the best performance story in the
project" and it costs twenty minutes. Use `cachegrind` for the cache-miss counts — WSL2 does not
reliably expose hardware performance counters to `perf`, so a simulator is the honest tool here.

### REVISED 2026-08-22 — measured, and the honest number is smaller than the folklore

Run on 200,000 SIFT vectors (98 MiB, comfortably past this CPU's 12 MB L3), 100 queries, 3
interleaved rounds. Record: `results/bench.jsonl`, tag `layout_ab_200k`.

| access pattern | flat | naive (fragmented) | flat advantage |
|---|---|---|---|
| sequential | 62.7 QPS | 54.8 QPS | **1.14×** |
| random | 25.3 QPS | 15.1 QPS | **1.67×** |

**The decision stands; the *argument* for it changes.** The claim "flat is 3–10× faster" — which is
what the folklore says and what this entry originally implied — **is not what this machine
measures.** What it measures is:

- **For a sequential scan, layout barely matters (1.14×).** The prefetcher handles both, because a
  contiguous array of pointers is itself perfectly predictable to walk.
- **Under random access the gap widens to 1.67×**, and *that* is the case this project actually
  cares about: brute force scans in order, but **HNSW does not**. Graph traversal visits neighbours
  in an order no prefetcher can anticipate, so a dependent pointer load costs a full DRAM round
  trip that a computed offset into a flat array does not.
- Random access costs the flat layout 2.5× against its own sequential number and the naive layout
  3.6×. The penalty for unpredictable access is large for both and *worse* for the layout carrying
  an extra indirection — which is the mechanism, stated in numbers.

**Two things had to be fixed before the benchmark could see any of this**, and both are recorded as
B-05: the naive store was built by one uninterrupted allocation loop, so `malloc` laid it out very
nearly contiguously and the "bad" layout was not bad; and only sequential access was being timed,
which is the case where the effect is smallest. The first version of the experiment reported the
naive layout as *faster*.

**Say the measured numbers, not the folklore.** "1.14× sequential, 1.67× random, and the second is
the one that matters because HNSW traverses a graph" is a defensible answer that invites a good
follow-up. "3–10×" is a number from a blog post, and the first interviewer who asks how it was
measured will find out it was not.

---

## D6 · The "from scratch" boundary · **ACCEPTED** · 2026-08-21

**We own, and must be able to defend line by line:** the vector store, both distance functions,
brute-force search, HNSW (insert, search, level assignment, the Algorithm 4 heuristic, the visited
list), k-means and PQ codebooks, ADC, the BM25 inverted index and scorer, RRF, the locking scheme,
and the benchmark harness.

**We use:** the C++ standard library, CMake, doctest, pybind11, numpy and matplotlib for plotting,
and **FAISS strictly as a benchmark baseline** — imported in `bench/faiss_baseline.py` and nowhere
else in the repo.

**Why this is the right line.** `02_VECCORE.md` §2 bans FAISS, hnswlib, Annoy, and ScaNN as
*dependencies*. It does not ban standard-library containers, and writing my own `priority_queue`
would demonstrate nothing that using `std::priority_queue` and explaining its heap property does
not.

**The one that will get probed: `std::priority_queue` in the HNSW search.** Be ready with: it is a
binary heap over a `std::vector`, push and pop are O(log n), and *the reason there are two of them
with opposite comparators* is the real content of the answer. If asked whether a custom flat
bounded heap would be faster — yes, probably, because `ef` is small and known, so a sorted insertion
into a fixed array can beat heap operations at that size. Say that; do not build it unless Phase 2
finishes early.

**Hard rule on hnswlib:** `02_VECCORE.md` permits reading it for C++ idiom. Read it *after* your
implementation works, or not at all. Reading it while stuck converts a bug you would have diagnosed
— and could have told a story about in December — into a line you copied.

---

## D7 · EdgeRAG integration: three honest claims, not one convenient one · **ACCEPTED** · 2026-08-21 · **DELIVERED at Phase 6, 2026-08-27**

**Decision.** The Phase 6.3 integration delivers:

1. **`VecCoreIndex` implementing the existing `RetrievalIndex` protocol**, with recall identical to
   `FlatIndex` on the 362-document corpus. The claim is **no regression** — that is the real and
   sufficient result for the drop-in.
2. **TF-IDF → BM25**, measured on EdgeRAG's 650 held-out queries. A genuine quality delta on a real
   corpus, independent of scale.
3. **A measured crossover curve** — brute force vs HNSW latency against corpus size on synthetic
   data, with EdgeRAG's n=362 marked on it.

**Why not the obvious claim ("HNSW makes EdgeRAG's retrieval faster").** Because it is false at
n=362, where HNSW is *slower* — you pay graph traversal to avoid work that costs microseconds. The
first interviewer to ask "at what corpus size?" would end that line of conversation badly. Recorded
in full as landmine **L-03**.

**Why BM25 is the substantive win here.** EdgeRAG's dense image-side signal was measured to be
effectively noise (`DEFAULT_ALPHA = 0.0`, and `hybrid == text_only` bit-for-bit across 165 held-out
queries), so its retrieval today is pure TF-IDF. BM25's term-frequency saturation and length
normalisation are a real improvement over raw TF-IDF on short OCR text — and unlike the HNSW story,
this one does not need scale to be true.

**Honesty item to pre-register in the README:** EdgeRAG's retrieval has a documented **structural
ceiling** — 112 of 362 documents have no OCR text at all and are unreachable by any text-based
retriever, capping recall at ~37.6%. **Quote any BM25 improvement against that ceiling, not against
100%.** Reporting "recall improved from 20% to 26%" without the ceiling invites exactly the
criticism `WHAT_IS_THIS.md` §10 levels at other people.

**The interface already exists**, which is worth pointing out in an interview: EdgeRAG's
`RetrievalIndex` protocol docstring reads *"``VecCore`` implements this later without touching a
caller."* Designing the seam before the implementation existed, and then having the implementation
drop in without changing a caller, is the whole argument for the abstraction.

### DELIVERED 2026-08-27 — all three claims, and the crossover came in lower than guessed

`python/veccore/edgerag.py` implements the protocol; `tests/test_edgerag_adapter.py` checks it
against the real 650-query set. All three claims hold:

1. **No regression** — same protocol, same call sites, recall at least as good as the TF-IDF index.
2. **TF-IDF → BM25: +4.17% relative recall@5 at 11× lower latency**, quoted against the 0.3846
   structural ceiling exactly as the honesty item above required.
3. **Crossover measured at n ≈ 1,189** (`bench/crossover.py`), so EdgeRAG's 362 documents sit **3.3×
   below it** and HNSW is **42% slower** there. `use_hnsw` therefore defaults to `False`.

**The one thing worth flagging against this decision:** what shipped is a *drop-in that EdgeRAG has
not yet adopted*. The adapter is here, tested against exported EdgeRAG data; EdgeRAG's own default
retriever is still TF-IDF and its source still mentions VecCore only in the protocol docstring. So
`02_VECCORE.md` §5's "swap VecCore into EdgeRAG; report the end-to-end delta" is delivered at the
**retrieval** layer and not at the **pipeline** layer — there is no end-to-end answer-quality or
token-cost number. That is a smaller claim than the spec's, it is the claim the measurements
actually support, and it is named here rather than left for a reader to notice.

---

## D8 · Coarse `shared_mutex` first; fine-grained locking described, not built · **ACCEPTED** · 2026-08-21 · **REVISED then AMENDED 2026-08-26**

**Decision.** One `std::shared_mutex` for the whole index. Readers take `shared_lock`, the inserter
takes `unique_lock`. Be able to describe the finer-grained design — per-node link locks plus an
atomic entry point, which is what hnswlib does — without shipping it.

**Why.** Coarse and correct with a *measured* statement of where it hurts beats fine-grained and
racy, every time, and that sentence is itself the interview answer. With ~30 h total, a subtle race
in a graph mutation path is exactly the bug that eats a day and produces nothing presentable.

**What must still be measured, because the honesty is the point:** read-only QPS at 1, 2, 4, 6, 8,
and 12 threads. Those numbers are chosen deliberately for a 6-core/12-thread CPU — predict the
shape before measuring (near-linear to 4, tailing by 6, well under linear at 8–12 because
hyperthreads share execution units and the distance loop already saturates them), then show the
plot. Predicting correctly and showing it is a much stronger story than the plot alone.

**And the trap that goes with it (P-30):** do not write "it stops scaling because of lock
contention" into the README without testing it. Re-run the sweep with the lock removed — read-only,
so it will not crash — and if the curve is unchanged, it was memory bandwidth or hyperthreading,
not the lock. A confident wrong explanation is worse than "I measured this and I am not certain
why."

### REVISED 2026-08-26 — measured, and P-30 fired exactly as predicted

**The lock is not the limiter, and the control proves it.** Read-only scaling with the lock
*entirely removed* (`LockMode::None`) is indistinguishable from `shared_mutex` at every thread
count — at 200k vectors and 12 threads: 14,994 / 15,172 / 14,701 QPS for none / shared_mutex /
writer_priority, a ~3% spread that is inside the trial-to-trial noise. **"It stops scaling because
of lock contention" would have been a confident, plausible, wrong sentence.**

**What does limit it, with evidence.** Running the same sweep at two working-set sizes:

| threads | n=10K (4.9 MiB, fits in 12 MB L3) | n=200K (98 MiB, exceeds L3) |
|---|---|---|
| 2 | 2.00× (100%) | 1.74× (87%) |
| 4 | 3.37× (84%) | 2.84× (71%) |
| 6 | 4.07× (68%) | 3.36× (56%) |
| 8 | 4.96× (62%) | 3.89× (49%) |
| 12 | 5.59× (47%) | 4.16× (35%) |

The out-of-cache workload loses **~12 points of scaling efficiency at every thread count**. That is
memory bandwidth, isolated by changing only the working-set size. Beyond 6 threads both curves
flatten because threads 7–12 are hyperthreads sharing execution units with a loop that already
saturates them — 8→12 threads buys 3.89→4.16× at 200k, almost nothing. Turbo clock reduction as
more cores engage is a third contributor and is not separated out here; saying so is more honest
than attributing the whole residual to bandwidth.

### AMENDED 2026-08-26 — the default lock mode changed, because the old one stops accepting writes

`LockMode::WriterPriority` is now the default, not `SharedMutex`. See **B-11**: glibc's
`shared_mutex` is reader-preferring, so four continuous readers starve the writer *completely* —
5 of 200 inserts in 36 seconds, insert p50 **8.2 seconds**. A turnstile mutex fixes it at no
measurable read cost: insert p99 under 4 readers goes **4,446 ms → 0.79 ms**.

This is the decision that changed most on contact with measurement, and the reason is worth
keeping: the original entry said "coarse and correct beats fine-grained and racy". That was right
about *correctness* and silent about *liveness*. A lock can be perfectly correct and still make the
index unable to accept a write.

---

## D9 · Cut order extends the spec's by one item · **ACCEPTED** · 2026-08-21

`02_VECCORE.md` §10 gives: cut reranking, then gRPC, then IVF. `PLAN.md` §0.3 keeps that ordering
and continues it: **the FastAPI service is cut fourth**, before hand-written SIMD, before PQ's
training-set size, and before concurrent insert.

**Why the service goes before those.** The pybind11 module is the artifact that matters — it is
what EdgeRAG calls, and it proves the C++/Python seam works. An HTTP wrapper on top of it adds one
interview answer ("what does the network layer cost?") that can be given from a whiteboard. Forty-
five minutes is cheap; forty-five minutes taken out of Phase 2 is not.

**Never cut, and this list is not negotiable at 2 a.m.:** brute-force ground truth and the recall
check against it, the `bench` harness, the neighbour-selection heuristic and the recall number
proving it works, the `ef_search` sweep, the FAISS head-to-head, and `BUGS.md`/`CONTEXT.md`/the
README's limitations section. Without the FAISS baseline there is no result — only an anecdote.

---

## D10 · Benchmark comparability rules · **ACCEPTED** · 2026-08-21

Adapted from EdgeRAG's "no perf number from an untrusted device," with the substitution that here
the CPU *is* the device.

1. **One environment.** WSL2, `Release` (`-O3 -march=native -DNDEBUG`), stated thread count. FAISS
   is measured in that same environment, in the same session, on the same data.
2. **Interleave compared configurations** — A, B, A, B — never all-A-then-all-B. This is a mobile
   CPU: run them in blocks and you are partly measuring how hot the laptop got.
3. **≥5 trials, report the spread.** A single number is not a result.
4. **Match on recall, not on parameters.** Comparing your `ef_search=64` to FAISS's default
   `efSearch` is not a comparison. That is what the recall/QPS curve exists for.
5. **Match thread counts explicitly.** `faiss.omp_set_num_threads(1)` against your single-threaded
   number. FAISS defaults to every core, and comparing that to one thread is a 6× error in its
   favour — and the first thing an interviewer will ask about.
6. **Every JSON record is stamped** with git SHA, CPU model, compiler version, flags, thread count,
   dataset, RNG seed, and every index parameter. A record missing those does not go in a plot.
7. **Every number in the README traces to a record in `results/`.** If you cannot find the record,
   delete the number.

---

## D12 · PQ is a memory result, not a speed result · **ACCEPTED** · 2026-08-26

**Measured on SIFT1M**, 1000 held-out queries, single thread:

| m | B/vector | compression | ADC recall@10 | +rerank top-100 | ADC QPS |
|---|---|---|---|---|---|
| 4 | 4 | 128× | 0.1071 | 0.4202 | 261 |
| 8 | 8 | 64× | 0.3125 | 0.7779 | 164 |
| 16 | 16 | 32× | 0.5344 | 0.9656 | 114 |
| 32 | 32 | 16× | 0.7186 | 0.9987 | 55 |

**The framing that matters, and it is not the obvious one.** PQ gives 16–128× *memory* compression
and only 4–20× *speed* over brute force — nothing like HNSW's 150×. That is not a shortfall, it is
what PQ is: it still scans every one of the million codes, it just makes each comparison cheaper
(4–32 bytes read instead of 512). **The win is bytes, not hops.** HNSW wins by not looking at most
of the data; PQ wins by making all of the data small. Saying that clearly is the difference between
understanding the technique and having implemented it.

**The honest memory number.** ADC-only at m=8 is ~8 MB of codes plus 128 KB of codebooks against
488 MB raw. But **rerank needs the full vectors resident**, so a reranking configuration's real
footprint includes all 488 MB. `bench` records those separately and never conflates them — quoting
64× compression for a configuration that keeps the uncompressed vectors in RAM would be exactly
the overclaiming `WHAT_IS_THIS.md` §10 criticises.

**Rejected: making the Phase 3 gate pass by loosening it.** ADC-only at 16× measured 0.7186 against
a written target of 0.80. Rather than adjust the number, FAISS `IndexPQ` was run on the same data
at the same `m`: **0.7059**. We are 1.8% ahead of the reference implementation. The target was an
estimate made before measuring; the code was never wrong. See `PLAN.md` §4.

---

## D13 · Hybrid retrieval on EdgeRAG: BM25 ships, RRF is reported as a negative · **ACCEPTED** · 2026-08-26

**Decision.** EdgeRAG's TF-IDF is replaced by BM25. RRF is implemented, tested, and **reported as a
measured negative on this corpus** rather than quietly omitted or presented with a flattering
configuration.

**Measured**, 362 documents, 650 real queries, structural ceiling 0.3846:

| retriever | recall@1 | recall@5 | recall@10 | ÷ ceiling | p50 |
|---|---|---|---|---|---|
| TF-IDF (EdgeRAG today) | 0.0400 | 0.1846 | 0.2738 | 48.0% | 0.0359 ms |
| **BM25** | **0.0446** | **0.1923** | **0.2862** | **50.0%** | **0.0034 ms** |
| RRF(BM25, TF-IDF) | 0.0400 | 0.1862 | 0.2769 | 48.4% | 0.0432 ms |

**BM25 wins on both axes: +4.17% relative recall@5 and 11× lower latency.** The latency half is not
a micro-optimisation — it is algorithmic. An inverted index touches only documents containing a
query term; the TF-IDF implementation scores every document in the corpus. Same tokenizer, same
data, different data structure.

**Why the recall lift is modest, stated so nobody has to ask.** Both are lexical retrievers over the
same tokens with the same tokenizer. The three things that differ are term-frequency saturation
(`k1`), explicit length normalisation (`b`), and the IDF form. On short OCR text with few repeated
terms, saturation has little to bite on. A +4% relative lift is what those three changes are worth
here, and claiming more would require a different corpus, not a different argument.

**Why RRF is reported as a negative.** See B-08. Briefly: RRF fuses by rank alone, which is what
makes it tuning-free and also what makes it authority-blind. It cannot know one input dominates.
Measured: 49% top-10 overlap, BM25 uniquely right on 13 queries against TF-IDF's 8, and an **oracle
fuser would reach 0.2046 against BM25's 0.1923** — so +6.4% of complementary signal genuinely
exists and RRF captures none of it.

**Rejected: dropping RRF from the report because the number is unflattering.** `PLAN.md` §4.5
pre-committed to reporting it either way, before the number was known. A null result that was
pre-registered and then investigated is worth more in an interview than a lift that was found by
trying configurations until one worked.

**Rejected: weighted score fusion to capture the headroom.** It would work here, and it would cost
the one property RRF exists for — a weight fitted to this corpus is wrong on the next one. The
trade is now stated with a number attached rather than asserted.

**Scope limitation, named rather than papered over.** A genuine dense+sparse hybrid — where the two
retrievers really are complementary — **could not be measured on this corpus**, because EdgeRAG's
dense signal was measured to be noise (`DEFAULT_ALPHA = 0.0`). It needs a real text embedding model,
which is out of scope. The `+3-10% hybrid lift` in `02_VECCORE.md` §6's target table is therefore
**not achievable on the data available**, and that is a data limitation, not an implementation one.

---

## D11 · No hand-written SIMD unless it is measured to matter · **ACCEPTED** · 2026-08-21 · **RESOLVED 2026-08-27**

**Decision.** Compile `-O3 -march=native`, verify with `-fopt-info-vec-optimized` that the distance
loop actually vectorised, and report that. Write AVX2 intrinsics only if a measurement shows the
compiler left something on the table *and* Phase 2 finished early.

**Why.** `02_VECCORE.md` marks SIMD "optional, high value," and the high value is real — but it is
the *understanding* that is valuable, not the code. On a simple contiguous `l2_sqr` loop, GCC 15
with `-march=native` will emit good AVX2 (and this CPU has AVX-512, though it downclocks under
sustained AVX-512 load, which is itself a talking point worth knowing). "I checked the generated
code, confirmed it was already vectorised, and measured that hand intrinsics gained nothing" is a
*better* answer than most hand-written SIMD — it demonstrates the same knowledge plus the judgement
to not spend a day on it.

**When this decision reverses:** if the auto-vectorisation report shows the loop was *not*
vectorised — most likely due to an aliasing assumption the compiler could not prove — then try
`__restrict` first, and only then reach for intrinsics.

### RESOLVED 2026-08-27 — measured. The loop vectorises, and AVX-512 is worth nothing here

Run it yourself; the report is a build option, not a one-off:

```bash
cmake -S . -B ~/veccore-build-vecreport -G Ninja -DCMAKE_BUILD_TYPE=Release -DVECCORE_VEC_REPORT=ON
cmake --build ~/veccore-build-vecreport --target veccore 2>&1 | grep distance.hpp
```

GCC 13.3.0, WSL2 Ubuntu 24.04, i5-11400H, `-O3 -march=native` — the same environment D10 requires of
every other number in this repo:

```
include/veccore/distance.hpp:35:29: optimized: loop vectorized using 32 byte vectors
include/veccore/distance.hpp:35:29: optimized: loop vectorized using 16 byte vectors
```

Line 35 is `L2Sqr`, and it is **the only distance line the library build reports** — which is itself
worth reading correctly rather than glossing. `NegInnerProduct` does not appear because nothing in
`libveccore.a` instantiates it in a hot loop; every index is built on squared L2 (D4). Compiling a
translation unit that *does* instantiate it reports `distance.hpp:62:29` with an identical
`32 byte vectors`, so both kernels vectorise — but only one of those two facts comes out of the
build command above, and saying otherwise would be exactly the sort of one-line overclaim B-12 is
about.

The 32-byte and 16-byte entries are the main loop and its epilogue, not two competing results:
32 bytes is 256 bits, so the body is AVX2, with a narrower tail for the remainder. The emitted
assembly confirms it — `ymm` registers and `vfmadd`, and **`-fopt-info-vec-missed` reports nothing at
all against either kernel**. `__restrict` was already on both kernels' parameters, so the aliasing
escape hatch named above was never needed.

**The interesting half is what the compiler declined to do.** `-march=native` resolves to
`tigerlake`, and on this machine:

```
-mavx2 [enabled]   -mavx512f [enabled]   -mavx512vl [enabled]   -mprefer-vector-width= 256
```

The hardware has AVX-512 and **GCC deliberately caps the vector width at 256 bits anyway** — the
licence-based downclocking this decision predicted, encoded as a compiler default. So the question
"did the compiler leave something on the table?" has a directly testable form: force the wider
width and measure. Same kernel, same data, 100,000 × 128-dim = 48.8 MiB (past this CPU's 12 MB L3,
like the real scan), 7 trials each, interleaved A/B/A/B:

| build | width chosen | `zmm` regs | median ns/distance, 3 runs |
|---|---|---|---|
| `-march=native` | 32 byte | 0 | 81.05 · 89.49 · 79.87 |
| `-march=native -mprefer-vector-width=512` | 64 byte | 88 | 83.85 · 86.62 · 76.86 |

**1.2% apart on the mean of the medians, against 13–43% spread within a single run.** The difference
is an order of magnitude smaller than the noise: there is no effect here to measure, let alone one
to hand-write intrinsics for. The reason is the same one Phase 5 arrived at from the other
direction — at a working set that exceeds L3, this loop is **memory-bandwidth bound, not ALU
bound**, and a wider multiply-add does not make the cache lines arrive faster. Widening the vector
unit optimises the half of the loop that was never the constraint.

**Decision stands, now on evidence rather than expectation: no hand-written AVX2.** The honest
interview answer is the one this was written to earn — *"I checked the vectorisation report, both
distance kernels were already AVX2+FMA with nothing missed, and I measured that forcing AVX-512 made
no difference because the loop is bandwidth-bound"* — and it is a better answer than intrinsics
would have been, because it comes with the measurement that says why.

**Caveat on provenance, since D10 is strict about this:** the vectorisation report reproduces from
the repo with the command above, but the A/B timing came from a standalone probe compiled against
`include/veccore/distance.hpp`, not from `bench`, so **it has no record in `results/bench.jsonl`.**
It is a compiler-configuration finding rather than an index benchmark, and it is labelled as one.

---

## D14 · Growth is the index's job, not the caller's · **ACCEPTED** · 2026-08-27

**Decision.** `ConcurrentHnsw` gains a constructor over a **mutable** `VectorStore` and an
`insert(const float*)` / `insert_batch(...)` pair that append to the store and link the new node into
the graph **inside one exclusive section**. `HnswIndex` grows its own per-node arrays (`grow_to`) and
`search` tops up an undersized `SearchScratch`. The original const-store constructor is untouched, so
every existing caller keeps the smaller, cheaper contract.

**Why this was not a binding-layer concern, which is where it looks like it belongs.**
`VectorStore::add` appends to a `std::vector<float>`, so it **may reallocate**. Every concurrent
reader is at that moment holding `const float*` into the old buffer — `search_layer` keeps one for
the whole beam search. Moving that buffer under a live reader is a use-after-free that **does not
crash**: it reads freed-but-still-mapped memory and returns k plausible, wrong neighbours. The append
therefore has to happen where the exclusive lock already is, and the exclusive lock is private to
`ConcurrentHnsw`.

**Rejected: let the binding own the store and call `add()` then `insert(id)` itself.** The obvious
shape, and it cannot be made safe from outside — the two steps are separately locked, so a reader can
enter between them and observe a reallocated buffer. Two correct operations, composed, are not a
correct operation.

**Rejected: reserve capacity up front so `add` never reallocates.** Cheap, and it works right up
until the caller exceeds the bound they guessed. The failure at that point is silent memory
corruption rather than an error, which makes it the worst kind of "works in testing".

**Rejected: make `HnswIndex` own its `VectorStore`.** This would make growth trivial and would
reverse the ownership model the whole repo rests on (D5, and `PLAN.md` §3): the index borrows the
data, which is what lets `bench` build several indexes over one 488 MB store without copying it.
Paying a 488 MB copy per index to simplify one method is the wrong trade.

**The part that was not obvious until TSan said so.** Growth does not only affect the code that
grows. `grow_to` reallocates `levels_`, `links0_` and `upper_offset_` — buffers that had been
fixed-size since construction for five phases, which made `ConcurrentHnsw::index()` handing out an
unsynchronised reference to them *harmless in practice*. It stopped being harmless without being
edited. That is B-13, and the fix is a set of locked accessors (`capacity`, `stats`, `graph_bytes`,
`check_invariants`) with `index()` kept for the after-join case and documented as unsafe.

**The same shape one level down:** a `SearchScratch` sized when the index held n nodes is a latent
out-of-bounds read the moment it holds n+1, because `VisitedList` indexes without a bounds check —
deliberately, since it is the hottest loop in the project. `search` now tops up an undersized scratch
before using it. It is a caller-owned, thread-local object by contract (B-10), so that write races
with nothing.

**What this does not buy.** Inserts still serialise against each other and against all readers — one
lock, coarse and correct (D8). Growth makes the index *incremental*, not *write-scalable*, and the
README says so.

---

## D15 · An index that cannot be saved is an index you rebuild every morning · **ACCEPTED** · 2026-08-27

**Decision.** `HnswIndex` and `ProductQuantizer` get `save`/`load` over a versioned, checksummed
binary format (`include/veccore/serialize.hpp`), exposed through the bindings. HNSW files are
**self-contained** — vectors included. PQ files are **not** — codes and codebooks only.

**Why at all, in one number each.** Building the SIFT1M graph takes **1,139 s**; training PQ
codebooks takes **86.5 s**. Both are pure recomputation of something deterministic. Until this
existed, every restart paid them again, which is the difference between a benchmark and something
a process can actually use.

### The format, and the four ways a binary dump corrupts data without erroring

Writing `reinterpret_cast<const char*>(v.data())` to a file is three lines and works until it does
not. Every header field exists for a specific silent failure:

| Failure | Defence | What it looks like without one |
|---|---|---|
| Wrong file entirely | `MAGIC`, checked first | a graph of garbage ids that still returns k results |
| Format drift between versions | `VERSION`, exact match | every field after the changed one misparses; recall just gets worse |
| Different machine | endian probe + `sizeof` fields | the same bytes mean different numbers; distances are wrong |
| Truncated or corrupted write | FNV-1a checksum, length guards, no-trailing-bytes | a file interrupted at 90% whose header parses cleanly |

**Only the first has an obvious symptom.** The other three produce *plausible* indexes, which on
this project means quietly wrong recall — the same class as B-01, B-07 and P-04. That asymmetry is
the whole argument for a header rather than a raw dump.

**Rejected: portability across architectures.** The format is native-endian and native-width. Making
it portable means byte-swapping every float on read, which costs real time on a 488 MB store to
serve a use case — moving an index from x86 to a big-endian machine — that does not exist here. The
claim made is that a mismatch is **detected**, not that it works. *"I detect it and refuse"* is a
defensible position; silently producing wrong distances is not.

**Rejected: a text or JSON format.** Human-readable is worth a lot for a config file and nothing for
629 MB of adjacency lists, where it would cost an order of magnitude in size and parse time.

**Rejected: no version field, on the grounds that there is only one version.** There is always
exactly one version until there are two, and the transition is the moment you need the field you
did not write. It costs four bytes.

### Two asymmetries that are deliberate

**HNSW saves its vectors; PQ does not.** A graph is a list of ids — without the vectors it indexes
nothing, and re-attaching it to a store the caller supplies requires proving that store is the right
one, which cannot be done cheaply. So HNSW files are self-contained at 629 MB for SIFT1M. PQ is the
opposite case: the **codes are the compressed form**, 30.6 MiB at m=32 against 488 MiB of vectors,
and writing the vectors beside them would inflate the artifact 16x to carry data that only
`search_rerank` needs. A loaded quantizer therefore searches immediately and reranks only if the
caller passes a store — which is already what that method's signature demands.

**The RNG state travels with the file.** Level assignment is a random draw, so an index whose
generator restarts after a load diverges from one that never stopped — silently, and only for nodes
inserted afterwards. Dropping it would reintroduce P-03 (an unseeded RNG making results
unreproducible) one level down, in a place nobody would think to look. `std::mt19937_64` has a
standard stream representation, so this is portable across libstdc++ versions in a way that
`memcpy`-ing the state would not be. `test_serialize.cpp` asserts the property directly: build,
save, load, insert 200 more, and the graph must match one that was never saved.

### What it cost to get right

**B-15**, and it is worth reading because the bug is in the layer this decision did not think about.
`HnswIndex::load` takes the store as an out-parameter precisely so the index can borrow it — and the
doc comment explains that a self-contained `{store; index;}` struct would dangle when moved. Then
`PyHnsw::load` returned by value, and `PyHnsw` **is** that struct. Wrong neighbours first, then a
segfault. A comment on the C++ API does not protect the binding that wraps it; deleted move
constructors do.

**Measured, on SIFT1M with 1,000 queries:** see the README's persistence table. The equivalence
check is not optional and `bench --index persist` refuses to report timings without it — a load that
is fast and wrong is not a result.

---

## Decisions ledger — closed

**Nothing is open. All fifteen decisions are closed as of 2026-08-27**, and this table is kept as the
ledger rather than deleted — a decision log that shows only the answers hides which questions were
genuinely live and for how long.

| # | Question | Blocked | Resolved |
|---|---|---|---|
| ~~D1~~ | ~~Schedule~~ | — | **2026-08-21: Aug 25 hard stop** (actual: Aug 22 / 26 / 27 — `PLAN.md` §0) |
| ~~D2~~ | ~~Build environment~~ | — | **2026-08-21: WSL2 Ubuntu** |
| ~~—~~ | ~~C++ ramp status~~ | — | **2026-08-21: done; §0.5 does not apply** |
| ~~D3~~ | ~~SIFT10K fixture size — confirm at Phase 0~~ | Phase 1 tests | **2026-08-22: confirmed at 10,000.** 140 tests run in seconds; SIFT1M only in `bench` |
| ~~D7~~ | ~~Is the EdgeRAG integration still in scope if Phase 2 runs long?~~ | Phase 6 | **2026-08-27: yes, all three claims shipped.** Crossover at n≈1,189; retrieval layer only, not end-to-end |
| ~~D8~~ | ~~Does concurrent *insert* ship, or only the read-scaling curve?~~ | Phase 5 scope | **2026-08-26: insert shipped.** And it is the reason B-11 was found — a read-only curve would never have exposed the starvation |
| ~~D11~~ | ~~Hand-written AVX2, or is the compiler already doing it?~~ | Nothing — pure stretch | **2026-08-27: not written.** Both kernels already AVX2+FMA, nothing missed, AVX-512 measured worthless |
| ~~D14~~ | ~~Who owns growth when a Python caller adds to a live index?~~ | Phase 6 bindings | **2026-08-27: the index does.** Append and link happen under one exclusive section; B-13 is what the alternative would have cost |
| ~~D15~~ | ~~Does an index that cannot be persisted count as usable?~~ | Phase 7 | **2026-08-27: no.** save/load over a versioned, checksummed format; HNSW self-contained, PQ codes-only |
| — | GIST1M as a second dataset? | Nothing — pure stretch | **Not attempted.** Phase 2 did not land early; §0.3's cut order outranks a stretch goal |
