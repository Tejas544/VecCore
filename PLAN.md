# VecCore — Execution Plan

**Status:** ACTIVE. D1 (schedule), D2 (environment) and the C++-ramp question were resolved
2026-08-21 — see `CONTEXT.md`. D3 is confirmed at Phase 0.
**Window:** **Aug 21 evening → Aug 25, hard stop.** The spec said Aug 19–24; see §0.1 for why.
**Source of truth for scope:** `02_VECCORE.md`. This file is *how*, not *what*.
**Read `WHAT_IS_THIS.md` first if any term below is unfamiliar.**

---

## 0. Honest schedule assessment — read before anything else

### 0.1 Two of the six days are already gone

`02_VECCORE.md` puts VecCore on **Aug 19–24, 6 days, ~28 h**. It is Aug 21 and this directory
contains markdown and no code. QuantKit's `PLAN.md` §0.1 already flagged this fork and recommended
starting VecCore now; that recommendation stands, but the arithmetic has to be redone honestly,
because 6 days is now 3½.

| Option | Consequence |
|---|---|
| **A. Aug 21 eve – Aug 24, hard stop** | Matches the plan of record and protects QuantKit's Aug 25 start. **~24 h.** Requires cutting roughly a third of the spec — see §0.3 |
| **B. Aug 21 eve – Aug 25, hard stop** ← **recommended** | **~30 h.** Costs QuantKit one day, which it buys back by doing its Phase 0 (pure scaffolding, no thinking) in evening slack, exactly as QuantKit's own plan proposes. Lets VecCore keep concurrency, which is the highest-value-per-hour item for an SDE panel |
| **C. Slip past Aug 25** | Rejected. QuantKit is already at or over budget on its own estimate, and it has a hard external dependency (Colab GPU time) that VecCore does not. Do not spend QuantKit's slack twice |
| **D. Abandon VecCore, roll into QuantKit** | Rejected while the portfolio still has no from-scratch systems project in it. This is the one repo an interviewer who does not do ML can grill you on. Revisit only if Phase 2 is not green by end of Aug 23 |

**✅ DECIDED 2026-08-21: option B.** Everything below is costed against ~30 h.

**✅ The Aug 15–18 modern-C++ ramp (`00_FOUNDATIONS.md` §5) is done**, so §0.2's estimates hold as
written and §0.5's inlined-ramp fallback does not apply. Keep §0.5 in view anyway for **Phase 5** —
`shared_mutex`, `atomic`, and memory ordering are the least-exercised corner of that ramp, and one
hour of cppreference before Phase 5 is cheaper than debugging a race you cannot reproduce.

### 0.2 The scope, costed honestly

| Phase | Spec's day | Estimate | Note |
|---|---|---|---|
| 0 · Rails | (day 1) | **2.5–3 h** | Toolchain is not free here — see §1. Do it tonight |
| 1 · Storage, brute force, harness | Day 1 | **4–5 h** | `00_FOUNDATIONS.md` §4: the harness exists before any feature |
| 2 · HNSW | Day 2 | **8–11 h** | **The project.** Widest error bar. Two-thirds of this is the neighbour heuristic and the insert path, not the search |
| 3 · Product Quantization | Day 3 | **4–6 h** | k-means is 1 h; the ADC table and the sweep are the rest |
| 4 · BM25 + RRF | Day 4 | **3–4 h** | Cheapest interview-answers-per-hour in the whole project |
| 5 · Concurrency | Day 5 | **3–4 h** | Read-write locking + the scaling curve. Concurrent *insert* is the expensive half |
| 6 · Bindings, FAISS head-to-head, README | Day 6 | **5–7 h** | Always underestimated. Plots and README are half a day, minimum |
| | | **29.5–40 h** | |

**Against ~30 h available, we are at the optimistic end of the range with zero bug tax.** That is
what §0.3 is for. It is not pessimism; it is the reason this plan will finish something defensible
instead of stopping mid-way through six things.

### 0.3 The cut order, decided now

Decided while calm, so it is not being decided at 2 a.m. on Aug 24.

1. **Cross-encoder reranking** — already gone. The spec cuts it first and it is the right call:
   it is a pretrained model you call, so it demonstrates nothing you own.
2. **gRPC.** FastAPI over the pybind11 module, or nothing.
3. **IVF on top of PQ.** Keep plain PQ + ADC. IVF is a coarse quantizer and inverted lists — one
   more hour, and it only matters at a scale you are not measuring at.
4. **The FastAPI service itself.** ← *this one goes further than the spec.* `02_VECCORE.md` §10
   stops its cut list at IVF and keeps the service. But the **pybind11 module** is what EdgeRAG
   actually calls, and it is the artifact that proves the C++/Python seam works; the HTTP wrapper
   on top of it adds exactly one interview answer ("what does the network layer cost?") which you
   can give from a whiteboard without having built it. If it survives, it is 45 minutes. It is not
   worth 45 minutes taken from Phase 2.
5. **Hand-written AVX2 intrinsics.** Compile with `-O3 -march=native`, *check the compiler
   actually vectorised the distance loop* (`-fopt-info-vec`), and report that. "I measured that
   GCC auto-vectorised it and hand-intrinsics gained nothing measurable" is a better answer than
   most hand-written SIMD, and it is honest.
6. **PQ trained on a 100k subsample instead of the full 1M.** Standard practice anyway; say so.
7. **Concurrent insert.** Last resort. Degrade Phase 5 to a read-only thread-scaling curve under a
   `shared_mutex`, and describe the insert path rather than shipping it.

**Never cut, in any scenario:**

- the brute-force **ground truth** and the recall check against it,
- `bench` — the measurement harness, written in Phase 1 before any index exists,
- the **neighbour-selection heuristic** and the proof (a recall number) that it is working,
- the **`ef_search` sweep** and the recall/QPS curve,
- the **FAISS head-to-head** on the same machine and the same data,
- `BUGS.md`, `CONTEXT.md`, and the README's limitations section.

Those six are what make every other number in the repo believable. Without the FAISS baseline you
do not have a result, you have an anecdote.

### 0.4 The rule that outranks this entire file

Same rule QuantKit records: **DSA wins every conflict.** An unfinished VecCore costs one CV
bullet. A weak October OA costs the season.

### 0.5 If the C++ ramp did not happen

Do **not** stop and do 20 hours of learncpp. Inline it instead — the phases are ordered so each
one needs exactly one new C++ idea, and you learn it against code you are about to write:

| Phase | The one C++ thing it forces you to learn | Budget |
|---|---|---|
| 0 | CMake targets, `-Wall -Wextra`, build types | 45 min |
| 1 | Contiguous layout, `span`-style views, `const` correctness, why `vector<vector<float>>` is a cache disaster | 60 min |
| 2 | RAII and ownership of large buffers, move semantics on return, `reserve` vs `resize` | 60 min |
| 3 | Templates/functors for the distance function, `alignas` | 30 min |
| 5 | `std::shared_mutex`, `std::atomic`, why `seq_cst` is the safe default | 60 min |
| 6 | pybind11's ownership model and the GIL | 45 min |

That is ~5 h of reading spread across the build, and every hour of it is immediately spent. It is
a worse way to learn C++ and a better way to finish this project — which is the correct trade with
3½ days left.

---

## 1. The environment — verified today, and it is not what you would assume

I probed the machine before writing this. These are measurements, not guesses.

| Fact | Status |
|---|---|
| `g++` 15.2.0 (MSYS2 UCRT64) | ✅ present |
| `cmake`, `ninja`, `make` | ❌ **none installed** |
| MSVC / Visual Studio | ❌ not installed |
| **`-fsanitize=address,undefined`** | ❌ **fails to link — MSYS2 UCRT64 ships no `libasan`/`libubsan`** |
| WSL2 | ⚠️ installed, but the only distro is `docker-desktop`. No Ubuntu |
| CPU | Intel i5-11400H — **6 physical cores, 12 threads**, AVX2 |
| RAM / disk | 15.6 GB / 180 GB free on `D:` |
| Python for bindings + plots | EdgeRAG's venv: 3.13.6, numpy 2.4.4, matplotlib. **No `faiss`, no `pybind11`** |

**The sanitizer line is the one that matters.** `02_VECCORE.md` §10 lists "memory bugs eat a day"
as a top-three risk and names ASan/UBSan-from-the-first-commit as the mitigation. That mitigation
is *not available* in the default local toolchain, and discovering this on Aug 23 while chasing a
heap corruption in the HNSW insert path would be the single most expensive way to find out. It is
logged as landmine **L-01** in `BUGS.md`.

### The decision that follows: build and measure in WSL2 Ubuntu

| | WSL2 Ubuntu ← **recommended** | MSYS2 UCRT64 | MSVC Build Tools |
|---|---|---|---|
| ASan / UBSan | ✅ | ❌ verified broken | ✅ (ASan only) |
| **TSan** (Phase 5) | ✅ | ❌ | ❌ not supported on Windows |
| `valgrind`/`cachegrind` for the cache story | ✅ | ❌ | ❌ |
| `pip install faiss-cpu` | ✅ reliable | ⚠️ | ⚠️ |
| pybind11 against CPython | ✅ | ❌ MinGW-built extension vs MSVC-built CPython is unsupported | ✅ |
| Setup cost | ~30 min | 10 min (`pacman -S cmake ninja`) | ~1 h, ~6 GB |

WSL2 wins on every axis that this project's risks live on. Setup:

```bash
wsl --install -d Ubuntu
```

then inside Ubuntu: `sudo apt install -y build-essential cmake ninja-build git python3-venv valgrind`.

**Where the files live — this specific layout, for a specific reason:**

- **Source stays at `D:\Placement Projects\VecCore`.** One working copy. Edit in your Windows
  editor, build from WSL. Two copies synced by hand is a bug factory.
- **Build directory lives on the Linux filesystem:**
  `cmake -S "/mnt/d/Placement Projects/VecCore" -B ~/veccore-build -G Ninja`.
  Object files and link steps are the I/O-heavy part; `/mnt/d` goes over the 9p protocol and is
  slow for many small files.
- **Datasets live on the Linux filesystem too**, at `~/veccore-data/`, and are gitignored anyway.
  Index build reads 512 MB of SIFT; reading that over 9p would make your build-time number a
  measurement of the filesystem bridge instead of your code.
- **Mind the space in the path.** Quote it everywhere. It will break at least one shell script or
  CMake invocation; that is landmine **L-02**.

**Fallback if WSL2 install is blocked:** MSYS2 (`pacman -S mingw-w64-ucrt-x86_64-cmake
mingw-w64-ucrt-x86_64-ninja`) for the day-to-day build, plus Colab for sanitizer runs and the FAISS
baseline. This costs you a slow debug loop and — worse — moves the FAISS comparison onto different
hardware, which weakens it. Take it only if you must, and say so in the README.

### The measurement rule, inherited from EdgeRAG

EdgeRAG enforces "no perf number from an untrusted device." VecCore's version, since the CPU *is*
the device:

> **Every published number comes from the same environment: WSL2, the same thread count, the same
> build flags, `Release` with `-O3 -march=native`, ≥5 trials.** FAISS is measured in that same
> environment, in the same session, on the same data. `bench` stamps hostname, CPU model, compiler
> version, flags, git SHA, and thread count into every JSON record. A record without those fields
> does not go in a plot.

**And one laptop-specific rule that matters more than it sounds:** an H-series mobile CPU thermally
throttles. If you run all the VecCore trials and then all the FAISS trials, you are partly
measuring how hot the laptop got. **Interleave them** — A, B, A, B, A, B — and report the spread.
That single sentence in your README is worth more than an extra point of recall.

---

## 2. Decisions needed from you

Full reasoning and rejected alternatives live in `CONTEXT.md`. The three that gate Phase 0:

| # | Decision | Status |
|---|---|---|
| **D1** | Schedule | ✅ **ACCEPTED** — Aug 21 evening through Aug 25, hard stop. QuantKit's Phase 0 moves to evening slack |
| **D2** | Build environment | ✅ **ACCEPTED** — **WSL2 Ubuntu**, source on `/mnt/d`, build dir and data on ext4 |
| **D3** | Dataset | Proposed — **SIFT1M** headline (128-dim, 1M, published ground truth), **SIFT10K as the inner-loop fixture** so tests run in seconds. Confirm at Phase 0 |

D3 is not a formality. `02_VECCORE.md` names SIFT1M; the thing it does not say is that a full
1M-vector HNSW build takes minutes, and a test suite that takes minutes stops being run — which is
EdgeRAG's D1 lesson, learned once already. **Every unit test and every correctness check runs on
10k vectors. SIFT1M appears only in `bench`.**

Decisions D4–D10 (distance metric defaults, graph memory layout, the "from scratch" boundary, the
EdgeRAG integration shape) are recorded in `CONTEXT.md` and do not block starting.

---

## 3. Repo layout

```
VecCore/
├─ CMakeLists.txt              # targets: veccore (lib), veccore_tests, bench, _veccore (pybind11)
├─ include/veccore/
│  ├─ types.hpp                # vec_id_t = uint32_t, dim_t, Metric enum
│  ├─ distance.hpp             # l2_sqr, inner_product — header-only, always inlined
│  ├─ storage.hpp              # VectorStore: flat row-major float buffer + accessors
│  ├─ flat_index.hpp           # exact brute force = ground truth
│  ├─ visited.hpp              # epoch-stamped visited list (see Phase 2)
│  ├─ hnsw.hpp
│  ├─ kmeans.hpp
│  ├─ pq.hpp                   # codebooks + ADC table
│  ├─ bm25.hpp                 # inverted index
│  └─ fusion.hpp               # RRF
├─ src/                        # one .cpp per header that needs one
├─ bindings/veccore_py.cpp     # pybind11 module
├─ tests/                      # doctest; every test runs on ≤10k vectors
├─ bench/
│  ├─ bench_main.cpp           # the C++ timing harness → JSON
│  ├─ sweep.py                 # drives bench across parameter grids
│  ├─ faiss_baseline.py        # FAISS, same data, same machine, interleaved
│  └─ plots.py                 # recall/QPS, PQ Pareto, thread scaling
├─ scripts/fetch_sift.sh
├─ results/                    # committed JSON records — the audit trail
├─ data/                       # gitignored; real data lives at ~/veccore-data
├─ WHAT_IS_THIS.md  PLAN.md  BUGS.md  CONTEXT.md  README.md
└─ 00_FOUNDATIONS.md  02_VECCORE.md    # the specs, kept in-repo
```

Two rules about this layout that are worth defending in an interview:

- **`veccore` the library knows nothing about pybind11, JSON, or FastAPI.** Bindings and serving
  are separate targets that depend on it, never the reverse. This is why the same code can be a
  library, a benchmark, and a service without a single `#ifdef`.
- **`results/` is committed.** Every number in the README traces back to a JSON record with the
  git SHA that produced it. That is the difference between a benchmark and a claim.

---

## 4. Phase gates

A phase is done when its **exit criterion** passes, not when the code compiles. Each gate is a
command that either succeeds or does not. Do not start the next phase with the previous gate red —
on a 3½-day budget, debugging two layers at once is how days disappear.

| Phase | Exit criterion |
|---|---|
| 0 | ✅ **PASSED 2026-08-21.** `asan_probe` aborts with a heap-buffer-overflow report; 22 tests green; `bench` writes a trusted record |
| 1 | ✅ **PASSED 2026-08-22.** Brute force = 1.0000 recall@10 vs the SIFT10K fixture (5 trials). On full SIFT1M vs the *published* ground truth: 0.9995, and every disagreement proven to be an exact distance tie (`scripts/diagnose_recall.py`), which is the real criterion — an exact id match is unachievable where the data ties. `bench` writes stamped JSON records |
| 2 | ✅ **PASSED 2026-08-22.** SIFT10K: 0.9952 @ ef=40. **SIFT1M: 0.9755 @ ef=80, 1,934 QPS — 150× brute force.** Recall monotone in `ef_search` across the whole sweep; level histogram matches 1/M to three decimals; 37/37 tests green under ASan+UBSan |
| 3 | ✅ **PASSED 2026-08-22, one criterion REVISED.** Reconstruction MSE monotone in `m` (39924→24165→10847→3665) ✅. **ADC-only recall at 16× is 0.7186, not the ≥0.80 written here — and that target was wrong, not the code.** FAISS `IndexPQ` on identical data at the same `m` gets **0.7059**, so we are 1.8% *ahead* of the reference implementation at 0.89× its QPS. The 0.80 was an estimate written before measuring. Replacement criterion, evidence-based: recall within 5% of FAISS at matched `m`, and **ADC + exact rerank top-100 ≥ 0.95** — measured 0.9656 at 32× and 0.9987 at 16× |
| 4 | ✅ **PASSED 2026-08-22.** BM25 matches hand-derived arithmetic to 1e-9 (the expected value is derived in the test from the formula, not pasted from a run). On EdgeRAG's real 362-doc / 650-query corpus: **BM25 beats TF-IDF +4.17% relative at recall@5** and is **11× faster**. RRF measured *negative* (0.1862 vs BM25's 0.1923) — investigated rather than reported flat, see B-08 |
| 5 | ✅ **PASSED 2026-08-22.** 4-thread read QPS **3.37×** single-thread (in-cache) / **2.84×** (out-of-cache). TSan clean over all 7 concurrency tests, **and `tsan_probe` proves TSan actually fires** (L-01's lesson, second sanitizer). Found and fixed two real concurrency defects: B-10 (a `const` method writing shared state) and **B-11 (4 readers starve the writer completely — insert p99 4,446 ms → 0.79 ms after the fix)** |
| 6 | FAISS and VecCore numbers in one table, produced by one interleaved script run |

---

## Phase 0 — Rails · ~2.5–3 h · **tonight, Aug 21**

No algorithms. This is the phase that makes the next three days cheap, and it is the phase that
gets skipped by people who then lose a day on Aug 23.

**0.1 · Environment (45 min).** WSL2 Ubuntu per §1. Verify: `g++ --version`, `cmake --version`,
`ninja --version`, and — critically — that ASan actually works:

```bash
printf 'int main(){int*p=new int[4];p[5]=1;delete[] p;}' > /tmp/t.cpp && g++ -fsanitize=address,undefined -g /tmp/t.cpp -o /tmp/t && /tmp/t
```

It must print a heap-buffer-overflow report. **If it does not, stop and fix that before writing
any VecCore code.** This is the check that would have caught L-01 four days earlier.

**0.2 · Start the SIFT1M download now, in the background (5 min to start).** It is ~168 MB
compressed from the TEXMEX corpus and everything in Phase 1 blocks on it. Kick it off first, then
do the rest of Phase 0 while it runs. Write `scripts/fetch_sift.sh` to fetch, verify size, and
extract to `~/veccore-data/sift/`. Also carve out the first 10,000 base vectors as the SIFT10K
fixture (D3) and compute its ground truth with a 20-line numpy script — you need a fixture that
does not depend on code you have not written yet.

*File format note, because it costs everyone an hour once:* `.fvecs`/`.ivecs` store each vector as
a 4-byte little-endian `int` dimension count **followed by** the values. The dimension is repeated
for every single vector. If your reader assumes a flat array you will get garbage that still looks
plausible — pre-registered as **P-01** in `BUGS.md`.

**0.3 · CMake skeleton (45 min).**

- Targets: `veccore` (static lib), `veccore_tests`, `bench`. Bindings come in Phase 6.
- `Debug`: `-O0 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined`.
- `Release`: `-O3 -march=native -DNDEBUG`. **Not `-Ofast`, not `-ffast-math`** — it reassociates
  floating-point sums, so your distances stop matching the ground truth bit-for-bit and you spend
  an evening hunting a "recall bug" that is a compiler flag. Pre-registered as **P-02**.
- `-Wall -Wextra` clean is a hard requirement, not an aspiration. Fix warnings the day they appear.
- doctest via `FetchContent`, or vendor the single header if the network is unreliable.

**0.4 · Git (20 min).** `git init`, `.gitignore` (`build/`, `data/`, `__pycache__/`, `.venv/`),
GitHub repo `Tejas544/veccore` to match your existing naming, first commit, push.
**Commit at every phase gate at minimum.** `00_FOUNDATIONS.md` §6 is blunt about why: a repo whose
entire history lands on one day is a visible red flag, and hiring managers do look.

**0.5 · Open `BUGS.md` and `CONTEXT.md` (already written — 10 min to read).** The rule: **any bug
costing more than 20 minutes gets an entry, written the same day.** Not on Aug 25. The diagnosis
is only interesting while it is fresh.

---

## Phase 1 — Storage, brute force, and the harness · ~4–5 h · **Sat Aug 22 morning**

`00_FOUNDATIONS.md` §4 is explicit: the harness comes before the features. It is also the phase
that produces the ground truth everything else is graded against, so nothing after this is
trustworthy if this is wrong.

**1.1 · `VectorStore` — the memory-layout decision that the whole project rests on.**

One `std::vector<float>` of size `n × d`, **row-major**, vector `i` starting at offset `i * d`.
Not `std::vector<std::vector<float>>`.

Be able to say why, with numbers, because it is spec question 13: a vector-of-vectors is `n`
separate heap allocations scattered across the address space, so every vector access is a pointer
dereference to an unpredictable location — a guaranteed cache miss, no hardware prefetching, plus
24 bytes of `std::vector` header per row. The flat version streams linearly, the prefetcher sees
it coming, and a 128-dim float vector is 512 bytes = exactly 8 cache lines with nothing wasted.
**Then measure it** — keep a deliberately naive `vector<vector<float>>` variant behind a compile
flag and A/B the brute-force scan. The spec calls this "the best performance story in the
project," and it costs 20 minutes.

**1.2 · Distance functions.** `l2_sqr` and `inner_product`, header-only, taking `const float*`
pairs and a dimension.

- **Never take the square root.** `sqrt` is monotonic, so it cannot change a ranking; skipping it
  saves a cycle-expensive instruction per comparison. Standard practice — say "squared L2" out
  loud so it is clear it is deliberate.
- Write them as simple loops first and let `-O3 -march=native` vectorise. Verify with
  `-fopt-info-vec-optimized` that it actually did. Templated functors, not `std::function` — a
  virtual or `std::function` call per distance computation would dominate the loop.

**1.3 · `FlatIndex`.** Exact top-k. Bounded max-heap of size k, not a full sort: `O(n log k)`
instead of `O(n log n)`, and at k=10 vs n=1M that is not a rounding error.

**1.4 · Ground truth + recall.** Verify `FlatIndex`'s top-10 on SIFT10K against the published
ground truth **exactly**. Then `recall_at_k(returned, truth)`. Watch for ties: when distances tie
at the k-th position, different implementations break them differently, and a "recall 0.999" that
is really a tie-break difference will waste your time. Compare **distances**, not just IDs, when
investigating.

**1.5 · `bench` — the harness.** Per `00_FOUNDATIONS.md` §4, non-negotiably:

- **Warmup**: discard the first 10–20 queries.
- **`steady_clock`**, timed per query, not amortised across a batch — you cannot get p99 from an
  average.
- **p50 / p95 / p99**, never a mean alone.
- **≥5 trials**, report std-dev, and **interleave** compared configurations (§1).
- **Peak RSS** — read it from `/proc/self/status` `VmHWM`, or `getrusage(RUSAGE_SELF).ru_maxrss`.
- One **JSON record per run** into `results/`, stamped with git SHA, CPU, compiler, flags, thread
  count, dataset, and every index parameter. Append incrementally so a crash at trial 4 does not
  destroy trials 1–3.
- Emit a markdown table from the JSON. You will paste it straight into the README.

**Gate:** brute force matches ground truth exactly on SIFT10K; `bench` writes a valid record.
**Commit.**

---

## Phase 2 — HNSW · ~8–11 h · **Sat Aug 22 afternoon → Sun Aug 23**

The project. Budget the most time here and protect it from every other phase.

Read Malkov & Yashunin ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)) **Algorithms 1–5
line by line** first — about 90 minutes. They are pseudocode you can transcribe almost directly.
Do not start typing before you have read Algorithm 4 twice; it is the one that decides whether
this works.

**2.1 · Layout — decide before writing the insert path.**

```
adjacency : std::vector<uint32_t>   // flat, stride = (M_max + 1)
                                    // slot 0 = neighbour count, slots 1.. = neighbour ids
levels    : std::vector<uint8_t>    // per node
upper     : per-level flat arrays, same scheme, stride (M + 1)
```

`M_max = 2 * M` at layer 0 (the paper's `M_0`), `M` above. Flat arrays with a fixed stride, not
`vector<vector<uint32_t>>` per node — same argument as §1.1, and this is spec question 5.

**2.2 · Level assignment.** `level = floor(-ln(uniform(0,1)) * mL)` with `mL = 1 / ln(M)`. Seed the
RNG from a config field and **record the seed in every JSON record**. Same data + different seed =
a different graph = different recall. An unseeded RNG makes every benchmark unreproducible, and it
looks like a real bug for hours. **P-03.**

**2.3 · Search (Algorithms 2 and 5).** Greedy descent through the upper layers with `ef = 1`, then
a beam search at layer 0 with `ef = ef_search`.

**The single most common implementation bug in HNSW is heap polarity.** You need two heaps with
*opposite* orderings:

- `candidates` — **min-heap by distance**: pop the closest unexplored node next.
- `W` (the working result set) — **max-heap by distance**: the *farthest* kept result sits on top,
  because that is the one you evict when a better candidate arrives, and its distance is the
  threshold that terminates the loop.

Get these the same way round and the search still runs, still returns k results, and recall is
quietly terrible. Note that `std::priority_queue` is a **max**-heap by default, so the min-heap is
the one needing `std::greater`. **P-04.**

**2.4 · The visited set — the cheap 5–10× that costs 15 minutes.** Do not use
`std::unordered_set` per query; hashing and allocating per query dominates the search. Use an
**epoch-stamped array**: a `std::vector<uint16_t>` of size `n`, plus a per-query `epoch` counter.
A node is visited iff `stamp[id] == epoch`. Increment `epoch` per query — no clearing at all. When
`epoch` wraps at 65535, zero the array once. This is what hnswlib does, and "I profiled the
visited-set allocation and replaced it with epoch stamping" is a genuinely good interview beat.

**2.5 · Insert (Algorithm 1) — where most of the time goes.** In order:

1. Draw the level; if it exceeds the current max, this node becomes the new entry point.
2. Greedy-descend from the entry point down to `level + 1`.
3. From `level` down to 0: beam-search with `ef_construction`, select `M` neighbours **with the
   heuristic**, link both directions.
4. **After linking backwards, if a neighbour now exceeds `M_max`, re-run the heuristic on its list
   and shrink it.** Skipping this step is silent: the graph still works, degree grows unbounded,
   memory and latency inflate, and nothing ever errors. **P-06.**

**2.6 · Algorithm 4, the neighbour-selection heuristic. Read this twice.**

For each candidate `e` in increasing distance order, keep it **only if `d(e, q) < d(e, r)` for
every `r` already kept**. In words: reject a candidate that is closer to a neighbour you already
have than it is to you — it is redundant, and the link it would consume is better spent on a
diverse direction.

Naive top-M instead produces tight clusters with no bridges between them; greedy search enters one
and cannot leave, because every neighbour is worse while the true answer sits in another cluster.
Symptom: **recall plateaus around 0.6–0.8 no matter how high you push `ef_search`.** That last
clause is the diagnostic — if raising `ef_search` does not raise recall, the problem is the graph,
not the search. This is spec question 2 and the best answer in the project.

Implement the plain heuristic first. `extendCandidates` and `keepPrunedConnections` are refinements;
add them only if recall is short and time allows.

**2.7 · Verify, in this order.** Do not skip ahead:

1. 100 vectors, `M=4`: dump the graph, check by hand it is connected and symmetric.
2. SIFT10K, `ef_search = 200`, k=10 → recall vs Phase 1 ground truth. **Expect ≥ 0.98.** If it is
   0.6–0.8, go straight to §2.6. If it is near 0, you have a heap-polarity or entry-point bug.
3. Sweep `ef_search` ∈ {10, 20, 40, 80, 160, 320} → the curve should rise monotonically. A
   non-monotonic curve means measurement noise or an unseeded RNG (§2.2).
4. Only then scale to SIFT1M.

**2.8 · The deliverable plot.** recall@10 on x, QPS on y, one point per `ef_search`, in
ann-benchmarks format. Sweep `M ∈ {8, 16, 32}` and `ef_construction ∈ {100, 200}` if time allows —
one curve per `M`.

**Gate:** recall@10 ≥ 0.95 on SIFT10K at some `ef_search`, verified against ground truth.
**Commit, and write up anything from §2.7 that cost you over 20 minutes.**

---

## Phase 3 — Product Quantization · ~4–6 h · **Mon Aug 24 morning**

**3.1 · k-means (Lloyd's).** Per subspace: `d/m` dimensions, 256 centroids.

- **k-means++ initialisation** — pick the first centroid at random, then each subsequent one with
  probability proportional to squared distance from the nearest existing centroid. Uniform random
  init converges to visibly worse codebooks and it is 15 lines.
- **Cap at 25 iterations.** Convergence is not the goal; a good-enough codebook is.
- **Handle empty clusters.** If a centroid claims no points, its new position is 0/0 = NaN, the NaN
  poisons every subsequent distance, and everything downstream returns garbage without a single
  error message. Re-seed an empty cluster from the largest cluster's farthest point. **P-07.**
- Train on a 100k subsample (§0.3 cut 6) — standard practice, and say so.

**3.2 · Encode.** Vector → `m` bytes, one per subspace. Store codes contiguously, stride `m`.

**3.3 · ADC — the part that gets asked about.** Per query, build a `m × 256` float table of
"distance from this query's subvector to each centroid". For m=8 that is 8 KB — **it fits in L1**,
which is exactly why this is fast. Then the distance to any code is `m` loads and `m-1` adds. No
multiplies in the inner loop at all.

Say why asymmetric beats symmetric without hesitating: quantizing the query too would add error on
the second side of the comparison for no speed gain, since the table is built once per query and
amortised over the whole scan.

**3.4 · Verify.** Reconstruction error must fall monotonically as `m` rises. Then recall@10 of
PQ-ADC against Phase 1 ground truth.

**3.5 · The Pareto plot.** Sweep `m ∈ {4, 8, 16, 32}`; plot recall@10 vs bytes-per-vector, with
latency as the point size or a third panel. **Report compression against the honest baseline:**
128 dims × 4 bytes = 512 B/vector raw. m=8 → 8 B → **64×**, but state the recall it lands at.
A compression ratio quoted without its recall is exactly the overclaiming `WHAT_IS_THIS.md` §10
criticises other people for.

**Gate:** monotone reconstruction error; recall@10 ≥ 0.80 at 16× compression. **Commit.**

---

## Phase 4 — BM25 + RRF · ~3–4 h · **Mon Aug 24 afternoon**

Cheapest phase per interview answer in the project, and the one that unlocks the EdgeRAG
integration (§Phase 6.2).

**4.1 · Inverted index.** `term → [(doc_id, term_freq), ...]`, plus per-document length and the
corpus average length. Tokenise simply — lowercase, split on non-alphanumerics. Do not add a
stemmer; it is out of scope and it changes the numbers under you.

**4.2 · BM25.** `k1 = 1.2`, `b = 0.75`.

```
idf(t)   = ln( 1 + (N - df + 0.5) / (df + 0.5) )
score(d) = Σ over query terms  idf(t) · ( tf · (k1+1) ) / ( tf + k1·(1 - b + b·len/avglen) )
```

Use the `1 + ...` (Lucene) form of the IDF. **The textbook form without the `1 +` goes negative for
terms appearing in more than half the documents**, which on a small corpus means a matching term
can *lower* a score. Real, confusing, and pre-registered as **P-08**.

Be able to say what each parameter does: `k1` sets how fast term-frequency saturates (`k1 = 0`
means presence-only); `b` sets how hard long documents are penalised (`b = 0` disables length
normalisation entirely).

**4.3 · Verify against arithmetic, not against vibes.** Three toy documents, compute the expected
score by hand, assert to 6 decimals. This is the whole gate, and it is why this phase does not
generate a bug — it is a formula with no state.

**4.4 · RRF.** `score(d) = Σ_retrievers 1 / (60 + rank_d)`. Four lines.

For "why not weighted score blending?": the two score distributions are incomparable — cosine sits
in [-1,1], BM25 is unbounded and corpus-dependent — so any fixed weight is fitted to today's
corpus and silently wrong on tomorrow's. RRF discards magnitudes and keeps only ordering, so it
needs neither normalisation nor tuning. (`k = 60` is from the original paper and is famously
insensitive; that insensitivity is the point.)

**4.5 · Measure the lift.** Hybrid recall vs dense-only vs sparse-only, on the same queries. Report
all three. If hybrid does not beat both, **say so and investigate** — on some datasets it does not,
and reporting a null result honestly is a stronger signal than a lift you cannot explain.

**Gate:** hand-computed BM25 matches to 6 decimals. **Commit.**

---

## Phase 5 — Concurrency · ~3–4 h · **Mon Aug 24 evening / Tue Aug 25 morning**

Highest value per hour for an SDE panel of anything left. Protect it if you can.

**5.1 · Locking.** `std::shared_mutex`: readers take `shared_lock`, the inserter takes
`unique_lock`. Start coarse — one lock for the whole index. Coarse and correct, with a measured
statement of where it hurts, beats fine-grained and racy every single time, and that sentence is
itself the interview answer.

Then be able to describe (whether or not you build it) the finer-grained design: a per-node link
lock plus an atomic entry point, which is what hnswlib does. Know the hazard it creates —
insertion mutates a *neighbour's* adjacency list, so a concurrent reader can observe a
half-updated list and walk off into a node that is momentarily unreachable.

**5.2 · Thread scaling.** Read-only QPS at **1, 2, 4, 6, 8, 12** threads. **Choose these numbers
deliberately, because your CPU has 6 physical cores and 12 hyperthreads** — expect near-linear to
4, tailing off by 6, and 8–12 giving well under linear because hyperthreads share execution units
and the distance loop is already saturating them. Predicting that shape *before* you measure it and
then showing the plot is a much stronger story than the plot alone.

Also: this is a laptop. Sustained all-core load thermally throttles. Report the spread across
trials and say so.

**5.3 · TSan.** Rebuild with `-fsanitize=thread` and run concurrent read + insert. TSan and ASan
cannot be combined; two separate builds. This is the check that only exists because of §1's
environment decision.

**5.4 · Incremental insert.** Adding vectors to a live index without a full rebuild. Make sure the
entry point update is atomic with respect to readers.

**Gate:** 4-thread QPS ≥ 3× single-thread; TSan clean. **Commit.**

---

## Phase 6 — Bindings, baseline, integration, README · ~5–7 h · **Tue Aug 25**

This phase is always underestimated. It is also the phase that decides whether anyone believes the
previous five.

**6.1 · pybind11 (~1 h).** `pip install pybind11`; `find_package(pybind11 CONFIG)` using
`pybind11-config --cmakedir`. Expose `build`, `add`, `search`, `save`, `load`. Take numpy arrays as
`py::array_t<float, py::array::c_style | py::array::forcecast>` — the `c_style` flag is what stops
a transposed or non-contiguous array from being silently reinterpreted as garbage. **P-34.**

**Release the GIL around the C++ search** (`py::gil_scoped_release`). Without it, your Phase 5
thread-scaling story evaporates the moment the caller is Python, and an interviewer who knows
pybind11 will ask.

**6.2 · The FAISS head-to-head (~1.5 h) — the thing you must not cut.** `pip install faiss-cpu`,
same WSL2 environment, same data, same k, **same thread count** (`faiss.omp_set_num_threads(1)`
against your single-threaded number — FAISS defaults to all cores and comparing that to your
single thread would be a straightforwardly dishonest benchmark, and an obvious one). One script,
**interleaved A/B/A/B**, emitting both sides into one JSON.

Compare: `IndexFlatL2` (validates your ground truth), `IndexHNSWFlat` (your real comparison), and
`IndexIVFPQ` if PQ landed. Report recall@10, QPS at matched recall, p99, peak memory, build time.

**Expect to lose on build time — probably by a lot.** FAISS has years of optimisation in its
construction path. Report it, explain it (single-threaded insert, no SIMD-optimised heuristic
evaluation), and move on. A table where you win everything is a table nobody believes.

**6.3 · EdgeRAG integration (~1.5 h).** Read `WHAT_IS_THIS.md` §13 before starting this — the
obvious framing is wrong and the honest one is better. EdgeRAG's `RetrievalIndex` protocol is
already the seam; its docstring already names VecCore. Deliver:

1. `VecCoreIndex` implementing `search(image_space_query, query_text, k) -> list[str]`, recall
   identical to `FlatIndex` on the 362-document corpus. The claim is **no regression**.
2. **TF-IDF → BM25**, measured on EdgeRAG's 650 held-out queries. This is a real quality delta on a
   real corpus, and it does not need scale to be meaningful.
3. A **crossover plot**: brute force vs HNSW latency against corpus size on synthetic data, with
   EdgeRAG's n=362 marked on it. This converts "HNSW is faster" — which is false at their scale —
   into "here is exactly when it becomes true," which is both true and more interesting.

**6.4 · README (~2 h).** The structure from `00_FOUNDATIONS.md` §6, in that order: one-line thesis,
architecture diagram (Mermaid — 20 minutes), benchmark table with the FAISS baseline, the plots
(recall/QPS, PQ Pareto, thread scaling, crossover), design decisions with rejected alternatives,
**known limitations**, how to reproduce.

The limitations section is not a disclaimer, it is a feature. Name at minimum: no deletes (tombstone
design described but not built), single-node and memory-resident only, no filtered search, PQ
codebooks trained on a subsample, whatever got cut from §0.3, and the fact that build time loses to
FAISS.

**6.5 · The CV bullet.** Fill in `02_VECCORE.md` §8 with the numbers you actually measured. Every
blank filled from a JSON record in `results/`, none from memory.

**Gate:** one table containing both VecCore and FAISS, produced by one interleaved run. **Commit.**

---

## 7. Definition of done

Ordered by what an interviewer checks first.

- [ ] **HNSW recall@10 ≥ 0.95** on SIFT1M, verified against brute-force ground truth
- [ ] **recall/QPS curve** from an `ef_search` sweep, ann-benchmarks format
- [ ] **FAISS head-to-head**, same machine, same data, matched thread count, interleaved
- [ ] **PQ Pareto plot**, compression ratios always quoted with their recall
- [ ] **BM25 + RRF** with the hybrid lift measured against both single-signal baselines
- [ ] **Thread-scaling curve** with the 6-core/12-thread shape explained
- [ ] **pybind11 module** importable from EdgeRAG's venv, GIL released
- [ ] **EdgeRAG integration**: no-regression drop-in, BM25 quality delta, crossover plot
- [ ] `BUGS.md` with **real entries**, including one you can tell as a story for five minutes
- [ ] `CONTEXT.md` with every decision and its rejected alternative
- [ ] README: diagram, table, ≥4 plots, limitations, reproduction steps
- [ ] `-Wall -Wextra` clean; ASan + UBSan clean; TSan clean if Phase 5 shipped
- [ ] Commits spread across every day of the build, not one bulk push
- [ ] You can answer all 19 questions in `02_VECCORE.md` §7 cold

---

## 8. Risk register

| Risk | Likelihood | Mitigation | Trigger to act |
|---|---|---|---|
| **Time — 3½ days for a 6-day spec** | Certain | §0.3 cut order, decided in advance | Phase 2 not green by end of Aug 23 → cut Phases 4 and 5, go straight to Phase 6 |
| **HNSW recall silently poor** | High | Ground truth exists before HNSW does; §2.7's ordered verification | Recall < 0.9 and flat in `ef_search` → Algorithm 4, immediately |
| **Heap polarity / visited-set bugs** | High | §2.3 and §2.4 pre-registered as P-04, P-09 | Recall near 0 → check heap comparators before anything else |
| **Memory bugs** | Medium | ASan/UBSan from commit 1 — **only works if §1's environment decision is made tonight** | Any unexplained wrong answer → rerun under ASan before theorising |
| **Toolchain rabbit hole (WSL, CMake, pybind11)** | Medium | Timeboxed to Phase 0's 45 min; MSYS2 + Colab fallback in §1 | 90 min gone with no compiling binary → take the fallback |
| **C++ ramp thinner than believed at Phase 5** | Low | §0.5's inlined ramp, applied to one phase | Reaching for `memory_order_relaxed` to look sophisticated → stop, use `seq_cst`, read for an hour |
| **Benchmarks unreproducible** | Medium | Seeded RNG, stamped JSON records, interleaved trials, ≥5 runs | Any non-monotonic recall curve |
| **FAISS install fails on WSL** | Low | It is a `pip install` on Linux; that is one reason §1 chose Linux | Any failure → Colab, and label the table as cross-machine |
| **Scope creep into IVF/SIMD/gRPC** | Medium | §0.3 says no | Wanting to write intrinsics before the README exists |

---

## 9. Reading, scheduled against the phases

Just-in-time, per `00_FOUNDATIONS.md` §2. Total ~7 h, all of it inside the build window.

| When | What | Time |
|---|---|---|
| Before Phase 2 | **Malkov & Yashunin**, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320) — Algorithms 1–5, line by line. Then Pinecone's HNSW explainer for intuition | 2 h |
| During Phase 2 | Re-read **Algorithm 4** whenever recall disappoints. Every time | 0 |
| Before Phase 3 | **Jégou, Douze & Schmid** (TPAMI 2011) §§II–IV; then the **FAISS wiki** — "Guidelines to choose an index" and "Faiss indexes", which are better written than most papers | 2 h |
| Before Phase 4 | **Robertson & Zaragoza**, *The Probabilistic Relevance Framework*, §3 only. **RRF**: Cormack et al., SIGIR 2009 — two pages | 1 h |
| Before Phase 5 | cppreference on `std::shared_mutex`; `00_FOUNDATIONS.md` §5 | 1 h |
| Before Phase 6 | [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) — how the field plots recall/QPS. Match their format | 1 h |

**One rule about hnswlib.** `02_VECCORE.md` §2 permits reading it for C++ idiom. Read it *after*
your own version works, or not at all. Reading it while stuck converts a bug you would have
diagnosed — and could have told a story about in December — into a line you copied.
