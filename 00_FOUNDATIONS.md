# 00 — Foundations (read this first)

Shared prerequisites for all three projects. Read once, refer back. Everything here is
cross-cutting; project-specific learning lives in files 01–03.

---

## 1. The Depth Framework

Every technology below is tagged **L1 / L2 / L3**. This is the answer to your core question
("to what depth?"), so internalise the scale:

| Level | Means | Interview test |
|---|---|---|
| **L1 — Use** | You can call the API and know what it does. Internals are a black box. | *"What did you use for X?"* → name it, one sentence on why. |
| **L2 — Explain** | You can whiteboard how it works, its tradeoffs, and when it fails. You could not implement it from scratch today. | *"How does X work? Why X and not Y?"* → 3–5 minutes of coherent explanation. |
| **L3 — Build** | You can implement it from a blank file with no reference. | *"Walk me through your implementation. What happens if...?"* → 15+ minutes, whiteboard code, handle curveballs. |

**The governing rule — this is the whole answer to "how do I own it":**

> **L3 for anything you personally wrote. L2 for anything you used. L1 for anything adjacent.**

You do not need L3 on PyTorch's autograd engine. You *absolutely* need L3 on your own paged
allocator, your HNSW graph, and your quantizer. An interviewer's job is to find the boundary
where your understanding stops. If that boundary sits *outside* what your CV claims, you pass.
If it sits *inside*, the project becomes a liability and you'd have been better off without it.

**Corollary:** if you run out of time, cut *scope*, never *depth*. A smaller project you own
completely beats a larger one you half-understand. Every time.

---

## 2. Learn Just-In-Time, Not Up-Front

You have 17 days. Do **not** spend the first week studying. The correct pattern is:

```
Hit a wall → read the specific thing for 60–90 min → implement → move on
```

Only two things get front-loaded, because everything else depends on them:

| Front-load | When | Hours | Why |
|---|---|---|---|
| Transformer internals + KV cache | Aug 8 | ~6 | EdgeRAG starts Day 1 and is built on this |
| Benchmarking methodology (§4 below) | Aug 8 | ~2 | The harness is written before any feature |

The modern-C++ ramp (§5) happens *during* Aug 15–18, just before VecCore starts — not now.

---

## 3. Hardware: Colab Free T4 — Facts and Constraints

**Buy Colab Pro (~$10/month).** Not for a better GPU — for runtime stability. On the free tier
you will lose multi-hour training and benchmark runs to disconnects, repeatedly, during the most
important month of your degree. Ten dollars is not a real decision.

### T4 hard specs — know these, they come up in interviews
- **Turing TU104, compute capability 7.5**
- **16 GB GDDR6, ~320 GB/s memory bandwidth** ← memorise this number, it anchors every roofline argument
- ~65 TFLOPS FP16 tensor core, ~130 TOPS INT8
- **No bfloat16. No FP8.** Both require Ampere (SM 80) or newer.

### What this means for your projects
- **EdgeRAG:** fine. An INT4-quantized 2B VLM plus a vector index fits comfortably in 16 GB.
- **QuantKit:** your INT4 kernel will dequantize INT4→FP16 inside the kernel and use FP16 tensor
  cores. Turing has no Triton-exposed INT4 tensor-core path. **This is still the correct design**
  — decode is memory-bandwidth-bound, so the win comes from moving 4× fewer bytes across HBM, not
  from faster math. Say exactly that in an interview; it's a specific, credible, senior-sounding
  answer that shows you understand *why* the optimisation works.
- **Use `float16`, never `bfloat16`,** in every script. bf16 will silently fall back or error.

### Colab workflow rules
1. Mount Google Drive. Checkpoint every artifact — weights, benchmark JSON, plots — to Drive immediately.
2. Keep all real code in a **GitHub repo**, `git clone` it into Colab. Never let Colab be the only copy.
3. Structure as importable `.py` modules with a thin notebook driver. Notebooks are terrible for
   version control and terrible for interviews. Your repo must read like software, not a lab book.
4. `nvidia-smi` at the start of every session — you don't always get a T4.
5. Long benchmark sweeps: write results incrementally to JSON as they complete, so a disconnect
   at hour 3 doesn't cost you hours 1 and 2.

---

## 4. Benchmarking Methodology — L3 required

This is the single most underrated skill in this entire plan. **Bad benchmarking is the fastest
way to lose an interviewer's trust**, and it is trivially detectable. A senior engineer will spot
a missing warmup in ten seconds and quietly discount every number on your CV.

### The rules
1. **Warmup.** Discard the first 10–20 iterations. First-call CUDA kernel compilation and cache
   warming will otherwise dominate your measurement.
2. **`torch.cuda.synchronize()` before *and* after timing.** CUDA is asynchronous. Without this
   you are timing kernel *launches*, not kernel *execution*. This is the classic beginner error.
3. **Report percentiles, not means.** p50, p95, p99. A mean latency hides exactly the tail
   behaviour that matters in serving.
4. **Report variance.** Run ≥ 5 trials. Show error bars or std-dev. A single number is not a result.
5. **Fix everything you're not measuring.** Same batch size, sequence length, dtype, and GPU clock
   state across compared runs. Note in the README what was held constant.
6. **Always have a baseline.** "11× faster" is meaningless without "than what, measured how."
   Your baselines: HuggingFace `generate()` for EdgeRAG, FAISS for VecCore, FP16 + round-to-nearest
   for QuantKit.
7. **Separate TTFT from throughput.** Time-to-first-token (prefill) and tokens/sec (decode) are
   different regimes with different bottlenecks. Conflating them is a red flag.
8. **Memory:** report `torch.cuda.max_memory_allocated()`, and reset the peak counter between runs.

### The harness comes first
Before any feature work on any project, write `bench.py`. It should emit a JSON record per run
and a markdown table. Every subsequent commit gets measured against it. This ordering is not
optional — it's what turns a hobby project into an engineering result.

---

## 5. Modern C++ Ramp (for VecCore) — ~20 focused hours

**Why C++ is the right choice here, confirming your preference:** VecCore's core is an in-memory
index — the work is memory layout, cache behaviour, and tight distance-computation loops. That is
C++'s home turf, and it's what every real vector database (FAISS, hnswlib, Milvus) is written in.
Go would be the better pick if the project were primarily a network service; it isn't. C++ also
signals systems seriousness to an SDE panel in a way Go does not, and — importantly — **it
double-counts with your DSA prep**, which you're doing in C++ anyway.

### The gap you actually have
You know **competitive-programming C++**: STL containers, algorithms, basic OOP, pointers. That is
a genuinely different language from **systems C++**. Here's precisely what to add:

| Topic | Depth | Why VecCore needs it |
|---|---|---|
| RAII, `unique_ptr` / `shared_ptr`, ownership | **L3** | Index owns large buffers; no leaks, no double-frees |
| Move semantics, rvalue refs, `std::move` | **L2** | Returning large vectors/graphs without copying |
| `const` correctness, references vs pointers | **L3** | Basic hygiene; interviewers notice its absence |
| Memory layout: struct padding, cache lines, contiguous vs pointer-chasing | **L3** | **The single biggest perf lever in HNSW.** Pointer-chasing a graph is a cache disaster; flat arrays with offset indices are 3–10× faster |
| Alignment (`alignas`, aligned alloc) | **L2** | Required for SIMD distance kernels |
| `std::thread`, `std::mutex`, `std::shared_mutex`, `std::atomic` | **L2** | Concurrent reads during inserts |
| Memory ordering (relaxed/acquire/release) | **L1–L2** | Know it exists, know why `seq_cst` is the safe default |
| Templates (basic) | **L2** | Distance functors, float/int8 vector types |
| CMake | **L1–L2** | Build the thing. Copy a good template, understand the pieces |
| SIMD intrinsics (AVX2) for L2/inner-product distance | **L2** *(optional, high value)* | 4–8× on the hottest loop; a great interview talking point |
| Sanitizers (`-fsanitize=address,undefined`), `perf` | **L2** | Finding your own bugs; shows engineering maturity |
| pybind11 | **L1** | Python bindings so the rest of the stack can call it |

### Resources (in priority order)
1. **[learncpp.com](https://www.learncpp.com/)** — chapters 12–17, 19, 21–22. The standard free
   reference. Skim what you know, read ownership/move/memory carefully. ~8 hrs.
2. **Scott Meyers, *Effective Modern C++*** — Items 1–6 (type deduction, `auto`), 17–22
   (smart pointers), 23–30 (move semantics). Read only these. ~5 hrs.
3. **[hnswlib source](https://github.com/nmslib/hnswlib)** — ~2000 lines of readable C++.
   **Read it to learn idiom, then close it and write your own.** Do not copy. You will be asked
   line-level questions about your implementation.
4. **CppCon on YouTube** — "Back to Basics: RAII", "Back to Basics: Move Semantics". ~2 hrs.
5. **[Agner Fog's optimization manuals](https://www.agner.org/optimize/)** — reference only, for
   cache and SIMD questions.

**Timing:** Aug 15–18, ~2 hrs/day alongside EdgeRAG. Not now.

---

## 6. Repository Hygiene

Applies to all three repos. This is cheap and disproportionately effective — it's the difference
between a hiring manager spending 20 seconds and 3 minutes on your GitHub.

- **Commit daily**, with real messages. A repo whose entire history lands on Aug 24 is a visible
  red flag to anyone who checks — and hiring managers do check the code behind projects you highlight.
- **README structure**, identical across all three:
  1. One-line thesis
  2. Architecture diagram (Excalidraw or Mermaid — 20 minutes, huge payoff)
  3. **Benchmark table** with baseline comparison
  4. **2+ plots** (matplotlib) — this is what makes someone stop scrolling
  5. Design decisions & tradeoffs
  6. **Known limitations** ← signals more maturity than any other section
  7. How to reproduce
- Type hints on Python, `-Wall -Wextra` clean on C++.
- Tests for the core logic. Not full coverage — just proof you know how.
- Pin all three repos on your GitHub profile. Consistent naming.

---

## 7. What "Owning It" Actually Looks Like in December

For each project you should be able to, cold, with no notes:

1. **Draw the architecture** on a whiteboard in under 3 minutes.
2. **Name the single hardest bug** and how you diagnosed it. *(Keep a `BUGS.md` in each repo as
   you build — you will not remember these in December. This is the highest-ROI habit in this
   entire document.)*
3. **Justify every design choice** against the alternative you rejected, with a reason that
   isn't "it was easier."
4. **Quote your numbers** and explain the measurement methodology behind them.
5. **State where it breaks** at 100× scale, and what you'd do about it.
6. **Explain what you'd do differently** with another month.

Point 2 is the one candidates fumble most and the one interviewers weight most heavily. Start
`BUGS.md` on day one.
