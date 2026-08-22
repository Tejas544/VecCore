# VecCore — Bug Log

Per `00_FOUNDATIONS.md` §7: *"name the single hardest bug and how you diagnosed it"* is the question
candidates fumble most and interviewers weight most heavily. **You will not remember these in
December. Write the entry the day the diagnosis is fresh.**

**Rule:** any bug costing more than 20 minutes gets an entry, written the same day.

**Why this file matters more here than on most projects.** An approximate index has no correct
output — it has a *recall number*. A wrong heap comparator, a missing bidirectional link shrink, or
a neighbour heuristic implemented as naive top-M all produce an index that builds without error,
searches without error, returns exactly `k` results, and is **quietly 30% worse than it should be.**
Nothing crashes. Nothing warns. **The default failure mode of this project is a plausible wrong
answer**, which is why the diagnosis story is usually more interesting than the fix — and that
story is what the interview is actually about.

---

## The answer, if you only read one section

> *To be written once there is a real answer.* Leave it empty rather than filling it with something
> minor — this section exists to hold the single hardest bug, and pre-filling it with a placeholder
> that reads like a diagnosis is exactly the failure mode described in P-30.

---

## Confirmed bugs at a glance

| # | Symptom | Class | Cost |
|---|---|---|---|
| B-01 | `TopK` kept the wrong id when distances tied | silent recall loss | ~5 min |
| B-02 | Fetch script hung 10 min on a blocked port, writing 0 bytes | measurement that measured nothing | ~20 min |

**Phase 0 scorecard:** 7 landmines defused, 0 confirmed bugs, ~1.5 h. Four of the seven
(L-01, L-02, L-05, L-06) were found by checking a tool *before* depending on it rather than after.
Two (L-03, L-07) came from reading output that was easy to scroll past — a docstring in someone
else's code, and a warning phrased in the future tense.

---

## Entry format

```
### B-nn · <one-line symptom> · <date> · <phase>
**Symptom:**      What you observed. The observation, not the cause.
**Wrong theory:** What you believed first, and why it was plausible. ← keep this; it is the most
                  interesting part of the story in an interview.
**Root cause:**   What was actually wrong.
**Diagnosis:**    How you found it. The *method*, not the answer.
**Fix:**          Commit SHA.
**Prevention:**   The test that now exists so it cannot come back.
**Cost:**         Hours.
```

Classes worth labelling, because they are the categories an interviewer will recognise:
**silent recall loss** · **measurement that measured nothing** · **cache-hostile layout** ·
**data race** · **uninitialised / undefined behaviour** · **format misread** ·
**baseline mismatch** · **unreproducible by construction**

---

## Confirmed bugs

### B-01 · The tie-break half of the heap comparator was backwards · 2026-08-22 · Phase 1

**Symptom:** `TopK` returned ids `{7, 9}` where `{3, 7}` was correct, for three candidates offered
at identical distance with k=2. Every test involving *distinct* distances passed.

**Wrong theory:** none, and that is the point — the test named the failure precisely enough that
there was no theorising to do. It cost about three minutes.

**Root cause:** `WorseFirst` builds a max-heap whose `top()` must be the *worst* result, i.e. the
eviction candidate. The distance clause was right (larger distance = worse = greater). The tie
clause returned `a.id > b.id`, which makes the *largest* id compare as "best" and therefore puts
the *smallest* id on top — so eviction threw away the id that should have been kept.

**Diagnosis:** the failing test was written in the same commit as the code, specifically to pin
tie-breaking, because P-12 predicts that ties at rank k produce recall like 0.997 and get
misdiagnosed as an algorithm bug.

**Fix:** `a.id < b.id`. Commit `see git log`.

**Prevention:** `TEST_CASE("TopK breaks ties on the smaller id, deterministically (P-12)")`.

**Cost:** ~5 minutes.

**Why a five-minute bug gets an entry.** Because it is **P-04's exact failure class** — comparator
polarity in a bounded heap — showing up in the simplest possible setting, where a hand-checkable
test caught it instantly. In Phase 2 the same mistake lives inside HNSW's search loop, where there
is no hand-checkable answer and the only symptom is that recall is quietly bad. The transferable
lesson is not "check your comparator", it is: **when a comparator has two clauses, the second one
is the one that will be wrong, and only a test with a deliberate tie will find it.** That test now
exists, and the same test shape gets written for HNSW before the search loop does.

---

### B-02 · The fetch script hung indefinitely on a blocked port, writing nothing · 2026-08-22 · Phase 1

**Symptom:** `scripts/fetch_sift.sh` ran for ten minutes producing `sift.tar.gz.part` at **0 bytes**,
no output, no error, no progress bar.

**Wrong theory:** that it was simply slow. A 168 MB download on a connection that had just pulled
142 MB of apt packages at 6 MB/s should have finished in under a minute, so "slow" stopped being
plausible quickly — but the script gave nothing to distinguish "slow" from "dead".

**Root cause, two layers.** The outer one: `ftp.irisa.fr` is the canonical SIFT distribution and it
is **FTP only**; outbound port 21 is blocked on this network, so the TCP connect never completed.
The inner one, which is the actual bug in our code: `wget` was invoked with **no connect timeout**,
so a blocked port presents as an indefinite hang rather than a failure.

**Diagnosis:** `curl -sSI --connect-timeout 8` against the FTP URL returned
`Operation timed out after 8002ms with 0 bytes received`, while an HTTP request to the same host's
web server returned `206`. Host reachable, protocol blocked.

**Fix:** the canonical source is now tried first with `--connect-timeout 15 --max-time 1800` so it
fails fast, then falls back to an HTTPS mirror. Downloads show a progress bar.

**Prevention:** every network call in this repo carries an explicit timeout.

**On trusting the mirror**, because swapping in a third-party source for your benchmark corpus is
not a small thing: the three files match the exact expected byte counts (516,000,000 / 5,160,000 /
4,040,000 = `n * (4 + 4d)` for SIFT1M's known geometry), the reader validates every record's
dimension (P-01), and — the strongest check — **Phase 1's gate is itself an authenticity test**:
brute force over the base vectors must reproduce the *published* ground truth exactly. Data passing
that is either real SIFT1M or an internally consistent forgery, and the latter is not a threat
model that applies to a benchmark corpus.

**Cost:** ~20 minutes, most of it spent believing the download was working.

**Lesson, and it is the same one as L-07 wearing different clothes:** a network call without a
timeout is not "patient", it is *unable to report failure*. Ten minutes of silence and zero bytes
carried exactly as much information as an instant error would have — except the instant error also
says what went wrong.

---

## Defused landmines

Things that would have cost hours, caught before they did. **These count.** "I checked whether my
tooling could actually catch memory bugs before I started writing code that would need it" is a
real engineering answer, and it is the kind of thing nobody thinks to claim because it never became
a war story.

### L-01 · The local toolchain cannot run the sanitizers the plan depends on · 2026-08-21 · Phase 0 (planning)

**Found by:** probing the toolchain before writing the plan, rather than assuming it.

```
$ g++ -fsanitize=address,undefined -g t.cpp -o t
collect2.exe: error: ld returned 1 exit status
```

MSYS2 UCRT64 ships **no `libasan` and no `libubsan`**. `-fsanitize=address,undefined` compiles and
then fails at link.

**Why this mattered.** `02_VECCORE.md` §10 lists "memory bugs eat a day" as a top-three risk and
names exactly one mitigation: *"ASan/UBSan on from the first commit, not after the bug appears."*
That mitigation was unavailable on the default local toolchain. The realistic path was: build
Phase 2's insert path, hit a heap corruption on Aug 23, reach for ASan, discover this then, and
lose the afternoon to a toolchain migration under time pressure — with a live memory bug.

**Resolution.** Build and measure in **WSL2 Ubuntu** (`PLAN.md` §1), where ASan, UBSan **and TSan**
all work — TSan matters independently, because it does not exist on Windows at all and Phase 5 is
concurrent. Phase 0 now contains an explicit check that ASan reports a *deliberately planted*
heap overflow before any VecCore code is written.

**Generalised lesson worth stating in an interview:** verify your debugging tools work *before* you
need them. A sanitizer you discover is broken while debugging is worse than no sanitizer, because
you have already committed to a plan that assumed it.

### L-02 · The repository path contains a space · 2026-08-21 · Phase 0 (planning)

`D:\Placement Projects\VecCore` → `/mnt/d/Placement Projects/VecCore`. CMake handles it; shell
scripts, `Makefile` recipes, and `pkg-config` invocations frequently do not, and the failure is
usually a confusing "no such file: /mnt/d/Placement" rather than anything that names the real
cause. Quote every path expansion. Prefer `"${VAR}"` over `$VAR` everywhere, including in
`scripts/fetch_sift.sh`.

### L-03 · The obvious EdgeRAG integration claim would not have survived a follow-up question · 2026-08-21 · Phase 6 (planning)

The natural plan was "swap `FlatIndex` for HNSW, measure the speedup." Reading EdgeRAG's code
first showed **its corpus is 362 documents** — a scale at which HNSW is *slower* than brute force,
because you pay graph traversal to avoid work that costs microseconds. And `DEFAULT_ALPHA = 0.0`
with a documented measurement behind it: EdgeRAG's dense image signal is effectively noise, so its
retrieval is pure TF-IDF text matching today.

Presenting "we made EdgeRAG's retrieval faster with HNSW" to anyone who then asked "at what corpus
size?" would have been unrecoverable. The replacement plan — no-regression drop-in, a TF-IDF→BM25
quality delta on 650 real held-out queries, and a *measured* crossover curve with n=362 marked on
it — is three defensible claims instead of one indefensible one. See `WHAT_IS_THIS.md` §13.

### L-04 · The baseline is not installed · 2026-08-21 · Phase 6 (planning)

EdgeRAG's venv has numpy, torch, and matplotlib, but **no `faiss` and no `pybind11`**. Both are
Phase 6 hard dependencies, and Phase 6 is the last day. `pip install faiss-cpu pybind11` inside
WSL during Phase 0, while the SIFT download runs — not on Aug 25, when a wheel resolution failure
would take the FAISS comparison out of the project, and `PLAN.md` §0.3 lists that comparison as
never-cut.

### L-05 · The system drive is full, and it presented as a WSL bug · 2026-08-21 · Phase 0

**Symptom.** `wsl --install -d Ubuntu` reported *"Distribution successfully installed"* and then
*"The distribution failed to start. Error code: 6, failure step: 2 — Wsl/Service/CreateInstance/E_FAIL."*
Retrying gave a different error, `Wsl/Service/E_UNEXPECTED — Catastrophic failure`, which is the
kind of message that invites an hour of searching for WSL-specific fixes.

**Wrong theory (mine, briefly).** A WSL platform problem — stale kernel, disabled Virtual Machine
Platform, or a bad `.wslconfig`. All three were checked and all three were fine: WSL 2.5.9.0,
kernel 6.6.87.2, `VirtualMachinePlatform` and `Microsoft-Windows-Subsystem-Linux` both enabled, no
`.wslconfig` present.

**Root cause.** `C:` had **0.3 GB free of 225 GB**. WSL could register the distribution but could
not start a VM with nowhere to write.

**Diagnosis method — the transferable part.** Two error codes for the same action, neither of them
specific, is a signal that the failing component is *not* the one reporting the error. So instead
of searching the error string, enumerate the resources the operation needs and check each: CPU
features, OS features, config files, **disk**. Disk was the fourth thing checked and should have
been the first, because it is the cheapest to check and the most commonly exhausted.

**Why this counts as defused rather than survived.** A system drive at 0.3 GB free would not have
stayed a WSL problem. It would have resurfaced as: the SIFT download failing at 90% into a temp
directory, `pip install faiss-cpu` failing to unpack a wheel on the last day of the project
(L-04's exact failure, from an unrelated cause), and Windows itself degrading during the most
important month of the placement season. Finding it on Phase 0 evening cost twenty minutes.
Finding it on Aug 25 would have cost the FAISS comparison.

**Resolution.** Free headroom on `C:` (`powercfg /h off` reclaims the 6.3 GB `hiberfil.sys` in one
reversible command), then reinstall Ubuntu onto `D:` with `wsl --install -d Ubuntu --location
D:\WSL\Ubuntu`. The distro belongs on `D:` regardless of the disk situation — `~/veccore-build` and
`~/veccore-data` live inside it, and `D:` has 179 GB free against `C:`'s nothing.

**Guard for the rest of the project.** `bench` should record free disk space alongside the CPU and
compiler stamps (D10). A benchmark run that fails or slows because the disk filled mid-sweep is
otherwise indistinguishable from a code regression.

### L-06 · Windows line endings would have broken the shell scripts inside WSL · 2026-08-21 · Phase 0

**Found by:** git printing `warning: LF will be replaced by CRLF the next time Git touches it` on
the first commit, and taking the warning seriously instead of scrolling past it.

**The trap.** D2 puts the source tree on Windows and the toolchain inside WSL — which is the right
call for every reason listed there, and it creates exactly one hazard: `core.autocrlf` is `true`
globally on this machine, so a fresh `git clone` (or any `git checkout`) writes `scripts/fetch_sift.sh`
with CRLF endings. Bash then reads the shebang as `#!/usr/bin/env bash\r` and reports:

```
/usr/bin/env: 'bash\r': No such file or directory
```

That message names neither the file nor the cause, and the file looks completely normal in every
editor. It is a reliable twenty-minute detour.

**Why it had not fired yet.** The working copy was written with LF, so everything ran locally. The
failure would have appeared on the first `git clone` — most likely on Aug 25, on a fresh machine or
after a reset, with no obvious connection to a commit made four days earlier.

**Fix.** `.gitattributes` with `* text=auto eol=lf`, plus explicit `binary` for `*.fvecs`/`*.ivecs`
so dataset files are never line-ending translated — that second half matters more than it looks,
because CRLF translation of a binary vector file corrupts it silently and it would present as
P-01, a completely unrelated diagnosis.

**Generalised lesson:** a warning that names a future tense — *"will be replaced the next time"* —
is describing a bug you have not hit yet. Those are the cheapest ones to fix.

### L-07 · The stamp reported "unknown" instead of the reason it was unknown · 2026-08-22 · Phase 0

**Symptom.** The first clean `bench` run on a Release build printed
`git : unknown (dirty)` and refused to write a record, citing *"git SHA could not be resolved."*
Git was installed, the repository was right there, and `git log` worked fine from Windows.

**Root cause.** Git's `safe.directory` protection. The repo lives on `/mnt/d`, so from inside WSL
its ownership does not match the calling user, and git refuses to operate on it:

```
fatal: detected dubious ownership in repository at '/mnt/d/Placement Projects/VecCore'
```

This is a direct and predictable consequence of D2's cross-boundary layout — source on Windows,
toolchain in Linux — and it will hit every git-touching tool in the project, not just the stamp.

**Fix.** `git config --global --add safe.directory '/mnt/d/Placement Projects/VecCore'` inside WSL.
One path, git's own prescribed remedy.

**The interesting half, and the reason this is an entry rather than a footnote.** The fix took
thirty seconds. Finding it took several minutes *only because the harness threw away the answer.*
`capture_env` ran git with `2>/dev/null`, so when git failed it recorded `git_sha: "unknown"` and
stopped. That is a true statement which hides the entire diagnosis — git had already printed both
the cause and the exact command to fix it, and the code discarded it on the floor.

`run()` now captures stderr into a `git_note` field, and `untrusted_reason()` surfaces it, so the
refusal message names the real problem instead of the symptom.

**Generalised lesson, and it is the one worth saying out loud in an interview:** *`2>/dev/null` on
a command whose failure you handle gracefully is a decision to be told less than you were offered.*
Graceful degradation and silent degradation are not the same thing. A field that can be missing
should always carry *why* it is missing.

**Worth noting as a design validation, not a bug.** `bench` did the right thing here — it refused
to write a record it could not attribute to a commit, rather than writing one with a blank
provenance field. That refusal is exactly what D10 rule 7 asks for, and it fired on its first real
encounter with a problem.

---

## Predicted failure modes

Written before the build. Each is a specific, named thing that these implementations get wrong,
with the symptom you will actually observe and the guard that catches it. **When something breaks,
read the triage table below first — most of what will go wrong is already on this list.**

### Triage: recall is wrong, what do I check?

| What you observe | Check first | Why |
|---|---|---|
| Recall ≈ 0.0 | Heap comparators (P-04), then the entry point (P-18) | Search is walking the wrong direction or starting nowhere |
| Recall plateaus at 0.6–0.8 **and rising `ef_search` does not help** | **Algorithm 4 — the neighbour heuristic (P-05)** | Flat in `ef_search` means the *graph* is wrong, not the search. This is the signature symptom |
| Recall high on small data, collapses at 1M | Bidirectional shrink (P-06), `size_t` offsets (P-14) | Degree explosion and index overflow are both scale-dependent |
| Recall is exactly 1.0 | You are querying with base vectors (P-19) | The query set must be the held-out one |
| Recall non-monotonic in `ef_search` | Unseeded RNG (P-03), or too few trials | Rebuilding between points changes the graph underneath the sweep |
| Recall slightly off, like 0.997 | Distance ties at rank k (P-12) | Not a bug. Compare distances, not IDs |
| Recall fine, results still wrong downstream | ID mapping between VecCore and the caller (P-20) | Internal index ≠ external doc key |

### Data and build — Phase 0

**P-01 · The `.fvecs` reader.** SIFT files store, per vector, a little-endian `int32` dimension
count **followed by** `d` floats — the dimension is repeated for *every* vector, not written once
as a header. Reading it as a flat float array yields data that is wrong but numerically plausible
(no NaNs, no crash), so recall comes out low and looks like an algorithm bug. **Guard:** assert
every per-vector dimension equals 128, and that `filesize == n * (4 + 4*128)`.

**P-02 · `-Ofast` / `-ffast-math` silently changes your answers.** They permit reassociating
floating-point sums, so the distance loop produces slightly different values than the ground truth
was computed with, and near-tied neighbours reorder. Recall drops by a fraction of a percent for no
visible reason. **Guard:** `Release` is `-O3 -march=native -DNDEBUG` and nothing else. If you ever
want `-ffast-math`, make it a separate build and a separate row in the results table.

**P-14 · `int` overflow on the offset arithmetic.** `i * d` for the flat store: fine at 1M×128
(128M), overflows `int32` around 2.1B elements. **Guard:** `size_t` for every offset computation,
never `int`. The failure mode at 1M is nothing; the failure mode at the scale you would brag about
is silent memory corruption.

### Storage, ground truth, and the harness — Phase 1

**P-10 · Recall computed against a misaligned ground truth.** SIFT's `groundtruth.ivecs` is indexed
by *query* position and holds *base* vector IDs. Off-by-one on either axis, or truncating to the
wrong k, produces a stable, believable, wrong recall number that poisons every phase after it.
**Guard:** brute force must match the published ground truth **exactly** on SIFT10K before HNSW
exists. That is Phase 1's gate for exactly this reason.

**P-11 · The harness measures the wrong thing.** No warmup, so the first query's page faults
dominate. Or the query vector copy is inside the timed region. Or you time a batch and divide,
which makes p99 mathematically unobtainable. **Guard:** `00_FOUNDATIONS.md` §4's checklist, applied
literally. Sanity check: single-query latency × QPS should be ≈ 1 for the single-threaded case. If
it is not, one of the two is measuring something else.

**P-12 · Distance ties at rank k look like a recall bug.** Multiple vectors at identical distance
from the query; you return one, ground truth lists another. Recall 0.997 and hours of hunting.
**Guard:** when investigating any recall shortfall, print the *distances* of your results and the
truth's, not just the IDs. If the distances match, you are done — it is a tie, not a bug.

**P-13 · Memory numbers that include the world.** Peak RSS covers the dataset, the ground truth
arrays, and the Python interpreter if you measured through the bindings. Reporting that as "index
memory" inflates your number and the comparison against FAISS becomes meaningless. **Guard:** report
index memory as a computed quantity (`n*d*4 + adjacency bytes + code bytes`) *and* peak RSS, as two
separate fields, and say which is which.

### HNSW — Phase 2, where most of the risk lives

**P-03 · Unseeded RNG makes the whole project unreproducible.** Level assignment is random. Rebuild
between two points of an `ef_search` sweep and you have swept two different graphs; the curve comes
out non-monotonic and you debug the search for an hour. **Guard:** seed from config, stamp the seed
into every JSON record, build the index **once** per sweep.

**P-04 · Heap polarity — the classic.** `candidates` must be a **min**-heap (explore closest first);
the result set `W` must be a **max**-heap (evict the farthest, and its distance is the termination
threshold). `std::priority_queue` is a max-heap by default, so `candidates` is the one that needs
`std::greater`. Reversed, the search still runs and still returns `k` results — recall is just
terrible. **Guard:** a unit test on 100 vectors where the exhaustive answer is known by hand.

**P-05 · Naive top-M instead of Algorithm 4.** The most consequential silent bug in the project.
Keeping the M closest neighbours creates tight clusters with no bridges; greedy search enters one
and cannot escape. **Signature symptom: recall plateaus at 0.6–0.8 and raising `ef_search` does
not move it.** **Guard:** implement both, A/B them once on SIFT10K, and keep the plot — that
comparison is a strong README figure and an even better interview answer.

**P-06 · The backward link shrink gets skipped.** After linking `q → neighbours`, each neighbour
gets `q` added to *its* list. If that list now exceeds `M_max`, it must be re-pruned **with the
same heuristic**. Skip it and node degree grows without bound: memory inflates, cache locality
degrades, latency rises — and nothing errors. **Guard:** assert `degree(v) <= M_max` for every node
after every build, in debug.

**P-09 · The visited set.** Two variants. (a) `std::unordered_set` per query — correct but slow
enough to dominate the search; you will conclude HNSW is slow when your data structure is.
(b) Epoch stamping done wrong: forget to increment the epoch and every query after the first sees
everything as already visited, returning almost nothing. Forget the wrap-around at 65535 and one
query in every 65k is silently broken — which is *worse*, because it is nearly invisible in an
aggregate recall number. **Guard:** unit-test the visited list directly, including a forced wrap.

**P-18 · Entry point not maintained.** When an inserted node draws a level above the current
maximum, it must become the new entry point. Miss it and the upper layers are never entered, so
the hierarchy does nothing and you get flat-NSW performance while believing you built HNSW.
**Guard:** log the level histogram after a build; it must look geometric, and the max level must
match the entry point's level.

**P-15 · `mL` wrong.** `mL = 1 / ln(M)`. Use `1/M` or `ln(M)` and the layer distribution is badly
shaped — too many layers (wasted descent) or too few (no long-range structure). **Guard:** the same
level histogram. With M=16, roughly 94% of nodes should be level 0 only.

**P-16 · Self-loops and duplicate neighbours.** A node linking to itself, or the same neighbour
twice, wastes degree budget and can trap greedy descent. **Guard:** assert no self-reference and no
duplicates in debug builds after each insert.

**P-17 · `ef_search < k`.** Then the result set physically cannot hold `k` items and you return
fewer, which reads as catastrophic recall. **Guard:** clamp `ef = max(ef_search, k)` and say so in
the docstring.

**P-19 · Querying with base vectors.** If your query set is drawn from the indexed vectors, every
query finds itself at distance 0 and recall@1 is trivially 1.0. Use SIFT's held-out `query.fvecs`.
**Guard:** be suspicious of any recall ≥ 0.999. It is more likely a leak than a triumph.

**P-20 · Internal ID vs external key.** VecCore indexes by dense `uint32_t`; EdgeRAG wants document
key strings. One off-by-one in the mapping and recall inside VecCore is perfect while EdgeRAG's
answers are nonsense. **Guard:** a round-trip test — `key → id → key` for every document.

### Product Quantization — Phase 3

**P-07 · k-means empty cluster → NaN.** A centroid claiming zero points gets `0/0` on the mean
update. The NaN propagates through every subsequent distance, `NaN < x` is false so comparisons
silently do nothing sensible, and the index returns arbitrary results with no error anywhere.
**Guard:** assert `std::isfinite` on every centroid after every iteration. Re-seed an empty cluster
from the farthest point of the largest cluster.

**P-21 · `d` not divisible by `m`.** 128/8 = 16, fine; 128/12 is not. Integer division silently
drops trailing dimensions from the encoding, so recall degrades in a way that looks like a
quantization-quality issue rather than a bug. **Guard:** hard `assert(d % m == 0)` at construction.

**P-22 · Squared vs non-squared mismatch in the ADC table.** Phase 1 uses squared L2 (no sqrt). If
the ADC table is built with unsquared distances, the sums are wrong in a way that still ranks
*almost* correctly — recall drops a few points and looks like normal PQ error. **Guard:** one
assertion: PQ distance for a vector encoded with `m = d` (one dimension per subspace, 256 centroids)
should closely match exact squared L2.

**P-23 · Training the codebooks on the query set.** Or on the full base set and then reporting
recall on the same vectors. Both inflate results. **Guard:** train on a subsample of *base*
vectors, evaluate on the held-out *query* vectors, and state the subsample size in the README.

**P-24 · Codes stored wider than a byte.** The entire premise is "256 centroids ⇒ one byte." Store
them in an `int` vector and you have written a 4× memory regression while reporting a 64×
compression ratio from a formula. **Guard:** the reported compression number must come from
`sizeof` on the actual buffer, never from arithmetic on paper.

### BM25 and RRF — Phase 4

**P-08 · Negative IDF.** The textbook form `ln((N - df + 0.5)/(df + 0.5))` goes **negative** for
terms appearing in more than half the documents. On a small corpus — like EdgeRAG's 362 — that
means matching a common term actively *reduces* a document's score. **Guard:** use the Lucene form,
`ln(1 + (N - df + 0.5)/(df + 0.5))`, which is strictly positive. It is one character and it is
`PLAN.md` §4.2.

**P-25 · Tokenizer drift between index time and query time.** Lowercase one side only, or split on
different characters, and terms simply never match. Recall craters and the formula is blameless.
**Guard:** exactly one tokenizer function, called from both paths. Not two functions that agree
today.

**P-26 · `avgdl` computed over a different document set than the lengths.** Or lengths measured
pre-filtering and `avgdl` post-filtering. The length normalisation term goes subtly wrong for every
document. **Guard:** compute both in the same pass, in the same function.

**P-27 · RRF rank base.** Ranks 0-based vs 1-based changes `1/(60+0)` to `1/(60+1)`. It barely
moves results — which is the problem: two runs disagree slightly and neither is reproducible from
the README. **Guard:** pick 1-based, write it in the docstring, and test it against a hand-computed
three-document example.

### Concurrency — Phase 5

**P-28 · The scaling benchmark that fakes its own result.** All threads issuing the *same* query
means the graph path is hot in L2 after the first, and you measure cache behaviour rather than
concurrency — producing a beautifully linear curve that means nothing. **Guard:** each thread draws
from a disjoint slice of the query set.

**P-29 · Thermal throttling read as a scaling limit.** On a 6-core mobile CPU, sustained all-core
load drops clock speed. The 12-thread point looks bad, you blame lock contention, and you write the
wrong explanation into the README. **Guard:** interleave thread counts across trials rather than
running them in ascending order, report the spread, and — if you can — log clock frequency.

**P-30 · A confident wrong explanation.** The most dangerous failure in this phase, and the one
`00_FOUNDATIONS.md` §7 is really warning about. "It stops scaling at 6 threads because of lock
contention" *sounds* right and may be entirely wrong — it could be memory bandwidth, hyperthread
sharing, or thermal. **Guard:** before writing a cause into the README, test it. Run the same sweep
with the lock removed (unsafe but read-only, so it will not crash) — if the curve is unchanged, it
was never the lock.

**P-31 · Publishing a node before its edges are visible.** The entry point becomes an atomic
pointing at a node whose adjacency writes have not been made visible to other threads. A concurrent
reader lands on a node with a garbage neighbour list. **Guard:** TSan, plus `seq_cst` by default —
`00_FOUNDATIONS.md` §5 explicitly rates memory ordering L1–L2, so *use the safe default and be able
to say why*, rather than reaching for `relaxed` to look sophisticated.

### The FAISS comparison and the bindings — Phase 6

**P-32 · Thread count mismatch.** FAISS uses OpenMP and defaults to every core. Comparing that to
your single-threaded search is a 6× error in FAISS's favour and, more importantly, it is the first
thing a competent interviewer will ask about. **Guard:** `faiss.omp_set_num_threads(1)` for the
single-threaded row, and state the thread count in every row of the table.

**P-33 · Metric or parameter mismatch.** `IndexFlatL2` vs `IndexFlatIP`; FAISS's `efSearch` default
vs yours; FAISS's `M` meaning the same thing as yours (it does, but verify). Comparing your ef=64
against FAISS's default is not a comparison. **Guard:** compare at *matched recall*, not matched
parameters. That is what the recall/QPS curve is for.

**P-34 · Non-contiguous numpy arrays through pybind11.** A sliced or transposed array has a stride
your C++ loop knows nothing about, so it reads the wrong memory — no crash, plausible garbage.
**Guard:** `py::array_t<float, py::array::c_style | py::array::forcecast>` in the signature. It
forces a copy when needed rather than reinterpreting.

**P-35 · Holding the GIL through the C++ search.** Every Phase 5 thread-scaling claim becomes false
the moment the caller is Python, and you will not notice because your C++ benchmark is unaffected.
**Guard:** `py::gil_scoped_release` around the search body, plus one Python-side multi-threaded
smoke test that would fail without it.

**P-36 · A README number with no JSON behind it.** Copied from a terminal during Phase 3, kept
after the code changed in Phase 5. **Guard:** every number in the README traces to a record in
`results/` with a git SHA. If you cannot find the record, delete the number.

---

## Scorecard

Updated at each phase gate. The point is not the count; it is that the pattern is visible when you
are preparing answers in December.

| Phase | Predicted | Hit | Novel | Hours lost |
|---|---|---|---|---|
| 0 · Rails | | | | |
| 1 · Storage + harness | | | | |
| 2 · HNSW | | | | |
| 3 · PQ | | | | |
| 4 · BM25 + RRF | | | | |
| 5 · Concurrency | | | | |
| 6 · Bindings + baseline | | | | |

**"Hit"** = a predicted failure that actually happened. **"Novel"** = something this file did not
anticipate — those are the most valuable entries in the whole document, because they are the ones
that prove the log is real rather than reconstructed.
