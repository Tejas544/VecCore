# What is VecCore? — the no-background explanation

**Read this first. It assumes you know nothing about vector search, databases, or C++.**
No jargon is used before it is defined. Where a term is standard, it is given in **bold** the
first time so you can recognise it later in a paper, a job description, or an interview.

`02_VECCORE.md` is the *specification* — what to build. `PLAN.md` is *how and when*.
This file is *why*, and it is the one you re-read the night before an interview.

---

## 1. The one-sentence version

> VecCore is a **search engine for meaning instead of keywords** — you give it a piece of text or
> an image, and it finds the most similar items out of a million, in under a millisecond, using
> a fraction of the memory that storing them plainly would take.

And the honest second sentence, which matters just as much:

> Tools that do this already exist and are excellent. We are building one from scratch anyway,
> because the *building* is the product here — not the tool.

Section 10 is where that honesty gets unpacked properly. Read it before you claim anything about
this project to anyone.

---

## 2. Start much further back: what is a "vector"?

Forget computers for a second.

Suppose you want a computer to understand that **"dog"** and **"puppy"** are related, but
**"dog"** and **"stapler"** are not. A computer has no idea what any of those words mean. It only
does arithmetic. So the trick the entire field is built on is this:

> **Turn every item into a list of numbers, arranged so that similar items get similar lists.**

That list of numbers is called a **vector**, or an **embedding**. A toy example with 3 numbers:

```
dog      →  [ 0.91,  0.12, -0.44 ]
puppy    →  [ 0.88,  0.19, -0.40 ]     ← very close to "dog"
stapler  →  [-0.32,  0.77,  0.05 ]     ← nowhere near it
```

Now "are these two things similar?" becomes "are these two lists of numbers close together?",
which is just arithmetic — and computers are extremely good at arithmetic.

**Where do the numbers come from?** From a neural network — an **embedding model** — trained on
enormous amounts of text or images. You feed it a sentence, it hands back the list. You never
write those numbers yourself, and nobody can tell you what any individual number "means". They
only mean something *collectively*, as a position in space.

**How long are the lists in practice?** Not 3. Typically **384, 768, 1024, or 1536 numbers**.
Each number is a **dimension**. A vector with 768 numbers is "768-dimensional" — a point in a
768-dimensional space. You cannot picture that, and you do not need to. Every piece of intuition
you have about points being near or far in 2D or 3D still works. (With one big exception, in §3,
which is exactly where this project's difficulty comes from.)

**How do you measure "close"?** Two ways, and VecCore implements both:

- **Euclidean distance (L2)** — straight-line distance, the Pythagoras you already know, extended
  to 768 dimensions instead of 2. Smaller = more similar.
- **Inner product / cosine similarity** — roughly, "do these two arrows point the same way?"
  Bigger = more similar. This is the one most text embedding models expect.

That is the entire mathematical foundation. Everything else in this project is engineering built
on top of those two formulas.

---

## 3. The actual problem: finding the nearest one, fast

Here is the job, stated plainly.

> You have **one million** vectors sitting in memory — one per document, image, product, or
> paragraph. A user query arrives, and it is also turned into a vector. **Find the 10 stored
> vectors closest to the query vector.**

This is called **k-nearest-neighbour search** (**k-NN**), where k = 10.

### The obvious solution, and why it is not enough

Compare the query against all one million, keep the best 10. That is **brute-force** or **flat**
search. It is about fifteen lines of code, it is *perfectly accurate by definition*, and VecCore
implements it on day one — not as a serious contender, but as the **ground truth**: the thing every
clever method gets graded against. You cannot claim your fast method is 95% accurate unless you
have the exact answer to compare it to.

The problem is cost. One query against one million 128-dimensional vectors is roughly
**128 million multiply-and-add operations**, and — more importantly — it has to **read all 512 MB
of vector data out of RAM**. Per query. On a laptop that lands in the tens of milliseconds.

Ten milliseconds sounds fine until you multiply it out:

| Situation | Brute force |
|---|---|
| One query, 1M vectors | ~10–30 ms — tolerable |
| 1,000 queries/second | Needs a rack of machines doing nothing else |
| 1M vectors → 100M vectors | 100× slower, and 50 GB of RAM |
| Inside a chatbot answering in real time | Retrieval alone eats the entire latency budget |

And the cost scales **linearly**: 10× the data is exactly 10× the work, forever. That is the wall.

### Why the classic computer-science answer fails here

Your instinct, correctly trained by DSA prep, is: *build an index.* A sorted array gives binary
search. A tree gives O(log n). Surely there is a tree for this?

There is — **k-d trees**, **ball trees**, **R-trees**. They work beautifully in 2 or 3 dimensions.
They are how map applications find nearby restaurants.

**They collapse above roughly 20 dimensions**, and we have 128 to 1536. This is the **curse of
dimensionality**, and it is worth being able to state out loud, because it is the reason this
entire field exists:

> In high-dimensional space, *everything is roughly equidistant from everything else*. The gap
> between the nearest point and the farthest point shrinks, relative to the distances themselves,
> as dimensions grow. Space-partitioning trees work by ruling out whole regions — "the answer
> cannot be over there" — and in high dimensions you can never rule anything out. The tree
> degenerates into checking almost every point, **plus** the overhead of walking the tree. It ends
> up slower than the brute-force scan it was meant to replace.

So: exact search is either brute force, or something worse than brute force. That is a genuinely
uncomfortable result, and it forces the move that defines the field.

---

## 4. The bargain that makes it work: approximate search

If exact answers are unaffordable, stop demanding exact answers.

> **Approximate Nearest Neighbour search (ANN):** return the *probably* nearest neighbours, 100 to
> 1000 times faster. Accept that occasionally, result #7 should really have been result #9.

This bargain is only reasonable because of what the results are *for*. If you are looking up a
bank balance, 99% correct is worthless. But if you are finding the 10 most relevant paragraphs to
show a language model, or 10 products a shopper might like, then:

- the "correct" answer was never crisp — it came out of an embedding model that is itself an
  approximation of meaning;
- something downstream (a re-ranker, or the language model itself) will re-judge them anyway;
- and a 300× speedup is the difference between a product that ships and one that does not.

**How the trade is measured — this is the single most important idea in the project.** Accuracy is
called **recall@k**: of the 10 truly-nearest items, how many did we return? Return 9 of the true
10 and recall@10 = 0.9. Speed is **QPS** (queries per second) or **latency** (milliseconds per
query).

Every ANN system has a knob that trades one against the other. So a single number is never a
result. **The result is a curve** — recall on one axis, QPS on the other — and systems are compared
by whose curve sits higher. Producing that curve honestly is most of what "benchmarking" means
here, and producing it *dishonestly by accident* is the most common way these projects fail.

Everything below is a different strategy for buying speed with a little accuracy.

---

## 5. Strategy one — HNSW: don't look at everything, follow a trail

**HNSW** stands for *Hierarchical Navigable Small World*. It is the algorithm at the heart of
VecCore, and it is the part of this project that is pure graph algorithms — which is exactly why
it is the right project to have in front of a systems interviewer.

### The intuition: six degrees of separation

You want to get a physical letter to a specific person in Tokyo, and you may only hand it to
someone you personally know. You do not know anyone in Tokyo. But you know someone who travels to
Asia; they know someone in Japan; that person knows someone in Tokyo; that person knows the
recipient. Four hops across the planet.

This is the **small-world property**: in a network where most connections are local but a few are
long-range, any two nodes are a handful of hops apart. Social networks have it. So can a graph
built on top of your vectors — deliberately.

### How the search actually runs

Build a graph where each vector is a node connected to a few dozen of its nearest neighbours.
Then to search:

1. Start at any node.
2. Look at its neighbours. Move to whichever is closest to the query.
3. Repeat.
4. Stop when no neighbour is closer than where you already are.

This is **greedy descent**, and it is the same shape as walking downhill in fog. Instead of
examining a million vectors you examine a few thousand along the path. That is where the
100–1000× comes from.

### The "hierarchical" part

Pure greedy descent has a flaw: if you start far away, you take thousands of tiny local steps to
cross the space — like navigating a country using only village roads.

HNSW's fix is to build **several layers**. The top layer holds a small random sample of nodes with
long-range links — motorways. Below it, a bigger sample with shorter links — A-roads. The bottom
layer holds every node with only local links — village streets. You search top-down: cross the
country on the motorway, exit near your destination, then take local roads to the door.

If you have met **skip lists** in DSA prep, this is precisely that idea generalised to many
dimensions, and saying so in an interview is a strong, cheap signal.

Which layer a node lands on is decided by a **coin flip with exponentially decaying probability** —
most nodes exist only at the bottom, a few reach higher, one or two reach the top. Nobody designs
the hierarchy; it falls out of the random draw. (Which also means: same data, different random
seed, slightly different index. Reproducible benchmarking therefore requires a fixed seed, and
forgetting that is a real and very confusing bug — it is pre-registered in `BUGS.md`.)

### The part that separates a working HNSW from a good one

When inserting a node, you must choose which neighbours to keep. The obvious choice — keep the
`M` closest — **is wrong**, and understanding why is the single best interview answer this project
contains.

If you keep only the closest, every node in a dense cluster links only to others in the same
cluster. The graph fragments into tight islands with no bridges between them. Greedy search walks
into an island and gets stuck, because *every* neighbour looks worse and yet the true answer is
somewhere else entirely. Recall quietly settles at something like 0.6 and nothing crashes.

The fix is a **neighbour-selection heuristic** (Algorithm 4 of the paper): keep a candidate only
if it is closer to *you* than it is to any neighbour you have already kept. This deliberately
retains a few diverse, long-range links instead of a redundant huddle of near-identical ones.
Those links are what make the graph *navigable* rather than merely connected.

**Three knobs you must be able to explain cold:**

| Knob | When it applies | What it controls |
|---|---|---|
| `M` | build time | Neighbours per node. Higher = better recall, more memory, slower build |
| `ef_construction` | build time | How hard you search for good neighbours while inserting. Higher = better graph, slower build |
| `ef_search` | **query time** | How wide a beam to keep while searching. **This is the recall/speed dial you sweep to draw the curve** |

The build-time versus query-time distinction is a standard interview probe, and it is the one
people get wrong under pressure.

### What HNSW costs

Memory. You store every vector at full size **plus** the graph edges — often 30–50% on top. Which
leads directly to the next idea.

---

## 6. Strategy two — Product Quantization: make each vector smaller

**Quantization** means storing numbers less precisely to make them smaller. (This is the same
theme as your QuantKit project, applied to *data* instead of to *model weights* — worth saying out
loud, because a portfolio that shares a thesis reads far stronger than three unrelated repos.)

Raw arithmetic: 1M vectors × 768 dimensions × 4 bytes per number = **3 GB**. At a billion vectors
it is 3 TB, and RAM is the dominant cost line of every vector database on earth.

**Product Quantization (PQ)** compresses by 8–32× with a genuinely clever trick.

### The intuition: postcodes

To describe where someone lives you do not need their exact GPS coordinates. A postcode gets you
within a few hundred metres using a handful of characters. You lost precision; you kept almost all
the useful information.

PQ does this to vectors, in two steps:

1. **Chop the vector into pieces.** A 128-dimensional vector becomes 8 chunks of 16 numbers each.
2. **Give each chunk a postcode.** For each chunk position, cluster all the training data into
   **256 representative chunks** (a **codebook**, learned with **k-means**). Now store, instead
   of 16 real numbers, the single ID of the closest representative — and 256 IDs fit in exactly
   **one byte**. That "256" is not a magic number: it is chosen precisely *because* it is one byte.

A 128-dim vector goes from **512 bytes → 8 bytes. 64× smaller.** Because the chunks are
independent, 8 codebooks of 256 entries can represent 256⁸ ≈ 1.8 × 10¹⁹ distinct vectors — the
combinatorial "product" the name refers to.

### The second trick, which is the one interviewers ask about

You now have compressed data, but the query is still a full-precision vector. Two options:

- **Symmetric:** compress the query too, then compare compressed-to-compressed. Fast, but you have
  now introduced error on *both* sides.
- **Asymmetric (ADC)**: keep the query exact. Before scanning, precompute a small table —
  "distance from my query's chunk 1 to each of the 256 representatives for chunk 1", and so on for
  all 8 chunks. That is 8 × 256 floats: **8 KB, which fits in L1 cache**. Then the distance to any
  stored vector is 8 table lookups and 7 additions. No multiplications at all.

**Asymmetric is both faster and more accurate**, because error is introduced on one side only. That
sentence, delivered without hesitation, is worth having memorised.

### What you give up

Accuracy — you are comparing postcodes, not addresses. In practice this is handled by
**over-fetching and re-ranking**: use PQ to cheaply shortlist 100 candidates, then compute exact
distances on just those 100. Cheap, and most of the accuracy comes back.

---

## 7. The other half of search, which people forget: words

Everything so far is **dense retrieval** — meaning-based. It has a specific, well-known weakness.

Search for `error code X-4417`. An embedding model has never seen that string, has no concept of
it, and will happily return paragraphs about *errors in general*. Meanwhile a 1970s keyword index
nails it instantly, because the exact token is either present or it is not.

**BM25** is that keyword index, and it is still the default baseline in serious information
retrieval after thirty years. It ranks documents by three sensible rules:

1. A document containing the query word more often is more relevant — **but with diminishing
   returns.** The tenth occurrence adds far less than the second. (The `k1` parameter sets how
   fast the returns diminish.)
2. Rare words matter more than common ones. Matching *X-4417* means much more than matching
   *the*. (This is **IDF**, inverse document frequency.)
3. Long documents match more words by luck, so their scores are discounted. (The `b` parameter
   sets how aggressively.)

That is the whole of BM25. It is a formula small enough to write from memory in an interview,
and you should be able to.

### Combining the two: why not just add the scores?

**Hybrid retrieval** runs both and merges. The naive merge — add a dense score to a BM25 score —
is broken, and knowing *why* is a standard senior probe: **the two scores live on incomparable
scales.** Cosine similarity sits in [-1, 1]. BM25 is unbounded and depends on corpus statistics.
Adding them means one silently dominates, and any weight you pick is over-fitted to today's data.

**Reciprocal Rank Fusion (RRF)** sidesteps it completely by throwing the scores away and keeping
only the **ranks**:

```
score(doc) = Σ over each retriever of   1 / (60 + rank of doc in that retriever)
```

Rank 1 contributes 1/61, rank 2 contributes 1/62, and so on. No normalisation, no tuning, no
scale mismatch — a rank is a rank. It is four lines of code, it routinely beats carefully-tuned
score blending, and that combination of trivial and effective is why it gets asked about.

---

## 8. Where is this used in the real world?

Vector search is not a niche. It is the retrieval layer under most of the AI products shipping
right now, plus a lot of things that predate the current wave:

- **RAG (Retrieval-Augmented Generation)** — the reason this is suddenly everywhere. A language
  model cannot know your company's documents. So before answering, you retrieve the handful of
  most relevant paragraphs and paste them into the prompt. **The quality ceiling of the whole
  system is set by the retrieval step** — if the right paragraph is not in the top 10, no model,
  however large, can answer correctly. This is exactly the role VecCore plays under EdgeRAG.
- **Semantic search** in documentation, support portals, and legal or medical archives — where
  users describe a problem in their own words rather than guessing the author's keywords.
- **Recommendations** — "customers who liked this also liked". Items and users become vectors;
  recommendation becomes nearest-neighbour lookup.
- **Image and video search** — reverse image search, visual duplicate detection, face matching.
  These were vector-search problems long before language models were.
- **Deduplication and clustering** — near-identical news articles, plagiarism detection, cleaning
  scraped training data.
- **Anomaly and fraud detection** — a transaction with no near neighbours in normal-behaviour space
  is worth a second look.
- **Long-term memory for AI agents** — every past interaction embedded and retrieved on relevance.

The common shape: *the data is fuzzy, the query is fuzzy, and exact matching cannot express what
the user wants.*

---

## 9. Where does a thing like this actually get deployed?

Four patterns, and it is worth knowing which one you are building, because it changes the design.

**A. Embedded library — in the same process as the application.**
Like SQLite. No network, no server, no serialisation; a function call. Lowest possible latency,
and this is what FAISS and hnswlib primarily are. **This is VecCore's primary form**: a C++ library
with Python bindings, so EdgeRAG can call it directly with no network in the path.

**B. Sidecar service — its own process, called over the network.**
A small HTTP or gRPC server on the same machine or nearby. You pay roughly a millisecond of
network cost and gain independent scaling, a language-agnostic interface, and the ability to
restart the index without restarting the application. **VecCore builds a thin version of this
too**, because "can you put a service in front of it, and do you understand what that costs?" is a
fair interview question.

**C. Dedicated cluster — a real vector database.**
Milvus, Qdrant, Weaviate, Vespa. Sharding across machines, replication, persistence, filtered
search, multi-tenancy, backups. This is where a *product* lives, and it is explicitly out of scope
here — but you must be able to describe what changes, which is what §18–19 of the spec's interview
list is testing.

**D. Managed cloud service.** Pinecone and similar: someone else operates C. You pay per vector
and per query and never think about `ef_search` again.

There is also **E: a column in a database you already have** — `pgvector` for Postgres, or vector
types in Elasticsearch and MongoDB. Not the fastest option, and frequently the *correct* one,
because it means one system to operate instead of two. Being willing to say that out loud is a
maturity signal; candidates who insist the specialised tool always wins get marked down.

**In this project specifically:** VecCore runs on your laptop (an i5-11400H, 6 physical cores,
16 GB RAM) against SIFT1M — a standard public benchmark of one million 128-dimensional vectors
*with published correct answers*, which is what makes recall measurable rather than assertable.
It is not deployed to production, and nothing in the README should imply it is.

---

## 10. Is this problem already solved? — the honest answer

**Yes. Substantially, and by people who have been at it for a decade.**

Anyone evaluating this project already knows that, so pretending otherwise is the fastest way to
lose them. State it first, in your own words:

- **FAISS** (Meta, 2017) — the reference implementation. C++ with Python bindings, GPU support,
  every index type in the literature. It is VecCore's **benchmark, not its dependency**, and the
  spec is emphatic on that point.
- **hnswlib** — a compact, extremely fast header-only HNSW written by the authors of the paper.
- **Milvus, Qdrant, Weaviate, Vespa, Chroma** — full databases built around the same algorithms.
- **pgvector** — ANN inside Postgres, which is how a great many teams should actually do this.
- The core algorithms are published, peer-reviewed, and stable: HNSW is Malkov & Yashunin (2016),
  PQ is Jégou, Douze & Schmid (2011), BM25 dates from the 1990s, RRF is Cormack et al. (2009).

**So the correct framing is not "nobody solved this."** It is: *this is a solved problem with a
deep, well-documented literature, which makes it an unusually good thing to implement from scratch
— because correctness is checkable against a public ground truth and performance is checkable
against a world-class baseline. There is nowhere to hide.* That property is precisely why it is a
good interview project and a bad startup idea.

### What is genuinely *not* solved

Worth knowing, because "where does the field still hurt?" is a question that separates people who
read papers from people who read blog posts:

- **Filtered search.** "Nearest neighbours, but only documents from 2024 owned by this tenant."
  Filter first and you scan too much; search first and your top-10 may all get filtered away.
  There is no clean answer, and every vendor solves it differently and imperfectly.
- **Updates and deletes.** HNSW has no real delete. You mark a node dead (a **tombstone**) and
  rebuild periodically, because physically removing a node severs the links that pass *through* it
  and quietly damages navigability for everyone else. For a live corpus this is a genuine
  operational headache.
- **Memory cost at billion scale.** Still the dominant expense. Disk-based indexes like
  **DiskANN** (Microsoft Research) are the active research direction and far from a settled answer.
- **Recall guarantees.** These systems are approximate with *no bound*. You measure recall on a
  test set and hope production looks like the test set. On drifting data, it may not.
- **The embedding model is usually the real bottleneck.** Teams spend weeks tuning `ef_search`
  when a better embedding model would have moved recall ten times further. Your own EdgeRAG is a
  live example: its image-side embedding was measured to be close to noise
  (`DEFAULT_ALPHA = 0.0`), and no index can fix a signal that is not there.

---

## 11. So why build it? — three reasons, in honest order

**1. It is the project that proves you can do classical systems engineering.** This is the real
reason and it belongs first. EdgeRAG and QuantKit are both ML-flavoured; an interviewer who does
not do ML can grill you on VecCore for forty minutes without saying "model" once. Graph algorithms,
cache-conscious memory layout, concurrency, and honest measurement — the skills every SDE panel
actually tests. `02_VECCORE.md` calls this *"your hedge, and the reason the portfolio doesn't read
as narrow,"* and that is exactly right.

**2. It completes a story that no single project can tell.** EdgeRAG currently ships a brute-force
`FlatIndex`, and its code already contains the seam — the docstring on its `RetrievalIndex`
protocol literally reads *"``VecCore`` implements this later without touching a caller."* Two
projects that compose into one system are worth more than three that sit side by side, because it
demonstrates you can design an interface before you have the implementation. That is a *senior*
behaviour rather than a student one.

**3. The depth is checkable, which is the whole point.** You can be asked "why is your recall 0.94
and not 0.99?" and there is a real, specific, defensible answer involving the neighbour heuristic
and `ef_search`. Compare that to a project where the honest answer is "that is what the library
returned." The spec puts it bluntly: *"I used a library" ends the conversation.*

**What is explicitly not a reason:** that VecCore will beat FAISS. It will not, and the spec says
so plainly — landing *within 2–3×* of FAISS is a strong solo result. Claiming more than you
measured is how a good project turns into a liability in the room.

---

## 12. Who are we solving it for?

Three answers, and being honest about the ordering is itself the point.

**First: for the person reading your CV in December.** This is a portfolio project built under a
deadline for a placement season. Every design decision — build HNSW rather than import it,
benchmark against FAISS rather than against nothing, keep a `BUGS.md` — is optimised for *"can
this candidate defend it under questioning."* Pretending the audience is anyone else would produce
different and worse decisions.

**Second: for EdgeRAG**, which is a real system with a real retrieval layer that is currently a
362-document brute-force scan over TF-IDF vectors. There is a genuine, measurable improvement
available there — and, importantly, *not the one you would guess*. See §13.

**Third, hypothetically: for a team that cannot use a managed service.** On-premise, air-gapped,
regulated, or cost-constrained deployments, where a few-thousand-line embeddable C++ index that
one person fully understands beats a dependency they cannot audit. This is a real category. It is
also not who this build is for, and you should not claim it is.

---

## 13. One thing to be honest about before you start

It is tempting to plan the EdgeRAG integration as "swap in HNSW, watch it get faster." **It will
not get faster, and you should know that today rather than discovering it on the last afternoon.**

EdgeRAG's corpus is **362 documents**. Brute-force scanning 362 vectors takes microseconds. HNSW
over 362 nodes is *slower* — you pay graph-traversal overhead to avoid work that was never
expensive in the first place. HNSW starts winning somewhere in the tens of thousands of vectors.

There is also a subtler fact sitting in EdgeRAG's own code: its dense image-side signal was
measured to be effectively noise, so its retrieval today is **pure TF-IDF text matching**.

So the honest — and considerably more interesting — integration story is:

1. **Correctness and interface:** VecCore implements the existing `RetrievalIndex` protocol as a
   drop-in, with recall identical to `FlatIndex` on the same 362 documents. "No regression" is the
   claim, and it is a real one.
2. **A genuine quality upgrade:** replace TF-IDF with **BM25**, and measure the recall change on
   EdgeRAG's 650 held-out queries. That is a real number on a real corpus, and it does not depend
   on scale.
3. **The scaling argument, measured rather than asserted:** show the crossover point on synthetic
   corpora — the corpus size at which HNSW overtakes brute force — and state where EdgeRAG sits on
   that curve today.

That reframing turns one claim that would not survive scrutiny into three that will. Notice it
only became visible by *reading EdgeRAG's code before planning VecCore's integration*. That habit
is most of what "senior" means.

---

## 14. What "finished" looks like

Not "the code runs." Specifically:

- **An HNSW index whose recall is verified against brute-force ground truth on SIFT1M**, with a
  recall/QPS curve produced by sweeping `ef_search` — the format the whole field uses.
- **A PQ implementation** with a Pareto plot showing recall against memory against latency, so the
  compression claim is a curve rather than a boast.
- **BM25 + RRF hybrid retrieval**, with the recall lift over dense-only measured.
- **A head-to-head against FAISS** on identical data on the same machine: recall@10, QPS, p99
  latency, memory, build time. **Including the numbers where FAISS wins**, with an explanation of
  why. Build time is expected to be one of them.
- **A thread-scaling curve** with a stated explanation of where it stops scaling and why.
- **A `BUGS.md` with real entries** — because "what was the hardest bug and how did you find it?"
  is the question candidates fumble most and interviewers weight most heavily.
- **A README with an architecture diagram, plots, design decisions, and a genuine limitations
  section.** The limitations section signals more maturity than any other part of it.

And the underlying test: on a whiteboard in December, with no notes, you can draw it, justify every
choice against the alternative you rejected, quote your numbers with the methodology behind them,
and say exactly where it breaks at 100× scale.

---

## 15. What could make this fail

Named now so they can be watched for. `BUGS.md` holds the technical versions; these are the
project-level ones:

| Failure | What it looks like | Guard |
|---|---|---|
| **Silently poor recall** | Everything runs, nothing crashes, recall is 0.6 and you do not notice for two days | Brute-force ground truth on day one, before HNSW exists |
| **Benchmarks that measure nothing** | Impressive numbers that do not survive a single follow-up question | The harness is written before the features, per `00_FOUNDATIONS.md` §4 |
| **Scope over depth** | Six half-built features instead of three you can defend | The cut order in `PLAN.md` §0.3, decided while calm |
| **Memory bugs eating a day** | Random crashes, corrupted results, hours lost | Sanitizers from the first commit — see `PLAN.md` §1; this is *not* free on Windows |
| **Time** | It is Aug 21 and the window was Aug 19–24 | `PLAN.md` §0 addresses this head-on |

---

## Glossary

| Term | Meaning |
|---|---|
| **Vector / embedding** | A list of numbers representing an item, arranged so similar items land near each other |
| **Dimension** | How many numbers in the list. 128 for SIFT1M, 768–1536 for modern text models |
| **k-NN** | k-nearest-neighbour search: find the k closest stored vectors to a query |
| **ANN** | Approximate nearest neighbour — trade a little accuracy for 100–1000× speed |
| **Brute force / flat** | Compare against everything. Exact, slow, and the ground truth for grading |
| **recall@k** | Of the k truly-nearest items, how many did we return. The accuracy metric |
| **QPS** | Queries per second. The speed metric |
| **p50 / p95 / p99** | Median / 95th / 99th percentile latency. Tail latency is what breaks services |
| **HNSW** | Hierarchical Navigable Small World — the layered graph index |
| **M / ef_construction** | HNSW build-time knobs: neighbours per node, search effort while inserting |
| **ef_search** | HNSW query-time knob. The dial you sweep to draw the recall/QPS curve |
| **Greedy descent** | Repeatedly step to whichever neighbour is closer to the query |
| **Small-world property** | Mostly-local links plus a few long-range ones ⇒ everything is a few hops away |
| **PQ** | Product Quantization — chop vectors into chunks, replace each chunk with a 1-byte codebook ID |
| **Codebook** | The 256 representative chunks learned by k-means for one chunk position |
| **ADC** | Asymmetric Distance Computation — exact query against compressed data, via a lookup table |
| **IVF** | Inverted File — pre-cluster the data, search only the nearest few clusters (`nprobe`) |
| **k-means** | Clustering algorithm used to learn codebooks. Lloyd's algorithm; k-means++ initialisation |
| **BM25** | The standard keyword-ranking formula. Term-frequency saturation (`k1`), length norm (`b`) |
| **IDF** | Inverse document frequency — rare terms count for more |
| **Dense / sparse retrieval** | Meaning-based (embeddings) / word-based (keyword index) |
| **Hybrid retrieval** | Running both and merging the results |
| **RRF** | Reciprocal Rank Fusion — merge by rank, `Σ 1/(60 + rank)`. No score normalisation needed |
| **Reranking** | Re-scoring a shortlist with a slower, more accurate model |
| **Tombstone** | Marking a node deleted rather than removing it, because removal breaks the graph |
| **SIFT1M** | Standard benchmark: 1M 128-dim vectors, 10k queries, published ground truth |
| **Curse of dimensionality** | In high dimensions everything is roughly equidistant, so partitioning fails |
| **Cache line** | The 64-byte block RAM is read in. Laying data out to respect it is the biggest perf lever |
| **SIMD / AVX2** | One CPU instruction doing 8 float operations at once |
| **RAII** | C++ resource ownership tied to object lifetime. No manual frees, no leaks |
| **pybind11** | Library that exposes C++ functions to Python |

---

## Where to go next

1. **`PLAN.md`** — the phase-by-phase build plan, the schedule reality, and the cut order.
2. **`02_VECCORE.md`** — the specification: scope, required depth per technology, and §7's
   nineteen interview questions. Those are the exam. This file is the revision notes.
3. **`BUGS.md`** — start it on day one, not on day four.
4. **`CONTEXT.md`** — every design decision with the alternative that was rejected and why.
   In December, this file is what turns "I built an HNSW index" into "here is why I built it this
   way, and here is what I would do differently."
