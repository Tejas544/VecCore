#!/usr/bin/env python3
"""Regenerate every plot in the README from results/bench.jsonl.

Nothing here recomputes or re-measures anything. Every figure is a rendering of
records that a `bench` run already wrote, each carrying the git SHA, CPU,
compiler, flags, seed and parameters it was produced under (CONTEXT.md D10).
That is what "reproducible" means in this repo: delete the plots, run this, get
the same plots.

Usage:
    ~/veccore-venv/bin/python bench/plots.py [--results results/bench.jsonl] [--out docs/plots]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: {e}")
    return records


def trusted(records: list[dict]) -> list[dict]:
    """Drop anything not fit to publish.

    D10: an ASan build runs several times slower than Release, and a latency
    copied out of one is indistinguishable from a real regression once it is in
    a plot. bench already refuses to write such records without
    --allow-untrusted; this is the second gate, because the flag exists.
    """
    keep = [r for r in records if r.get("env", {}).get("trusted")]
    dropped = len(records) - len(keep)
    if dropped:
        print(f"  dropped {dropped} untrusted record(s)")
    return keep


#: Fields every plottable record must carry. Records written by an older build
#: legitimately lack some of them -- results/bench.jsonl is append-only and
#: outlives any single schema, which is the point of an audit trail (B-06).
#: Skipping them is correct; crashing on them means one stale line from Phase 0
#: can stop every plot in the repo from regenerating.
REQUIRED = ("index", "index_params", "measurements", "n_base", "env")


def plottable(records: list[dict]) -> list[dict]:
    keep, skipped = [], 0
    for r in records:
        if all(k in r for k in REQUIRED):
            keep.append(r)
        else:
            skipped += 1
    if skipped:
        missing = sorted({k for r in records for k in REQUIRED if k not in r})
        print(f"  skipped {skipped} record(s) from an older schema (missing: {', '.join(missing)})")
    return keep


def recall_qps_curve(records: list[dict], out: Path) -> None:
    """The field-standard figure: recall on x, QPS on y, one point per ef_search."""
    hnsw = [r for r in records
            if r["index"] in ("hnsw", "faiss_hnsw") and r["measurements"].get("recall_at_k")]
    flat = [r for r in records if r["index"] == "flat" and r["measurements"].get("recall_at_k")]
    if not hnsw:
        print("  no hnsw records with recall; skipping recall/QPS curve")
        return

    groups: dict[tuple, list[dict]] = {}
    for r in hnsw:
        p = r["index_params"]
        groups.setdefault((r["n_base"], p["M"], p["ef_construction"], r["index"]), []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (n, m, efc, which), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["index_params"]["ef_search"])
        xs = [r["measurements"]["recall_at_k"] for r in rs]
        ys = [r["measurements"]["qps_mean"] for r in rs]
        errs = [r["measurements"]["qps_stddev"] for r in rs]
        # FAISS dashed, VecCore solid -- the comparison is the point of the figure.
        name = "FAISS IndexHNSWFlat" if which == "faiss_hnsw" else "VecCore HNSW"
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3,
                    ls="--" if which == "faiss_hnsw" else "-",
                    label=f"{name}  n={n:,} M={m} efC={efc}")
        for r, x, y in zip(rs, xs, ys):
            ax.annotate(f"ef={r['index_params']['ef_search']}", (x, y),
                        textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)

    for r in flat:
        # The baseline. Exact search sits at recall 1.0 by definition; the whole
        # point of the curve is how much QPS the approximation buys back.
        ax.axhline(r["measurements"]["qps_mean"], ls="--", lw=1, color="grey", alpha=0.7)
        ax.annotate(f"brute force, n={r['n_base']:,}  ({r['measurements']['qps_mean']:.0f} QPS)",
                    (0.02, r["measurements"]["qps_mean"]), xycoords=("axes fraction", "data"),
                    fontsize=8, va="bottom", color="grey")

    ax.set_yscale("log")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("queries/sec (log scale)")
    ax.set_title("Recall vs throughput, single thread, matched parameters\nsweeping ef_search; error bars are stddev over trials")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "recall_qps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def layout_ab(records: list[dict], out: Path) -> None:
    """D5: does flat storage actually beat pointer-chasing, and when?"""
    rs = [r for r in records if r["index_params"].get("kind") == "layout_ab"]
    if not rs:
        print("  no layout_ab records; skipping")
        return
    r = rs[-1]
    m = r["measurements"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["sequential", "random"]
    flat_v = [m["ab_flat_seq_qps"], m["ab_flat_rand_qps"]]
    naive_v = [m["ab_naive_seq_qps"], m["ab_naive_rand_qps"]]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], flat_v, width=0.4, label="flat vector<float>")
    ax.bar([i + 0.2 for i in x], naive_v, width=0.4, label="vector<vector<float>> (fragmented)")
    for i, (f, nv) in enumerate(zip(flat_v, naive_v)):
        ax.annotate(f"{f/nv:.2f}x", (i, max(f, nv)), ha="center",
                    textcoords="offset points", xytext=(0, 6), fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("queries/sec")
    ax.set_title(f"Memory layout, {r['n_base']:,} vectors\n"
                 "the gap is small for a scan and large for random access -- which is what HNSW does")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "layout_ab.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def pq_pareto(records: list[dict], out: Path) -> None:
    """Recall vs memory vs latency for PQ.

    A compression ratio quoted without the recall it lands at is exactly the
    overclaiming WHAT_IS_THIS.md section 10 criticises other people for, so the
    three axes stay together in one figure.
    """
    pq = [r for r in records
          if r["index_params"].get("kind") == "pq" and r["measurements"].get("recall_at_k")]
    if not pq:
        print("  no pq records; skipping Pareto plot")
        return

    flat = [r for r in records if r["index"] == "flat" and r["measurements"].get("recall_at_k")]
    n = max(r["n_base"] for r in pq)
    pq = [r for r in pq if r["n_base"] == n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    by_rerank: dict[int, list[dict]] = {}
    for r in pq:
        by_rerank.setdefault(r["index_params"]["rerank"], []).append(r)

    for rr, rs in sorted(by_rerank.items()):
        rs.sort(key=lambda r: r["measurements"]["bytes_per_vector"])
        xs = [r["measurements"]["bytes_per_vector"] for r in rs]
        ys = [r["measurements"]["recall_at_k"] for r in rs]
        label = "ADC only" if rr == 0 else f"ADC + exact rerank top-{rr}"
        ax1.plot(xs, ys, marker="o", label=label)
        for r, x, y in zip(rs, xs, ys):
            ax1.annotate(f"m={r['index_params']['m']}\n{r['measurements']['compression_ratio']:.0f}x",
                         (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7, alpha=0.8)

        ax2.plot([r["measurements"]["qps_mean"] for r in rs], ys, marker="o", label=label)

    raw = next((r["measurements"]["bytes_per_vector"] for rs in by_rerank.values() for r in rs), 0)
    ax1.axhline(1.0, ls="--", lw=1, color="grey", alpha=0.7)
    ax1.annotate("exact search (512 B/vector)", (0.98, 1.0), xycoords=("axes fraction", "data"),
                 ha="right", va="bottom", fontsize=8, color="grey")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("bytes per vector (log scale)")
    ax1.set_ylabel(f"recall@10, n={n:,}")
    ax1.set_title("PQ: what the compression costs in recall")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8)

    for r in flat:
        if r["n_base"] == n:
            ax2.axvline(r["measurements"]["qps_mean"], ls="--", lw=1, color="grey", alpha=0.7)
            ax2.annotate(f"brute force ({r['measurements']['qps_mean']:.0f} QPS)",
                         (r["measurements"]["qps_mean"], 0.02), xycoords=("data", "axes fraction"),
                         rotation=90, fontsize=8, color="grey", va="bottom")
    ax2.set_xscale("log")
    ax2.set_xlabel("queries/sec (log scale)")
    ax2.set_ylabel("recall@10")
    ax2.set_title("PQ: recall vs throughput")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Product Quantization Pareto, SIFT1M ({raw and ''}single thread)", y=1.0)
    fig.tight_layout()
    path = out / "pq_pareto.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def crossover(records: list[dict], out: Path) -> None:
    """Where HNSW starts beating brute force, with EdgeRAG's corpus marked.

    D7 / L-03: "HNSW made EdgeRAG faster" is false at 362 documents. This is the
    figure that replaces the claim with a number.
    """
    rs = [r for r in records if r["index_params"].get("kind") == "crossover"]
    if not rs:
        print("  no crossover records; skipping")
        return
    r = rs[-1]
    rows = r["measurements"]["rows"]
    ns = [x["n"] for x in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, [x["flat_ms"] for x in rows], marker="o", label="brute force (exact)")
    ax.plot(ns, [x["hnsw_ms"] for x in rows], marker="s",
            label=f"HNSW (ef_search={r['index_params']['ef_search']})")

    cx = r["measurements"].get("crossover_n")
    if cx:
        ax.axvline(cx, ls="--", lw=1, color="seagreen")
        ax.annotate(f"crossover\nn = {cx:,}", (cx, min(x["hnsw_ms"] for x in rows)),
                    fontsize=9, color="seagreen", ha="left",
                    textcoords="offset points", xytext=(6, 0))

    eg = r["index_params"]["edgerag_n"]
    eg_row = next((x for x in rows if x["n"] == eg), None)
    slower = f"\nHNSW is {(1 - eg_row['speedup']) * 100:.0f}% SLOWER here" if eg_row else ""
    ax.axvline(eg, ls="--", lw=1.5, color="firebrick")
    ax.annotate(f"EdgeRAG's corpus\nn = {eg}{slower}",
                (eg, max(x["flat_ms"] for x in rows) * 0.25), fontsize=9,
                color="firebrick", ha="right", textcoords="offset points", xytext=(-8, 0))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size (log scale)")
    ax.set_ylabel("ms per query (log scale)")
    ax.set_title("When does HNSW actually beat brute force?\n"
                 "128-dim, k=10, single thread -- the graph overhead has to be paid back first")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    path = out / "crossover.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def thread_scaling(records: list[dict], out: Path) -> None:
    """Read throughput vs thread count, per lock mode and per working-set size.

    Two things this figure has to show, because P-30 warns that the obvious
    explanation is usually wrong:
      1. the three lock modes lie on top of each other -- the lock is NOT what
         limits scaling, and `none` is the control that proves it;
      2. the cache-resident and memory-resident curves diverge, which is what
         does limit it.
    """
    rs = [r for r in records if r["index_params"].get("kind") == "thread_scaling"]
    if not rs:
        print("  no thread_scaling records; skipping")
        return

    by_n: dict[int, list[dict]] = {}
    for r in rs:
        by_n.setdefault(r["n_base"], []).append(r)

    fig, axes = plt.subplots(1, len(by_n), figsize=(6.5 * len(by_n), 5), squeeze=False)
    for ax, (n, group) in zip(axes[0], sorted(by_n.items())):
        by_mode: dict[str, list[dict]] = {}
        for r in group:
            by_mode.setdefault(r["index_params"]["lock_mode"], []).append(r)

        max_t = 1
        for mode, items in sorted(by_mode.items()):
            items.sort(key=lambda r: r["index_params"]["threads"])
            xs = [r["index_params"]["threads"] for r in items]
            ys = [r["measurements"]["speedup"] for r in items]
            max_t = max(max_t, max(xs))
            ax.plot(xs, ys, marker="o", label=mode)

        ax.plot([1, max_t], [1, max_t], ls=":", color="grey", label="linear")
        ax.axvline(6, ls="--", lw=1, color="firebrick", alpha=0.6)
        ax.annotate("6 physical cores\n(12 are hyperthreads)", (6, 1.2),
                    fontsize=8, color="firebrick", ha="center")
        mib = n * 128 * 4 / (1024 * 1024)
        cache = "fits in 12 MB L3" if mib < 12 else "exceeds 12 MB L3"
        ax.set_title(f"n = {n:,}  ({mib:.0f} MiB, {cache})")
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup vs 1 thread")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Read scaling, HNSW ef_search=64 -- the three lock modes coincide, "
                 "so the lock is not the limiter (P-30)", y=1.0)
    fig.tight_layout()
    path = out / "thread_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def writer_starvation(records: list[dict], out: Path) -> None:
    """Insert latency under continuous read load. B-11."""
    rs = [r for r in records if r["index_params"].get("kind") == "writer_starvation"]
    if not rs:
        print("  no writer_starvation records; skipping")
        return
    n = max(r["n_base"] for r in rs)
    rs = [r for r in rs if r["n_base"] == n]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    by_mode: dict[str, list[dict]] = {}
    for r in rs:
        by_mode.setdefault(r["index_params"]["lock_mode"], []).append(r)

    for mode, items in sorted(by_mode.items()):
        items.sort(key=lambda r: r["index_params"]["readers"])
        xs = [r["index_params"]["readers"] for r in items]
        ys = [r["measurements"]["insert_p99_ms"] for r in items]
        ax.plot(xs, ys, marker="o", label=mode)
        for r, x, y in zip(items, xs, ys):
            if r["measurements"].get("starved"):
                ax.annotate(f"STARVED\n{r['measurements']['inserts_completed']}/200 inserts",
                            (x, y), textcoords="offset points", xytext=(-70, -6),
                            fontsize=8, color="firebrick", fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlabel("concurrent reader threads")
    ax.set_ylabel("insert p99 latency, ms (log scale)")
    ax.set_title(f"Writer starvation, n = {n:,}\n"
                 "glibc's shared_mutex is reader-preferring; a turnstile fixes it (B-11)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    path = out / "writer_starvation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def latency_distribution(records: list[dict], out: Path) -> None:
    """p50 vs p99, because a mean hides the tail that matters in serving."""
    hnsw = [r for r in records if r["index"] == "hnsw" and r["measurements"].get("recall_at_k")]
    if not hnsw:
        return
    n = max(r["n_base"] for r in hnsw)
    rs = sorted((r for r in hnsw if r["n_base"] == n),
                key=lambda r: r["index_params"]["ef_search"])

    efs = [r["index_params"]["ef_search"] for r in rs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in (("latency_p50_ms", "p50"), ("latency_p95_ms", "p95"), ("latency_p99_ms", "p99")):
        ax.plot(efs, [r["measurements"][key] for r in rs], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ef_search")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"Query latency percentiles, n={n:,}\nthe p50/p99 gap is what a mean would hide")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    path = out / "latency_percentiles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def markdown_table(records: list[dict], out: Path) -> None:
    """The README's benchmark table, generated rather than typed."""
    rows = []
    for r in records:
        m, p = r["measurements"], r["index_params"]
        if p.get("kind") in ("layout_ab", "thread_scaling", "writer_starvation",
                             "edgerag_hybrid", "crossover"):
            continue
        rows.append({
            "index": r["index"],
            "ef": p.get("ef_search", 0),
            "n": r["n_base"],
            "params": (f"M={p['M']} efC={p['ef_construction']} efS={p['ef_search']}"
                       if r["index"] == "hnsw" else "exact"),
            "recall": m.get("recall_at_k"),
            "qps": m["qps_mean"],
            "sd": m["qps_stddev"],
            "p50": m["latency_p50_ms"],
            "p99": m["latency_p99_ms"],
            "mib": m.get("index_bytes", 0) / (1024 * 1024),
            "sha": r["env"]["git_sha"],
        })
    # Sort by ef_search numerically. Sorting the formatted string puts
    # efS=160 between efS=10 and efS=20, which makes a monotone curve look
    # scrambled in the one place a reader checks it by eye.
    rows.sort(key=lambda r: (r["n"], r["index"], r["ef"]))

    lines = ["| index | n | params | recall@10 | QPS | p50 ms | p99 ms | index MiB | commit |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rec = f"{r['recall']:.4f}" if r["recall"] is not None else "1.0000 (exact)"
        lines.append(f"| {r['index']} | {r['n']:,} | {r['params']} | {rec} | "
                     f"{r['qps']:,.0f} ± {r['sd']:,.0f} | {r['p50']:.3f} | {r['p99']:.3f} | "
                     f"{r['mib']:.1f} | `{r['sha']}` |")
    path = out / "benchmark_table.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def head_to_head_table(records: list[dict], out: Path) -> None:
    """The VecCore-vs-FAISS table, paired by ef_search and generated from records.

    This one is separate from `markdown_table` because pairing is the whole
    point: a row that shows VecCore's number next to FAISS's is only meaningful
    if both came from the same interleaved session at the same parameters, and
    building it by hand is how a p99 column goes missing for a month. Rows are
    emitted only where both sides exist at the same ef_search, so a half-measured
    sweep produces a shorter table rather than a misaligned one.
    """
    ours: dict[int, dict] = {}
    theirs: dict[int, dict] = {}
    for r in records:
        p = r["index_params"]
        if r["n_base"] != 1_000_000:
            continue
        ef = p.get("ef_search")
        if r["index"] == "hnsw" and ef:
            ours[int(ef)] = r
        elif r["index"] == "faiss_hnsw" and ef:
            theirs[int(ef)] = r

    shared = sorted(set(ours) & set(theirs))
    if not shared:
        print("  head_to_head: no paired ef_search points; skipping")
        return

    lines = [
        "| ef_search | VecCore recall@10 | FAISS recall@10 | VecCore QPS | FAISS QPS "
        "| VecCore p99 ms | FAISS p99 ms | FAISS faster by |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ef in shared:
        a, b = ours[ef]["measurements"], theirs[ef]["measurements"]
        ratio = b["qps_mean"] / a["qps_mean"] if a["qps_mean"] else float("nan")
        lines.append(
            f"| {ef} | {a['recall_at_k']:.4f} | {b['recall_at_k']:.4f} | "
            f"{a['qps_mean']:,.0f} | {b['qps_mean']:,.0f} | "
            f"{a['latency_p99_ms']:.3f} | {b['latency_p99_ms']:.3f} | {ratio:.2f}x |"
        )

    # Build time is a property of the index, not of an ef_search point, so it is
    # a footer row rather than a column repeated identically down the table.
    a_build = ours[shared[0]]["measurements"].get("build_ms")
    b_build = theirs[shared[0]]["measurements"].get("build_ms")
    if a_build and b_build:
        lines.append(
            f"| **build** | | | | | | | **{a_build / 1000:,.0f} s vs {b_build / 1000:,.0f} s "
            f"= {a_build / b_build:.2f}x** |"
        )

    path = out / "head_to_head.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/bench.jsonl")
    ap.add_argument("--out", default="docs/plots")
    args = ap.parse_args()

    results = Path(args.results)
    if not results.exists():
        raise SystemExit(f"{results} not found -- run bench first")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = load(results)
    print(f"loaded {len(records)} record(s) from {results}")
    records = plottable(trusted(records))
    if not records:
        raise SystemExit("no plottable records -- run bench first")

    recall_qps_curve(records, out)
    latency_distribution(records, out)
    pq_pareto(records, out)
    crossover(records, out)
    thread_scaling(records, out)
    writer_starvation(records, out)
    layout_ab(records, out)
    markdown_table(records, out)
    head_to_head_table(records, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
