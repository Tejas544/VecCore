#!/usr/bin/env python3
"""Export EdgeRAG's corpus and queries to TSV for the C++ hybrid evaluation.

CONTEXT.md D7: the honest EdgeRAG integration is not "HNSW made it faster" --
at 362 documents HNSW is *slower* than brute force. It is (1) a no-regression
drop-in, (2) a real TF-IDF -> BM25 quality delta on 650 held-out queries, and
(3) a measured crossover curve. This script feeds (2).

Why TSV rather than reading JSON from C++: the C++ side has a JSON *writer* and
no parser, and adding a parser to the library to read a fixture would be the
wrong dependency in the wrong place (D6). Tabs and newlines are replaced with
spaces on export, which cannot change the result -- both are non-alphanumeric,
so both tokenizers already treat them as separators.

Usage:
    ~/veccore-venv/bin/python scripts/export_edgerag.py [--edgerag PATH]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def clean(text: str) -> str:
    return " ".join(text.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgerag", default="/mnt/d/Placement Projects/EdgeRag")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.edgerag) / "data"
    if not src.is_dir():
        raise SystemExit(f"{src} not found -- pass --edgerag with EdgeRAG's repo root")

    out = Path(args.out or (Path(os.environ.get("VECCORE_DATA", Path.home() / "veccore-data"))
                            / "edgerag"))
    out.mkdir(parents=True, exist_ok=True)

    docs = [json.loads(line) for line in (src / "corpus.jsonl").open(encoding="utf-8") if line.strip()]
    queries = [json.loads(line) for line in (src / "queries.jsonl").open(encoding="utf-8") if line.strip()]

    keys = {d["doc_key"]: i for i, d in enumerate(docs)}
    with (out / "corpus.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for d in docs:
            f.write(f"{d['doc_key']}\t{clean(d.get('text') or '')}\n")

    # Queries whose gold document is not in the corpus would silently depress
    # recall for a reason unrelated to retrieval -- drop them and say how many.
    kept, orphaned = [], 0
    for q in queries:
        if q["gold_doc_key"] in keys:
            kept.append(q)
        else:
            orphaned += 1

    with (out / "queries.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for q in kept:
            f.write(f"{q['query_id']}\t{q['gold_doc_key']}\t{clean(q['question'])}\n")

    # The structural ceiling. 112 of 362 documents have no OCR text at all, so
    # any query whose gold document is one of them is unreachable by ANY
    # text-based retriever. Quoting a recall improvement against 100% instead of
    # against this number is the overclaiming WHAT_IS_THIS.md section 10
    # criticises other people for -- so it is computed here and carried through.
    empty_docs = sum(1 for d in docs if not (d.get("text") or "").strip())
    reachable = sum(1 for q in kept if (docs[keys[q["gold_doc_key"]]].get("text") or "").strip())
    ceiling = reachable / len(kept) if kept else 0.0

    summary = {
        "n_docs": len(docs),
        "n_docs_without_text": empty_docs,
        "n_queries": len(kept),
        "n_queries_orphaned": orphaned,
        "n_queries_reachable_by_text": reachable,
        "text_recall_ceiling": ceiling,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  STRUCTURAL CEILING: no text-only retriever can exceed recall {ceiling:.4f}")
    print("  Every recall number from this corpus must be quoted against that, not against 1.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
