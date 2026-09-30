#!/usr/bin/env python3
"""
summarize.py
------------
Roll up every scored question folder into the head-to-head result table.

Reads code/rag/eval/runs/q*/scored_rag.json + scored_plain.json (written by
judge.py) and reports, per mode:
    - coverage        mean of (coverage_score / max_score)   [0..1]
    - citation        mean faithfulness (RAG only; baseline has none)
    - hallucinations  mean count of flagged fabricated/contradicting features
    - free_text_win   mean pairwise win (1 = better explanation, 0.5 = tie)

Usage (from project root):
    python -m code.rag.summarize
    python -m code.rag.summarize --runs code/rag/eval/runs --out code/rag/eval/summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _cov_fraction(scored):
    cov = scored.get("coverage") or {}
    score = cov.get("coverage_score")
    mx = cov.get("max_score")
    if score is None or not mx:
        return None
    return round(score / mx, 3)


def _faithfulness(scored):
    cit = scored.get("citation") or {}
    if not cit.get("applicable"):
        return None
    return cit.get("faithfulness")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="summarize")
    p.add_argument("--runs", default="code/rag/eval/runs")
    p.add_argument("--out", default="code/rag/eval/summary.json")
    args = p.parse_args(argv)

    rows = []  # per-question records
    for folder in sorted(glob.glob(os.path.join(args.runs, "q*"))):
        sr = os.path.join(folder, "scored_rag.json")
        sp = os.path.join(folder, "scored_plain.json")
        if not (os.path.exists(sr) and os.path.exists(sp)):
            continue
        rag = json.load(open(sr, encoding="utf-8"))
        plain = json.load(open(sp, encoding="utf-8"))
        rows.append({
            "question_id": rag.get("question_id"),
            "topic": rag.get("topic"),
            "rag_coverage": _cov_fraction(rag),
            "plain_coverage": _cov_fraction(plain),
            "rag_faithfulness": _faithfulness(rag),
            "rag_hallucinations": len(rag.get("hallucinated_features", [])),
            "plain_hallucinations": len(plain.get("hallucinated_features", [])),
            "rag_freetext_win": rag.get("free_text_win"),
            "plain_freetext_win": plain.get("free_text_win"),
        })

    if not rows:
        print(f"[summarize] no scored folders found under {args.runs}", file=sys.stderr)
        return 1

    summary = {
        "n_questions": len(rows),
        "rag_no_bouncer": {
            "coverage":       _mean([r["rag_coverage"] for r in rows]),
            "citation":       _mean([r["rag_faithfulness"] for r in rows]),
            "hallucinations": _mean([r["rag_hallucinations"] for r in rows]),
            "freetext_win":   _mean([r["rag_freetext_win"] for r in rows]),
        },
        "no_retrieval": {
            "coverage":       _mean([r["plain_coverage"] for r in rows]),
            "citation":       None,  # baseline has no citations by construction
            "hallucinations": _mean([r["plain_hallucinations"] for r in rows]),
            "freetext_win":   _mean([r["plain_freetext_win"] for r in rows]),
        },
        "per_question": rows,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # pretty console table
    def fmt(x):
        return "  n/a" if x is None else f"{x:>5}"
    print("\n" + "=" * 62)
    print(f"CherryPicker eval — {len(rows)} question(s)")
    print("=" * 62)
    print(f"{'metric':<18}{'RAG (no bouncer)':>20}{'no-retrieval':>18}")
    print("-" * 62)
    for label, key in (("coverage", "coverage"), ("citation", "citation"),
                       ("hallucinations", "hallucinations"),
                       ("free-text win", "freetext_win")):
        r = summary["rag_no_bouncer"][key]
        pl = summary["no_retrieval"][key]
        print(f"{label:<18}{fmt(r):>20}{fmt(pl):>18}")
    print("=" * 62)
    print(f"[summarize] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
