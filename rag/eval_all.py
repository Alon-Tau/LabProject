#!/usr/bin/env python3
"""
eval_all.py
-----------
Run every holdout question (q*.json in the eval dir) through BOTH modes
(RAG-without-bouncer + plain no-retrieval) in a single run, and lay out the
results so each question gets its own folder:

    code/rag/eval/runs/
        q1/
            gt.json      <- ground-truth key (for the human scorer)
            rag.json     <- RAG (no bouncer) structured answer + chunks
            plain.json   <- no-retrieval structured answer (chunks: [])
        q2/ ...
        q3/ ...
        q4/ ...

Usage (from project root):
    python -m code.rag.eval_all --model gpt-4o-mini --k 8
    python -m code.rag.eval_all --only 1          # just question 1
    python -m code.rag.eval_all --model claude-sonnet-4-5 --k 10

Prereqs: OPENAI_API_KEY (always); ANTHROPIC_API_KEY for claude-* models.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from .eval_run import run_question
from . import cost_tracker


# Only these ground_truth keys count toward coverage_score. Each is worth
# 1.0 total, split evenly among its items. Any OTHER ground_truth key present
# on a question (currently just conflicting_or_alternative_findings) is still
# carried through to gt.json as unscored context -- the judge sees it (so it
# can flag a candidate that hallucinates or contradicts that material) but it
# never contributes points and the model is never required to populate it.
SCORED_COMPONENTS = ("microbiota", "metabolites", "mechanisms")


def _write_gt(cfg, path):
    """Write the ground-truth key for a question, with per-item point weights."""
    gt = cfg.get("ground_truth", {})
    rubric = []
    unscored = []
    for component, items in gt.items():
        if component in SCORED_COMPONENTS:
            n = len(items) if items else 1
            rubric.append({
                "component": component,
                "ground_truth_items": items,
                "n_items": n,
                "per_item": round(1.0 / n, 3),
            })
        else:
            unscored.append({"component": component, "items": items})
    out = {
        "question_id":  cfg.get("question_id"),
        "topic":        cfg.get("topic"),
        "source_pmid":  cfg.get("source_pmid"),
        "source_pmcid": cfg.get("source_pmcid"),
        "question":     cfg.get("question"),
        "rubric":       rubric,
        "unscored_components": unscored,
        "ground_truth_free_text": cfg.get("ground_truth_free_text", ""),
        "max_score":    len(rubric),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eval_all")
    p.add_argument("--eval-dir", default="code/rag/eval",
                   help="Folder holding q*.json question specs.")
    p.add_argument("--out-dir", default="code/rag/eval/runs",
                   help="Folder to write per-question result folders into.")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--k", type=int, default=8, help="Chunks to retrieve (rag mode).")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Cap on the answering model's generated JSON (components + "
                        "free_text). Raised 1500 -> 2500 -> 8000; the latest bump "
                        "matches eval_run.py's default after the systematic "
                        "per-source review instruction was added (longer, more "
                        "exhaustive output needs more headroom).")
    p.add_argument("--only", type=int, default=None,
                   help="Run only this question_id (default: all).")
    args = p.parse_args(argv)

    specs = sorted(glob.glob(os.path.join(args.eval_dir, "q*.json")))
    if not specs:
        print(f"[eval_all] no q*.json specs found in {args.eval_dir}", file=sys.stderr)
        return 1

    ran = 0
    for spec in specs:
        with open(spec, encoding="utf-8") as f:
            cfg = json.load(f)
        qid = cfg.get("question_id")
        if args.only is not None and qid != args.only:
            continue

        folder = os.path.join(args.out_dir, f"q{qid}")
        os.makedirs(folder, exist_ok=True)
        print(f"\n=== Q{qid}: {cfg.get('topic')} ({os.path.basename(spec)}) ===",
              file=sys.stderr)

        _write_gt(cfg, os.path.join(folder, "gt.json"))
        print("  [gt]    wrote gt.json", file=sys.stderr)

        for mode, name in (("rag", "rag.json"), ("plain", "plain.json")):
            try:
                res = run_question(cfg, mode=mode, model=args.model, k=args.k,
                                   max_tokens=args.max_tokens)
                with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
                print(f"  [{mode}]  wrote {name} (chunks={len(res['chunks'])})",
                      file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"  [{mode}]  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        ran += 1

    print(f"\n[eval_all] done — {ran} question(s) → {args.out_dir}", file=sys.stderr)
    cost_tracker.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
