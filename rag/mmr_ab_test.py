#!/usr/bin/env python3
"""
mmr_ab_test.py
---------------
One-shot A/B test: does MMR diversity re-ranking help or hurt RAG answer
quality? Runs each question TWICE in RAG mode -- once with MMR on, once with
MMR off (retrieve.search's use_mmr toggle) -- everything else held identical
(same model, same k, same excluded source paper), then judges both against
the same ground-truth key and reports coverage, citation faithfulness,
hallucination counts, and a pairwise free-text winner -- same axes as
run_eval_pipeline.py's RAG-vs-plain comparison, but here BOTH sides are RAG,
which isolates MMR's effect specifically.

Why a separate script instead of a run_eval_pipeline.py flag:
This is a standalone script so this A/B test can NEVER touch the existing
RAG-vs-plain --out-dir, summary.json, or report.md -- point --out-dir at a
brand-new folder and the established 64-question run stays exactly as-is.
It reuses the real answer-generation call (eval_run.run_question, already
mode-blind and MMR-aware) and the real judge scoring functions from judge.py
(coverage / citation / structural_citation / pairwise free-text -- all
mode-blind by that module's own design, see its docstring) -- only the
labeling and report layout differ, because judge.py's built-in judge_folder()
assumes the "plain" slot is a no-retrieval baseline (chunks=[], citation
skipped by construction). That assumption is FALSE for an MMR-off RAG
answer -- it still has retrieved chunks and citations worth checking -- so
this script scores both sides identically and symmetrically instead.

Output layout:
    <out-dir>/qN/
        gt.json                ground-truth rubric (same helper run_eval_pipeline.py uses)
        mmr_on.json             RAG answer, use_mmr=True
        mmr_off.json            RAG answer, use_mmr=False
        scored_mmr_on.json      judge verdict on the MMR-on answer
        scored_mmr_off.json     judge verdict on the MMR-off answer
        judge_prompts.json      exact prompts sent to each judge call
    <out-dir>/summary.json      head-to-head rollup (mean coverage/citation/
                                 hallucinations/free-text-win, MMR-on vs MMR-off)
    <out-dir>/report.md         detailed per-question report, both variants
                                 shown side by side (answers, right/wrong,
                                 citations, retrieved chunks, free-text winner)

Usage (from project root):
    # full 64-question A/B test, same settings as the established run:
    python -m code.rag.mmr_ab_test --eval-dir code/rag/eval/questions_64_v2 \
        --out-dir code/rag/eval/runs_mmr_ab --model gpt-4o-mini \
        --judge gemini-3.1-pro-preview --citation-judge gpt-4o-mini --k 50

    # quick smoke test on the first 2 questions:
    python -m code.rag.mmr_ab_test --eval-dir code/rag/eval/questions_64_v2 \
        --out-dir code/rag/eval/runs_mmr_ab_test --k 50 --limit 2

Resume support: same as run_eval_pipeline.py -- rerun the same command after
a partial failure and already-generated/-judged files are reused unless
--force is passed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import TimeoutError as FutureTimeoutError

from .eval_run import run_question
from .run_eval_pipeline import _write_gt, _install_stack_dump_handler, _run_with_timeout
from .judge import (
    _score_coverage, _score_citation, _score_structural_citation,
    _freetext_pairwise, COVERAGE_JUDGE_SYS, CITATION_JUDGE_SYS,
    _coverage_prompt, _citation_prompt, _structural_citation_prompt,
    _pairwise_prompt,
)
from . import cost_tracker


# ----------------------------------------------------------------------------
# Judging -- generic, symmetric, mode-blind (both sides are RAG here)
# ----------------------------------------------------------------------------

def judge_pair(folder, judge_model="gemini-3.1-pro-preview",
              citation_judge_model="gpt-4o-mini"):
    """
    Judge the two RAG variants (mmr_on.json / mmr_off.json) already written
    into `folder`. Mirrors judge.py's judge_folder() but treats both sides
    symmetrically -- judge.py's judge_folder() only runs a full citation /
    structural-citation check on the "rag" slot and skips the "plain" slot
    by construction (chunks=[] there); here BOTH variants have retrieved
    chunks, so both get the full check.
    """
    def _load(name):
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            return json.load(f)

    gt = _load("gt.json")
    on = _load("mmr_on.json")
    off = _load("mmr_off.json")
    qtag = gt.get("question_id", "?")

    print(f"[mmr_ab] scoring coverage (mmr_on, judge={judge_model}) ...", file=sys.stderr)
    cov_on = _score_coverage(gt, on, judge_model, label=f"judge:q{qtag}:coverage:mmr_on")
    print(f"[mmr_ab] scoring coverage (mmr_off, judge={judge_model}) ...", file=sys.stderr)
    cov_off = _score_coverage(gt, off, judge_model, label=f"judge:q{qtag}:coverage:mmr_off")
    print(f"[mmr_ab] scoring citation (mmr_on, judge={citation_judge_model}) ...", file=sys.stderr)
    cit_on = _score_citation(on, citation_judge_model, label=f"judge:q{qtag}:citation:mmr_on")
    print(f"[mmr_ab] scoring citation (mmr_off, judge={citation_judge_model}) ...", file=sys.stderr)
    cit_off = _score_citation(off, citation_judge_model, label=f"judge:q{qtag}:citation:mmr_off")
    print(f"[mmr_ab] scoring structural citation (mmr_on) ...", file=sys.stderr)
    struct_on = _score_structural_citation(on, citation_judge_model, label=f"judge:q{qtag}:structural_citation:mmr_on")
    print(f"[mmr_ab] scoring structural citation (mmr_off) ...", file=sys.stderr)
    struct_off = _score_structural_citation(off, citation_judge_model, label=f"judge:q{qtag}:structural_citation:mmr_off")
    print(f"[mmr_ab] pairwise free-text (both orders, judge={judge_model}) ...", file=sys.stderr)
    # _freetext_pairwise's internal labels are literally "rag"/"plain" -- here
    # "rag" position = mmr_on, "plain" position = mmr_off. Relabeled below.
    ft = _freetext_pairwise(gt, on, off, judge_model)

    on_win = ft["rag_win"]
    off_win = ft["plain_win"]
    _relabel = {"rag": "mmr_on", "plain": "mmr_off", "tie": "tie"}
    ft_relabeled = {
        "mmr_on_win": on_win, "mmr_off_win": off_win,
        "order1_favored": _relabel[ft["order1_favored"]],
        "order2_favored": _relabel[ft["order2_favored"]],
        "order1_reason": ft["order1_reason"], "order2_reason": ft["order2_reason"],
        "consistent": ft["consistent"],
    }
    if "error" in ft:
        ft_relabeled["error"] = ft["error"]

    on_ft_text = on.get("free_text", "")
    off_ft_text = off.get("free_text", "")
    judge_prompts = {
        "judge_model": judge_model, "citation_judge_model": citation_judge_model,
        "coverage_system_prompt": COVERAGE_JUDGE_SYS,
        "citation_system_prompt": CITATION_JUDGE_SYS,
        "coverage_mmr_on_prompt": _coverage_prompt(gt, on),
        "coverage_mmr_off_prompt": _coverage_prompt(gt, off),
        "citation_mmr_on_prompt": _citation_prompt(on),
        "citation_mmr_off_prompt": _citation_prompt(off),
        "structural_citation_mmr_on_prompt": _structural_citation_prompt(on),
        "structural_citation_mmr_off_prompt": _structural_citation_prompt(off),
        "pairwise_order1_prompt": _pairwise_prompt(gt, on_ft_text, off_ft_text),
        "pairwise_order2_prompt": _pairwise_prompt(gt, off_ft_text, on_ft_text),
    }
    with open(os.path.join(folder, "judge_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(judge_prompts, f, ensure_ascii=False, indent=2)

    scored_on = {
        "question_id": gt.get("question_id"), "topic": gt.get("topic"),
        "variant": "mmr_on",
        "judge": judge_model, "citation_judge": citation_judge_model,
        "coverage": cov_on.get("coverage"),
        "hallucinated_features": cov_on.get("hallucinated_features", []),
        "citation": cit_on, "structural_citation": struct_on,
        "free_text_win": on_win, "free_text_detail": ft_relabeled,
    }
    scored_off = {
        "question_id": gt.get("question_id"), "topic": gt.get("topic"),
        "variant": "mmr_off",
        "judge": judge_model, "citation_judge": citation_judge_model,
        "coverage": cov_off.get("coverage"),
        "hallucinated_features": cov_off.get("hallucinated_features", []),
        "citation": cit_off, "structural_citation": struct_off,
        "free_text_win": off_win, "free_text_detail": ft_relabeled,
    }

    for name, obj in (("scored_mmr_on.json", scored_on), ("scored_mmr_off.json", scored_off)):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    cov_r = (scored_on["coverage"] or {}).get("coverage_score")
    cov_p = (scored_off["coverage"] or {}).get("coverage_score")
    print(f"[mmr_ab] {folder}: coverage mmr_on={cov_r} mmr_off={cov_p}  "
          f"free-text mmr_on_win={on_win} (consistent={ft_relabeled['consistent']})",
          file=sys.stderr)
    return scored_on, scored_off


# ----------------------------------------------------------------------------
# Report -- both variants shown side by side, symmetrically
# ----------------------------------------------------------------------------

def _fmt_answer(components, free_text):
    lines = []
    if components:
        for comp, items in components.items():
            lines.append(f"- **{comp}**: {', '.join(str(i) for i in items) if items else '_(nothing named)_'}")
    else:
        lines.append("_(no structured components)_")
    lines.append("")
    lines.append(f"> {free_text or '_(empty)_'}")
    return "\n".join(lines)


def _fmt_right_wrong(scored):
    cov = (scored or {}).get("coverage") or {}
    comps = cov.get("components") or []
    right, wrong_missing = [], []
    for c in comps:
        comp_name = c.get("component")
        for item in c.get("matched", []):
            right.append(f"{item} _(component: {comp_name})_")
        for item in c.get("missing", []):
            wrong_missing.append(f"{item} _(component: {comp_name})_")
        if not c.get("direction_ok", True):
            wrong_missing.append(f"**wrong direction** on component: {comp_name}")
    hallucinated = (scored or {}).get("hallucinated_features") or []
    lines = ["**Right (correctly identified):**"]
    lines.append("\n".join(f"- {x}" for x in right) if right else "_None._")
    lines.append("")
    lines.append("**Wrong -- missed (should have been mentioned):**")
    lines.append("\n".join(f"- {x}" for x in wrong_missing) if wrong_missing else "_None missed._")
    lines.append("")
    lines.append("**Wrong -- hallucinated / contradicted the source:**")
    lines.append("\n".join(f"- {x}" for x in hallucinated) if hallucinated else "_None flagged._")
    return "\n".join(lines)


def _fmt_citations(scored, chunks):
    cit = (scored or {}).get("citation") or {}
    if not cit.get("applicable"):
        return "_Not applicable._"
    claims = cit.get("claims") or []
    if not claims:
        return "_No distinct citable claims identified._"
    chunk_by_rank = {c.get("rank"): c for c in (chunks or [])}
    lines = [
        f"Overall faithfulness: **{cit.get('faithfulness')}** "
        f"({cit.get('n_supported', 0)} supported / {cit.get('n_unsupported', 0)} unsupported).",
        "",
    ]
    for c in claims:
        verdict = "GOOD" if c.get("supported") else "BAD"
        cited_ranks = c.get("cited_chunks", [])
        ranks_str = ", ".join(str(n) for n in cited_ranks)
        lines.append(f"- **[{verdict}]** cited chunk {ranks_str} -- \"{c.get('claim')}\"")
        lines.append(f"  - *why*: {c.get('note', '') or '_(no note given)_'}")
        for rank in cited_ranks:
            ch = chunk_by_rank.get(rank)
            if ch:
                text = (ch.get("text") or "").strip()
                snippet = text[:600] + ("..." if len(text) > 600 else "")
                lines.append(f"  - *chunk {rank} -- {ch.get('pmcid')} "
                            f"(chunk_id={ch.get('chunk_id', 'n/a')})*: \"{snippet}\"")
            else:
                lines.append(f"  - *chunk {rank}*: _(not found among retrieved chunks)_")
        lines.append("")
    return "\n".join(lines)


def _chunks_section(chunks, heading):
    if not chunks:
        return []
    out = [f"### {heading} (k={len(chunks)})", ""]
    for ch in chunks:
        out.append(
            f"- **[{ch.get('rank')}]** {ch.get('pmcid')} ({ch.get('year')}, "
            f"score={ch.get('score')}, source={ch.get('source')}, "
            f"chunk_id={ch.get('chunk_id', 'n/a')})"
        )
    out.append("")
    return out


def _question_section(folder):
    gt_path = os.path.join(folder, "gt.json")
    on_path = os.path.join(folder, "mmr_on.json")
    off_path = os.path.join(folder, "mmr_off.json")
    scored_on_path = os.path.join(folder, "scored_mmr_on.json")
    scored_off_path = os.path.join(folder, "scored_mmr_off.json")
    if not all(os.path.exists(p) for p in
              (gt_path, on_path, off_path, scored_on_path, scored_off_path)):
        return None

    gt = json.load(open(gt_path, encoding="utf-8"))
    on = json.load(open(on_path, encoding="utf-8"))
    off = json.load(open(off_path, encoding="utf-8"))
    scored_on = json.load(open(scored_on_path, encoding="utf-8"))
    scored_off = json.load(open(scored_off_path, encoding="utf-8"))

    qid = gt.get("question_id")
    topic = gt.get("topic")
    on_chunks = on.get("chunks") or []
    off_chunks = off.get("chunks") or []
    cov_on = scored_on.get("coverage") or {}
    cov_off = scored_off.get("coverage") or {}

    out = [
        f"## Q{qid}: {topic}",
        "",
        f"**Question:** {gt.get('question')}",
        "",
        f"**Source:** {gt.get('source_pmcid') or gt.get('source_pmid') or 'n/a'}",
        "",
        "### RAG answer -- MMR ON",
        "", _fmt_answer(on.get("components"), on.get("free_text")), "",
        "### RAG answer -- MMR OFF",
        "", _fmt_answer(off.get("components"), off.get("free_text")), "",
        f"### Judge verdict -- MMR ON (coverage {cov_on.get('coverage_score')}/{cov_on.get('max_score')})",
        "", _fmt_right_wrong(scored_on), "",
        "**Citations -- MMR ON:**", "", _fmt_citations(scored_on, on_chunks), "",
        f"### Judge verdict -- MMR OFF (coverage {cov_off.get('coverage_score')}/{cov_off.get('max_score')})",
        "", _fmt_right_wrong(scored_off), "",
        "**Citations -- MMR OFF:**", "", _fmt_citations(scored_off, off_chunks), "",
        "### Free-text head-to-head -- what won, and why", "",
    ]
    ft = scored_on.get("free_text_detail") or {}
    win = scored_on.get("free_text_win")
    if win is None:
        out.append("_Not judged._")
    else:
        winner = "MMR ON" if win > 0.5 else ("tie" if win == 0.5 else "MMR OFF")
        out += [
            f"**Winner: {winner}**", "",
            f"- Order 1 (MMR-ON shown first) favored **{ft.get('order1_favored')}** -- "
            f"{ft.get('order1_reason') or '_(no reason given)_'}",
            f"- Order 2 (MMR-OFF shown first) favored **{ft.get('order2_favored')}** -- "
            f"{ft.get('order2_reason') or '_(no reason given)_'}",
            f"- Consistent across both orders (no position bias): **{ft.get('consistent')}**",
        ]
    out.append("")
    out += _chunks_section(on_chunks, "All retrieved chunks -- MMR ON")
    out += _chunks_section(off_chunks, "All retrieved chunks -- MMR OFF")
    out.append("---")
    out.append("")
    return "\n".join(out)


def build_report(out_dir, report_path):
    def _qnum(p):
        digits = ''.join(ch for ch in os.path.basename(p) if ch.isdigit())
        return int(digits) if digits else 0

    sections = []
    for folder in sorted(glob.glob(os.path.join(out_dir, "q*")), key=_qnum):
        sec = _question_section(folder)
        if sec:
            sections.append(sec)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# CherryPicker MMR A/B report -- {len(sections)} question(s)\n\n")
        f.write("\n".join(sections))
    print(f"[mmr_ab] wrote {report_path}", file=sys.stderr)


# ----------------------------------------------------------------------------
# Summary rollup
# ----------------------------------------------------------------------------

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


def build_summary(out_dir, summary_path):
    rows = []
    for folder in sorted(glob.glob(os.path.join(out_dir, "q*"))):
        sp_on = os.path.join(folder, "scored_mmr_on.json")
        sp_off = os.path.join(folder, "scored_mmr_off.json")
        if not (os.path.exists(sp_on) and os.path.exists(sp_off)):
            continue
        on = json.load(open(sp_on, encoding="utf-8"))
        off = json.load(open(sp_off, encoding="utf-8"))
        rows.append({
            "question_id": on.get("question_id"), "topic": on.get("topic"),
            "mmr_on_coverage": _cov_fraction(on), "mmr_off_coverage": _cov_fraction(off),
            "mmr_on_faithfulness": _faithfulness(on), "mmr_off_faithfulness": _faithfulness(off),
            "mmr_on_hallucinations": len(on.get("hallucinated_features", [])),
            "mmr_off_hallucinations": len(off.get("hallucinated_features", [])),
            "mmr_on_freetext_win": on.get("free_text_win"),
            "mmr_off_freetext_win": off.get("free_text_win"),
        })
    if not rows:
        print(f"[mmr_ab] no scored folders found under {out_dir}", file=sys.stderr)
        return

    on_wins = sum(1 for r in rows if r["mmr_on_freetext_win"] == 1.0)
    off_wins = sum(1 for r in rows if r["mmr_on_freetext_win"] == 0.0)
    ties = sum(1 for r in rows if r["mmr_on_freetext_win"] == 0.5)

    summary = {
        "n_questions": len(rows),
        "freetext_win_counts": {"mmr_on_wins": on_wins, "mmr_off_wins": off_wins, "ties": ties},
        "mmr_on": {
            "coverage": _mean([r["mmr_on_coverage"] for r in rows]),
            "citation": _mean([r["mmr_on_faithfulness"] for r in rows]),
            "hallucinations": _mean([r["mmr_on_hallucinations"] for r in rows]),
            "freetext_win": _mean([r["mmr_on_freetext_win"] for r in rows]),
        },
        "mmr_off": {
            "coverage": _mean([r["mmr_off_coverage"] for r in rows]),
            "citation": _mean([r["mmr_off_faithfulness"] for r in rows]),
            "hallucinations": _mean([r["mmr_off_hallucinations"] for r in rows]),
            "freetext_win": _mean([r["mmr_off_freetext_win"] for r in rows]),
        },
        "per_question": rows,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    def fmt(x):
        return "  n/a" if x is None else f"{x:>5}"
    print("\n" + "=" * 62)
    print(f"CherryPicker MMR A/B -- {len(rows)} question(s)")
    print("=" * 62)
    print(f"{'metric':<18}{'MMR ON':>12}{'MMR OFF':>12}")
    print("-" * 62)
    for label, key in (("coverage", "coverage"), ("citation", "citation"),
                       ("hallucinations", "hallucinations"), ("free-text win", "freetext_win")):
        print(f"{label:<18}{fmt(summary['mmr_on'][key]):>12}{fmt(summary['mmr_off'][key]):>12}")
    print("-" * 62)
    print(f"free-text wins:  MMR-ON={on_wins}   MMR-OFF={off_wins}   ties={ties}  (of {len(rows)} judged)")
    print("=" * 62)
    print(f"[mmr_ab] wrote {summary_path}", file=sys.stderr)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="mmr_ab_test")
    p.add_argument("--eval-dir", default="code/rag/eval/questions_64_v2",
                   help="Folder holding q*.json question specs.")
    p.add_argument("--out-dir", required=True,
                   help="FRESH folder for this A/B test's results. Never point "
                        "this at the existing rag-vs-plain --out-dir -- this "
                        "script writes its own file names (mmr_on.json / "
                        "mmr_off.json / scored_mmr_on.json / ...), but shares "
                        "no state with that pipeline and should stay physically "
                        "separate so the established run is never touched.")
    p.add_argument("--model", default="gpt-4o-mini", help="Answer-generation model.")
    p.add_argument("--judge", default="gemini-3.1-pro-preview",
                   help="Judge model for coverage/hallucination scoring and the "
                        "pairwise free-text winner call.")
    p.add_argument("--citation-judge", default="gpt-4o-mini",
                   help="Cheaper model for citation faithfulness checking only.")
    p.add_argument("--k", type=int, default=50,
                   help="Chunks to retrieve. Default 50 to match the settings "
                        "of the established full 64-question run, so this A/B "
                        "test is comparable to it.")
    p.add_argument("--lambda-mmr", type=float, default=0.5,
                   help="MMR relevance/diversity tradeoff for the mmr_on side "
                        "(1.0=relevance only, 0.0=diversity only). Default 0.5, "
                        "same as retrieve.py's default.")
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--reranker", action="store_true",
                   help="Apply the cross-encoder reranker to BOTH variants "
                        "(held constant either way -- this test isolates MMR "
                        "only). Default off.")
    p.add_argument("--only", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", type=str, default=None,
                   help="Comma-separated question_ids, e.g. --ids 1,2,3,4,5")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--force", action="store_true",
                   help="Regenerate/re-judge even if output files already exist.")
    args = p.parse_args(argv)

    _install_stack_dump_handler()

    specs = sorted(glob.glob(os.path.join(args.eval_dir, "q*.json")))
    if not specs:
        print(f"[mmr_ab] no q*.json specs found in {args.eval_dir}", file=sys.stderr)
        return 1

    target_ids = None
    if args.ids:
        target_ids = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
    elif args.only is not None:
        target_ids = {args.only}
    if args.limit is not None:
        specs = specs[:args.limit]

    print(f"[mmr_ab] {len(specs)} spec file(s) considered"
          f"{f', filtering to ids={sorted(target_ids)}' if target_ids else ''} "
          f"-> {args.out_dir} (k={args.k}, model={args.model}) ...", file=sys.stderr)

    ran = 0
    for spec in specs:
        cfg = json.load(open(spec, encoding="utf-8"))
        qid = cfg.get("question_id")
        if target_ids is not None and qid not in target_ids:
            continue

        folder = os.path.join(args.out_dir, f"q{qid}")
        os.makedirs(folder, exist_ok=True)
        print(f"\n{'='*62}\n[mmr_ab] Q{qid}: {cfg.get('topic')}\n{'='*62}", file=sys.stderr)

        already_judged = (
            os.path.exists(os.path.join(folder, "scored_mmr_on.json")) and
            os.path.exists(os.path.join(folder, "scored_mmr_off.json"))
        )
        if already_judged and not args.force:
            print(f"[mmr_ab] Q{qid}: already generated + judged -- skipping "
                  f"(use --force to redo).", file=sys.stderr)
            ran += 1
            continue

        _write_gt(cfg, os.path.join(folder, "gt.json"))
        ok = True
        for variant, name, use_mmr in (("mmr_on", "mmr_on.json", True),
                                        ("mmr_off", "mmr_off.json", False)):
            out_path = os.path.join(folder, name)
            if os.path.exists(out_path) and not args.force:
                print(f"  [{variant}] {name} already exists -- reusing "
                      f"(use --force to redo)", file=sys.stderr)
                continue
            try:
                res = _run_with_timeout(
                    run_question, cfg, mode="rag", model=args.model, k=args.k,
                    max_tokens=args.max_tokens, timeout=args.timeout,
                    use_reranker=args.reranker, use_mmr=use_mmr,
                    lambda_mmr=args.lambda_mmr)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
                print(f"  [{variant}] wrote {name} (chunks={len(res['chunks'])})",
                      file=sys.stderr)
            except FutureTimeoutError:
                print(f"  [{variant}] TIMED OUT after {args.timeout}s.", file=sys.stderr)
                ok = False
            except Exception as e:  # noqa: BLE001
                print(f"  [{variant}] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                ok = False
        if not ok:
            print(f"[mmr_ab] Q{qid}: generation failed/timed out -- skipping judge step.",
                  file=sys.stderr)
            continue

        print(f"[mmr_ab] judging (judge={args.judge}, citation-judge={args.citation_judge}) ...",
              file=sys.stderr)
        try:
            _run_with_timeout(judge_pair, folder, judge_model=args.judge,
                              citation_judge_model=args.citation_judge,
                              timeout=args.timeout)
        except FutureTimeoutError:
            print(f"[mmr_ab] Q{qid}: judge TIMED OUT after {args.timeout}s.", file=sys.stderr)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"[mmr_ab] Q{qid}: judge ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        ran += 1

    print(f"\n[mmr_ab] generated + judged {ran} question(s) -> {args.out_dir}", file=sys.stderr)
    build_summary(args.out_dir, os.path.join(args.out_dir, "summary.json"))
    build_report(args.out_dir, os.path.join(args.out_dir, "report.md"))
    cost_tracker.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
