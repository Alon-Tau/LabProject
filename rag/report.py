#!/usr/bin/env python3
"""
report.py
---------
Build human-readable markdown reports from every judged question folder,
both as ONE consolidated file and as ONE FILE PER QUESTION.

summarize.py rolls up just the aggregate NUMBERS (mean coverage, mean
hallucination count, etc). This script pulls the actual CONTENT that backs
those numbers -- the questions, both full answers, which specific citations
were verified good/bad and WHY (with the actual cited chunk text inline so
you don't have to cross-reference rag.json separately), which specific
features each model got right vs wrong, and WHY the free-text judge picked
the winner it did.

Reads, per runs/qN/ folder:
    gt.json            question text + rubric
    rag.json           RAG answer (components, free_text, retrieved chunks)
    plain.json         no-retrieval answer (components, free_text)
    scored_rag.json    judge's coverage/citation/hallucination verdict on rag
    scored_plain.json  judge's coverage/hallucination verdict on plain

build_report() / build_per_question_files() are importable so
run_eval_pipeline.py calls both directly at the end of a full run -- no
separate command needed.

Usage (from project root, standalone):
    python -m code.rag.report
    python -m code.rag.report --runs code/rag/runs --out code/rag/report.md \
        --per-question-dir code/rag/reports
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _load(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
    """
    What this model got RIGHT (matched GT items) vs WRONG (missing GT items,
    plus anything flagged as hallucinated / contradicting the key).
    """
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
    """
    Per-citation good/bad verdict with the judge's stated reason, AND the
    actual text of the chunk(s) it cited inline -- so you can verify the
    claim right here without opening rag.json separately.
    """
    cit = (scored or {}).get("citation") or {}
    if not cit.get("applicable"):
        return "_Not applicable (no retrieval, so nothing was cited)._"
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
                lines.append(
                    f"  - *chunk {rank} -- {ch.get('pmcid')} ({ch.get('year')})*: \"{snippet}\""
                )
            else:
                lines.append(f"  - *chunk {rank}*: _(not found among retrieved chunks)_")
        lines.append("")
    return "\n".join(lines)


def _fmt_freetext_verdict(scored_rag):
    ft = (scored_rag or {}).get("free_text_detail") or {}
    win = (scored_rag or {}).get("free_text_win")
    if win is None:
        return "_Not judged._"
    winner = "RAG" if win > 0.5 else ("tie" if win == 0.5 else "no-retrieval")
    lines = [f"**Winner: {winner}**", ""]
    lines.append(f"- Order 1 (RAG shown first) favored **{ft.get('order1_favored')}** -- "
                 f"{ft.get('order1_reason') or '_(no reason given)_'}")
    lines.append(f"- Order 2 (no-retrieval shown first) favored **{ft.get('order2_favored')}** -- "
                 f"{ft.get('order2_reason') or '_(no reason given)_'}")
    lines.append(f"- Consistent across both orders (no position bias): **{ft.get('consistent')}**")
    return "\n".join(lines)


def _question_section(folder):
    gt = _load(folder, "gt.json")
    rag = _load(folder, "rag.json")
    plain = _load(folder, "plain.json")
    scored_rag = _load(folder, "scored_rag.json")
    scored_plain = _load(folder, "scored_plain.json")

    if gt is None:
        return None  # folder not fully populated yet

    qid = gt.get("question_id")
    topic = gt.get("topic")
    question = gt.get("question")
    rag_chunks = (rag or {}).get("chunks") or []

    out = [
        f"## Q{qid}: {topic}",
        "",
        f"**Question:** {question}",
        "",
        f"**Source:** {gt.get('source_pmcid') or gt.get('source_pmid') or 'n/a'}",
        "",
        "### RAG answer",
        "",
        _fmt_answer((rag or {}).get("components"), (rag or {}).get("free_text")),
        "",
        "### No-retrieval (plain) answer",
        "",
        _fmt_answer((plain or {}).get("components"), (plain or {}).get("free_text")),
        "",
    ]

    if scored_rag is not None:
        cov = scored_rag.get("coverage") or {}
        out += [
            f"### Judge verdict -- RAG (coverage {cov.get('coverage_score')}/{cov.get('max_score')})",
            "",
            _fmt_right_wrong(scored_rag),
            "",
            "**Citations -- good vs bad, why, and the actual cited text:**",
            "",
            _fmt_citations(scored_rag, rag_chunks),
            "",
        ]
    if scored_plain is not None:
        cov = scored_plain.get("coverage") or {}
        out += [
            f"### Judge verdict -- no-retrieval (coverage {cov.get('coverage_score')}/{cov.get('max_score')})",
            "",
            _fmt_right_wrong(scored_plain),
            "",
        ]
    if scored_rag is not None:
        out += [
            "### Free-text head-to-head -- what won, and why",
            "",
            _fmt_freetext_verdict(scored_rag),
            "",
        ]

    if rag_chunks:
        out += [
            f"### All retrieved chunks (k={len(rag_chunks)}) -- for reference",
            "",
        ]
        for ch in rag_chunks:
            out.append(
                f"- **[{ch.get('rank')}]** {ch.get('pmcid')} ({ch.get('year')}, "
                f"score={ch.get('score')}, source={ch.get('source')}, "
                f"chunk_id={ch.get('chunk_id', 'n/a')})"
            )
        out.append("")

    out.append("---")
    out.append("")
    return "\n".join(out)


def build_report(runs_dir, out_path):
    """
    Scan runs_dir for qN/ folders and write one consolidated markdown report
    to out_path. Returns the number of questions actually included (folders
    missing gt.json are skipped -- e.g. still mid-run).
    """
    folders = sorted(
        glob.glob(os.path.join(runs_dir, "q*")),
        key=lambda f: int(os.path.basename(f).lstrip("q")) if os.path.basename(f).lstrip("q").isdigit() else 0,
    )

    sections = []
    for folder in folders:
        sec = _question_section(folder)
        if sec:
            sections.append(sec)

    if not sections:
        print(f"[report] no populated qN/ folders found under {runs_dir}", file=sys.stderr)
        return 0

    header = [
        "# CherryPicker Eval -- Detailed Report",
        "",
        f"{len(sections)} question(s). Generated from `{runs_dir}`.",
        "",
        "Each question below shows both full answers (RAG vs no-retrieval), what "
        "each model got right vs wrong against the ground-truth key, a per-citation "
        "good/bad verdict with the judge's reasoning and the actual cited chunk text, "
        "and which free-text answer won the head-to-head comparison and why.",
        "",
        "---",
        "",
    ]

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
        f.write("\n".join(sections))

    print(f"[report] wrote {out_path} ({len(sections)} question(s))", file=sys.stderr)
    return len(sections)


def build_per_question_files(runs_dir, out_dir):
    """
    Same content as build_report(), but written as ONE FILE PER QUESTION
    (out_dir/q1.md, q2.md, ...) instead of one consolidated document --
    handy if you want to open/share/skim a single question at a time.
    """
    folders = sorted(
        glob.glob(os.path.join(runs_dir, "q*")),
        key=lambda f: int(os.path.basename(f).lstrip("q")) if os.path.basename(f).lstrip("q").isdigit() else 0,
    )

    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for folder in folders:
        sec = _question_section(folder)
        if not sec:
            continue
        qid_str = os.path.basename(folder).lstrip("q")
        out_path = os.path.join(out_dir, f"q{qid_str}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(sec)
        n += 1

    print(f"[report] wrote {n} per-question file(s) to {out_dir}", file=sys.stderr)
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="report")
    p.add_argument("--runs", default="code/rag/runs",
                   help="Folder holding qN/ result folders (same as run_eval_pipeline --out-dir).")
    p.add_argument("--out", default="code/rag/report.md",
                   help="Path for the single consolidated report.")
    p.add_argument("--per-question-dir", default="code/rag/reports",
                   help="Folder to write one file per question into (q1.md, q2.md, ...).")
    args = p.parse_args(argv)

    n1 = build_report(args.runs, args.out)
    n2 = build_per_question_files(args.runs, args.per_question_dir)
    return 0 if (n1 > 0 or n2 > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
