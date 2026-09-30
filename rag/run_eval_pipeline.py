#!/usr/bin/env python3
"""
run_eval_pipeline.py
---------------------
One-shot driver: generate -> judge -> summarize, in a single process.

Wraps eval_all.py's generation logic + judge.py's judge_folder() +
summarize.py's main(), so the whole pipeline runs as ONE command instead of
three, and cost_tracker accumulates a SINGLE cumulative total across
generation AND judging. (Running the three scripts separately means the
judge step's cost ledger resets on every CLI invocation, since each is its
own process -- this driver avoids that by doing everything in one process.)

Two safety features, added after a real run stalled silently for 26+ minutes
inside retrieval with no way to diagnose it (no sudo -> no py-spy):

  1. faulthandler on SIGUSR1 -- at startup this process prints its own PID
     and registers a signal handler so you can get a full Python stack trace
     of every thread, at any time, with zero special privileges:
         kill -USR1 <pid>
     The trace goes to this process's stderr (so use `tee` / a log file to
     capture it). This tells you exactly which line it's stuck on -- inside
     Chroma, inside a network call, wherever -- without needing py-spy/sudo.

  2. A per-call timeout (--timeout, default 3600s) around each generate call
     and the judge step. If a single call exceeds it, this prints a clear
     "TIMED OUT" message and MOVES ON to the next question/step instead of
     hanging silently forever. Note: this does NOT forcibly kill a stuck
     call -- Python can't safely kill a thread blocked inside a C extension
     -- it just stops waiting for it and tells you clearly what happened.
     If you hit a timeout, the underlying stuck thread is still alive in the
     background holding resources; kill and restart the whole process.

Resume support: if you rerun the same command after a partial failure (e.g. a
transient Gemini 503), already-generated rag.json/plain.json and already-
judged scored_rag.json/scored_plain.json are reused instead of redone. Pass
--force to regenerate/re-judge everything anyway.

Usage (from project root):
    # test one question end-to-end:
    python -m code.rag.run_eval_pipeline --eval-dir code/rag/eval/questions_64_v2 \
        --model gpt-4o-mini --judge gemini-3.1-pro-preview --k 8 --only 1

    # quick test batch of 5:
    python -m code.rag.run_eval_pipeline --eval-dir code/rag/eval/questions_64_v2 \
        --model gpt-4o-mini --judge gemini-3.1-pro-preview --k 8 --limit 5

    # specific questions:
    python -m code.rag.run_eval_pipeline --eval-dir code/rag/eval/questions_64_v2 \
        --model gpt-4o-mini --judge gemini-3.1-pro-preview --k 8 --ids 1,2,3,4,5

    # full 64-question run:
    python -m code.rag.run_eval_pipeline --eval-dir code/rag/eval/questions_64_v2 \
        --model gpt-4o-mini --judge gemini-3.1-pro-preview --k 8

    # if you suspect something's stuck, from another terminal:
    kill -USR1 <pid>          # dumps all thread stacks to this process's stderr/log

Prereqs: OPENAI_API_KEY (always); GEMINI_API_KEY for the judge;
ANTHROPIC_API_KEY only if --model is claude-*.

Output layout (identical to running eval_all.py + judge.py separately):
    code/rag/eval/runs/qN/
        gt.json           ground-truth rubric
        rag.json          RAG answer + chunks
        plain.json        no-retrieval answer
        scored_rag.json   judge's scoring of the RAG answer
        scored_plain.json judge's scoring of the plain answer
    code/rag/eval/summary.json   head-to-head rollup across all judged questions
    code/rag/report.md           detailed per-question report (answers, right/
                                  wrong breakdown, citation good/bad + why,
                                  free-text winner + why) -- written every run
"""

from __future__ import annotations

import argparse
import faulthandler
import glob
import json
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from .eval_run import run_question
from .eval_all import SCORED_COMPONENTS
from .judge import judge_folder
from .summarize import main as summarize_main
from .report import build_report, build_per_question_files
from . import cost_tracker


def _install_stack_dump_handler():
    """
    Register SIGUSR1 -> full thread-stack dump, no special privileges needed.
    Prints the PID up front so you don't have to go hunt for it with ps/grep.
    """
    if not hasattr(signal, "SIGUSR1"):
        return  # not available on this platform; skip quietly
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    print(f"[pipeline] PID={os.getpid()} -- if this looks stuck, run "
          f"'kill -USR1 {os.getpid()}' from another terminal to dump every "
          f"thread's stack trace here (no sudo needed).", file=sys.stderr)


def _run_with_timeout(fn, *args, timeout=None, **kwargs):
    """
    Run fn(*args, **kwargs), giving up after `timeout` seconds if it hasn't
    returned. Raises concurrent.futures.TimeoutError on expiry.

    IMPORTANT: this does not (and cannot safely) kill the underlying call if
    it's blocked inside a C extension (e.g. Chroma's HNSW code) -- it just
    stops waiting for it. The stuck worker thread keeps running in the
    background. Treat a timeout as a signal to kill and restart the whole
    process, not as proof the stuck work has actually stopped.
    """
    if timeout is None:
        return fn(*args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=timeout)


def _write_gt(cfg, path):
    """
    Write the ground-truth key for a question, with per-item point weights.

    Kept in sync with eval_all.py::_write_gt -- only SCORED_COMPONENTS
    (microbiota / metabolites / mechanisms) count toward coverage_score.
    Any other ground_truth key (currently just
    conflicting_or_alternative_findings) is carried through as
    unscored_components: visible to the judge for hallucination-checking,
    but never scored. ground_truth_free_text is passed through too, as
    background context for the judge prompts.
    """
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
    p = argparse.ArgumentParser(prog="run_eval_pipeline")
    p.add_argument("--eval-dir", default="code/rag/eval/questions_64_v2",
                   help="Folder holding q*.json question specs.")
    p.add_argument("--out-dir", default="code/rag/eval/runs",
                   help="Folder to write per-question result folders into.")
    p.add_argument("--model", default="gpt-4o-mini", help="Answer-generation model.")
    p.add_argument("--judge", default="gemini-3.1-pro-preview",
                   help="Judge model for coverage/hallucination scoring and the "
                        "pairwise free-text winner call.")
    p.add_argument("--citation-judge", default="gpt-4o-mini",
                   help="Cheaper model used ONLY for citation faithfulness "
                        "checking (does each [N]-cited chunk exist and support "
                        "the claim?) -- a narrow grounding check against chunk "
                        "text the judge is given in full, so it does not need "
                        "the expensive --judge model. See judge.py.")
    p.add_argument("--k", type=int, default=8, help="Chunks to retrieve (rag mode).")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Cap on the answering model's generated JSON (components + "
                        "free_text). Raised 1500 -> 2500 -> 8000; the latest bump "
                        "matches eval_run.py's default after the systematic "
                        "per-source review instruction was added (longer, more "
                        "exhaustive output needs more headroom).")
    p.add_argument("--reranker", action="store_true",
                   help="Rescore retrieved chunks with a cross-encoder and drop "
                        "per-query outliers before building the prompt (see "
                        "rerank.py). Default off. NOTE: this only affects newly "
                        "generated answers -- if rag.json already exists on disk "
                        "for a question, it's reused as-is unless you also pass "
                        "--force, or point --out-dir at a fresh folder (recommended "
                        "for an A/B comparison against a non-reranked baseline run).")
    p.add_argument("--no-mmr", action="store_true",
                   help="Disable MMR diversity re-ranking in retrieval (see "
                        "retrieve.py) -- take the raw top-k by similarity instead "
                        "of fetching a larger pool and MMR-reranking down to k. "
                        "Default off (MMR ON), matching production behavior. Same "
                        "caveat as --reranker: only affects newly generated "
                        "rag.json files, so point --out-dir at a FRESH folder to "
                        "run a clean with/without-MMR A/B comparison against an "
                        "existing MMR-on run without touching it.")
    p.add_argument("--lambda-mmr", type=float, default=0.5,
                   help="MMR relevance/diversity tradeoff, 1.0=relevance only, "
                        "0.0=diversity only. Only used when MMR is on (default 0.5, "
                        "same as retrieve.py's default).")
    p.add_argument("--only", type=int, default=None,
                   help="Run only this single question_id (default: all). For "
                        "multiple specific questions use --ids instead.")
    p.add_argument("--limit", type=int, default=None,
                   help="Run only the first N questions, in sorted file order "
                        "(e.g. --limit 5 for a quick test batch).")
    p.add_argument("--ids", type=str, default=None,
                   help="Comma-separated list of specific question_ids to run, "
                        "e.g. --ids 1,2,3,4,5")
    p.add_argument("--summary-out", default="code/rag/eval/summary.json")
    p.add_argument("--report-out", default="code/rag/report.md",
                   help="Where to write the ONE consolidated markdown report "
                        "(full answers, right/wrong breakdown, citation good/bad "
                        "verdicts + cited chunk text, free-text winner + why).")
    p.add_argument("--report-per-question-dir", default="code/rag/reports",
                   help="Folder to write ONE FILE PER QUESTION into (q1.md, "
                        "q2.md, ...) with the same content as --report-out.")
    p.add_argument("--timeout", type=int, default=3600,
                   help="Max seconds to wait for a single generate/judge call "
                        "before giving up and moving on (default 3600 = 1hr, "
                        "generous enough to cover a cold Chroma index load on "
                        "the first call).")
    p.add_argument("--force", action="store_true",
                   help="Regenerate/re-judge even if output files already exist "
                        "from a prior run. Default: skip whatever's already done "
                        "on disk, so rerunning after a partial failure (e.g. a "
                        "transient 503) only redoes the missing pieces.")
    args = p.parse_args(argv)

    _install_stack_dump_handler()

    specs = sorted(glob.glob(os.path.join(args.eval_dir, "q*.json")))
    if not specs:
        print(f"[pipeline] no q*.json specs found in {args.eval_dir}", file=sys.stderr)
        return 1

    # --ids takes priority over --only if both are given; --limit trims the
    # candidate file list itself (first N in sorted order) and composes with
    # either of the above.
    target_ids = None
    if args.ids:
        target_ids = {int(x.strip()) for x in args.ids.split(",") if x.strip()}
    elif args.only is not None:
        target_ids = {args.only}

    if args.limit is not None:
        specs = specs[:args.limit]

    print(f"[pipeline] {len(specs)} spec file(s) considered"
          f"{f', filtering to ids={sorted(target_ids)}' if target_ids else ''} ...",
          file=sys.stderr)

    ran = 0
    for spec in specs:
        with open(spec, encoding="utf-8") as f:
            cfg = json.load(f)
        qid = cfg.get("question_id")
        if target_ids is not None and qid not in target_ids:
            continue

        folder = os.path.join(args.out_dir, f"q{qid}")
        os.makedirs(folder, exist_ok=True)
        print(f"\n{'='*62}\n[pipeline] Q{qid}: {cfg.get('topic')}\n{'='*62}",
              file=sys.stderr)

        # already fully judged from a prior run? skip the whole question --
        # this is what makes rerunning after a partial failure (e.g. a
        # transient 503 on one question) cheap instead of redoing everything.
        already_judged = (
            os.path.exists(os.path.join(folder, "scored_rag.json")) and
            os.path.exists(os.path.join(folder, "scored_plain.json"))
        )
        if already_judged and not args.force:
            print(f"[pipeline] Q{qid}: already generated + judged -- skipping "
                  f"(use --force to redo).", file=sys.stderr)
            ran += 1
            continue

        # ---- step 1/4: generate (rag + plain) ----
        print(f"[pipeline] step 1/4 -- generating answers (model={args.model}, k={args.k}, "
              f"reranker={args.reranker}, mmr={not args.no_mmr}, timeout={args.timeout}s) ...",
              file=sys.stderr)
        _write_gt(cfg, os.path.join(folder, "gt.json"))
        ok = True
        for mode, name in (("rag", "rag.json"), ("plain", "plain.json")):
            out_path = os.path.join(folder, name)
            if os.path.exists(out_path) and not args.force:
                print(f"  [{mode}] {name} already exists -- reusing (use --force to redo)",
                      file=sys.stderr)
                continue
            try:
                res = _run_with_timeout(run_question, cfg, mode=mode, model=args.model,
                                        k=args.k, max_tokens=args.max_tokens,
                                        timeout=args.timeout,
                                        use_reranker=args.reranker,
                                        use_mmr=not args.no_mmr,
                                        lambda_mmr=args.lambda_mmr)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)
                print(f"  [{mode}] wrote {name} (chunks={len(res['chunks'])})",
                      file=sys.stderr)
            except FutureTimeoutError:
                print(f"  [{mode}] TIMED OUT after {args.timeout}s -- the underlying call "
                      f"may still be running in the background. Consider killing and "
                      f"restarting this process (a lingering stuck thread won't free "
                      f"itself). Use kill -USR1 {os.getpid()} first if you want a stack "
                      f"trace before killing.", file=sys.stderr)
                ok = False
            except Exception as e:  # noqa: BLE001
                print(f"  [{mode}] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                ok = False
        if not ok:
            print(f"[pipeline] Q{qid}: generation failed/timed out -- skipping judge step.",
                  file=sys.stderr)
            continue

        # ---- step 2/4: judge (rag vs plain) ----
        print(f"[pipeline] step 2/4 -- judging (judge={args.judge}, "
              f"citation-judge={args.citation_judge}) ...", file=sys.stderr)
        try:
            _run_with_timeout(judge_folder, folder, judge_model=args.judge,
                              citation_judge_model=args.citation_judge,
                              timeout=args.timeout)
        except FutureTimeoutError:
            print(f"[pipeline] Q{qid}: judge TIMED OUT after {args.timeout}s -- see note "
                  f"above about killing/restarting.", file=sys.stderr)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"[pipeline] Q{qid}: judge ERROR: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue

        ran += 1

    print(f"\n[pipeline] generated + judged {ran} question(s) -> {args.out_dir}",
          file=sys.stderr)

    # ---- step 3/4: summarize (rolls up every scored qN/ folder found on disk) ----
    print(f"[pipeline] step 3/4 -- summarizing ...", file=sys.stderr)
    summarize_main(["--runs", args.out_dir, "--out", args.summary_out])

    # ---- step 4/4: detailed report(s) -- answers, right/wrong, citations +
    #      cited chunk text, free-text winner + why. Written automatically
    #      every run, both as one consolidated file and one file per question.
    print(f"[pipeline] step 4/4 -- writing detailed report(s) ...", file=sys.stderr)
    build_report(args.out_dir, args.report_out)
    build_per_question_files(args.out_dir, args.report_per_question_dir)

    cost_tracker.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
