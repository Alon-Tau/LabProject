#!/usr/bin/env python3
"""
ask_group.py
------------
Two-stage RAG: given a group of microbiome features (bacteria, metabolites,
pathways, ...) plus a question, retrieve from only those articles that
mention at least K features from each requested category, then answer.

Pipeline:
    1. Read a JSON config (--query) with the question + features + thresholds
    2. resolve user feature names -> canonical IDs (entity_resolver)
    3. find eligible PMCIDs from the SQLite entity index (feature_filter)
    4. vector search restricted to those PMCIDs (retrieve.search via answer())
    5. apply MMR re-rank, build prompt, call LLM (answer())
    6. print the answer plus optional debugging detail

Example config (my_query.json):

    {
      "question": "Explain the biological mechanism connecting these microbiome features to CRC.",
      "features": {
        "bacteria":   ["Bacteroides dorei", "Bilophila wadsworthia",
                       "Clostridium bolteae", "Parabacteroides merdae",
                       "Pseudoflavonifractor capillosus"],
        "metabolite": ["Pyruvic acid", "alpha-ketoglutaric acid", "D-mannose",
                       "Isoleucine", "Glutamic acid"]
      },
      "thresholds": {"bacteria": 3, "metabolite": 3}
    }

Usage:
    python -m code.rag.ask_group --query my_query.json
    python -m code.rag.ask_group --query my_query.json --model claude-sonnet-4-5
    python -m code.rag.ask_group --query my_query.json --show-chunks --show-resolution
    python -m code.rag.ask_group --query my_query.json --retrieve-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap

from .answer            import answer, AnswerResult
from .retrieve          import search
from bouncer.entity_resolver import resolve_features, DEFAULT_VOCAB_PATH
from bouncer.feature_filter import eligible_pmcids, per_article_breakdown, DEFAULT_INDEX_PATH


# --------------------------------------------------------------------------
# Pretty printing helpers (re-used from ask.py style)
# --------------------------------------------------------------------------

def _wrap(s: str, width: int = 100, indent: str = "  ") -> str:
    return textwrap.fill(s, width=width, initial_indent=indent,
                         subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False)


def _print_chunks(res: AnswerResult, max_chars: int = 400) -> None:
    print("\n" + "=" * 80)
    print(f"RETRIEVED CHUNKS ({len(res.chunks)}):")
    print("=" * 80)
    for i, c in enumerate(res.chunks, start=1):
        prov_bits = []
        if c.year:   prov_bits.append(str(c.year))
        if c.pmcid:  prov_bits.append(c.pmcid)
        elif c.pmid: prov_bits.append(f"PMID:{c.pmid}")
        if c.source: prov_bits.append(c.source)
        prov = " | ".join(prov_bits) if prov_bits else "(no provenance)"
        cited_marker = " ★ cited" if i in res.cited_ranks else ""
        print(f"\n[{i}] score={c.score:.3f}  {prov}{cited_marker}")
        text = c.text.strip().replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        print(_wrap(text))


def _print_prompt(res: AnswerResult) -> None:
    print("\n" + "=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(res.system_prompt)
    print("\n" + "=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(res.user_prompt)


def _print_answer(res: AnswerResult) -> None:
    print("\n" + "=" * 80)
    print(f"ANSWER  (model={res.model}, k={res.retrieval_params.get('k', '?')}, "
          f"cited={res.cited_ranks or 'none'})")
    print("=" * 80)
    print(res.answer_text)
    print()


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------

def load_query_config(path: str) -> dict:
    """Read + validate a query JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise SystemExit(f"{path}: top-level JSON must be an object.")
    if "question" not in cfg or not isinstance(cfg["question"], str) or not cfg["question"].strip():
        raise SystemExit(f"{path}: missing or empty 'question' field.")
    if "features" not in cfg or not isinstance(cfg["features"], dict):
        raise SystemExit(f"{path}: missing 'features' object.")
    for cat, terms in cfg["features"].items():
        if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
            raise SystemExit(f"{path}: features.{cat} must be a list of strings.")
    if "thresholds" in cfg:
        if not isinstance(cfg["thresholds"], dict):
            raise SystemExit(f"{path}: 'thresholds' must be an object.")
        for cat, k in cfg["thresholds"].items():
            if not isinstance(k, int) or k < 0:
                raise SystemExit(f"{path}: thresholds.{cat} must be a non-negative int.")
    return cfg


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ask_group",
        description="Two-stage RAG: feature-filter article candidates, then "
                    "vector-search + answer over only those articles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--query", required=True,
                   help="Path to a JSON config file (see module docstring for schema).")
    p.add_argument("--model",      default="gpt-4o")
    p.add_argument("--k",          type=int, default=5)
    p.add_argument("--no-mmr",     action="store_true")
    p.add_argument("--lambda-mmr", type=float, default=0.5)
    p.add_argument("--fetch-n",    type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--min-score",  type=float, default=None)
    p.add_argument("--chroma-dir", default=None)
    p.add_argument("--vocab",      default=DEFAULT_VOCAB_PATH)
    p.add_argument("--entity-db",  default=DEFAULT_INDEX_PATH)
    p.add_argument("--retrieve-only", action="store_true",
                   help="Skip the LLM call. Print stage-1 PMCIDs and stage-2 chunks.")
    p.add_argument("--show-chunks", action="store_true",
                   help="Print retrieved chunks alongside the answer.")
    p.add_argument("--show-prompt", action="store_true",
                   help="Print the full system+user prompt that was sent.")
    p.add_argument("--show-resolution", action="store_true",
                   help="Print how user feature names mapped to canonical IDs.")
    p.add_argument("--max-pmcids", type=int, default=5000,
                   help="Hard cap on PMCIDs passed to Chroma's $in filter. "
                        "If stage 1 returns more than this, the run aborts "
                        "with an error (tighten your K thresholds).")
    p.add_argument("--exclude-pmcids", default=None,
                   help="Comma-separated PMCIDs to drop from the stage-1 set "
                        "before vector search (e.g. the source paper, to avoid "
                        "the RAG echoing it).")
    args = p.parse_args(argv)

    cfg = load_query_config(args.query)
    question  = cfg["question"]
    features  = cfg["features"]
    thresholds = cfg.get("thresholds")   # may be None

    # ---- Stage 1a: resolve user terms -> canonical IDs ----
    print("[ask_group] resolving feature names ...", file=sys.stderr)
    rfg = resolve_features(features, vocab_path=args.vocab)
    if args.show_resolution:
        print(rfg.summary())
    if not rfg.all_resolved():
        print("WARNING: some user terms did not resolve to any canonical ID. "
              "They were dropped silently. Use --show-resolution to see which.",
              file=sys.stderr)

    # ---- Stage 1b: find eligible PMCIDs from the entity index ----
    print("[ask_group] querying entity index ...", file=sys.stderr)
    pmcids = eligible_pmcids(rfg.resolved, thresholds, index_path=args.entity_db)
    print(f"[ask_group] stage 1: {len(pmcids):,} eligible articles", file=sys.stderr)

    if args.exclude_pmcids:
        drop = {x.strip() for x in args.exclude_pmcids.split(",") if x.strip()}
        before = len(pmcids)
        pmcids = pmcids - drop
        print(f"[ask_group] excluded {before - len(pmcids)} PMCID(s): {sorted(drop)}",
              file=sys.stderr)
    
    if not pmcids:
        print("\nNo articles satisfied the feature-group constraints.")
        print("Try a lower K threshold, fewer required categories, or check "
              "--show-resolution to confirm your features resolved.")
        return 0

    if len(pmcids) > args.max_pmcids:
        raise SystemExit(
            f"Stage 1 returned {len(pmcids):,} PMCIDs but --max-pmcids={args.max_pmcids}. "
            f"Raise --max-pmcids or tighten your thresholds."
        )

    # ---- Stage 2 (retrieve-only branch) ----
    if args.retrieve_only:
        print("[ask_group] --retrieve-only: vector search, no LLM call", file=sys.stderr)
        search_kwargs = dict(
            k=args.k,
            use_mmr=not args.no_mmr,
            lambda_mmr=args.lambda_mmr,
            fetch_n=args.fetch_n,
            restrict_to_pmcids=pmcids,
        )
        if args.chroma_dir is not None:
            search_kwargs["chroma_dir"] = args.chroma_dir
        try:
            chunks = search(question, **search_kwargs)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        shim = AnswerResult(
            question=question,
            answer_text="(retrieve-only mode: no LLM call)",
            model="(none)",
            chunks=chunks,
            cited_ranks=[],
            top_score=chunks[0].score if chunks else None,
            retrieval_params={**{k: v for k, v in search_kwargs.items()
                                  if k != "restrict_to_pmcids" and v is not None},
                              "stage1_pmcids": len(pmcids)},
        )
        _print_answer(shim)
        _print_chunks(shim)
        return 0

    # ---- Stage 2 + 3: real answer ----
    try:
        res = answer(
            question=question,
            model=args.model,
            k=args.k,
            use_mmr=not args.no_mmr,
            lambda_mmr=args.lambda_mmr,
            fetch_n=args.fetch_n,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            min_score=args.min_score,
            restrict_to_pmcids=pmcids,
            chroma_dir=args.chroma_dir,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    _print_answer(res)
    if res.invalid_cited_ranks:
        print(f"WARNING: LLM cited non-existent chunks: {res.invalid_cited_ranks}",
              file=sys.stderr)
    if res.refused:
        if res.top_score is not None:
            print(f"NOTE: refused (top_score={res.top_score:.3f} < --min-score)",
                  file=sys.stderr)
        else:
            print("NOTE: refused (no chunks retrieved)", file=sys.stderr)
    if args.show_chunks:
        _print_chunks(res)
    if args.show_prompt:
        _print_prompt(res)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
