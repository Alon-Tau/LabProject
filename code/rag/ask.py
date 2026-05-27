#!/usr/bin/env python3
"""
ask.py
------
CLI front-end for running a single RAG query.

Usage (from the project root, e.g. /home/elhanan/PROJECTS/CHERRY_PICKER_AR):

    python -m code.rag.ask "Which bacterial species are associated with colorectal cancer?"

    # specify a model and inspect the retrieved chunks
    python -m code.rag.ask "..." --model claude-sonnet-4-5 --show-chunks

    # filter by year and source (Tier 1 / full-text only, recent)
    python -m code.rag.ask "..." --year-min 2020 --source FT

    # turn MMR off, fetch more candidates
    python -m code.rag.ask "..." --no-mmr --k 10

    # print the EXACT prompt sent to the LLM (great for debugging)
    python -m code.rag.ask "..." --show-prompt

Prereqs:
    * OPENAI_API_KEY exported (always, for query embedding)
    * ANTHROPIC_API_KEY exported when using a claude-* model
    * Chroma DB built at the default path or pass --chroma-dir
"""

import sys
import argparse
import textwrap
import logging
from typing import Optional, Tuple


from .answer import answer, AnswerResult
from .retrieve import search


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


def _parse_year_range(year_min: Optional[int],
                      year_max: Optional[int]) -> Optional[Tuple[int, int]]:
    if year_min is None and year_max is None:
        return None
    return (year_min, year_max)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ask",
        description="Single-question RAG query against the CherryPicker corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("question", help="The question to ask, as a string.")
    p.add_argument("--model",      default="gpt-4o",
                   help="LLM to call. Examples: gpt-4o, gpt-4o-mini, "
                        "claude-sonnet-4-5, claude-opus-4-5. (default: gpt-4o)")
    p.add_argument("--k",          type=int, default=5,
                   help="Number of chunks to retrieve (default: 5)")
    p.add_argument("--year-min",   type=int, default=None,
                   help="Only retrieve chunks from year >= this.")
    p.add_argument("--year-max",   type=int, default=None,
                   help="Only retrieve chunks from year <= this.")
    p.add_argument("--source",     choices=["FT", "MO"], default=None,
                   help="Restrict to full-text (FT) or metadata-only (MO).")
    p.add_argument("--no-mmr",     action="store_true",
                   help="Disable MMR re-ranking; use plain top-k.")
    p.add_argument("--lambda-mmr", type=float, default=0.5,
                   help="MMR tradeoff in [0,1] (default 0.5). "
                        "1 = pure relevance, 0 = pure diversity.")
    p.add_argument("--fetch-n",    type=int, default=None,
                   help="How many candidates to fetch from Chroma before "
                        "MMR. Default = max(4*k, 20).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="LLM temperature (default 0.0 for reproducibility).")
    p.add_argument("--max-tokens", type=int, default=2000,
                   help="Hard cap on LLM output length (default 2000).")
    p.add_argument("--chroma-dir", default=None,
                   help="Override the Chroma DB path.")
    p.add_argument("--min-score", type=float, default=None,
                   help="If the top retrieved chunk scores below this, "
                        "the LLM is NOT called and the system replies with a "
                        "refusal. Typical range 0.20-0.30. Default: disabled.")
    p.add_argument("--retrieve-only", action="store_true",
                   help="Skip the LLM call entirely. Print retrieved chunks "
                        "and exit. Great for debugging retrieval quality "
                        "without spending tokens.")
    p.add_argument("--show-chunks", action="store_true",
                   help="Print retrieved chunks alongside the answer.")
    p.add_argument("--show-prompt", action="store_true",
                   help="Print the full system+user prompt that was sent.")
    args = p.parse_args(argv)

    # ---- retrieve-only short-circuit (no LLM call) ----
    if args.retrieve_only:
        # Build kwargs conditionally so we never pass chroma_dir=None and
        # override the default; mirrors the pattern in answer.answer().
        search_kwargs = dict(
            k=args.k,
            year_range=_parse_year_range(args.year_min, args.year_max),
            source=args.source,
            use_mmr=not args.no_mmr,
            lambda_mmr=args.lambda_mmr,
            fetch_n=args.fetch_n,
        )
        if args.chroma_dir is not None:
            search_kwargs["chroma_dir"] = args.chroma_dir
        try:
            chunks = search(args.question, **search_kwargs)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        # Build a minimal AnswerResult-like shim so _print_chunks works.
        # Include the real retrieval_params and top_score so the header
        # reports the actual k, filters, MMR settings, and top similarity.
        shim = AnswerResult(
            question=args.question,
            answer_text="(retrieve-only mode: no LLM call)",
            model="(none)",
            chunks=chunks,
            cited_ranks=[],
            top_score=chunks[0].score if chunks else None,
            retrieval_params={k: v for k, v in search_kwargs.items() if v is not None},
        )
        _print_answer(shim)
        _print_chunks(shim)
        return 0

    try:
        res = answer(
            question=args.question,
            model=args.model,
            k=args.k,
            year_range=_parse_year_range(args.year_min, args.year_max),
            source=args.source,
            use_mmr=not args.no_mmr,
            lambda_mmr=args.lambda_mmr,
            fetch_n=args.fetch_n,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            min_score=args.min_score,
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
    # Configure logging BEFORE main() so tenacity retry warnings from the
    # embedding and LLM wrappers are visible during a CLI run.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())

