#!/usr/bin/env python3
"""
MMR Retrieval (LangChain) with dynamic k + per-article cap

What it does
- Loads an existing Chroma vector store (persisted on disk)
- Runs MMR retrieval with configurable:
    - k (final number of chunks returned)
    - fetch_k (candidates before MMR selection)
    - lambda_mult (relevance vs diversity)
    - optional metadata filter (vector-store dependent)
- Applies a post-step: cap max chunks per article (pmid/pmcid/article_id)

Usage examples
1) Simple (defaults):
    python retrieve_mmr.py --query "CFTR mucus viscosity mechanism"

2) Control number of chunks:
    python retrieve_mmr.py --query "..." --k 8

3) More diversity:
    python retrieve_mmr.py --query "..." --lambda-mult 0.25

4) Print more text per chunk:
    python retrieve_mmr.py --query "..." --preview-chars 800
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


# =========================
# CONFIG (EDIT THESE)
# =========================
PERSIST_DIR = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/vectorstore_chroma" #need to update after embedding
COLLECTION = "cherry_picker_ar"  #need to update after embedding
EMBEDDING_MODEL = "text-embedding-3-small"
# =========================


def build_vectorstore() -> Chroma:
    """Load a persisted Chroma collection."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )


def cap_per_article(
    docs: Sequence[Document],
    max_per_article: int = 2,
    id_keys: Tuple[str, ...] = ("pmid", "pmcid", "article_id"),
) -> List[Document]:
    """
    Keep at most `max_per_article` chunks per article. we will take at most max_per_article chunks from each article.

    We identify the article by the first existing metadata key in `id_keys`.
    If none exist, we treat it as "unknown" (still capped as one bucket).
    """
    counts: Dict[str, int] = defaultdict(int)
    out: List[Document] = []

    for d in docs:
        meta = d.metadata or {}
        aid = None
        for k in id_keys:
            if meta.get(k):
                aid = str(meta[k])
                break
        if aid is None:
            aid = "unknown"

        if counts[aid] < max_per_article:
            out.append(d)
            counts[aid] += 1

    return out


def mmr_retrieve(
    vs: Chroma,
    query: str,
    k: int = 6,
    fetch_k: Optional[int] = None,
    lambda_mult: float = 0.35,
    filter_dict: Optional[Dict[str, Any]] = None,
    max_per_article: int = 2,
) -> List[Document]:
    """
    MMR retrieval:
      - k: final docs returned after MMR
      - fetch_k: candidates before MMR selects k (defaults to max(30, k*8))
      - lambda_mult: 0..1 (lower = more diversity, higher = more relevance)
      - filter_dict: optional metadata filter (DB-specific)
      - max_per_article: cap chunks per article AFTER retrieval
    """
    if fetch_k is None:
        fetch_k = max(30, k * 8)

    search_kwargs: Dict[str, Any] = {
        "k": k,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
    }
    if filter_dict:
        # NOTE: Chroma supports simple equality filtering; advanced operators depend on your setup.
        # If you use Qdrant/Pinecone/etc., you’ll adapt filter syntax accordingly.
        search_kwargs["filter"] = filter_dict

    retriever = vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    docs = retriever.invoke(query)

    if max_per_article and max_per_article > 0:
        docs = cap_per_article(docs, max_per_article=max_per_article)

    return docs[:k]


def pretty_print_docs(docs: Sequence[Document], preview_chars: int = 500) -> None:
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        source = (
            meta.get("pmid")
            or meta.get("pmcid")
            or meta.get("doi")
            or meta.get("article_id")
            or meta.get("chunk_id")
            or "unknown"
        )
        journal = meta.get("journal", "unknown_journal")
        year = meta.get("year", "unknown_year")

        text = (d.page_content or "").strip().replace("\n", " ")
        if len(text) > preview_chars:
            text = text[:preview_chars] + "..."

        print(f"\n--- DOC {i} ---")
        print(f"source: {source} | journal: {journal} | year: {year}")
        print(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMR retrieval with per-article cap")
    p.add_argument("--query", required=True, type=str, help="User query text")
    p.add_argument("--k", type=int, default=6, help="Final number of chunks to return")
    p.add_argument(
        "--fetch-k",
        type=int,
        default=None,
        help="Candidate pool size before MMR selects k (default: max(30, k*8))",
    )
    p.add_argument(
        "--lambda-mult",
        type=float,
        default=0.35,
        help="MMR relevance/diversity tradeoff (0..1). Lower=more diverse.",
    )
    p.add_argument(
        "--max-per-article",
        type=int,
        default=2,
        help="Max chunks allowed per article (based on pmid/pmcid/article_id).",
    )
    p.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="How many characters of each chunk to print",
    )

    # Optional: simple filter example (key=value). You can pass multiple.
    # Note: advanced operators are vectorstore-specific.
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        help='Metadata filter as key=value (can repeat). Example: --filter journal="Nature"',
    )
    return p.parse_args()


def build_filter_dict(filter_args: List[str]) -> Optional[Dict[str, Any]]:
    """
    Build a simple dict filter from repeated --filter key=value arguments.
    Example:
      --filter journal=Nature --filter year=2021
    -> {"journal": "Nature", "year": 2021}
    """
    if not filter_args:
        return None

    filt: Dict[str, Any] = {}
    for item in filter_args:
        if "=" not in item:
            raise ValueError(f"Bad --filter '{item}'. Use key=value.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")

        # Try int cast for convenience (e.g., year=2021)
        if v.isdigit():
            filt[k] = int(v)
        else:
            filt[k] = v

    return filt if filt else None


def main() -> None:
    args = parse_args()
    filter_dict = build_filter_dict(args.filter)

    vs = build_vectorstore()
    docs = mmr_retrieve(
        vs=vs,
        query=args.query,
        k=args.k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
        filter_dict=filter_dict,
        max_per_article=args.max_per_article,
    )

    print(f"\nRetrieved {len(docs)} document chunks (k={args.k}, max_per_article={args.max_per_article}).")
    pretty_print_docs(docs, preview_chars=args.preview_chars)


if __name__ == "__main__":
    main()
