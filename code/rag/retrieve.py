"""
retrieve.py
-----------
Vector search over the ChromaDB collection populated by ingest_to_chroma.py,
with optional MMR (Maximal Marginal Relevance) re-ranking for diversity.

Public API:
    search(query, k=5, year_range=None, source=None,
           use_mmr=True, lambda_mmr=0.5, fetch_n=None) -> list[RetrievedChunk]

Why MMR matters here:
    Biomedical questions like "which bacteria are linked to colorectal cancer?"
    expect a *set* of distinct species. Plain top-k will often surface the same
    species over and over from different paragraphs of the same article. MMR
    re-ranks so chunks are both relevant AND diverse.

The actual MMR formula:
    score(c) = lambda * sim(query, c) - (1 - lambda) * max sim(c, s in selected)

IMPORTANT — distance metric assumption:
    This module assumes the Chroma collection was created with
    `metadata={"hnsw:space": "cosine"}` (which is exactly what
    ingest_to_chroma.py does). Under that setting, Chroma returns
    `distance = 1 - cosine_similarity`, so we recover similarity as
    `1 - distance`. If you change the ingest to use l2 or ip distance,
    update _distance_to_similarity() below or scores will be wrong.
"""

import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install numpy", file=sys.stderr)
    raise

try:
    import chromadb
except ImportError:
    print("ERROR: chromadb not installed. Run: pip install chromadb", file=sys.stderr)
    raise

from .embeddings import embed_query  # same-package relative import


DEFAULT_CHROMA_DIR  = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/chroma_db"
COLLECTION_NAME     = "cherrypicker_chunks"


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """One chunk returned by retrieval, with provenance and scores."""
    chunk_id: str            # the original OpenAI custom_id (globally unique)
    text: str
    score: float             # similarity in [0, 1] (1 - cosine distance)
    year: Optional[int] = None
    source: Optional[str] = None   # "FT" or "MO"
    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    split: Optional[str] = None
    n_chars: Optional[int] = None
    mmr_rank: Optional[int] = None    # populated when use_mmr=True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------
# Chroma collection access (cached at module level)
# ----------------------------------------------------------------------------

_collection = None
_collection_dir = None


def get_collection(chroma_dir: str = DEFAULT_CHROMA_DIR):
    """Open (and cache) the persistent Chroma collection."""
    global _collection, _collection_dir
    if _collection is not None and _collection_dir == chroma_dir:
        return _collection
    if not os.path.isdir(chroma_dir):
        raise FileNotFoundError(
            f"Chroma directory not found: {chroma_dir}. "
            f"Run ingest_to_chroma.py first."
        )
    client = chromadb.PersistentClient(path=chroma_dir)
    _collection = client.get_collection(name=COLLECTION_NAME)
    _collection_dir = chroma_dir
    return _collection


# ----------------------------------------------------------------------------
# Metadata filter construction
# ----------------------------------------------------------------------------

def _build_where(year_range: Optional[Tuple[int, int]],
                 source: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Build a Chroma `where` filter dict from high-level args.

    Chroma uses Mongo-style operators: {"year": {"$gte": 2020}}
    Multiple conditions are AND-ed via {"$and": [...]}.
    Returns None if no filters are requested.
    """
    clauses = []
    if year_range is not None:
        ymin, ymax = year_range
        if ymin is not None:
            clauses.append({"year": {"$gte": int(ymin)}})
        if ymax is not None:
            clauses.append({"year": {"$lte": int(ymax)}})
    if source:
        clauses.append({"source": {"$eq": source}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ----------------------------------------------------------------------------
# MMR
# ----------------------------------------------------------------------------

def _distance_to_similarity(distance: float) -> float:
    """
    Convert a Chroma distance to a [0, 1] cosine-similarity-like score.

    Assumes the collection was created with hnsw:space=cosine, where
    distance = 1 - cos_sim and ranges in [0, 2]. Clamping to [0, 1] guards
    against numerical noise and against the rare case where cosine yields
    a slightly-negative similarity (semantically opposite vectors).
    """
    sim = 1.0 - float(distance)
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Assumes neither is zero."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mmr_select(query_vec: np.ndarray,
                doc_vecs: np.ndarray,
                relevance: np.ndarray,
                k: int,
                lambda_mmr: float) -> List[int]:
    """
    Iteratively pick k indices from doc_vecs maximizing the MMR objective.

    Args:
        query_vec  : (D,)         query embedding
        doc_vecs   : (N, D)       candidate doc embeddings
        relevance  : (N,)         similarity of each doc to the query
        k          : how many to select
        lambda_mmr : 0..1  (1 = pure relevance, 0 = pure diversity)
    """
    n = doc_vecs.shape[0]
    if k >= n:
        # Nothing to re-rank; return everything in relevance order
        return list(np.argsort(-relevance))

    # Normalize for fast cosine via dot product
    norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_vecs_n = doc_vecs / norms

    selected: List[int] = []
    remaining = set(range(n))

    # First pick = pure max relevance
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    # Pre-compute pairwise sim against selected as we go
    while len(selected) < k and remaining:
        rem_list = list(remaining)
        sel_matrix = doc_vecs_n[selected]                  # (s, D)
        rem_matrix = doc_vecs_n[rem_list]                  # (r, D)
        # diversity_penalty[i] = max over selected of sim(rem_i, selected_j)
        sims_to_selected = rem_matrix @ sel_matrix.T       # (r, s)
        diversity_penalty = sims_to_selected.max(axis=1)   # (r,)

        rem_relevance = relevance[rem_list]
        mmr_scores = lambda_mmr * rem_relevance - (1.0 - lambda_mmr) * diversity_penalty

        best_in_rem = int(np.argmax(mmr_scores))
        best_idx = rem_list[best_in_rem]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# ----------------------------------------------------------------------------
# Public search function
# ----------------------------------------------------------------------------

def search(query: str,
           k: int = 5,
           year_range: Optional[Tuple[int, int]] = None,
           source: Optional[str] = None,
           use_mmr: bool = True,
           lambda_mmr: float = 0.5,
           fetch_n: Optional[int] = None,
           chroma_dir: str = DEFAULT_CHROMA_DIR) -> List[RetrievedChunk]:
    """
    Run a similarity search and return the top-k chunks as RetrievedChunk objects.

    Args:
        query       : natural-language question
        k           : how many chunks to return (default 5)
        year_range  : optional (min_year, max_year) tuple, inclusive
        source      : optional "FT" (full-text) or "MO" (metadata-only)
        use_mmr     : if True, fetch more candidates and MMR-rerank to k
        lambda_mmr  : MMR tradeoff: 1.0 = relevance only, 0.0 = diversity only
        fetch_n     : how many candidates to pull from Chroma before MMR.
                      Default = max(4*k, 20) when MMR is on, else k.
        chroma_dir  : where the persistent Chroma DB lives
    """
    if not query or not query.strip():
        raise ValueError("search() received empty query.")
    if k < 1:
        raise ValueError("k must be >= 1.")
    if not 0.0 <= lambda_mmr <= 1.0:
        raise ValueError(f"lambda_mmr must be in [0, 1], got {lambda_mmr}.")
    if fetch_n is not None and fetch_n < k:
        raise ValueError(f"fetch_n ({fetch_n}) must be >= k ({k}).")
    if source is not None and source not in {"FT", "MO"}:
        raise ValueError(
            f"source must be 'FT' (full-text), 'MO' (metadata-only), or None; got {source!r}."
        )

    coll = get_collection(chroma_dir)

    # Embed the query with the same model used on the corpus
    q_vec = embed_query(query)

    # Decide how many candidates to fetch from Chroma
    if not use_mmr:
        n_results = k
    else:
        n_results = fetch_n if fetch_n is not None else max(4 * k, 20)

    where = _build_where(year_range, source)

    res = coll.query(
        query_embeddings=[q_vec],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    # Chroma returns nested lists keyed by the (single) query
    ids        = res.get("ids",       [[]])[0]
    docs       = res.get("documents", [[]])[0]
    metas      = res.get("metadatas", [[]])[0]
    distances  = res.get("distances", [[]])[0]
    embeddings = res.get("embeddings", [[]])[0]

    if not ids:
        return []

    # Convert Chroma distances to similarities. See module docstring:
    # we rely on the collection being created with hnsw:space=cosine.
    relevance = np.array([_distance_to_similarity(d) for d in distances], dtype=float)

    if use_mmr and len(ids) > k:
        q_arr = np.array(q_vec, dtype=float)
        doc_arr = np.array(embeddings, dtype=float)
        selected_idx = _mmr_select(q_arr, doc_arr, relevance, k, lambda_mmr)
    else:
        selected_idx = list(range(min(k, len(ids))))

    out: List[RetrievedChunk] = []
    for rank, idx in enumerate(selected_idx, start=1):
        md = metas[idx] or {}
        out.append(RetrievedChunk(
            chunk_id=ids[idx],
            text=docs[idx] or "",
            score=float(relevance[idx]),
            year=md.get("year"),
            source=md.get("source"),
            pmcid=md.get("pmcid"),
            pmid=md.get("pmid"),
            split=md.get("split"),
            n_chars=md.get("n_chars"),
            mmr_rank=rank if use_mmr else None,
        ))
    return out
