"""
rerank.py
---------
Cross-encoder re-ranking of already-retrieved chunks.

Where this sits in the pipeline:
    retrieve.search()   ->  [50 chunks, ordered by bi-encoder cosine similarity]
                                  |
                                  v
                         rerank.rerank_chunks()      <-- THIS MODULE
                                  |
                                  v
                         rerank.filter_by_gap()
                                  |
                                  v
                         prompts.build_rag_prompt()

Why a cross-encoder instead of trusting the embedding score:
    retrieve.py's `score` is a bi-encoder cosine similarity: the query and
    each chunk were embedded INDEPENDENTLY (chunks at ingest time, query at
    search time) and never seen together. That captures topical closeness
    ("this is about gut microbiome and bile acids") but not fine-grained
    entailment ("does this passage state THIS specific compound / direction /
    species"). The July 2026 audit of all 100 eval questions found GOOD and
    BAD citation chunks were statistically indistinguishable by that cosine
    score (GOOD mean 0.6234 vs BAD mean 0.6220, n=859 individual citations) --
    so a threshold on it has no power to separate faithful from unfaithful
    citations.

    A cross-encoder instead feeds the query and ONE chunk together into a
    single forward pass ("[CLS] query [SEP] chunk [SEP]"), so its attention
    layers can directly contrast tokens between the two texts. It should be
    better at catching exactly the failure mode the audit found most often:
    a topically-adjacent chunk about the wrong specific compound / species /
    direction (e.g. spermine vs. spermidine, or a sibling Bacteroides
    species) scoring as "relevant" as a genuinely correct one.

    It is NOT a fact-checker: it still can't surface evidence that was never
    retrieved at all (the Q2/Q59-style cases, where the correct answer is
    absent from all 50 chunks). That gap is what the prompt-side abstention
    instruction (see prompts.py) is for, not this module.

Public API:
    rerank_chunks(query, chunks, model_name=DEFAULT_RERANK_MODEL)
        -> list[RetrievedChunk]
        Scores every chunk against the query with the cross-encoder and
        returns a NEW list sorted by rerank_score descending. Also sets
        each chunk's `.rerank_score` in place.

    filter_by_gap(chunks, min_keep=25, max_keep=50, gap_sigma=1.0) -> list[RetrievedChunk]
        Drops chunks whose rerank_score falls more than `gap_sigma` standard
        deviations below the top score for THIS query -- a per-query
        relative cutoff, not a fixed top-N truncation. If fewer than
        `min_keep` chunks would survive, falls back to the top `min_keep`
        by rerank_score instead (never hands the generator a too-thin or
        empty evidence set).

    rerank_and_filter(query, chunks, ...) -> list[RetrievedChunk]
        Convenience wrapper: rerank_chunks() then filter_by_gap().

Model choice:
    Default is a general-purpose MS MARCO cross-encoder -- small, CPU-only,
    no API cost, ~50 pairs scored in well under a second. Swap
    DEFAULT_RERANK_MODEL for a biomedical-tuned cross-encoder later if the
    general one isn't sharp enough on entity-level distinctions; every
    function here takes model_name as an argument, so nothing else in the
    pipeline needs to change to try a different reranker.
"""

import sys
from functools import lru_cache
from typing import List, Optional, TYPE_CHECKING

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install numpy", file=sys.stderr)
    raise

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print(
        "ERROR: sentence-transformers not installed. Run:\n"
        "    pip install sentence-transformers\n"
        "(this pulls in torch; the first call also downloads the model "
        "weights once, then caches them under "
        "~/.cache/torch/sentence_transformers/).",
        file=sys.stderr,
    )
    raise

if TYPE_CHECKING:
    from .retrieve import RetrievedChunk


# General-purpose MS MARCO cross-encoder: small, CPU-friendly, well-tested.
# Swap this for a biomedical-tuned cross-encoder (e.g. a PubMedBERT-based
# one) if entity-level precision (compound/species/gene names) needs
# sharpening -- nothing else in this module or its callers needs to change
# to try a different model, just pass model_name=... through.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@lru_cache(maxsize=4)
def _model(model_name: str) -> "CrossEncoder":
    """Load (and cache) a CrossEncoder by name, so repeated calls within one
    process don't reload weights from disk every time."""
    print(f"[rerank] loading cross-encoder '{model_name}' ...", file=sys.stderr)
    return CrossEncoder(model_name)


def rerank_chunks(query: str,
                  chunks: List["RetrievedChunk"],
                  model_name: str = DEFAULT_RERANK_MODEL) -> List["RetrievedChunk"]:
    """
    Score every chunk against `query` with a cross-encoder and return a NEW
    list sorted by that score, descending. Mutates each chunk's
    `.rerank_score` in place, so the original list's chunk objects also
    carry the score even if a caller kept a separate reference to them.

    Does NOT drop anything -- this only rescores and reorders. Pair with
    filter_by_gap() if you also want to drop clearly-irrelevant chunks.
    """
    if not query or not query.strip():
        raise ValueError("rerank_chunks() received an empty query.")
    if not chunks:
        return []

    pairs = [(query, c.text) for c in chunks]
    scores = _model(model_name).predict(pairs)  # raw logits; only relative order matters

    for c, s in zip(chunks, scores):
        c.rerank_score = float(s)

    return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)


def filter_by_gap(chunks: List["RetrievedChunk"],
                  min_keep: int = 25,
                  max_keep: Optional[int] = 50,
                  gap_sigma: float = 1.0) -> List["RetrievedChunk"]:
    """
    Drop chunks whose rerank_score is far below the best score for THIS
    query -- a per-query relative cutoff, not a fixed top-N truncation.

    Rationale (from the July 2026 rank-based analysis of all 100 eval
    questions): truncating to a fixed k (e.g. top 10) strips away
    scattered-but-good evidence that RAG's best answers actually depend
    on -- at k=10, only ~19% of GOOD citations survived vs. ~22% of BAD
    ones, i.e. fixed truncation removes good and bad evidence at about the
    same rate, and disproportionately hurts well-supported questions whose
    evidence happens to be spread past rank 10. A relative gap only removes
    chunks that are clear outliers *for this specific query* (this is what
    would actually help a case like Q2, where all 50 chunks were uniformly
    mediocre topic-adjacent noise with no real standout), and leaves a
    well-supported query's full spread of evidence untouched even if it
    runs deep.

    Args:
        chunks     : output of rerank_chunks() -- every chunk must already
                     have .rerank_score set. Does not need to be pre-sorted;
                     this function sorts by rerank_score descending itself.
        min_keep   : never return FEWER than this many chunks. Guards
                     against a degenerate case (e.g. near-zero score
                     variance, or a query where nothing clears the gap)
                     collapsing the evidence set to almost nothing.
        max_keep   : never return MORE than this many chunks, even if the
                     gap-based cutoff would otherwise pass more. This is a
                     ceiling on context size / attention dilution for the
                     generator (long, mostly-similar-scoring evidence sets
                     can bury the genuinely load-bearing chunks in the
                     middle of the prompt), NOT a re-introduction of blind
                     k-truncation -- it only bites when a query's own score
                     distribution says more than `max_keep` chunks are
                     genuinely close to the top, which the earlier k=10
                     analysis never actually observed happening at the low
                     end. Set to None to disable (unbounded above).
        gap_sigma  : cutoff = max(top_score - gap_sigma * std, mean_score).
                     Larger gap_sigma keeps more chunks (more permissive);
                     smaller cuts harder. Start at 1.0 and tune against a
                     side-by-side eval run, not by intuition -- exactly the
                     same discipline used to rule out a flat score
                     threshold and k=10 truncation earlier.
    """
    if not chunks:
        return []
    if any(c.rerank_score is None for c in chunks):
        raise ValueError(
            "filter_by_gap() requires rerank_score to be set on every "
            "chunk -- call rerank_chunks() first."
        )

    # Sort defensively so both the min_keep and max_keep fallbacks below can
    # safely assume descending order, regardless of the order chunks arrived in.
    chunks = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

    if len(chunks) <= min_keep:
        return chunks

    scores = np.array([c.rerank_score for c in chunks], dtype=float)
    mu, sigma = float(scores.mean()), float(scores.std())
    cutoff = max(scores.max() - gap_sigma * sigma, mu)

    kept = [c for c in chunks if c.rerank_score >= cutoff]

    if len(kept) < min_keep:
        # Guard: never hand the generator fewer than min_keep chunks, even
        # if the gap-based cutoff happened to be aggressive this time.
        return chunks[:min_keep]
    if max_keep is not None and len(kept) > max_keep:
        # Guard: never hand the generator more than max_keep chunks, even
        # if the gap-based cutoff was permissive this time (e.g. a query
        # where a large fraction of the 50 chunks all score similarly
        # well). `kept` is already sorted descending, so this is just
        # "top max_keep of the ones that passed the gap cutoff."
        return kept[:max_keep]
    return kept


def rerank_and_filter(query: str,
                      chunks: List["RetrievedChunk"],
                      model_name: str = DEFAULT_RERANK_MODEL,
                      min_keep: int = 25,
                      max_keep: Optional[int] = 50,
                      gap_sigma: float = 1.0) -> List["RetrievedChunk"]:
    """Convenience: rerank_chunks() then filter_by_gap() in one call."""
    reranked = rerank_chunks(query, chunks, model_name=model_name)
    return filter_by_gap(reranked, min_keep=min_keep, max_keep=max_keep,
                         gap_sigma=gap_sigma)


# ----------------------------------------------------------------------------
# Manual smoke test -- run directly against the real pipeline, e.g.:
#     python -m rag.rerank "does spermidine protect against gulf war illness?"
# Requires the same environment as retrieve.py (OPENAI_API_KEY set, populated
# Chroma dir). Prints the original bi-encoder order next to the reranked
# order and the gap-filter decision, so you can eyeball whether reordering
# makes sense on a real question before wiring it into eval_run.py (step 4).
# ----------------------------------------------------------------------------

def _smoke_test(argv=None) -> int:
    from .retrieve import search

    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m rag.rerank \"your question here\"", file=sys.stderr)
        return 1
    query = " ".join(argv)

    print(f"[rerank smoke test] retrieving top 200 for: {query!r}", file=sys.stderr)
    chunks = search(query, k=200, use_mmr=False)
    if not chunks:
        print("[rerank smoke test] no chunks retrieved -- check Chroma dir / query.",
              file=sys.stderr)
        return 1

    reranked = rerank_chunks(query, list(chunks))
    filtered = filter_by_gap(reranked)

    orig_rank = {c.chunk_id: i + 1 for i, c in enumerate(chunks)}
    kept_ids = {c.chunk_id for c in filtered}

    print(f"\n{'new#':>4}  {'orig#':>5}  {'embed_score':>11}  {'rerank_score':>12}  "
          f"{'kept?':>5}  pmcid")
    for new_rank, c in enumerate(reranked, start=1):
        kept = "yes" if c.chunk_id in kept_ids else "no"
        print(f"{new_rank:>4}  {orig_rank[c.chunk_id]:>5}  {c.score:>11.4f}  "
              f"{c.rerank_score:>12.4f}  {kept:>5}  {c.pmcid}")

    print(f"\n[rerank smoke test] kept {len(filtered)}/{len(chunks)} chunks after "
          f"gap filter.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
