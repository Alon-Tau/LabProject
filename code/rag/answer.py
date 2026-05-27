"""
answer.py
---------
End-to-end RAG orchestrator: takes a question, runs retrieval, builds the
prompt, calls the LLM, parses out citation markers, and returns an
AnswerResult that contains EVERY intermediate artifact for debugging and
evaluation.

The AnswerResult is intentionally rich because eval requires answering
questions like "the model dropped Phenylalanine — did retrieval surface
a chunk that mentioned it?". You need the chunks AND the answer AND the
prompt to triangulate where each failure came from.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any  # Optional used by AnswerResult.top_score

from .retrieve import RetrievedChunk, search
from .prompts  import build_rag_prompt
from .llm      import generate


@dataclass
class AnswerResult:
    """Everything produced by one end-to-end answer() call."""
    question:     str
    answer_text:  str
    model:        str
    chunks:       List[RetrievedChunk]
    cited_ranks:          List[int]    # valid 1-based ranks the answer cited
    invalid_cited_ranks:  List[int] = field(default_factory=list)
                                       # citations to ranks that don't exist
                                       # (a sign the model hallucinated indices)
    refused:              bool = False  # True if we short-circuited due to
                                       # low retrieval evidence and never
                                       # called the LLM
    top_score:           Optional[float] = None
                                       # similarity of the top retrieved chunk
                                       # (None if no chunks retrieved). Stored
                                       # explicitly so eval scripts can tune
                                       # --min-score across many questions.
    system_prompt: str = ""
    user_prompt:   str = ""
    retrieval_params: Dict[str, Any] = field(default_factory=dict)

    def cited_chunks(self) -> List[RetrievedChunk]:
        """Subset of self.chunks that the answer actually cited (valid only)."""
        return [self.chunks[i - 1] for i in self.cited_ranks
                if 1 <= i <= len(self.chunks)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question":             self.question,
            "answer_text":          self.answer_text,
            "model":                self.model,
            "retrieval_params":     self.retrieval_params,
            "cited_ranks":          self.cited_ranks,
            "invalid_cited_ranks":  self.invalid_cited_ranks,
            "refused":              self.refused,
            "top_score":            self.top_score,
            "chunks":               [c.to_dict() for c in self.chunks],
            "system_prompt":        self.system_prompt,
            "user_prompt":          self.user_prompt,
        }


# Matches citation markers like [1], [2,4], [3, 5, 8], or even "[ 7 , 8 ]"
# (LLMs sometimes pad whitespace inside the brackets).
_CITATION_RE = re.compile(r"\[\s*((?:\d+\s*,\s*)*\d+)\s*\]")


def _parse_cited_ranks(answer_text: str) -> List[int]:
    """Extract the unique 1-based citation indices the LLM used, in order of appearance."""
    seen: List[int] = []
    for m in _CITATION_RE.finditer(answer_text):
        for s in m.group(1).split(","):
            try:
                n = int(s.strip())
            except ValueError:
                continue
            if n not in seen:
                seen.append(n)
    return seen


def _split_citations(cited: List[int], n_chunks: int) -> tuple:
    """Partition raw citation indices into (valid, invalid)."""
    valid   = [i for i in cited if 1 <= i <= n_chunks]
    invalid = [i for i in cited if i < 1 or i > n_chunks]
    return valid, invalid


REFUSAL_TEXT = (
    "I do not have enough retrieved evidence in the CherryPicker corpus to "
    "answer this question reliably."
)


def answer(question: str,
           model: str = "gpt-4o",
           k: int = 5,
           year_range: Optional[Tuple[int, int]] = None,
           source: Optional[str] = None,
           use_mmr: bool = True,
           lambda_mmr: float = 0.5,
           fetch_n: Optional[int] = None,
           temperature: float = 0.0,
           max_tokens: int = 2000,
           min_score: Optional[float] = None,
           chroma_dir: Optional[str] = None) -> AnswerResult:
    """
    Run the full RAG pipeline on `question` and return an AnswerResult.

    Retrieval params (k, year_range, source, use_mmr, lambda_mmr, fetch_n) are
    passed through to retrieve.search().

    LLM params (model, temperature, max_tokens) are passed to llm.generate().

    min_score:
        Optional minimum similarity for the TOP retrieved chunk. If the top
        chunk scores below this threshold, the LLM is NOT called and the
        AnswerResult comes back with refused=True. Use this as an evidence
        guardrail during eval; set to None to disable. Typical values: 0.20-0.30
        for cosine. Tune against your eval set.

    Returns:
        AnswerResult with the model's answer text, the retrieved chunks, the
        citation indices the answer references, and the exact prompts sent —
        everything needed for evaluation and debugging.
    """
    # ---- 1. retrieve ----
    search_kwargs: Dict[str, Any] = dict(
        k=k,
        year_range=year_range,
        source=source,
        use_mmr=use_mmr,
        lambda_mmr=lambda_mmr,
        fetch_n=fetch_n,
    )
    if chroma_dir is not None:
        search_kwargs["chroma_dir"] = chroma_dir
    chunks = search(question, **search_kwargs)

    retrieval_params = {k_: v for k_, v in search_kwargs.items() if v is not None}
    if min_score is not None:
        retrieval_params["min_score"] = min_score
    top_score = chunks[0].score if chunks else None

    # ---- 2. evidence guardrail ----
    # If the top chunk doesn't clear `min_score`, don't burn LLM tokens
    # answering. This both saves money and gives the eval pipeline a
    # clean signal: "the corpus didn't support an answer here."
    if min_score is not None and (not chunks or chunks[0].score < min_score):
        return AnswerResult(
            question=question,
            answer_text=REFUSAL_TEXT,
            model=model,
            chunks=chunks,
            cited_ranks=[],
            invalid_cited_ranks=[],
            refused=True,
            top_score=top_score,
            system_prompt="",
            user_prompt="",
            retrieval_params=retrieval_params,
        )

    # ---- 3. build prompt ----
    system_prompt, user_prompt = build_rag_prompt(question, chunks)

    # ---- 4. call LLM ----
    answer_text = generate(system=system_prompt,
                           user=user_prompt,
                           model=model,
                           temperature=temperature,
                           max_tokens=max_tokens)

    # ---- 5. parse + validate citations ----
    raw_cited = _parse_cited_ranks(answer_text)
    valid_cited, invalid_cited = _split_citations(raw_cited, len(chunks))

    return AnswerResult(
        question=question,
        answer_text=answer_text,
        model=model,
        chunks=chunks,
        cited_ranks=valid_cited,
        invalid_cited_ranks=invalid_cited,
        refused=False,
        top_score=top_score,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        retrieval_params=retrieval_params,
    )
