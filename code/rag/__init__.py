"""
CherryPicker RAG package.

Modules:
    embeddings  - wraps OpenAI text-embedding-3-small for query embedding
    retrieve    - vector search over Chroma with optional MMR re-ranking
    prompts     - prompt templates for the RAG answer step
    llm         - multi-provider LLM adapter (OpenAI, Anthropic, ...)
    answer      - end-to-end orchestrator (question -> AnswerResult)
    ask         - CLI entry point

Re-exports the most-used types/functions so callers can write
    from code.rag import answer, search
instead of digging through submodules.

Recommended invocation (run from the project root, e.g. CHERRY_PICKER_AR/):
    python -m code.rag.ask "your question here"
which requires `code/__init__.py` and `code/rag/__init__.py` to both exist.
"""

from .answer   import answer, AnswerResult
from .retrieve import search, RetrievedChunk
from .embeddings import embed_query

__all__ = [
    "answer",
    "AnswerResult",
    "search",
    "RetrievedChunk",
    "embed_query",
]
