"""
embeddings.py
-------------
Thin wrapper around OpenAI's text-embedding-3-small for *query-time* embedding.

The corpus was already embedded with this exact model in the batch pipeline
(submit_embedding_batches.py), so query and corpus must agree on the model
choice -- mixing models silently breaks similarity scores.

Network resilience:
    The actual API call is wrapped with tenacity's exponential-backoff retry
    on transient errors (rate limits, network blips, 5xx). Auth errors and
    bad-request errors are NOT retried; those need human attention.

We cache the OpenAI client at module level so repeated queries within one
process don't pay TCP connection setup on every call.
"""

import os
import sys
from functools import lru_cache
from typing import List

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("ERROR: openai SDK not installed. Run: pip install openai", file=sys.stderr)
    raise

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
    )
except ImportError:
    print("ERROR: tenacity not installed. Run: pip install tenacity", file=sys.stderr)
    raise

import logging
_log = logging.getLogger(__name__)


CORPUS_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small returns 1536 floats

# Exceptions worth retrying. Everything else (AuthenticationError,
# BadRequestError, ...) is a real bug and should propagate.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key, timeout=45.0)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    before_sleep=before_sleep_log(_log, logging.WARNING),
    reraise=True,
)
def _embed_with_retry(model: str, text: str):
    return _client().embeddings.create(model=model, input=text)


def embed_query(text: str, model: str = CORPUS_EMBEDDING_MODEL) -> List[float]:
    """
    Embed a single query string and return its 1536-dim vector.

    Always uses CORPUS_EMBEDDING_MODEL by default -- overriding is allowed
    only if the corpus is also re-embedded with the same alternate model.

    Retries automatically on transient API errors (up to 5 attempts with
    exponential backoff: 2s, 4s, 8s, 16s, 32s capped at 60s).
    """
    if not text or not text.strip():
        raise ValueError("embed_query received empty text.")
    resp = _embed_with_retry(model, text)
    vec = resp.data[0].embedding
    if len(vec) != EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected embedding length {len(vec)} for model {model}; "
            f"expected {EMBEDDING_DIM}. Did the model change?"
        )
    return vec
