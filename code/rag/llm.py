"""
llm.py
------
Multi-provider LLM adapter.

Dispatch is based on the model name prefix:
    "gpt-..."     -> OpenAI Chat Completions
    "claude-..."  -> Anthropic Messages API
    "llama-..."   -> Placeholder; raises NotImplementedError until wired up
                     to a local server (vLLM, Ollama, TGI, etc.)

The single entry point is generate(system, user, model, ...). Each provider
adapter is a small function with the same signature so we have ONE call site
and zero surprises about which provider got which request.

Network resilience:
    Both provider call sites are wrapped with tenacity exponential-backoff
    retries on transient errors (rate limits, network blips, 5xx). Auth
    and bad-request errors are NOT retried; those need human attention.
    Synchronous I/O on purpose -- eval-time concurrency will be added later
    via an async wrapper when we need to run hundreds of questions in a row.

Why we don't use a framework here:
    LangChain hides exactly which bytes are sent to each provider. For a
    research project that needs to compare models rigorously, that's a
    liability. Keep it explicit, keep it auditable.
"""

import os
import sys
import logging
from functools import lru_cache
from typing import Optional

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


_log = logging.getLogger(__name__)


def _retry_decorator(retryable_exceptions):
    """Return a tenacity decorator configured for our standard retry policy."""
    return retry(
        retry=retry_if_exception_type(retryable_exceptions),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        before_sleep=before_sleep_log(_log, logging.WARNING),
        reraise=True,
    )


# ----------------------------------------------------------------------------
# OpenAI
# ----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai SDK not installed. Run: pip install openai", file=sys.stderr)
        raise
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)


@lru_cache(maxsize=1)
def _openai_retry():
    """Build the retry decorator lazily so we don't import openai at module load."""
    import openai
    return _retry_decorator((
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ))


def _generate_openai(system: str, user: str, model: str,
                     temperature: float, max_tokens: int) -> str:
    client = _openai_client()

    @_openai_retry()
    def _call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    resp = _call()
    return resp.choices[0].message.content or ""


# ----------------------------------------------------------------------------
# Anthropic
# ----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _anthropic_client():
    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: anthropic SDK not installed. Run: pip install anthropic", file=sys.stderr)
        raise
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return Anthropic(api_key=key, timeout=45.0)


@lru_cache(maxsize=1)
def _anthropic_retry():
    import anthropic
    return _retry_decorator((
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.InternalServerError,
    ))


def _generate_anthropic(system: str, user: str, model: str,
                        temperature: float, max_tokens: int) -> str:
    client = _anthropic_client()

    @_anthropic_retry()
    def _call():
        return client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    resp = _call()
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


# ----------------------------------------------------------------------------
# Placeholder for local / open-source models
# ----------------------------------------------------------------------------

def _generate_llama(system: str, user: str, model: str,
                    temperature: float, max_tokens: int) -> str:
    raise NotImplementedError(
        f"Local LLM '{model}' not wired up yet. Add an HTTP call to your "
        f"vLLM/Ollama/TGI endpoint here."
    )


# ----------------------------------------------------------------------------
# Public dispatch
# ----------------------------------------------------------------------------

def generate(system: str, user: str, model: str,
             temperature: float = 0.0, max_tokens: int = 2000) -> str:
    """
    Send a single prompt to the named LLM and return the text response.

    Args:
        system        : system instruction (role, constraints)
        user          : the user message (question + sources)
        model         : provider-prefixed model name, e.g.
                          "gpt-4o", "gpt-4o-mini",
                          "claude-sonnet-4-5", "claude-opus-4-5",
                          "llama-3.1-70b" (placeholder)
        temperature   : 0.0 for reproducibility (recommended for eval runs)
        max_tokens    : hard cap on output length

    Retries on transient API errors up to 5 attempts with exponential backoff
    (2s, 4s, 8s, 16s, 32s, capped at 60s).
    """
    name = model.lower()
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return _generate_openai(system, user, model, temperature, max_tokens)
    if name.startswith("claude"):
        return _generate_anthropic(system, user, model, temperature, max_tokens)
    if name.startswith("llama") or name.startswith("mistral") or name.startswith("mixtral"):
        return _generate_llama(system, user, model, temperature, max_tokens)
    raise ValueError(
        f"Unknown model prefix: '{model}'. "
        f"Expected one of: gpt-*, claude-*, llama-*, mistral-*, mixtral-*."
    )
