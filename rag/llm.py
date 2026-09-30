"""
llm.py
------
Multi-provider LLM adapter.

Dispatch is based on the model name prefix:
    "gpt-..."     -> OpenAI Chat Completions
    "claude-..."  -> Anthropic Messages API
    "gemini-..."  -> Google Gemini (google-genai SDK)
    "llama-..."   -> Placeholder; raises NotImplementedError until wired up
                     to a local server (vLLM, Ollama, TGI, etc.)

The single entry point is generate(system, user, model, ...). Each provider
adapter is a small function with the same signature so we have ONE call site
and zero surprises about which provider got which request.

Network resilience:
    All provider call sites are wrapped with tenacity exponential-backoff
    retries on transient errors (rate limits, network blips, 5xx). Auth
    and bad-request errors are NOT retried; those need human attention.
    Synchronous I/O on purpose -- eval-time concurrency will be added later
    via an async wrapper when we need to run hundreds of questions in a row.

Cost tracking:
    Every provider function logs its token usage to cost_tracker.record()
    right after the API call returns, tagged with an optional `label` so
    callers can trace which run/question/step a cost came from. cost_tracker
    prints a running-total line to stderr per call; call
    cost_tracker.print_summary() at the end of a script for a final tally.
    See cost_tracker.py.

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

from . import cost_tracker


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
                     temperature: float, max_tokens: int, label: str = "") -> str:
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
    usage = getattr(resp, "usage", None)
    if usage is not None:
        cost_tracker.record(label or "llm.generate", model,
                            getattr(usage, "prompt_tokens", 0) or 0,
                            getattr(usage, "completion_tokens", 0) or 0)
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
                        temperature: float, max_tokens: int, label: str = "") -> str:
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
    usage = getattr(resp, "usage", None)
    if usage is not None:
        cost_tracker.record(label or "llm.generate", model,
                            getattr(usage, "input_tokens", 0) or 0,
                            getattr(usage, "output_tokens", 0) or 0)
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)

# ----------------------------------------------------------------------------
# Google Gemini  (google-genai SDK:  pip install google-genai)
# ----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _gemini_client():
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai SDK not installed. Run: pip install google-genai",
              file=sys.stderr)
        raise
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
    return genai.Client(api_key=key)


@lru_cache(maxsize=1)
def _gemini_retry():
    from google.genai import errors as gerr
    from tenacity import retry_if_exception

    def _is_transient(e):
        # retry 5xx server errors and 429 rate limits; NOT auth/bad-request
        if isinstance(e, gerr.ServerError):
            return True
        return isinstance(e, gerr.APIError) and getattr(e, "code", None) == 429

    # Gemini preview models see real demand-based 503 spikes (observed live:
    # "This model is currently experiencing high demand"). 5 attempts capped
    # at 60s tops out around ~30s of total backoff -- often not enough to ride
    # out a spike. 8 attempts with the same 60s cap gives ~4min of total
    # patience before giving up, much more likely to survive a transient spike
    # without needing a manual rerun.
    return retry(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        before_sleep=before_sleep_log(_log, logging.WARNING),
        reraise=True,
    )


def _generate_gemini(system: str, user: str, model: str,
                     temperature: float, max_tokens: int, label: str = "") -> str:
    from google.genai import types
    client = _gemini_client()

    config_kwargs = dict(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    # Gemini 3.x models cannot disable "thinking" outright (unlike pre-3 models,
    # which use thinking_budget). Left uncapped, thinking tokens eat into the
    # same max_output_tokens budget as the actual answer -- for a structured-
    # JSON judge prompt this can consume the whole budget and leave nothing for
    # the real output (exactly what happened: 75 output tokens for a 7080-token
    # prompt, and _extract_json found no JSON at all). thinking_level=LOW steers
    # the model to spend less of that budget on reasoning. Guarded in case the
    # installed google-genai version predates this API shape.
    if model.lower().startswith("gemini-3"):
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW
            )
        except AttributeError:
            pass  # older SDK -- fall back to default thinking behavior

    @_gemini_retry()
    def _call():
        return client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        )

    resp = _call()
    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        # Thinking tokens are billed as output tokens by Google but reported in
        # a separate usage field -- fold them in so cost_tracker isn't silently
        # undercounting real spend on reasoning-heavy models.
        out_tokens = (getattr(usage, "candidates_token_count", 0) or 0) + \
                     (getattr(usage, "thoughts_token_count", 0) or 0)
        cost_tracker.record(label or "llm.generate", model,
                            getattr(usage, "prompt_token_count", 0) or 0,
                            out_tokens)
    return getattr(resp, "text", None) or ""

# ----------------------------------------------------------------------------
# Placeholder for local / open-source models
# ----------------------------------------------------------------------------

def _generate_llama(system: str, user: str, model: str,
                    temperature: float, max_tokens: int, label: str = "") -> str:
    raise NotImplementedError(
        f"Local LLM '{model}' not wired up yet. Add an HTTP call to your "
        f"vLLM/Ollama/TGI endpoint here."
    )


# ----------------------------------------------------------------------------
# Public dispatch
# ----------------------------------------------------------------------------

def generate(system: str, user: str, model: str,
             temperature: float = 0.0, max_tokens: int = 2000,
             label: str = "") -> str:
    """
    Send a single prompt to the named LLM and return the text response.

    Args:
        system        : system instruction (role, constraints)
        user          : the user message (question + sources)
        model         : provider-prefixed model name, e.g.
                          "gpt-4o", "gpt-4o-mini",
                          "claude-sonnet-4-5", "claude-opus-4-5",
                          "gemini-3.1-pro-preview",
                          "llama-3.1-70b" (placeholder)
        temperature   : 0.0 for reproducibility (recommended for eval runs)
        max_tokens    : hard cap on output length
        label         : short tag for cost_tracker, e.g. "eval_run:q3:rag"
                        or "judge:q3:coverage:rag". Purely cosmetic -- shows
                        up in the [cost] lines printed to stderr; safe to
                        leave blank.

    Retries on transient API errors up to 5 attempts with exponential backoff
    (2s, 4s, 8s, 16s, 32s, capped at 60s).

    Every call's token usage is logged to cost_tracker (see cost_tracker.py)
    and a running-total cost line is printed to stderr as it happens. Call
    cost_tracker.print_summary() at the end of your script for a final tally.
    """
    name = model.lower()
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return _generate_openai(system, user, model, temperature, max_tokens, label)
    if name.startswith("claude"):
        return _generate_anthropic(system, user, model, temperature, max_tokens, label)
    if name.startswith("gemini"):
        return _generate_gemini(system, user, model, temperature, max_tokens, label)
    if name.startswith("llama") or name.startswith("mistral") or name.startswith("mixtral"):
        return _generate_llama(system, user, model, temperature, max_tokens, label)
    raise ValueError(
        f"Unknown model prefix: '{model}'. "
        f"Expected one of: gpt-*, claude-*, gemini-*, llama-*, mistral-*, mixtral-*."
    )
