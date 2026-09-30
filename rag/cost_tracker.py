"""
cost_tracker.py
----------------
Lightweight, in-process cost tracking for eval runs.

Every call to llm.generate() and embeddings.embed_query() logs its token
usage here and prints a running total to stderr AS IT HAPPENS -- so during
a real run (e.g. `eval_all.py --k 30 --only 1`) you see the cost of each
API call live, plus a final summary line when the script finishes. No new
commands, no external tools -- it's just an in-memory ledger for the life
of one Python process.

Pricing is USD per 1,000,000 tokens: (input_rate, output_rate). Embeddings
have no output cost. These are current as of July 2026 -- update PRICING
if OpenAI / Google change their rates.

    gpt-4o-mini                     $0.15 / $0.60   per 1M tokens (in/out)
    gemini-3.1-pro-preview          $2.00 / $12.00  per 1M tokens (in/out), <=200k ctx
    gemini-3.1-pro-preview (>200k)  $4.00 / $18.00  per 1M tokens (in/out)
    text-embedding-3-small          $0.02 / --      per 1M tokens (input only)
"""

from __future__ import annotations

import sys
from typing import Optional


PRICING = {
    # model name prefix -> (usd per 1M input tokens, usd per 1M output tokens)
    "gpt-4o-mini":            (0.15, 0.60),
    "gpt-4o":                 (2.50, 10.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),   # standard tier, <=200k context
    "claude-sonnet-4-5":      (3.00, 15.00),
    "claude-opus-4-5":        (15.00, 75.00),
    "text-embedding-3-small": (0.02, 0.00),
}

# threshold above which gemini-3.1-pro-preview's long-context pricing kicks in
GEMINI_LONG_CONTEXT_TOKENS = 200_000
GEMINI_LONG_RATE = (4.00, 18.00)

_log = []  # list of dicts: label, model, input_tokens, output_tokens, cost


def _rate_for(model: str, input_tokens: int = 0):
    """Longest-prefix match against PRICING; handles gemini's long-context tier."""
    name = model.lower()
    best_key = None
    for key in PRICING:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    if best_key is None:
        return None
    if best_key == "gemini-3.1-pro-preview" and input_tokens > GEMINI_LONG_CONTEXT_TOKENS:
        return GEMINI_LONG_RATE
    return PRICING[best_key]


def record(label: str, model: str, input_tokens: int, output_tokens: int = 0) -> Optional[float]:
    """
    Log one API call's token usage, print its cost + running total to stderr,
    and return the cost (None if the model isn't in PRICING).
    """
    rate = _rate_for(model, input_tokens)
    if rate is None:
        cost = None
        print(f"[cost] {label} ({model}): {input_tokens} in / {output_tokens} out "
              f"tokens -- UNKNOWN MODEL, add it to cost_tracker.PRICING to track $.",
              file=sys.stderr)
    else:
        rin, rout = rate
        cost = (input_tokens / 1_000_000) * rin + (output_tokens / 1_000_000) * rout
        running = sum(r["cost"] for r in _log if r["cost"] is not None) + cost
        print(f"[cost] {label} ({model}): {input_tokens} in / {output_tokens} out "
              f"tokens = ${cost:.5f}   (running total ${running:.4f})", file=sys.stderr)
    _log.append({"label": label, "model": model, "input_tokens": input_tokens,
                 "output_tokens": output_tokens, "cost": cost})
    return cost


def print_summary() -> float:
    """Print a final tally. Call this once at the end of a script's main()."""
    total = sum(r["cost"] for r in _log if r["cost"] is not None)
    n_unknown = sum(1 for r in _log if r["cost"] is None)
    print("\n" + "=" * 56, file=sys.stderr)
    print(f"[cost] {len(_log)} API call(s) tracked in this run -- total ${total:.4f}",
          file=sys.stderr)
    if n_unknown:
        print(f"[cost] WARNING: {n_unknown} call(s) used a model not in PRICING "
              f"and are EXCLUDED from this total.", file=sys.stderr)
    by_model = {}
    for r in _log:
        by_model.setdefault(r["model"], []).append(r)
    for model, rows in by_model.items():
        c = sum(r["cost"] for r in rows if r["cost"] is not None)
        it = sum(r["input_tokens"] for r in rows)
        ot = sum(r["output_tokens"] for r in rows)
        print(f"[cost]   {model}: {len(rows)} call(s), {it} in / {ot} out tokens, ${c:.4f}",
              file=sys.stderr)
    print("=" * 56, file=sys.stderr)
    return total


def reset() -> None:
    """Clear the ledger (useful for tests; not needed for normal CLI runs)."""
    _log.clear()
