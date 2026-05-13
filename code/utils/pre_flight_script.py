#!/usr/bin/env python3
"""
preflight_batches.py — READ-ONLY pre-flight check on OpenAI Batch input shards.

Scans every `<out-root>/<YEAR>/batch_*.jsonl` file and verifies:
  • Each line is valid JSON
  • Each request has the required shape: {custom_id, method, url, body}
  • Each `body.input` is non-empty
  • No `body.input` exceeds 8192 tokens (the embedding limit)
  • Every custom_id is unique GLOBALLY across all shards (any duplicate kills a batch)
  • No shard exceeds 100MB or 50,000 requests (OpenAI's hard limits)
  • Counts tokens precisely with tiktoken and projects the exact cost

Outputs:
  preflight_report.json — per-shard + per-year + global stats
  preflight_issues.txt  — every detected issue with shard + line context
  (printed to terminal) — human-readable summary

This script DOES NOT modify any input file. DOES NOT submit anything.

Run:
  python preflight_batches.py
  python preflight_batches.py --batch-inputs-root /home/elhanan/.../batch_inputs
  python preflight_batches.py --first-n-shards 3   # quick spot check
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, Iterator, Tuple, Optional

DEFAULT_BATCH_INPUTS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs"

EMBED_TOKEN_LIMIT = 8192
SHARD_MAX_BYTES = 100 * 1024 * 1024   # 100 MB OpenAI cap
SHARD_MAX_REQUESTS = 50_000           # OpenAI cap
PRICE_PER_1K_BATCH = 0.00001          # $0.01 / 1M tokens = $0.00001 / 1k tokens

YEAR_RE = re.compile(r"^\d{4}$")

# --- tiktoken (required for exact tokens) ---
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(t: str) -> int:
        if not t: return 0
        return len(_ENC.encode(t, disallowed_special=()))
    TOKEN_MODE = "tiktoken_cl100k_base"
except Exception:
    def count_tokens(t: str) -> int:
        return int(len((t or "").split()) * 1.33)
    TOKEN_MODE = "word_count_x_1.33_approx (tiktoken not installed — install for exact counts)"


def iter_shards(root: Path):
    """Yield (year_str, shard_path) for each year/*.jsonl."""
    if not root.exists():
        raise SystemExit(f"Batch inputs root does not exist: {root}")
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir():
            continue
        for shard in sorted(year_dir.glob("batch_*.jsonl")):
            yield year_dir.name, shard


def check_request_shape(obj: Any) -> Optional[str]:
    """Return an error string if the request line is malformed, else None."""
    if not isinstance(obj, dict):
        return "line is not a JSON object"
    if "custom_id" not in obj or not isinstance(obj["custom_id"], str) or not obj["custom_id"]:
        return "missing or empty custom_id"
    if obj.get("method") != "POST":
        return f"method != POST (got {obj.get('method')!r})"
    if obj.get("url") != "/v1/embeddings":
        return f"url != /v1/embeddings (got {obj.get('url')!r})"
    body = obj.get("body")
    if not isinstance(body, dict):
        return "missing body"
    if "model" not in body or not body["model"]:
        return "missing body.model"
    if "input" not in body:
        return "missing body.input"
    inp = body["input"]
    if not isinstance(inp, str):
        return f"body.input is not a string (type: {type(inp).__name__})"
    if not inp.strip():
        return "body.input is empty"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-inputs-root", default=DEFAULT_BATCH_INPUTS_ROOT)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--first-n-shards", type=int, default=None,
                    help="Limit to first N shards (for quick spot check)")
    args = ap.parse_args()

    root = Path(args.batch_inputs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch inputs root: {root}")
    print(f"Token mode:        {TOKEN_MODE}")
    print()

    # Global accumulators
    g_total_requests = 0
    g_total_tokens = 0
    g_total_bytes = 0
    g_oversize_inputs = 0     # inputs > 8192 tokens
    g_oversize_shards_bytes = 0
    g_oversize_shards_count = 0
    g_bad_lines = 0
    g_shape_errors = 0
    g_empty_input = 0
    g_token_lens = []  # for distribution

    custom_id_seen = {}  # custom_id -> first (shard_path, line)
    duplicates = []      # list of (custom_id, first_loc, dup_loc)
    issues = []          # general issues for the report
    per_year: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "shards": 0, "requests": 0, "tokens": 0, "bytes": 0,
        "oversize_inputs": 0, "bad_lines": 0, "shape_errors": 0
    })
    per_shard: Dict[str, Dict[str, Any]] = {}

    shards = list(iter_shards(root))
    if args.first_n_shards:
        shards = shards[:args.first_n_shards]
    print(f"Shards found: {len(shards)}\n")

    for year, shard in shards:
        shard_path = str(shard)
        shard_bytes = shard.stat().st_size
        shard_requests = 0
        shard_tokens = 0
        shard_bad_lines = 0
        shard_shape_errors = 0
        shard_oversize_inputs = 0
        shard_token_lens = []
        first_dup_in_shard = None

        with shard.open("r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    shard_bad_lines += 1
                    g_bad_lines += 1
                    issues.append(f"{shard_path}:{line_num} BAD_JSON: {e}")
                    continue

                err = check_request_shape(obj)
                if err:
                    shard_shape_errors += 1
                    g_shape_errors += 1
                    issues.append(f"{shard_path}:{line_num} BAD_SHAPE: {err}")
                    continue

                shard_requests += 1
                cid = obj["custom_id"]
                inp = obj["body"]["input"]

                # Token check
                tok = count_tokens(inp)
                shard_tokens += tok
                shard_token_lens.append(tok)
                if tok > EMBED_TOKEN_LIMIT:
                    shard_oversize_inputs += 1
                    g_oversize_inputs += 1
                    issues.append(f"{shard_path}:{line_num} OVERSIZE: {tok} tokens > {EMBED_TOKEN_LIMIT} (cid={cid[:60]})")

                # Cross-shard custom_id dedup
                prev = custom_id_seen.get(cid)
                if prev:
                    duplicates.append((cid, prev, f"{shard_path}:{line_num}"))
                else:
                    custom_id_seen[cid] = f"{shard_path}:{line_num}"

        # Shard caps
        if shard_bytes > SHARD_MAX_BYTES:
            g_oversize_shards_bytes += 1
            issues.append(f"{shard_path} EXCEEDS BYTE CAP: {shard_bytes:,} > {SHARD_MAX_BYTES:,}")
        if shard_requests > SHARD_MAX_REQUESTS:
            g_oversize_shards_count += 1
            issues.append(f"{shard_path} EXCEEDS REQUEST CAP: {shard_requests:,} > {SHARD_MAX_REQUESTS:,}")

        per_shard[shard_path] = {
            "year": year,
            "bytes": shard_bytes,
            "requests": shard_requests,
            "tokens": shard_tokens,
            "bad_lines": shard_bad_lines,
            "shape_errors": shard_shape_errors,
            "oversize_inputs": shard_oversize_inputs,
            "token_p95": (sorted(shard_token_lens)[int(0.95*(len(shard_token_lens)-1))] if shard_token_lens else 0),
            "token_max": max(shard_token_lens) if shard_token_lens else 0,
        }
        py = per_year[year]
        py["shards"] += 1
        py["requests"] += shard_requests
        py["tokens"] += shard_tokens
        py["bytes"] += shard_bytes
        py["oversize_inputs"] += shard_oversize_inputs
        py["bad_lines"] += shard_bad_lines
        py["shape_errors"] += shard_shape_errors

        g_total_requests += shard_requests
        g_total_tokens += shard_tokens
        g_total_bytes += shard_bytes
        g_token_lens.extend(shard_token_lens)

        # progress line
        print(f"  [{year}] {shard.name:<22} requests={shard_requests:>6,} tokens={shard_tokens:>10,} "
              f"bytes={shard_bytes:>10,} dup_in_shard={'(see below)' if first_dup_in_shard else 'no'}")

    # ---- Final report ----
    cost = (g_total_tokens / 1000) * PRICE_PER_1K_BATCH
    print()
    print("=" * 78)
    print("PRE-FLIGHT SUMMARY")
    print("=" * 78)
    print(f"  Shards:                       {len(per_shard):,}")
    print(f"  Total requests:               {g_total_requests:,}")
    print(f"  Total tokens (input):         {g_total_tokens:,}")
    print(f"  Total bytes:                  {g_total_bytes:,}")
    print(f"  Cost projection (batch):      ${cost:,.2f}")
    print()
    print(f"  Duplicate custom_ids (cross-shard):  {len(duplicates):,}  <-- MUST BE 0")
    print(f"  Inputs > 8192 tokens:                {g_oversize_inputs:,}  <-- MUST BE 0")
    print(f"  Shards exceeding 100 MB:             {g_oversize_shards_bytes:,}  <-- MUST BE 0")
    print(f"  Shards exceeding 50k requests:       {g_oversize_shards_count:,}  <-- MUST BE 0")
    print(f"  Malformed JSON lines:                {g_bad_lines:,}")
    print(f"  Malformed request shapes:            {g_shape_errors:,}")

    if g_token_lens:
        st = sorted(g_token_lens)
        n = len(st)
        print()
        print(f"  Token-per-input distribution:")
        print(f"     min: {st[0]}  p05: {st[int(0.05*(n-1))]}  p50: {st[n//2]}  p95: {st[int(0.95*(n-1))]}  p99: {st[int(0.99*(n-1))]}  max: {st[-1]}")

    print()
    print("  Per-year breakdown:")
    for y in sorted(per_year.keys()):
        v = per_year[y]
        print(f"     {y}: shards={v['shards']:>3} requests={v['requests']:>7,} tokens={v['tokens']:>11,} cost=${(v['tokens']/1000)*PRICE_PER_1K_BATCH:>6.2f}")

    # ---- Write outputs ----
    report = {
        "token_mode": TOKEN_MODE,
        "summary": {
            "shards": len(per_shard),
            "total_requests": g_total_requests,
            "total_tokens": g_total_tokens,
            "total_bytes": g_total_bytes,
            "cost_projection_batch_usd": round(cost, 4),
            "duplicate_custom_ids_cross_shard": len(duplicates),
            "inputs_over_8192_tokens": g_oversize_inputs,
            "shards_over_100MB": g_oversize_shards_bytes,
            "shards_over_50k_requests": g_oversize_shards_count,
            "bad_json_lines": g_bad_lines,
            "shape_errors": g_shape_errors,
        },
        "per_year": dict(per_year),
        "per_shard": per_shard,
        "duplicate_custom_ids_first_10": [
            {"custom_id": d[0], "first": d[1], "duplicate": d[2]}
            for d in duplicates[:10]
        ],
    }
    (out_dir / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Truncate issues file to first 500 to keep it small but useful
    issue_path = out_dir / "preflight_issues.txt"
    with issue_path.open("w", encoding="utf-8") as f:
        f.write(f"Pre-flight issues — first 500 of {len(issues)} shown\n")
        f.write("=" * 70 + "\n")
        for line in issues[:500]:
            f.write(line + "\n")

    print()
    print(f"  Report:   {out_dir / 'preflight_report.json'}")
    print(f"  Issues:   {out_dir / 'preflight_issues.txt'}  (first 500 of {len(issues)})")

    # Final verdict
    print()
    all_clean = (
        len(duplicates) == 0
        and g_oversize_inputs == 0
        and g_oversize_shards_bytes == 0
        and g_oversize_shards_count == 0
        and g_bad_lines == 0
        and g_shape_errors == 0
    )
    if all_clean:
        print("VERDICT: ALL CHECKS PASSED — your batch inputs are ready to submit.")
    else:
        print("VERDICT: ISSUES FOUND — see preflight_issues.txt; fix before re-submitting.")


if __name__ == "__main__":
    main()