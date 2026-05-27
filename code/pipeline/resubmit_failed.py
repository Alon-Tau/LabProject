#!/usr/bin/env python3
"""
resubmit_failed.py
------------------
Find every batch in the manifest whose status is `failed` on OpenAI, and
re-submit its input_path with safer throttling than the original submitter.

Why this exists:
  The original submit_embedding_batches.py uses a 20M-token enqueue ceiling.
  OpenAI validates batches asynchronously, so a batch can be accepted by the
  API and *then* rejected with `token_limit_exceeded` once other batches start
  processing. ~22% of the v2 run hit this race condition.

  This script:
    * uses a lower SAFE_LIMIT (15M) so we always leave headroom
    * after submitting a batch, waits 30s and verifies it didn't immediately
      fail with token_limit_exceeded — if it did, exponentially backs off and
      retries up to MAX_RETRIES times
    * never re-submits a shard that has a non-failed batch in the manifest
    * appends new manifest entries with `retry_of: <old_batch_id>` so you keep
      a full audit trail

Usage:
  python resubmit_failed.py                    # do the work
  python resubmit_failed.py --dry-run          # show what would be submitted, do nothing
  python resubmit_failed.py --max-shards 5     # limit how many to resubmit this run

Run inside a tmux session — it can take many hours because of the throttling.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai SDK not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

# ---- Paths ----
DEFAULT_MANIFEST = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs_v2/batches_manifest.jsonl"
DEFAULT_REPORT   = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/conda_envs/alons_env/preflight_report.json"

# ---- Throttling ----
SAFE_LIMIT       = 15_000_000   # leave 5M headroom under the 20M org cap
POST_SUBMIT_WAIT = 30           # seconds to wait before verifying a batch didn't auto-fail
MAX_RETRIES      = 3            # per-shard retry budget for token_limit_exceeded
POLL_INTERVAL    = 600          # 10 min sleep when over the limit
INTER_BATCH_GAP  = 5            # seconds between successful submissions

ACTIVE_STATUSES = {"validating", "in_progress", "finalizing"}


# --------------- helpers ---------------

def load_manifest(path):
    if not os.path.exists(path):
        raise SystemExit(f"Manifest not found at {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_token_map(report_path):
    """Maps absolute path -> token count, from preflight_report.json's per_shard dict."""
    if not os.path.exists(report_path):
        print(f"WARNING: pre-flight report not found at {report_path}; "
              f"will fall back to a 10M default per shard for throttling.")
        return {}
    with open(report_path, "r") as f:
        report = json.load(f)
    per_shard = report.get("per_shard") or {}
    return {path: info["tokens"] for path, info in per_shard.items()}


def find_failed_inputs(client, records):
    """
    Walk the manifest; for each batch, ask OpenAI for the current status.
    Return:
      - failed_paths: list of input_paths whose latest manifest entry is `failed`
      - skip_paths:   set of input_paths that have at least one non-failed (good or pending) entry
    A shard is only resubmitted if ALL its entries are failed.
    """
    # Group manifest entries by input_path
    by_path = defaultdict(list)
    for r in records:
        if r.get("input_path"):
            by_path[r["input_path"]].append(r)

    failed_paths = []
    skip_paths = set()
    print(f"Polling OpenAI for status of {len(records)} manifest entries...")

    statuses = {}
    for r in records:
        bid = r.get("batch_id")
        if not bid:
            continue
        try:
            b = client.batches.retrieve(bid)
            statuses[bid] = b.status
        except Exception as e:
            print(f"  fetch_error for {bid}: {e}")
            statuses[bid] = "fetch_error"

    for path, entries in by_path.items():
        entry_statuses = [statuses.get(e["batch_id"], "fetch_error") for e in entries]
        if any(s in ACTIVE_STATUSES or s == "completed" for s in entry_statuses):
            skip_paths.add(path)
            continue
        if all(s == "failed" for s in entry_statuses):
            failed_paths.append(path)
        else:
            # Mixed bag (some fetch_error etc.) — be conservative and skip
            skip_paths.add(path)

    return failed_paths, skip_paths


def current_active_tokens(client, records, token_map):
    """Sum token counts of all batches currently active (validating/in_progress/finalizing)."""
    total = 0
    for r in records:
        bid = r.get("batch_id")
        if not bid:
            continue
        try:
            b = client.batches.retrieve(bid)
            if b.status in ACTIVE_STATUSES:
                total += token_map.get(r.get("input_path", ""), 10_000_000)
        except Exception:
            continue
    return total


def wait_until_under_limit(client, manifest_records_ref, token_map, next_shard_tokens):
    """Block until adding next_shard_tokens to the queue stays under SAFE_LIMIT."""
    while True:
        active = current_active_tokens(client, manifest_records_ref, token_map)
        if active + next_shard_tokens < SAFE_LIMIT:
            print(f"  Load check OK: active={active:,} + adding={next_shard_tokens:,} "
                  f"< safe_limit={SAFE_LIMIT:,}")
            return
        print(f"  Queue full ({active:,} tokens active, need {next_shard_tokens:,} more). "
              f"Sleeping {POLL_INTERVAL//60} min...")
        time.sleep(POLL_INTERVAL)


def submit_one(client, path, year, tokens, retry_of=None):
    """
    Upload the input file, create the batch. Returns (batch_id, status) on success,
    raises on hard failure.
    """
    with open(path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={"year": str(year), "input_path": path,
                  "retry_of": retry_of or ""},
    )
    return batch.id, batch.status


def verify_not_auto_failed(client, batch_id, wait_seconds=POST_SUBMIT_WAIT):
    """
    Wait `wait_seconds`, then poll. If the batch is already `failed` with
    token_limit_exceeded, return (False, error_message). Otherwise return (True, status).
    """
    time.sleep(wait_seconds)
    b = client.batches.retrieve(batch_id)
    if b.status == "failed":
        msg = ""
        if b.errors and getattr(b.errors, "data", None):
            for e in b.errors.data:
                msg = getattr(e, "message", "") or msg
        return False, msg
    return True, b.status


# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser(description="Re-submit failed v2 embedding batches.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--report",   default=DEFAULT_REPORT)
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be submitted, do nothing.")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="Limit how many shards to (re)submit this run. 0 = no limit.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    manifest_path = Path(args.manifest)
    records = load_manifest(manifest_path)
    if not records:
        raise SystemExit(f"Manifest is empty: {manifest_path}")
    print(f"Manifest: {manifest_path}  ({len(records)} entries)")

    token_map = load_token_map(args.report)

    failed_paths, skip_paths = find_failed_inputs(client, records)

    by_year = defaultdict(int)
    for p in failed_paths:
        try:
            y = Path(p).parent.name
            by_year[y] += 1
        except Exception:
            by_year["?"] += 1
    print(f"\nFound {len(failed_paths)} shards that need resubmission.")
    print(f"({len(skip_paths)} shards have at least one non-failed entry and will be skipped.)")
    if by_year:
        print("Per year:")
        for y in sorted(by_year):
            print(f"  {y}: {by_year[y]}")

    if not failed_paths:
        print("\nNothing to resubmit. Done.")
        return

    if args.dry_run:
        print("\n--dry-run: not submitting. Paths that would be resubmitted:")
        for p in failed_paths:
            print(f"  {p}")
        return

    if args.max_shards and args.max_shards < len(failed_paths):
        print(f"\n--max-shards={args.max_shards}: trimming from {len(failed_paths)} to {args.max_shards} shards.")
        failed_paths = failed_paths[:args.max_shards]

    # Build a map for new entries we append, so the throttler counts our own new
    # batches as well as the original ones.
    live_records = list(records)  # we'll extend this in place
    submitted_ok = 0
    gave_up      = 0

    with manifest_path.open("a", encoding="utf-8") as mf:
        for i, path in enumerate(failed_paths, 1):
            year = Path(path).parent.name
            shard_tokens = token_map.get(path, 10_000_000)
            print(f"\n[{i}/{len(failed_paths)}] {year} :: {path} "
                  f"(~{shard_tokens:,} tokens)")

            attempt = 0
            backoff = POLL_INTERVAL
            success = False

            while attempt < MAX_RETRIES and not success:
                attempt += 1
                wait_until_under_limit(client, live_records, token_map, shard_tokens)

                try:
                    bid, status0 = submit_one(client, path, year, shard_tokens)
                    print(f"  Submitted: {bid} (initial status: {status0}). Verifying in {POST_SUBMIT_WAIT}s...")
                except Exception as e:
                    print(f"  CRITICAL submission error: {e}")
                    time.sleep(60)
                    continue

                # Record provisionally so the throttler sees it
                record = {
                    "year": year,
                    "input_path": path,
                    "batch_id": bid,
                    "status": status0,
                    "tokens": shard_tokens,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "retry_of": "resubmit_failed.py",
                    "attempt": attempt,
                }
                mf.write(json.dumps(record) + "\n")
                mf.flush()
                live_records.append(record)

                ok, info = verify_not_auto_failed(client, bid)
                if ok:
                    print(f"  Verified: status={info}. Counts as in-flight. Moving on.")
                    success = True
                    submitted_ok += 1
                    time.sleep(INTER_BATCH_GAP)
                else:
                    print(f"  AUTO-FAILED: {info}")
                    if "token_limit" in (info or "").lower():
                        print(f"  Backing off {backoff//60} min and retrying...")
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        print(f"  Non-token-limit failure; not retrying this shard.")
                        break

            if not success:
                print(f"  Gave up on {path} after {attempt} attempts.")
                gave_up += 1

    print(f"\n=== Resubmit run complete ===")
    print(f"  shards resubmitted successfully: {submitted_ok}")
    print(f"  shards given up on:              {gave_up}")
    print(f"  total processed:                 {submitted_ok + gave_up}")
    print(f"\nRun:")
    print(f"  python code/research/new_check_and_download.py --status-only")
    print(f"to see the new summary, or wait ~24h for OpenAI to finish processing.")


if __name__ == "__main__":
    main()
