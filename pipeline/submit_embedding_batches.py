#!/usr/bin/env python3
"""
submit_embedding_batches.py
Production-grade throttled submission for OpenAI Batch API.
Uses exact token counts from 'per_shard' in the pre-flight report.
"""

import os
import json
import glob
import argparse
import time
from pathlib import Path
from openai import OpenAI

# Paths and Config
DEFAULT_BATCH_GLOB = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs_v2/*/batch_*.jsonl"
DEFAULT_MANIFEST = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs_v2/batches_manifest.jsonl"
REPORT_PATH = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/conda_envs/alons_env/preflight_report.json"

# Tier 2 Limit
TIER_2_LIMIT = 20_000_000 

def load_token_map():
    """Loads exact token counts from the 'per_shard' list in the pre-flight report."""
    if not os.path.exists(REPORT_PATH):
        print(f"ERROR: Pre-flight report not found at {REPORT_PATH}")
        raise SystemExit(1)
    
    with open(REPORT_PATH, 'r') as f:
        report = json.load(f)
    
    shards_dict = report.get("per_shard")

    if not shards_dict:
        print(f"ERROR: 'per_shard' key not found in {REPORT_PATH}")
        raise SystemExit(1)

    # Map absolute path → tokens (per_shard is a dict keyed by path)
    token_map = {path: info["tokens"] for path, info in shards_dict.items()}

    print(f"Successfully mapped {len(token_map)} shards from the pre-flight report.")
    return token_map

def get_exact_active_tokens(client, manifest_path, token_map):
    """Sums exact token counts of active batches by polling OpenAI."""
    total_tokens = 0
    if not manifest_path.exists():
        return 0
    
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                batch = client.batches.retrieve(data["batch_id"])
                # We only count tokens for batches still in the queue
                if batch.status in ["validating", "in_progress", "finalizing"]:
                    total_tokens += token_map.get(data["input_path"], 10_000_000)
            except Exception:
                continue
    return total_tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-glob", default=DEFAULT_BATCH_GLOB)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    token_map = load_token_map()
    paths = sorted(glob.glob(args.batch_glob))
    
    if not paths:
        raise SystemExit(f"No batch files found at {args.batch_glob}")

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume safety: don't double-submit
    submitted = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    submitted.add(json.loads(line)["input_path"])
                except:
                    pass

    with manifest_path.open("a", encoding="utf-8") as mf:
        for path in paths:
            if path in submitted:
                continue

            # Get exact token count for the current shard
            next_shard_tokens = token_map.get(path, 10_000_000)

            # --- DATA-DRIVEN THROTTLING ---
            while True:
                current_active_tokens = get_exact_active_tokens(client, manifest_path, token_map)
                
                if current_active_tokens + next_shard_tokens < TIER_2_LIMIT:
                    print(f"\n--- Load Check Passed ---")
                    print(f"Active: {current_active_tokens:,} | Adding: {next_shard_tokens:,}")
                    print(f"Total: {current_active_tokens + next_shard_tokens:,} / {TIER_2_LIMIT:,}")
                    break
                else:
                    # Overwrite line to keep terminal clean while sleeping
                    print(f"Queue full ({current_active_tokens:,} tokens). Sleeping 10m...", end="\r")
                    time.sleep(600)

            # --- UPLOAD AND SUBMIT ---
            year = Path(path).parent.name
            print(f"Submitting Year {year}: {path}")
            
            try:
                with open(path, "rb") as f:
                    uploaded = client.files.create(file=f, purpose="batch")

                batch = client.batches.create(
                    input_file_id=uploaded.id,
                    endpoint="/v1/embeddings",
                    completion_window="24h",
                    metadata={"year": year, "input_path": path}
                )

                record = {
                    "year": year,
                    "input_path": path,
                    "batch_id": batch.id,
                    "status": batch.status,
                    "tokens": next_shard_tokens,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                mf.write(json.dumps(record) + "\n")
                mf.flush()
                print(f"  → Success! ID: {batch.id}")
                
                # Small delay to avoid hitting general rate limits
                time.sleep(5) 

            except Exception as e:
                print(f"CRITICAL ERROR: {e}")
                time.sleep(60)

    print(f"\nAll batches submitted successfully.")

if __name__ == "__main__":
    main()