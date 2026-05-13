#!/usr/bin/env python3
"""
submit_embedding_batches.py

Smart throttled submission for OpenAI Batch API. 
Respects Tier 2 (20M token) limits by polling active batch statuses.
"""

import os
import json
import glob
import argparse
import time
from pathlib import Path
from openai import OpenAI

# Default paths for your resharded v2 data
DEFAULT_BATCH_GLOB = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs_v2/*/batch_*.jsonl"
DEFAULT_MANIFEST = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs_v2/batches_manifest.jsonl"

# Tier 2 Constraints
TIER_2_LIMIT = 20_000_000 
BUFFER_TOKENS = 11_000_000 # Enough room for the largest ~10.6M shard found in pre-flight

def get_enqueued_tokens(client, manifest_path):
    """
    Retrieves the status of previously submitted batches from the API.
    Returns a conservative estimate of tokens currently in the queue.
    """
    active_batches = 0
    if not manifest_path.exists():
        return 0
    
    # We read the manifest to find IDs we need to check
    batch_ids = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                batch_ids.append(json.loads(line)["batch_id"])
            except:
                continue

    # Only check the most recent 20 batches to stay efficient
    for b_id in batch_ids[-20:]:
        try:
            batch = client.batches.retrieve(b_id)
            if batch.status in ["validating", "in_progress", "finalizing"]:
                active_batches += 1
        except Exception as e:
            print(f"Warning: Could not retrieve status for {b_id}: {e}")
    
    # We assume each active shard is ~9M tokens for safety
    return active_batches * 9_000_000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-glob", default=DEFAULT_BATCH_GLOB)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    paths = sorted(glob.glob(args.batch_glob))
    
    if not paths:
        raise SystemExit(f"No batch files found at {args.batch_glob}")

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-submitted files (Resume safety)
    submitted = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    submitted.add(json.loads(line)["input_path"])
                except:
                    pass

    submitted_now = 0

    with manifest_path.open("a", encoding="utf-8") as mf:
        for path in paths:
            if path in submitted:
                continue

            if args.max_files is not None and submitted_now >= args.max_files:
                break

            # --- SMART THROTTLING ---
            while True:
                current_load = get_enqueued_tokens(client, manifest_path)
                if current_load + BUFFER_TOKENS <= TIER_2_LIMIT:
                    print(f"Current load ~{current_load/1e6:.1f}M tokens. Space available.")
                    break
                else:
                    print(f"Queue full (~{current_load/1e6:.1f}M tokens). Waiting 10 mins...")
                    time.sleep(600) 

            # --- UPLOAD AND SUBMIT ---
            year = Path(path).parent.name
            print(f"Submitting: {path} (Year: {year})")
            
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
                    "uploaded_file_id": uploaded.id,
                    "batch_id": batch.id,
                    "status": batch.status,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                mf.write(json.dumps(record) + "\n")
                mf.flush()
                submitted_now += 1
                print(f"  → Successfully enqueued. ID: {batch.id}")
                
                # Small cooldown to avoid hitting API rate limits on the 'create' call
                time.sleep(10) 

            except Exception as e:
                print(f"ERROR submitting {path}: {e}")
                print("Waiting 60 seconds before next attempt...")
                time.sleep(60)

    print(f"Done. Submitted {submitted_now} new batches.")

if __name__ == "__main__":
    main()