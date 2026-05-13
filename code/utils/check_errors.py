#!/usr/bin/env python3
"""
check_errors.py — inspect why your OpenAI batches failed.

Reads batches_status.json (produced by retrieve_embedding_batches.py --status-only)
and prints the error details for failed batches.
"""
import json
import sys
from pathlib import Path

STATUS_PATH = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_outputs/batches_status.json"

p = Path(STATUS_PATH)
if not p.exists():
    print(f"ERROR: file not found: {p}")
    sys.exit(1)

with p.open("r", encoding="utf-8") as f:
    data = json.load(f)

failed = [s for s in data if s.get("status") == "failed"]
completed = [s for s in data if s.get("status") == "completed"]
other = [s for s in data if s.get("status") not in ("failed", "completed")]

print(f"Total batches:    {len(data)}")
print(f"Failed:           {len(failed)}")
print(f"Completed:        {len(completed)}")
print(f"Other:            {len(other)}")
print()

# Show errors for the first 5 failed batches
print("=" * 70)
print("Error details for first 5 failed batches:")
print("=" * 70)
for s in failed[:5]:
    print(f"\n--- {s.get('batch_id')} (year={s.get('metadata', {}).get('year')}) ---")
    errs = s.get("errors")
    if not errs:
        print("  (no errors field — batch may have just been rejected at submission)")
    else:
        print(json.dumps(errs, indent=2, default=str))

# Also show input_file_id and dates so we have full context
print()
print("=" * 70)
print("Timeline of first 3 failed batches:")
print("=" * 70)
for s in failed[:3]:
    print(f"\n  batch_id:        {s.get('batch_id')}")
    print(f"  year:            {s.get('metadata', {}).get('year')}")
    print(f"  input_path:      {s.get('metadata', {}).get('input_path')}")
    print(f"  created_at:      {s.get('created_at')}")
    print(f"  failed_at:       {s.get('failed_at')}")
    print(f"  input_file_id:   {s.get('input_file_id')}")