#!/usr/bin/env python3
import os
import json
from openai import OpenAI

# 1. Setup API Key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("❌ ERROR: OPENAI_API_KEY is not set in your environment.")
    
client = OpenAI(api_key=API_KEY)

# 2. Paths (Using your exact Linux server paths)
MANIFEST_PATH = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs/batches_manifest.jsonl"
OUTPUT_DIR = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_embeddings/batch_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found at {MANIFEST_PATH}. No batches to check.")
        return

    print("Checking batch statuses with OpenAI...\n")
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            
            try:
                record = json.loads(line)
                batch_id = record["batch_id"]
                year = record.get("year", "UNKNOWN")
            except Exception:
                continue
            
            # Check the current status of the batch
            try:
                batch_job = client.batches.retrieve(batch_id)
                status = batch_job.status
                
                print(f"Batch {batch_id} (Year: {year}) -> Status: {status.upper()}")
                
                if status == "completed":
                    output_file_id = batch_job.output_file_id
                    save_path = os.path.join(OUTPUT_DIR, f"{year}_{batch_id}_results.jsonl")
                    
                    # Check if we already downloaded it
                    if os.path.exists(save_path):
                        print(f"  -> Already downloaded: {save_path}")
                        continue
                    
                    print(f"  -> DOWNLOADING results to {save_path}...")
                    
                    # Download the file contents
                    file_response = client.files.content(output_file_id)
                    with open(save_path, "wb") as out_f:
                        out_f.write(file_response.read())
                    print("  -> ✅ Download complete!")
                    
                elif status in ["failed", "expired", "cancelled"]:
                    print(f"  -> ⚠️ NOTE: This batch is marked as {status.upper()}.")
                    
            except Exception as e:
                print(f"  -> ❌ Error retrieving batch {batch_id}: {e}")

    print(f"\nAll done! Check your files in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()