import os
import json
import time
from typing import List, Dict, Any
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# ============ CONFIG ============

# 1. API Key (Best practice: set this in your environment variables, or paste it here temporarily)
# os.environ["OPENAI_API_KEY"] = "sk-..." 
API_KEY = os.getenv("OPENAI_API_KEY") 

# 2. Paths
CHUNKS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_chunks/pilot_paragraph"
OUTPUT_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_embeddings/pilot_paragraph"

# 3. Model Settings
# "text-embedding-3-small" is the standard (1536 dims).
# "text-embedding-3-large" is higher res (3072 dims).
MODEL_NAME = "text-embedding-3-small"
BATCH_SIZE = 50  # Number of chunks to send in one request (Max is usually 2048, but 50-100 is safer)

# ================================

client = OpenAI(api_key=API_KEY)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

@retry(
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(Exception) # Retries on any API error (RateLimit, ServerError)
)
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Sends a batch of text strings to OpenAI and returns a list of vectors.
    Includes exponential backoff (wait 2s, 4s, 8s...) if rate limited.
    """
    # Replace newlines with spaces to improve performance (recommended by OpenAI)
    cleaned_texts = [t.replace("\n", " ") for t in texts]
    
    response = client.embeddings.create(
        input=cleaned_texts,
        model=MODEL_NAME
    )
    
    # Extract the embeddings in the correct order
    return [data.embedding for data in response.data]

def process_file(input_path: str, output_path: str):
    print(f"Processing: {input_path}")
    
    # 1. Load all chunks
    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    if not chunks:
        print("  File is empty.")
        return

    # 2. check for existing progress (Resume capability)
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["chunk_id"])
                except:
                    pass
    
    chunks_to_process = [c for c in chunks if c["chunk_id"] not in processed_ids]
    print(f"  Total chunks: {len(chunks)}. Already processed: {len(processed_ids)}. To do: {len(chunks_to_process)}")
    
    if not chunks_to_process:
        print("  All done for this file!")
        return

    # 3. Process in Batches
    total_batches = (len(chunks_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with open(output_path, "a", encoding="utf-8") as f_out:
        for i in range(0, len(chunks_to_process), BATCH_SIZE):
            batch = chunks_to_process[i : i + BATCH_SIZE]
            batch_texts = [c["text"] for c in batch]
            
            print(f"  Batch {i // BATCH_SIZE + 1}/{total_batches} (Size: {len(batch)})...")
            
            try:
                # Call API
                vectors = get_embeddings_batch(batch_texts)
                
                # Attach vectors to chunks and save immediately
                for chunk, vector in zip(batch, vectors):
                    chunk["embedding"] = vector
                    chunk["embedding_model"] = MODEL_NAME
                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                
                # Flush to disk safety
                f_out.flush()
                
            except Exception as e:
                print(f"  CRITICAL ERROR on batch {i}: {e}")
                # We stop the script here so you don't burn credits on broken loops
                raise e

def main():
    if not API_KEY:
        print("❌ ERROR: No API Key found. Please set 'API_KEY' variable inside the script.")
        return

    ensure_dir(OUTPUT_ROOT)
    
    # Identify JSONL files in the chunks directory
    files = [f for f in os.listdir(CHUNKS_ROOT) if f.endswith(".jsonl")]
    files.sort()
    
    print(f"Found {len(files)} chunk files to embed.")
    
    for filename in files:
        in_path = os.path.join(CHUNKS_ROOT, filename)
        out_path = os.path.join(OUTPUT_ROOT, f"embedded_{filename}")
        
        process_file(in_path, out_path)
    
    print("\n✅ Embedding Complete.")

if __name__ == "__main__":
    main()