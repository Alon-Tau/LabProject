#!/usr/bin/env python3
import os
import re
import json
from typing import Iterator

import tiktoken

CHUNKS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_chunks/new_corpus_chunks"

MODEL = "text-embedding-3-small"
PRICE_PER_1K_TOKENS = 0.00001  # Batch price

def iter_chunk_files(root: str) -> Iterator[str]:
    for name in sorted(os.listdir(root)):
        year_dir = os.path.join(root, name)
        if not os.path.isdir(year_dir):
            continue
        if name != "UNKNOWN_YEAR" and not re.fullmatch(r"\d{4}", name):
            continue
        for fn in os.listdir(year_dir):
            if fn.endswith(".jsonl") and "chunks_combined" in fn:
                yield os.path.join(year_dir, fn)

def main():
    enc = tiktoken.encoding_for_model(MODEL)

    total_chunks = 0
    total_tokens = 0

    for path in iter_chunk_files(CHUNKS_ROOT):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text")
                if not text:
                    continue
                total_chunks += 1
                total_tokens += len(enc.encode(text))

    cost = (total_tokens / 1000) * PRICE_PER_1K_TOKENS

    print("==== EMBEDDING COST ESTIMATE ====")
    print(f"Chunks: {total_chunks:,}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Estimated cost (Batch): ${cost:,.2f}")
    print("================================")

if __name__ == "__main__":
    main()
