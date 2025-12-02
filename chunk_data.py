import os
import json
from typing import List, Dict

# ---------- CONFIG ----------

CORPUS_ROOT = "/path/to/corpus_root"       # ← change this
CHUNKS_ROOT = "/path/to/corpus_chunks"     # ← change this
YEAR = 2015                                # ← choose a year to test
MAX_ARTICLES = 20                          # first N articles for the pilot

# medium-sized chunks
TARGET_WORDS = 650     # ~800 tokens
OVERLAP_WORDS = 150    # overlap between chunks

# ----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_text(article_dir: str) -> str:
    txt_path = os.path.join(article_dir, "clean_text.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def load_metadata(article_dir: str) -> Dict:
    meta_path = os.path.join(article_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_into_chunks(text: str, target_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    n = len(words)

    if n == 0:
        return []

    while start < n:
        end = min(start + target_words, n)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        if end == n:
            break

        # move start forward with overlap
        start = end - overlap_words
        if start < 0:
            start = 0

    return chunks

def chunk_article(year: int, pmcid: str, article_dir: str) -> List[Dict]:
    text = load_text(article_dir)
    meta = load_metadata(article_dir)

    journal = meta.get("journal") or meta.get("journal_title") or None
    title = meta.get("title") or meta.get("article_title") or None
    section = None  # we can improve later if we detect sections

    raw_chunks = split_into_chunks(text, TARGET_WORDS, OVERLAP_WORDS)
    chunk_objs = []

    for idx, ch_text in enumerate(raw_chunks, start=1):
        chunk_id = f"{year}_{pmcid}_{idx:04d}"
        chunk = {
            "chunk_id": chunk_id,
            "pmcid": pmcid,
            "year": year,
            "journal": journal,
            "title": title,
            "section": section,
            "chunk_index": idx,
            "text": ch_text
        }
        chunk_objs.append(chunk)

    return chunk_objs

def pilot_chunk_year(year: int, max_articles: int):
    year_dir = os.path.join(CORPUS_ROOT, str(year))
    if not os.path.isdir(year_dir):
        raise FileNotFoundError(f"Year directory not found: {year_dir}")

    ensure_dir(CHUNKS_ROOT)
    out_path = os.path.join(CHUNKS_ROOT, f"{year}_pilot.jsonl")

    # list article directories (e.g. PMCIDs)
    pmcid_dirs = [
        d for d in os.listdir(year_dir)
        if os.path.isdir(os.path.join(year_dir, d))
    ]
    pmcid_dirs.sort()
    pmcid_dirs = pmcid_dirs[:max_articles]

    total_chunks = 0
    total_articles = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for pmcid in pmcid_dirs:
            article_dir = os.path.join(year_dir, pmcid)
            try:
                chunks = chunk_article(year, pmcid, article_dir)
            except FileNotFoundError as e:
                print(f"Skipping {pmcid}: {e}")
                continue

            for ch in chunks:
                out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            total_articles += 1
            print(f"Chunked {pmcid}: {len(chunks)} chunks")

    print(f"\nPilot finished for year {year}.")
    print(f"Articles processed: {total_articles}")
    print(f"Total chunks written: {total_chunks}")
    print(f"Output file: {out_path}")

if __name__ == "__main__":
    pilot_chunk_year(YEAR, MAX_ARTICLES)
