#!/usr/bin/env python3
"""
Backfill metadata-only chunks INTO an existing full-text chunk root (e.g., pilot_paragraph).

Goal:
- For each year, chunk metadata ONLY for articles (PMCIDs) that do NOT already have chunks
  in that year's full-text combined JSONLs inside the target chunk root.
- Write metadata-only chunks into separate METAONLY combined JSONL per year
- Write separate METAONLY stats per year (tokens/chunks/articles + over-soft-limit)
- Also writes a global METAONLY stats JSON across all processed years

Metadata input:
- JSONL only (one JSON object per line) is supported (your metadata_all.jsonl)

Safe behavior:
- Does NOT modify existing full-text combined files.
- Uses per-year scanning of existing combined files to decide if PMCID is already chunked.

Example output files (target_words=650):
pilot_paragraph/2005/
  - 2005_chunks_combined_650w_METAONLY.jsonl
  - 2005_chunking_stats_650w_METAONLY.json
pilot_paragraph/
  - ALL_chunking_stats_650w_METAONLY.json
"""

import os
import re
import json
import argparse
from typing import Dict, Any, Optional, List, Tuple, Set, Iterable
from collections import defaultdict

import tiktoken  # pip install tiktoken

ENCODING_NAME = "cl100k_base"

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def norm_pmcid(raw: Any) -> str:
    if raw is None:
        return ""
    x = str(raw).strip().upper()
    if not x:
        return ""
    if x.endswith(".TXT"):
        x = x[:-4]
    m = re.match(r"^(PMC)?(\d+)$", x)
    if m:
        return "PMC" + m.group(2)
    m2 = re.search(r"(PMC\d+)", x)
    if m2:
        return m2.group(1)
    return x

def count_tokens_factory():
    enc = tiktoken.get_encoding(ENCODING_NAME)
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text, disallowed_special=()))
    return count_tokens

def is_open_access(meta: Dict[str, Any]) -> bool:
    v = meta.get("isOpenAccess")
    return str(v).strip().upper() == "Y" if v is not None else False

def get_journal_title(meta: Dict[str, Any]) -> Optional[str]:
    try:
        jt = meta.get("journalInfo", {}).get("journal", {}).get("title")
        if jt:
            return str(jt).strip()
    except Exception:
        pass
    for k in ("journalTitle", "journal"):
        v = meta.get(k)
        if v:
            return str(v).strip()
    v = meta.get("source")
    return str(v).strip() if v else None

def meta_year(meta: Dict[str, Any]) -> Optional[int]:
    for k in ("year", "pubYear", "publicationYear", "firstPublicationDate", "pubDate", "date"):
        v = meta.get(k)
        if not v:
            continue
        m = re.search(r"(19|20)\d{2}", str(v))
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass
    try:
        y = meta.get("journalInfo", {}).get("yearOfPublication")
        if y:
            return int(y)
    except Exception:
        pass
    return None

def meta_text_payload(meta: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = meta.get("title") or meta.get("article_title")
    if title:
        parts.append(str(title).strip())
    abstract = meta.get("abstract") or meta.get("abstractText")
    if abstract:
        parts.append(str(abstract).strip())
    kw = meta.get("keywords")
    if isinstance(kw, list) and kw:
        parts.append("Keywords: " + ", ".join(map(str, kw)))
    elif isinstance(kw, str) and kw.strip():
        parts.append("Keywords: " + kw.strip())
    return "\n\n".join([p for p in parts if p]).strip()

def split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    raw = re.split(r"\n\s*\n+", text)
    return [re.sub(r"\s+", " ", p).strip() for p in raw if p.strip()]

def split_long_paragraph(p: str, target_words: int) -> List[str]:
    words = p.split()
    if len(words) <= target_words:
        return [p]
    return [" ".join(words[i:i + target_words]) for i in range(0, len(words), target_words)]

def paragraphs_to_chunks(paragraphs: List[str], target_words: int, max_overlap: int) -> List[str]:
    chunks: List[str] = []
    current_group: List[str] = []
    current_count = 0

    for p in paragraphs:
        wlen = len(p.split())

        if wlen > target_words:
            if current_group:
                chunks.append("\n\n".join(current_group))
                current_group, current_count = [], 0
            chunks.extend(split_long_paragraph(p, target_words))
            continue

        if current_group and (current_count + wlen > target_words):
            chunks.append("\n\n".join(current_group))

            if max_overlap > 0:
                last_words = current_group[-1].split()
                bridge_words = last_words[-max_overlap:] if len(last_words) > max_overlap else last_words
                bridge = "[...] " + " ".join(bridge_words) if bridge_words else ""
                if bridge:
                    current_group = [bridge, p]
                    current_count = len(bridge.split()) + wlen
                else:
                    current_group = [p]
                    current_count = wlen
            else:
                current_group = [p]
                current_count = wlen
        else:
            current_group.append(p)
            current_count += wlen

    if current_group:
        chunks.append("\n\n".join(current_group))

    return chunks

def ensure_newline_at_eof(file_path: str) -> None:
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return
    with open(file_path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------------
# Paths
# -----------------------------
def year_dir(chunks_root: str, year: Optional[int]) -> str:
    folder = str(year) if year is not None else "UNKNOWN_YEAR"
    return os.path.join(chunks_root, folder)

def metaonly_paths(chunks_root: str, year: Optional[int], target_words: int) -> Tuple[str, str]:
    folder = str(year) if year is not None else "UNKNOWN_YEAR"
    out_dir = os.path.join(chunks_root, folder)
    combined = os.path.join(out_dir, f"{folder}_chunks_combined_{target_words}w_METAONLY.jsonl")
    stats = os.path.join(out_dir, f"{folder}_chunking_stats_{target_words}w_METAONLY.json")
    return combined, stats

def global_metaonly_stats_path(chunks_root: str, target_words: int) -> str:
    return os.path.join(chunks_root, f"ALL_chunking_stats_{target_words}w_METAONLY.json")

# -----------------------------
# Existing fulltext index per year
# -----------------------------
def combined_jsonls_in_year_folder(year_folder: str) -> List[str]:
    """
    Any combined jsonl inside the year folder counts as "already chunked" for pmcid.
    We intentionally include:
      - fulltext combined files (whatever you named them)
      - previously written METAONLY file too (so re-runs are idempotent)
    """
    if not os.path.isdir(year_folder):
        return []
    out: List[str] = []
    for fn in os.listdir(year_folder):
        if fn.endswith(".jsonl") and "chunks_combined" in fn:
            out.append(os.path.join(year_folder, fn))
    return sorted(out)

def build_chunked_pmcids_index_for_year(year_folder: str) -> Set[str]:
    chunked: Set[str] = set()
    for path in combined_jsonls_in_year_folder(year_folder):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    pmcid = obj.get("pmcid") or obj.get("PMCID")
                    if pmcid:
                        chunked.add(norm_pmcid(pmcid))
        except Exception:
            continue
    return chunked

# -----------------------------
# Stats
# -----------------------------
def init_metaonly_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    if stats:
        return stats
    return {
        "articles_meta_only": 0,
        "chunks_meta_only": 0,
        "tokens_meta_only_total": 0,
        "chunks_meta_only_over_soft_limit": 0,
        "avg_tokens_per_chunk_meta_only": 0,
    }

def update_metaonly_stats(stats_path: str, added_art: int, added_ch: int, added_tok: int, added_over: int) -> Dict[str, Any]:
    stats = init_metaonly_stats(load_json(stats_path))

    stats["articles_meta_only"] += added_art
    stats["chunks_meta_only"] += added_ch
    stats["tokens_meta_only_total"] += added_tok
    stats["chunks_meta_only_over_soft_limit"] += added_over

    c = stats["chunks_meta_only"]
    t = stats["tokens_meta_only_total"]
    stats["avg_tokens_per_chunk_meta_only"] = int(t / c) if c else 0

    save_json(stats_path, stats)
    return stats

# -----------------------------
# Metadata streaming (JSONL)
# -----------------------------
def iter_metadata_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj

# -----------------------------
# Chunk builder
# -----------------------------
def build_metadata_chunks_for_article(
    year: Optional[int],
    pmcid: str,
    meta: Dict[str, Any],
    target_words: int,
    max_overlap_words: int,
) -> List[Dict[str, Any]]:
    text = meta_text_payload(meta)
    if not text:
        return []

    paragraphs = split_into_paragraphs(text)
    raw_chunks = paragraphs_to_chunks(paragraphs, target_words, max_overlap_words)

    year_label = str(year) if year is not None else "UNKNOWN_YEAR"
    chunks: List[Dict[str, Any]] = []
    for idx, ch_text in enumerate(raw_chunks, start=1):
        chunks.append({
            "chunk_id": f"{year_label}_{pmcid}_META_{idx:04d}",
            "pmcid": pmcid,
            "year": year,
            "chunk_index": idx,
            "title": meta.get("title") or meta.get("article_title"),
            "journal": get_journal_title(meta),
            "source_type": "metadata_only",
            "text": ch_text,
        })
    return chunks

# -----------------------------
# Runner
# -----------------------------
def run_backfill_to_pilot(
    metadata_jsonl: str,
    chunks_root: str,
    target_words: int,
    max_overlap_words: int,
    soft_token_limit: int,
    include_oa: bool,
    year_min: Optional[int],
    year_max: Optional[int],
    dry_run: bool,
) -> None:
    count_tokens = count_tokens_factory()

    # Group metadata records by year (streaming)
    by_year: Dict[Optional[int], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)

    print(f"Streaming metadata from: {metadata_jsonl}")
    total_records = 0
    skipped_oa = 0
    skipped_year = 0
    skipped_no_pmcid = 0

    for meta in iter_metadata_jsonl(metadata_jsonl):
        total_records += 1
        pmcid = norm_pmcid(meta.get("pmcid") or meta.get("PMCID") or meta.get("id"))
        if not pmcid:
            skipped_no_pmcid += 1
            continue
        if (not include_oa) and is_open_access(meta):
            skipped_oa += 1
            continue

        y = meta_year(meta)
        if y is not None:
            if (year_min is not None and y < year_min) or (year_max is not None and y > year_max):
                skipped_year += 1
                continue

        by_year[y].append((pmcid, meta))

    years_sorted = sorted([y for y in by_year.keys() if y is not None])
    if None in by_year:
        years_sorted.append(None)

    print(f"Loaded {total_records} metadata records (streamed).")
    print(f"Years to process: {len(years_sorted)}")

    global_stats = global_metaonly_stats_path(chunks_root, target_words)

    total_added_art = 0
    total_added_chunks = 0
    total_added_tokens = 0

    for year in years_sorted:
        y_folder = year_dir(chunks_root, year)
        ensure_dir(y_folder)

        # Build per-year chunked index from whatever combined jsonls exist in that year folder
        already_chunked_year = build_chunked_pmcids_index_for_year(y_folder)

        combined_path, stats_path = metaonly_paths(chunks_root, year, target_words)
        if not dry_run:
            ensure_newline_at_eof(combined_path)
            f_out = open(combined_path, "a", encoding="utf-8")
        else:
            f_out = None

        y_added_art = 0
        y_added_chunks = 0
        y_added_tokens = 0
        y_over = 0

        label = str(year) if year is not None else "UNKNOWN_YEAR"
        print(f"\nProcessing {label}: metadata records={len(by_year[year])}, already_chunked_in_year={len(already_chunked_year)}")

        for pmcid, meta in by_year[year]:
            # Key requirement: only if NOT already chunked in FULLTEXT (or any combined) for that year
            if pmcid in already_chunked_year:
                continue

            chunks = build_metadata_chunks_for_article(year, pmcid, meta, target_words, max_overlap_words)
            if not chunks:
                already_chunked_year.add(pmcid)
                continue

            y_added_art += 1
            for ch in chunks:
                tok = count_tokens(ch["text"])
                y_added_tokens += tok
                if tok > soft_token_limit:
                    y_over += 1

                if f_out:
                    f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                y_added_chunks += 1

            already_chunked_year.add(pmcid)

        if f_out:
            f_out.close()

        total_added_art += y_added_art
        total_added_chunks += y_added_chunks
        total_added_tokens += y_added_tokens

        if y_added_art > 0:
            print(f" -> Added METAONLY: {y_added_art} articles, {y_added_chunks} chunks, tokens={y_added_tokens:,}, over_soft={y_over}")
            if not dry_run:
                update_metaonly_stats(stats_path, y_added_art, y_added_chunks, y_added_tokens, y_over)
                update_metaonly_stats(global_stats, y_added_art, y_added_chunks, y_added_tokens, y_over)
        else:
            print(" -> No METAONLY additions for this year.")

    print("\n====================")
    print("METAONLY backfill complete.")
    print(f"Total METAONLY articles added: {total_added_art}")
    print(f"Total METAONLY chunks added:   {total_added_chunks}")
    print(f"Total METAONLY tokens added:   {total_added_tokens:,} (text field only)")
    print("====================")
    print(f"Skipped (no PMCID): {skipped_no_pmcid}")
    print(f"Skipped (OA excluded): {skipped_oa}  (use --include-oa to include)")
    print(f"Skipped (year filter): {skipped_year}")
    if dry_run:
        print("\n(DRY RUN) No files were written and no stats were updated.")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata-jsonl", required=True, help="Path to metadata_all.jsonl")
    p.add_argument("--chunks-root", required=True, help="Existing fulltext chunk root (e.g., pilot_paragraph)")
    p.add_argument("--target-words", type=int, default=650)
    p.add_argument("--max-overlap", type=int, default=120)
    p.add_argument("--soft-token-limit", type=int, default=1000)
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)
    p.add_argument("--include-oa", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_backfill_to_pilot(
        metadata_jsonl=args.metadata_jsonl,
        chunks_root=args.chunks_root,
        target_words=args.target_words,
        max_overlap_words=args.max_overlap,
        soft_token_limit=args.soft_token_limit,
        include_oa=args.include_oa,
        year_min=args.year_min,
        year_max=args.year_max,
        dry_run=args.dry_run,
    )
