#!/usr/bin/env python3
"""
Metadata-driven chunking for NEW corpus (fulltext preferred, metadata fallback),
PLUS a final integrity pass that chunks any fulltext article that somehow wasn't chunked yet.

Defaults:
  CORPUS_ROOT   = /home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus
  METADATA_PATH = /home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus/metadata_all.jsonl
  CHUNKS_ROOT   = /home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_chunks/new_corpus_chunks

High-level logic:
PASS 1 (metadata-driven):
  For each metadata record (grouped by year):
    - Normalize PMCID
    - If already chunked -> skip
    - If fulltext exists in corpus (any year) -> chunk fulltext (written under metadata year)
    - Else -> chunk metadata-only
    - Append to <YEAR>_chunks_combined_<target>w.jsonl and update year + global stats

PASS 2 (integrity fix):
  For each fulltext article in corpus:
    - If already chunked -> skip
    - Chunk fulltext and append to appropriate year file (prefer metadata year if available, else folder year)

Outputs (per year):
  <YEAR>_chunks_combined_<target>w.jsonl   (fulltext + metadata_only)
  <YEAR>_chunking_stats_<target>w.json

Global:
  ALL_chunking_stats_<target>w.json

Run examples:
  python chunk_new_corpus_metadata_driven.py --year-min 2025 --year-max 2025
  python chunk_new_corpus_metadata_driven.py --overwrite
"""

import os
import json
import re
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

import tiktoken  # pip install tiktoken

# ============ CONFIG ============
DEFAULT_CORPUS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus"
DEFAULT_METADATA_PATH = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus/metadata_all.jsonl"
DEFAULT_CHUNKS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_chunks/new_corpus_chunks"

TARGET_WORDS_DEFAULT = 650
MAX_OVERLAP_WORDS_DEFAULT = 120

ENCODING_NAME = "cl100k_base"
SOFT_TOKEN_LIMIT_DEFAULT = 1000
# ===============================


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
    # digits only -> PMC + digits
    if x.isdigit():
        return "PMC" + x
    # contains PMC\d+
    m = re.search(r"(PMC\d+)", x)
    if m:
        return m.group(1)
    # fallback: leave as-is (but usually not wanted)
    return x

def safe_remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)

def ensure_newline_at_eof(file_path: str) -> None:
    """Ensures JSONL ends with newline before appending (prevents corruption)."""
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

def count_tokens_factory():
    enc = tiktoken.get_encoding(ENCODING_NAME)
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text, disallowed_special=()))
    return count_tokens

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


# -----------------------------
# Metadata loading / parsing
# -----------------------------
def load_metadata_index(metadata_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads jsonl metadata into index keyed by normalized PMCID.
    Handles records that look like: {"SOMEKEY": {...actual meta...}}
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(metadata_path):
        print(f"⚠️ Metadata file not found: {metadata_path}")
        return index

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # handle single top-level key weirdness
                if isinstance(obj, dict) and len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
                    obj = next(iter(obj.values()))
                pmcid_raw = obj.get("pmcid") or obj.get("PMCID") or obj.get("id")
                pmcid = norm_pmcid(pmcid_raw)
                if pmcid:
                    index[pmcid] = obj
            except Exception:
                continue
    return index

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


# -----------------------------
# Chunking helpers
# -----------------------------
def split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    raw = re.split(r"\n\s*\n+", text)
    return [re.sub(r"\s+", " ", p).strip() for p in raw if p.strip()]

def split_long_paragraph(p: str, target_words: int) -> List[str]:
    words = p.split()
    if len(words) <= target_words:
        return [p]
    return [" ".join(words[i: i + target_words]) for i in range(0, len(words), target_words)]

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

            last_p_words = current_group[-1].split()
            bridge = (
                "[...] " + " ".join(last_p_words[-max_overlap:])
                if len(last_p_words) > max_overlap
                else current_group[-1]
            )

            current_group = [bridge, p]
            current_count = len(bridge.split()) + wlen
        else:
            current_group.append(p)
            current_count += wlen

    if current_group:
        chunks.append("\n\n".join(current_group))

    return chunks


# -----------------------------
# Fulltext index (PMCID -> (folder_year, path))
# -----------------------------
def iter_year_dirs(corpus_root: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    if not os.path.isdir(corpus_root):
        return out
    for name in os.listdir(corpus_root):
        p = os.path.join(corpus_root, name)
        if not os.path.isdir(p):
            continue
        if re.fullmatch(r"\d{4}", name):
            out.append((int(name), p))
    out.sort(key=lambda x: x[0])
    return out

def build_fulltext_index(corpus_root: str,
                         year_min: Optional[int],
                         year_max: Optional[int]) -> Dict[str, Tuple[int, str]]:
    """
    Build global map of pmcid -> (folder_year, txt_path) across the corpus.
    """
    idx: Dict[str, Tuple[int, str]] = {}
    for year, year_dir in iter_year_dirs(corpus_root):
        if year_min is not None and year < year_min:
            continue
        if year_max is not None and year > year_max:
            continue
        try:
            for fn in os.listdir(year_dir):
                if not fn.lower().endswith(".txt"):
                    continue
                pmcid = norm_pmcid(os.path.splitext(fn)[0])
                if pmcid and pmcid not in idx:
                    idx[pmcid] = (year, os.path.join(year_dir, fn))
        except Exception:
            continue
    return idx


# -----------------------------
# Chunk builders (LEAN schema)
# -----------------------------
def build_fulltext_chunks(
    out_year: int,
    pmcid: str,
    txt_path: str,
    metadata_index: Dict[str, Dict[str, Any]],
    target_words: int,
    max_overlap_words: int,
) -> List[Dict[str, Any]]:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return []

    paragraphs = split_into_paragraphs(text)
    raw_chunks = paragraphs_to_chunks(paragraphs, target_words, max_overlap_words)
    meta = metadata_index.get(pmcid, {})

    chunks: List[Dict[str, Any]] = []
    for idx, ch_text in enumerate(raw_chunks, start=1):
        chunks.append({
            "chunk_id": f"{out_year}_{pmcid}_{idx:04d}",
            "pmcid": pmcid,
            "year": out_year,
            "chunk_index": idx,
            "title": meta.get("title") or meta.get("article_title"),
            "journal": get_journal_title(meta),
            "source_type": "fulltext",
            "text": ch_text,
        })
    return chunks

def build_metadata_only_chunks(
    out_year: int,
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

    chunks: List[Dict[str, Any]] = []
    for idx, ch_text in enumerate(raw_chunks, start=1):
        chunks.append({
            "chunk_id": f"{out_year}_{pmcid}_META_{idx:04d}",
            "pmcid": pmcid,
            "year": out_year,
            "chunk_index": idx,
            "title": meta.get("title") or meta.get("article_title"),
            "journal": get_journal_title(meta),
            "source_type": "metadata_only",
            "text": ch_text,
        })
    return chunks


# -----------------------------
# Chunked PMCID index from outputs
# -----------------------------
def combined_jsonls_under_root(chunks_root: str, target_words: int) -> List[str]:
    out = []
    if not os.path.isdir(chunks_root):
        return out
    for name in os.listdir(chunks_root):
        p = os.path.join(chunks_root, name)
        if not os.path.isdir(p):
            continue
        if name != "UNKNOWN_YEAR" and not re.fullmatch(r"\d{4}", name):
            continue
        fn = os.path.join(p, f"{name}_chunks_combined_{target_words}w.jsonl")
        if os.path.exists(fn):
            out.append(fn)
    return out

def build_chunked_pmcids_global(chunks_root: str, target_words: int) -> Set[str]:
    chunked: Set[str] = set()
    for path in combined_jsonls_under_root(chunks_root, target_words):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
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
# Stats update (incremental)
# -----------------------------
def init_stats_template(target_words: int, max_overlap_words: int, soft_token_limit: int) -> Dict[str, Any]:
    return {
        "target_words": target_words,
        "max_overlap_words": max_overlap_words,
        "soft_token_limit": soft_token_limit,

        "articles_fulltext": 0,
        "chunks_fulltext": 0,
        "tokens_fulltext_total": 0,
        "chunks_fulltext_over_soft_limit": 0,

        "articles_meta_only": 0,
        "chunks_meta_only": 0,
        "tokens_meta_only_total": 0,
        "chunks_meta_only_over_soft_limit": 0,

        "avg_tokens_per_chunk_fulltext": 0,
        "avg_tokens_per_chunk_meta_only": 0,
    }

def update_stats(stats_path: str,
                 added_articles_full: int, added_chunks_full: int, added_tokens_full: int, added_over_full: int,
                 added_articles_meta: int, added_chunks_meta: int, added_tokens_meta: int, added_over_meta: int,
                 target_words: int, max_overlap_words: int, soft_token_limit: int) -> Dict[str, Any]:
    stats = load_json(stats_path)
    if not stats:
        stats = init_stats_template(target_words, max_overlap_words, soft_token_limit)

    stats["articles_fulltext"] += added_articles_full
    stats["chunks_fulltext"] += added_chunks_full
    stats["tokens_fulltext_total"] += added_tokens_full
    stats["chunks_fulltext_over_soft_limit"] += added_over_full

    stats["articles_meta_only"] += added_articles_meta
    stats["chunks_meta_only"] += added_chunks_meta
    stats["tokens_meta_only_total"] += added_tokens_meta
    stats["chunks_meta_only_over_soft_limit"] += added_over_meta

    stats["avg_tokens_per_chunk_fulltext"] = (
        int(stats["tokens_fulltext_total"] / stats["chunks_fulltext"]) if stats["chunks_fulltext"] else 0
    )
    stats["avg_tokens_per_chunk_meta_only"] = (
        int(stats["tokens_meta_only_total"] / stats["chunks_meta_only"]) if stats["chunks_meta_only"] else 0
    )

    save_json(stats_path, stats)
    return stats


# -----------------------------
# Main runner
# -----------------------------
def run(
    corpus_root: str,
    metadata_path: str,
    chunks_root: str,
    target_words: int,
    max_overlap_words: int,
    soft_token_limit: int,
    overwrite: bool,
    year_min: Optional[int],
    year_max: Optional[int],
    do_integrity_pass: bool,
    dry_run: bool,
) -> None:
    ensure_dir(chunks_root)
    count_tokens = count_tokens_factory()

    print("Loading metadata index...")
    metadata_index = load_metadata_index(metadata_path)
    print(f"Metadata entries indexed: {len(metadata_index):,}")

    print("Building fulltext index from corpus...")
    fulltext_idx = build_fulltext_index(corpus_root, year_min, year_max)
    print(f"Fulltext .txt indexed: {len(fulltext_idx):,}")

    # Group metadata by year (including None)
    meta_by_year: Dict[Optional[int], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for pmcid, meta in metadata_index.items():
        y = meta_year(meta)
        if y is not None:
            if year_min is not None and y < year_min:
                continue
            if year_max is not None and y > year_max:
                continue
        meta_by_year[y].append((pmcid, meta))

    # Overwrite cleanup
    if overwrite and not dry_run:
        # remove per-year outputs we will touch (within year_min/year_max)
        for y in list(meta_by_year.keys()):
            if y is None:
                continue
            out_dir = os.path.join(chunks_root, str(y))
            out_combined = os.path.join(out_dir, f"{y}_chunks_combined_{target_words}w.jsonl")
            out_stats = os.path.join(out_dir, f"{y}_chunking_stats_{target_words}w.json")
            safe_remove(out_combined)
            safe_remove(out_stats)
        # remove global stats
        safe_remove(os.path.join(chunks_root, f"ALL_chunking_stats_{target_words}w.json"))

    print("Building global index of already-chunked PMCIDs from outputs...")
    chunked_global = build_chunked_pmcids_global(chunks_root, target_words)
    print(f"Already-chunked PMCIDs found: {len(chunked_global):,}")

    global_stats_path = os.path.join(chunks_root, f"ALL_chunking_stats_{target_words}w.json")

    # ---------------
    # PASS 1: metadata-driven
    # ---------------
    years = sorted([y for y in meta_by_year.keys() if y is not None])
    if None in meta_by_year:
        years.append(None)

    print("\n=== PASS 1: metadata-driven chunking ===")
    for y in years:
        if y is None:
            # keep unknown-year behavior, but still prefer fulltext if exists
            year_label = "UNKNOWN_YEAR"
            out_dir = os.path.join(chunks_root, year_label)
            ensure_dir(out_dir)
            out_combined = os.path.join(out_dir, f"{year_label}_chunks_combined_{target_words}w.jsonl")
            out_stats = os.path.join(out_dir, f"{year_label}_chunking_stats_{target_words}w.json")
            out_year = None  # stored in chunk 'year' as None
        else:
            year_label = str(y)
            out_dir = os.path.join(chunks_root, year_label)
            ensure_dir(out_dir)
            out_combined = os.path.join(out_dir, f"{y}_chunks_combined_{target_words}w.jsonl")
            out_stats = os.path.join(out_dir, f"{y}_chunking_stats_{target_words}w.json")
            out_year = y

        if not dry_run:
            ensure_newline_at_eof(out_combined)
            f_out = open(out_combined, "a", encoding="utf-8")
        else:
            f_out = None

        added_articles_full = added_chunks_full = added_tokens_full = added_over_full = 0
        added_articles_meta = added_chunks_meta = added_tokens_meta = added_over_meta = 0

        records = meta_by_year.get(y, [])
        print(f"\nYear {year_label}: metadata records={len(records):,}")

        for pmcid, meta in records:
            if pmcid in chunked_global:
                continue

            # prefer fulltext if exists anywhere
            if pmcid in fulltext_idx and out_year is not None:
                _, txt_path = fulltext_idx[pmcid]
                chunks = build_fulltext_chunks(out_year=out_year, pmcid=pmcid, txt_path=txt_path,
                                              metadata_index=metadata_index,
                                              target_words=target_words, max_overlap_words=max_overlap_words)
                if chunks:
                    added_articles_full += 1
                    for ch in chunks:
                        t = count_tokens(ch["text"])
                        added_tokens_full += t
                        if t > soft_token_limit:
                            added_over_full += 1
                        if f_out:
                            f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                        added_chunks_full += 1
                    chunked_global.add(pmcid)
                    continue

            # fallback: metadata-only
            # for UNKNOWN_YEAR we store year=None in chunk object, but chunk_id should remain stable
            if out_year is None:
                # store as None in chunk, but still keep chunk_id prefix stable using UNKNOWN_YEAR
                # we reuse build_metadata_only_chunks by passing out_year=0 then override year field to None
                meta_chunks = build_metadata_only_chunks(out_year=0, pmcid=pmcid, meta=meta,
                                                        target_words=target_words, max_overlap_words=max_overlap_words)
                for ch in meta_chunks:
                    ch["chunk_id"] = ch["chunk_id"].replace("0_", "UNKNOWN_YEAR_", 1)
                    ch["year"] = None
            else:
                meta_chunks = build_metadata_only_chunks(out_year=out_year, pmcid=pmcid, meta=meta,
                                                        target_words=target_words, max_overlap_words=max_overlap_words)

            if not meta_chunks:
                chunked_global.add(pmcid)
                continue

            added_articles_meta += 1
            for ch in meta_chunks:
                t = count_tokens(ch["text"])
                added_tokens_meta += t
                if t > soft_token_limit:
                    added_over_meta += 1
                if f_out:
                    f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                added_chunks_meta += 1

            chunked_global.add(pmcid)

        if f_out:
            f_out.close()

        if (added_articles_full or added_articles_meta) and (not dry_run):
            update_stats(out_stats,
                         added_articles_full, added_chunks_full, added_tokens_full, added_over_full,
                         added_articles_meta, added_chunks_meta, added_tokens_meta, added_over_meta,
                         target_words, max_overlap_words, soft_token_limit)
            update_stats(global_stats_path,
                         added_articles_full, added_chunks_full, added_tokens_full, added_over_full,
                         added_articles_meta, added_chunks_meta, added_tokens_meta, added_over_meta,
                         target_words, max_overlap_words, soft_token_limit)

        print(f"  Added fulltext:  articles={added_articles_full:,} chunks={added_chunks_full:,} tokens={added_tokens_full:,}")
        print(f"  Added meta-only: articles={added_articles_meta:,} chunks={added_chunks_meta:,} tokens={added_tokens_meta:,}")

    # ---------------
    # PASS 2: integrity fix (fulltext-driven)
    # ---------------
    if do_integrity_pass:
        print("\n=== PASS 2: integrity fix over fulltext corpus ===")
        fixed = 0

        # quick lookup for metadata year
        meta_year_map: Dict[str, Optional[int]] = {}
        for pmcid, meta in metadata_index.items():
            meta_year_map[pmcid] = meta_year(meta)

        for pmcid, (folder_year, txt_path) in fulltext_idx.items():
            if pmcid in chunked_global:
                continue

            # choose output year: prefer metadata year if present else folder year
            oy = meta_year_map.get(pmcid) or folder_year
            if year_min is not None and oy < year_min:
                continue
            if year_max is not None and oy > year_max:
                continue

            out_dir = os.path.join(chunks_root, str(oy))
            ensure_dir(out_dir)
            out_combined = os.path.join(out_dir, f"{oy}_chunks_combined_{target_words}w.jsonl")
            out_stats = os.path.join(out_dir, f"{oy}_chunking_stats_{target_words}w.json")

            chunks = build_fulltext_chunks(out_year=oy, pmcid=pmcid, txt_path=txt_path,
                                          metadata_index=metadata_index,
                                          target_words=target_words, max_overlap_words=max_overlap_words)
            if not chunks:
                chunked_global.add(pmcid)
                continue

            added_articles_full = 1
            added_chunks_full = 0
            added_tokens_full = 0
            added_over_full = 0

            if not dry_run:
                ensure_newline_at_eof(out_combined)
                with open(out_combined, "a", encoding="utf-8") as f_out:
                    for ch in chunks:
                        t = count_tokens(ch["text"])
                        added_tokens_full += t
                        if t > soft_token_limit:
                            added_over_full += 1
                        f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                        added_chunks_full += 1

                update_stats(out_stats,
                             added_articles_full, added_chunks_full, added_tokens_full, added_over_full,
                             0, 0, 0, 0,
                             target_words, max_overlap_words, soft_token_limit)
                update_stats(global_stats_path,
                             added_articles_full, added_chunks_full, added_tokens_full, added_over_full,
                             0, 0, 0, 0,
                             target_words, max_overlap_words, soft_token_limit)

            chunked_global.add(pmcid)
            fixed += 1
            if fixed % 100 == 0:
                print(f"  fixed {fixed} missing fulltext articles...")

        print(f"Integrity pass complete. Newly chunked fulltext articles: {fixed:,}")

    print("\n=== DONE ===")
    print(f"Chunks root: {chunks_root}")
    print(f"Global stats: {global_stats_path}")
    if dry_run:
        print("(DRY RUN) No files were written.")


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Metadata-driven chunking + integrity fix pass (fulltext preferred).")
    p.add_argument("--corpus-root", default=DEFAULT_CORPUS_ROOT)
    p.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--chunks-root", default=DEFAULT_CHUNKS_ROOT)

    p.add_argument("--target-words", type=int, default=TARGET_WORDS_DEFAULT)
    p.add_argument("--max-overlap-words", type=int, default=MAX_OVERLAP_WORDS_DEFAULT)
    p.add_argument("--soft-token-limit", type=int, default=SOFT_TOKEN_LIMIT_DEFAULT)

    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs for years in range + global stats.")
    p.add_argument("--year-min", type=int, default=None)
    p.add_argument("--year-max", type=int, default=None)

    p.add_argument("--no-integrity-pass", action="store_true", help="Disable final integrity fix pass.")
    p.add_argument("--dry-run", action="store_true", help="Run without writing files.")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(
        corpus_root=args.corpus_root,
        metadata_path=args.metadata_path,
        chunks_root=args.chunks_root,
        target_words=args.target_words,
        max_overlap_words=args.max_overlap_words,
        soft_token_limit=args.soft_token_limit,
        overwrite=args.overwrite,
        year_min=args.year_min,
        year_max=args.year_max,
        do_integrity_pass=(not args.no_integrity_pass),
        dry_run=args.dry_run,
    )
