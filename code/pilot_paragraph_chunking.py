#!/usr/bin/env python3
"""
Chunk the entire EuropePMC corpus into retrieval-friendly JSONL files.

What this version does (includes BOTH suggestions you asked for):
1) ✅ Writes ONE combined JSONL per year that includes BOTH:
   - fulltext chunks (source_type="fulltext")
   - metadata-only chunks for articles missing a .txt (source_type="metadata_only")

2) ✅ Adds a fallback for metadata records with no usable year:
   - CHUNKS_ROOT/UNKNOWN_YEAR/UNKNOWN_YEAR_meta_only_chunks_<TARGET>w.jsonl
   - CHUNKS_ROOT/UNKNOWN_YEAR/UNKNOWN_YEAR_chunking_stats_<TARGET>w.json

Also:
- Chunks are LEAN (no token_count/word_count/source_path stored inside chunk JSON).
- Token accounting is computed and saved into stats JSON (per year + global).

Outputs:
  CHUNKS_ROOT/<YEAR>/<YEAR>_chunks_combined_<TARGET>w.jsonl
  CHUNKS_ROOT/<YEAR>/<YEAR>_chunking_stats_<TARGET>w.json
  CHUNKS_ROOT/<YEAR>/<YEAR>_DONE_<TARGET>w.txt

  CHUNKS_ROOT/UNKNOWN_YEAR/UNKNOWN_YEAR_meta_only_chunks_<TARGET>w.jsonl
  CHUNKS_ROOT/UNKNOWN_YEAR/UNKNOWN_YEAR_chunking_stats_<TARGET>w.json

  CHUNKS_ROOT/ALL_chunking_stats_<TARGET>w.json

Usage examples:
  python chunk_full_corpus_v4_combined.py
  python chunk_full_corpus_v4_combined.py --year-min 2019 --year-max 2020
  python chunk_full_corpus_v4_combined.py --max-years 2 --max-articles-per-year 10
  python chunk_full_corpus_v4_combined.py --overwrite
"""

import os
import json
import re
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set

import tiktoken  # pip install tiktoken


# ============ DEFAULT CONFIG ============
DEFAULT_CORPUS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_europepmc"
DEFAULT_METADATA_PATH = os.path.join(
    DEFAULT_CORPUS_ROOT,
    "metadata_all_1990_2025.BACKUP_before_remove_2020.jsonl",
)
DEFAULT_CHUNKS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_chunks/pilot_paragraph"

TARGET_WORDS_DEFAULT = 650
MAX_OVERLAP_WORDS_DEFAULT = 120

ENCODING_NAME = "cl100k_base"
SOFT_TOKEN_LIMIT_DEFAULT = 1000
# ======================================


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def norm_pmcid(raw: str) -> str:
    if not raw:
        return ""
    x = str(raw).strip().upper()
    if x.endswith(".TXT"):
        x = x[:-4]
    return "PMC" + x if x.isdigit() else x


def count_tokens_factory():
    enc = tiktoken.get_encoding(ENCODING_NAME)

    def count_tokens(text: str) -> int:
        return len(enc.encode(text))

    return count_tokens


# -----------------------------
# Metadata loading / parsing
# -----------------------------
def load_metadata_index(metadata_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads a jsonl metadata file into an index keyed by normalized PMCID.
    Handles the "single top-level key" records (dict with one key whose value is dict).
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
                if isinstance(obj, dict) and len(obj) == 1 and isinstance(list(obj.values())[0], dict):
                    obj = list(obj.values())[0]

                pmcid_raw = obj.get("pmcid") or obj.get("PMCID") or obj.get("id")
                if pmcid_raw:
                    index[norm_pmcid(pmcid_raw)] = obj
            except Exception:
                continue

    return index


def meta_year(meta: Dict[str, Any]) -> Optional[int]:
    """
    Extract a year from common EuropePMC metadata keys.
    """
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
    return None


def meta_text_payload(meta: Dict[str, Any]) -> str:
    """
    Build a metadata-only text payload to chunk (title + abstract + keywords).
    """
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
    """
    Groups paragraphs up to target_words (approx), with a capped overlap bridge
    from the last paragraph of the previous chunk.
    """
    chunks: List[str] = []
    current_group: List[str] = []
    current_count = 0

    for p in paragraphs:
        wlen = len(p.split())

        # Huge paragraph -> split into blocks (no bridge across those blocks)
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
# Chunk builders (LEAN schema)
# -----------------------------
def build_fulltext_chunks(
    year: int,
    txt_path: str,
    metadata_index: Dict[str, Dict[str, Any]],
    target_words: int,
    max_overlap_words: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (pmcid, chunks_list). Chunks are lean (no token_count stored).
    """
    filename = os.path.basename(txt_path)
    pmcid = norm_pmcid(os.path.splitext(filename)[0])

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return pmcid, []

    paragraphs = split_into_paragraphs(text)
    raw_chunks = paragraphs_to_chunks(paragraphs, target_words, max_overlap_words)

    meta = metadata_index.get(pmcid, {})

    chunks: List[Dict[str, Any]] = []
    for idx, ch_text in enumerate(raw_chunks, start=1):
        chunks.append(
            {
                "chunk_id": f"{year}_{pmcid}_{idx:04d}",
                "pmcid": pmcid,
                "year": year,
                "chunk_index": idx,
                "title": meta.get("title") or meta.get("article_title"),
                "journal": meta.get("journal") or meta.get("journalTitle") or meta.get("source"),
                "source_type": "fulltext",
                "text": ch_text,
            }
        )

    return pmcid, chunks


def build_metadata_only_chunks(
    year: int,
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
        chunks.append(
            {
                "chunk_id": f"{year}_{pmcid}_META_{idx:04d}",
                "pmcid": pmcid,
                "year": year,
                "chunk_index": idx,
                "title": meta.get("title") or meta.get("article_title"),
                "journal": meta.get("journal") or meta.get("journalTitle") or meta.get("source"),
                "source_type": "metadata_only",
                "text": ch_text,
            }
        )
    return chunks


# -----------------------------
# Corpus traversal
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
            try:
                out.append((int(name), p))
            except ValueError:
                pass

    out.sort(key=lambda x: x[0])
    return out


def safe_remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


# -----------------------------
# Main runner
# -----------------------------
def run_full_corpus(
    corpus_root: str,
    metadata_path: str,
    chunks_root: str,
    target_words: int,
    max_overlap_words: int,
    soft_token_limit: int,
    overwrite: bool,
    max_years: Optional[int],
    max_articles_per_year: Optional[int],
    year_min: Optional[int],
    year_max: Optional[int],
) -> None:
    ensure_dir(chunks_root)

    print("Loading metadata index...")
    metadata_index = load_metadata_index(metadata_path)
    print(f"Metadata entries indexed: {len(metadata_index):,}")

    count_tokens = count_tokens_factory()

    year_dirs = iter_year_dirs(corpus_root)
    if year_min is not None:
        year_dirs = [yd for yd in year_dirs if yd[0] >= year_min]
    if year_max is not None:
        year_dirs = [yd for yd in year_dirs if yd[0] <= year_max]
    if max_years is not None:
        year_dirs = year_dirs[:max_years]

    if not year_dirs:
        print(f"❌ No year directories found under: {corpus_root}")
        return

    # Pre-group metadata by year (including None -> unknown)
    meta_by_year: Dict[Optional[int], List[Tuple[str, Dict[str, Any]]]] = {}
    for pmcid, meta in metadata_index.items():
        y = meta_year(meta)
        meta_by_year.setdefault(y, []).append((pmcid, meta))

    global_stats = {
        "target_words": target_words,
        "max_overlap_words": max_overlap_words,
        "soft_token_limit": soft_token_limit,
        "years_processed": 0,

        "articles_fulltext": 0,
        "chunks_fulltext": 0,
        "tokens_fulltext_total": 0,
        "chunks_fulltext_over_soft_limit": 0,

        "articles_meta_only": 0,
        "chunks_meta_only": 0,
        "tokens_meta_only_total": 0,
        "chunks_meta_only_over_soft_limit": 0,

        "unknown_year_articles_meta_only": 0,
        "unknown_year_chunks_meta_only": 0,
        "unknown_year_tokens_meta_only_total": 0,
        "unknown_year_chunks_meta_only_over_soft_limit": 0,
    }

    # -------------------------
    # Process normal years (YYYY dirs)
    # -------------------------
    for year, year_dir in year_dirs:
        out_dir = os.path.join(chunks_root, str(year))
        ensure_dir(out_dir)

        out_combined = os.path.join(out_dir, f"{year}_chunks_combined_{target_words}w.jsonl")
        out_stats = os.path.join(out_dir, f"{year}_chunking_stats_{target_words}w.json")
        done_flag = os.path.join(out_dir, f"{year}_DONE_{target_words}w.txt")

        if (not overwrite) and os.path.exists(out_combined) and os.path.exists(out_stats) and os.path.exists(done_flag):
            print(f"⏭️  Skip {year}: already done")
            continue

        txt_files = sorted(
            os.path.join(year_dir, f)
            for f in os.listdir(year_dir)
            if f.lower().endswith(".txt")
        )
        if max_articles_per_year is not None:
            txt_files = txt_files[:max_articles_per_year]

        # overwrite cleanup
        if overwrite:
            safe_remove(out_combined)
            safe_remove(out_stats)
            safe_remove(done_flag)
        else:
            # remove done flag if partial work to avoid "false done"
            if os.path.exists(done_flag):
                safe_remove(done_flag)

        print(f"\n=== Year {year} ===")
        print(f"Fulltext articles (.txt): {len(txt_files):,}")

        year_stats = {
            "year": year,
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
        }

        seen_fulltext_pmcids: Set[str] = set()

        # Write combined file: fulltext first, then metadata-only for missing fulltext.
        with open(out_combined, "w", encoding="utf-8") as f_out:
            # ---- fulltext chunks
            for i, txt_path in enumerate(txt_files, start=1):
                pmcid, chunks = build_fulltext_chunks(
                    year=year,
                    txt_path=txt_path,
                    metadata_index=metadata_index,
                    target_words=target_words,
                    max_overlap_words=max_overlap_words,
                )
                if not chunks:
                    continue

                seen_fulltext_pmcids.add(pmcid)
                year_stats["articles_fulltext"] += 1

                for ch in chunks:
                    t = count_tokens(ch["text"])
                    year_stats["tokens_fulltext_total"] += t
                    if t > soft_token_limit:
                        year_stats["chunks_fulltext_over_soft_limit"] += 1

                    f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                    year_stats["chunks_fulltext"] += 1

                if i % 50 == 0:
                    print(f"  processed {i}/{len(txt_files)} fulltext articles...")

            # ---- metadata-only chunks for this year (excluding those with fulltext)
            for pmcid, meta in meta_by_year.get(year, []):
                if pmcid in seen_fulltext_pmcids:
                    continue

                meta_chunks = build_metadata_only_chunks(
                    year=year,
                    pmcid=pmcid,
                    meta=meta,
                    target_words=target_words,
                    max_overlap_words=max_overlap_words,
                )
                if not meta_chunks:
                    continue

                year_stats["articles_meta_only"] += 1
                for ch in meta_chunks:
                    t = count_tokens(ch["text"])
                    year_stats["tokens_meta_only_total"] += t
                    if t > soft_token_limit:
                        year_stats["chunks_meta_only_over_soft_limit"] += 1

                    f_out.write(json.dumps(ch, ensure_ascii=False) + "\n")
                    year_stats["chunks_meta_only"] += 1

        # Derived stats
        year_stats["avg_tokens_per_chunk_fulltext"] = (
            int(year_stats["tokens_fulltext_total"] / year_stats["chunks_fulltext"])
            if year_stats["chunks_fulltext"] else 0
        )
        year_stats["avg_tokens_per_chunk_meta_only"] = (
            int(year_stats["tokens_meta_only_total"] / year_stats["chunks_meta_only"])
            if year_stats["chunks_meta_only"] else 0
        )

        # Print summary
        print(
            f"Fulltext: articles={year_stats['articles_fulltext']:,} "
            f"chunks={year_stats['chunks_fulltext']:,} "
            f"tokens={year_stats['tokens_fulltext_total']:,} "
            f"over_limit={year_stats['chunks_fulltext_over_soft_limit']:,}"
        )
        print(
            f"Meta-only: articles={year_stats['articles_meta_only']:,} "
            f"chunks={year_stats['chunks_meta_only']:,} "
            f"tokens={year_stats['tokens_meta_only_total']:,} "
            f"over_limit={year_stats['chunks_meta_only_over_soft_limit']:,}"
        )

        # Save year stats
        with open(out_stats, "w", encoding="utf-8") as sf:
            json.dump(year_stats, sf, ensure_ascii=False, indent=2)

        # Mark done
        with open(done_flag, "w", encoding="utf-8") as df:
            df.write("done\n")

        # Update global stats
        global_stats["years_processed"] += 1

        global_stats["articles_fulltext"] += year_stats["articles_fulltext"]
        global_stats["chunks_fulltext"] += year_stats["chunks_fulltext"]
        global_stats["tokens_fulltext_total"] += year_stats["tokens_fulltext_total"]
        global_stats["chunks_fulltext_over_soft_limit"] += year_stats["chunks_fulltext_over_soft_limit"]

        global_stats["articles_meta_only"] += year_stats["articles_meta_only"]
        global_stats["chunks_meta_only"] += year_stats["chunks_meta_only"]
        global_stats["tokens_meta_only_total"] += year_stats["tokens_meta_only_total"]
        global_stats["chunks_meta_only_over_soft_limit"] += year_stats["chunks_meta_only_over_soft_limit"]

    # -------------------------
    # Process UNKNOWN YEAR metadata-only
    # -------------------------
    unknown_records = meta_by_year.get(None, [])
    if unknown_records:
        unknown_dir = os.path.join(chunks_root, "UNKNOWN_YEAR")
        ensure_dir(unknown_dir)

        unknown_out = os.path.join(unknown_dir, f"UNKNOWN_YEAR_meta_only_chunks_{target_words}w.jsonl")
        unknown_stats_path = os.path.join(unknown_dir, f"UNKNOWN_YEAR_chunking_stats_{target_words}w.json")

        if overwrite:
            safe_remove(unknown_out)
            safe_remove(unknown_stats_path)

        unknown_stats = {
            "year": None,
            "label": "UNKNOWN_YEAR",
            "target_words": target_words,
            "max_overlap_words": max_overlap_words,
            "soft_token_limit": soft_token_limit,

            "articles_meta_only": 0,
            "chunks_meta_only": 0,
            "tokens_meta_only_total": 0,
            "chunks_meta_only_over_soft_limit": 0,
        }

        with open(unknown_out, "w", encoding="utf-8") as fu:
            for pmcid, meta in unknown_records:
                # Assign a pseudo-year 0 in chunk_id prefix while storing year=None in record.
                # (Helps keep chunk_id unique and readable.)
                pseudo_year = 0

                meta_chunks = build_metadata_only_chunks(
                    year=pseudo_year,
                    pmcid=pmcid,
                    meta=meta,
                    target_words=target_words,
                    max_overlap_words=max_overlap_words,
                )
                if not meta_chunks:
                    continue

                # Patch year field to None and chunk_id to start with UNKNOWN_YEAR
                # so you don't accidentally treat these as real year=0 later.
                unknown_stats["articles_meta_only"] += 1
                for ch in meta_chunks:
                    ch["year"] = None
                    ch["chunk_id"] = "UNKNOWN_YEAR_" + ch["chunk_id"][2:]  # replace leading "0_"
                    t = count_tokens(ch["text"])
                    unknown_stats["tokens_meta_only_total"] += t
                    if t > soft_token_limit:
                        unknown_stats["chunks_meta_only_over_soft_limit"] += 1

                    fu.write(json.dumps(ch, ensure_ascii=False) + "\n")
                    unknown_stats["chunks_meta_only"] += 1

        unknown_stats["avg_tokens_per_chunk_meta_only"] = (
            int(unknown_stats["tokens_meta_only_total"] / unknown_stats["chunks_meta_only"])
            if unknown_stats["chunks_meta_only"] else 0
        )

        with open(unknown_stats_path, "w", encoding="utf-8") as sf:
            json.dump(unknown_stats, sf, ensure_ascii=False, indent=2)

        print(f"\n=== UNKNOWN_YEAR metadata-only ===")
        print(
            f"Meta-only: articles={unknown_stats['articles_meta_only']:,} "
            f"chunks={unknown_stats['chunks_meta_only']:,} "
            f"tokens={unknown_stats['tokens_meta_only_total']:,} "
            f"over_limit={unknown_stats['chunks_meta_only_over_soft_limit']:,}"
        )

        global_stats["unknown_year_articles_meta_only"] = unknown_stats["articles_meta_only"]
        global_stats["unknown_year_chunks_meta_only"] = unknown_stats["chunks_meta_only"]
        global_stats["unknown_year_tokens_meta_only_total"] = unknown_stats["tokens_meta_only_total"]
        global_stats["unknown_year_chunks_meta_only_over_soft_limit"] = unknown_stats["chunks_meta_only_over_soft_limit"]

    # Global derived stats
    global_stats["avg_tokens_per_chunk_fulltext"] = (
        int(global_stats["tokens_fulltext_total"] / global_stats["chunks_fulltext"])
        if global_stats["chunks_fulltext"] else 0
    )
    global_stats["avg_tokens_per_chunk_meta_only"] = (
        int(global_stats["tokens_meta_only_total"] / global_stats["chunks_meta_only"])
        if global_stats["chunks_meta_only"] else 0
    )

    global_stats_path = os.path.join(chunks_root, f"ALL_chunking_stats_{target_words}w.json")
    with open(global_stats_path, "w", encoding="utf-8") as gf:
        json.dump(global_stats, gf, ensure_ascii=False, indent=2)

    print("\n=== ALL DONE ===")
    print(f"Years processed: {global_stats['years_processed']:,}")
    print(
        f"Fulltext: articles={global_stats['articles_fulltext']:,} "
        f"chunks={global_stats['chunks_fulltext']:,} "
        f"tokens={global_stats['tokens_fulltext_total']:,}"
    )
    print(
        f"Meta-only (year-assigned): articles={global_stats['articles_meta_only']:,} "
        f"chunks={global_stats['chunks_meta_only']:,} "
        f"tokens={global_stats['tokens_meta_only_total']:,}"
    )
    if global_stats["unknown_year_chunks_meta_only"] > 0:
        print(
            f"Meta-only (UNKNOWN_YEAR): articles={global_stats['unknown_year_articles_meta_only']:,} "
            f"chunks={global_stats['unknown_year_chunks_meta_only']:,} "
            f"tokens={global_stats['unknown_year_tokens_meta_only_total']:,}"
        )
    print(f"Global stats written: {global_stats_path}")


# -----------------------------
# CLI
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Chunk the entire EuropePMC corpus into JSONL files (per year, combined).")
    p.add_argument("--corpus-root", default=DEFAULT_CORPUS_ROOT)
    p.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    p.add_argument("--chunks-root", default=DEFAULT_CHUNKS_ROOT)

    p.add_argument("--target-words", type=int, default=TARGET_WORDS_DEFAULT)
    p.add_argument("--max-overlap-words", type=int, default=MAX_OVERLAP_WORDS_DEFAULT)
    p.add_argument("--soft-token-limit", type=int, default=SOFT_TOKEN_LIMIT_DEFAULT)

    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--max-years", type=int, default=None, help="Process only first N years (debug).")
    p.add_argument("--max-articles-per-year", type=int, default=None, help="Limit articles per year (debug).")

    p.add_argument("--year-min", type=int, default=None, help="Only process years >= this.")
    p.add_argument("--year-max", type=int, default=None, help="Only process years <= this.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    run_full_corpus(
        corpus_root=args.corpus_root,
        metadata_path=args.metadata_path,
        chunks_root=args.chunks_root,
        target_words=args.target_words,
        max_overlap_words=args.max_overlap_words,
        soft_token_limit=args.soft_token_limit,
        overwrite=args.overwrite,
        max_years=args.max_years,
        max_articles_per_year=args.max_articles_per_year,
        year_min=args.year_min,
        year_max=args.year_max,
    )
