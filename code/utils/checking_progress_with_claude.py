#!/usr/bin/env python3
"""
audit_corpus.py — READ-ONLY audit of the CherryPicker corpus.

Scans:
  1. Chunk JSONL files under <chunks-root>/<YEAR>/ (both *_650w.jsonl and *_650w_METAONLY.jsonl)
  2. Cleaned full-text .txt files under <corpus-root>/<YEAR>/
  3. metadata_all.jsonl

Produces (all in --out-dir):
  audit_report.json   — aggregate stats (per-year + global) — small, the main deliverable
  audit_samples.json  — up to N example chunks/files per "suspicious" category
  audit_summary.txt   — human-readable plain-text summary

The script DOES NOT modify any file in the corpus.

Run examples:
  python audit_corpus.py
  python audit_corpus.py --chunks-root /path/to/chunks --corpus-root /path/to/new_corpus
  python audit_corpus.py --out-dir /tmp/audit
  python audit_corpus.py --skip-txt        # if you want a faster run (chunks + metadata only)
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Iterator

# -------------------------------------------------------------------
# Optional tiktoken — exact tokens if available, word-count proxy otherwise.
# -------------------------------------------------------------------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(_ENC.encode(text, disallowed_special=()))
    TOKEN_MODE = "tiktoken_cl100k_base"
except Exception:
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        # cheap approximation: ~1.33 tokens per word, close enough for distribution stats
        return int(len(text.split()) * 1.33)
    TOKEN_MODE = "word_count_x_1.33_approx"


# -------------------------------------------------------------------
# Defaults match your code; override with CLI args if needed.
# -------------------------------------------------------------------
DEFAULT_CHUNKS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_chunks/new_corpus_chunks"
DEFAULT_CORPUS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus"
DEFAULT_METADATA    = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus/metadata_all.jsonl"

EMBED_TOKEN_LIMIT = 8192      # OpenAI embedding hard limit
SOFT_TOKEN_LIMIT  = 1000      # what your chunker tracks
MIN_USEFUL_WORDS  = 80        # below this we flag as "very short"

SAMPLES_PER_CATEGORY = 8

# Heuristic patterns to detect noise that should probably not be embedded.
JUNK_PATTERNS = [
    ("nature_reporting_summary",      re.compile(r"\breporting summary\b", re.I)),
    ("source_data_footer",            re.compile(r"\bsource data\b", re.I)),
    ("see_also_figure",               re.compile(r"\bsee also (fig|figure|table)\b", re.I)),
    ("scale_bar",                     re.compile(r"\bscale bar\b", re.I)),
    ("figure_caption_pvalue",         re.compile(r"\bP\s*[<=]\s*0(\.\d+)?\b")),
    ("figure_caption_n_equals",       re.compile(r"\bn\s*=\s*\d+\b.*\bn\s*=\s*\d+\b")),
    ("supplementary_pointer",         re.compile(r"\bsupplementary (table|figure|materials?|information|methods?)\b", re.I)),
    ("extended_data_pointer",         re.compile(r"\bextended data (fig|figure|table)\b", re.I)),
    ("references_residue",            re.compile(r"\b(et al\.,?\s*\d{4}|doi:\s*10\.)", re.I)),
]

# Normalize PMCID consistent with your codebase
def norm_pmcid(raw: Any) -> str:
    if raw is None:
        return ""
    x = str(raw).strip().upper()
    if not x:
        return ""
    if x.endswith(".TXT"):
        x = x[:-4]
    if x.isdigit():
        return "PMC" + x
    m = re.search(r"(PMC\d+)", x)
    if m:
        return m.group(1)
    return x

YEAR_RE = re.compile(r"^\d{4}$")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def iter_jsonl(path: Path) -> Iterator[Tuple[int, Optional[dict], Optional[str]]]:
    """Yield (line_num, obj, error) per line. obj=None on parse error."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line), None
            except json.JSONDecodeError as e:
                yield i, None, f"line {i}: {e}"

def percentile(sorted_vals: List[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]

def summarize_dist(vals: List[int]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "min": s[0],
        "p05": percentile(s, 0.05),
        "p50": percentile(s, 0.50),
        "p95": percentile(s, 0.95),
        "p99": percentile(s, 0.99),
        "max": s[-1],
        "sum": sum(s),
        "mean": round(sum(s) / len(s), 1),
    }

def bucket_words(words: int) -> str:
    if words < 50:    return "lt_50"
    if words < 100:   return "50_99"
    if words < 200:   return "100_199"
    if words < 500:   return "200_499"
    if words < 800:   return "500_799_target"
    if words < 1200:  return "800_1199"
    return "ge_1200_oversize"

def bucket_tokens(toks: int) -> str:
    if toks < 100:    return "lt_100"
    if toks < 500:    return "100_499"
    if toks < 1000:   return "500_999"
    if toks < 1500:   return "1000_1499"
    if toks < 2000:   return "1500_1999"
    if toks < 4000:   return "2000_3999"
    if toks < 8192:   return "4000_8191"
    return "ge_8192_OVERSIZE_FOR_EMBED"


# -------------------------------------------------------------------
# Chunks audit
# -------------------------------------------------------------------
def audit_chunks(chunks_root: Path) -> Dict[str, Any]:
    print(f"\n[1/3] Auditing chunks under: {chunks_root}", flush=True)

    per_year: Dict[str, Any] = {}
    global_stats = {
        "files_seen": 0,
        "total_chunks": 0,
        "bad_lines": 0,
        "chunks_by_source": Counter(),
        "word_buckets": Counter(),
        "token_buckets": Counter(),
        "junk_pattern_hits": Counter(),
        "null_title": 0,
        "null_journal": 0,
        "null_pmcid": 0,
        "chunks_over_embed_limit": 0,
        "chunks_under_min_useful": 0,
        "duplicate_chunk_ids": 0,
        "pmcid_in_both_ft_and_mo_same_year": 0,
    }
    global_word_lens: List[int] = []
    global_token_lens: List[int] = []
    global_chunks_per_pmcid: Counter = Counter()
    global_journals: Counter = Counter()
    global_chunk_id_seen: set = set()
    global_pmcid_source_types: Dict[str, set] = defaultdict(set)
    global_pmcid_year_seen: Dict[str, set] = defaultdict(set)

    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    year_dirs = sorted([
        d for d in chunks_root.iterdir()
        if d.is_dir() and (YEAR_RE.match(d.name) or d.name == "UNKNOWN_YEAR")
    ], key=lambda p: p.name)

    for year_dir in year_dirs:
        year = year_dir.name
        files = sorted([
            f for f in year_dir.iterdir()
            if f.is_file() and f.suffix == ".jsonl" and "chunks_combined" in f.name
        ])
        if not files:
            continue

        y_stats = {
            "files": [f.name for f in files],
            "total_chunks": 0,
            "bad_lines": 0,
            "chunks_by_source": Counter(),
            "unique_pmcids": 0,
            "unique_journals": 0,
            "word_buckets": Counter(),
            "token_buckets": Counter(),
            "junk_pattern_hits": Counter(),
            "null_title": 0,
            "null_journal": 0,
            "null_pmcid": 0,
            "chunks_over_embed_limit": 0,
            "chunks_under_min_useful": 0,
            "duplicate_chunk_ids": 0,
            "pmcid_in_both_ft_and_mo": 0,
            "word_lens": [],
            "token_lens": [],
            "top_journals": [],
            "chunks_per_pmcid_dist": {},
            "chunks_per_article_dist_by_source": {},
        }

        y_chunks_per_pmcid: Counter = Counter()
        y_chunks_per_pmcid_by_source: Dict[str, Counter] = defaultdict(Counter)
        y_chunk_id_seen: set = set()
        y_pmcid_source_types: Dict[str, set] = defaultdict(set)
        y_journals: Counter = Counter()
        y_word_lens: List[int] = []
        y_token_lens: List[int] = []

        for fpath in files:
            global_stats["files_seen"] += 1
            print(f"   - scanning {fpath.name}", flush=True)
            for line_num, obj, err in iter_jsonl(fpath):
                if obj is None:
                    y_stats["bad_lines"] += 1
                    global_stats["bad_lines"] += 1
                    if len(samples["bad_lines"]) < SAMPLES_PER_CATEGORY:
                        samples["bad_lines"].append({
                            "file": str(fpath),
                            "error": err,
                        })
                    continue

                y_stats["total_chunks"] += 1
                global_stats["total_chunks"] += 1

                chunk_id = obj.get("chunk_id") or ""
                pmcid    = norm_pmcid(obj.get("pmcid"))
                src      = obj.get("source_type") or "unknown"
                title    = obj.get("title")
                journal  = obj.get("journal")
                text     = obj.get("text") or ""

                # Source type counts
                y_stats["chunks_by_source"][src] += 1
                global_stats["chunks_by_source"][src] += 1

                # Duplicate chunk_ids?
                if chunk_id in y_chunk_id_seen:
                    y_stats["duplicate_chunk_ids"] += 1
                    if len(samples["duplicate_chunk_ids"]) < SAMPLES_PER_CATEGORY:
                        samples["duplicate_chunk_ids"].append({"year": year, "chunk_id": chunk_id})
                else:
                    y_chunk_id_seen.add(chunk_id)
                if chunk_id in global_chunk_id_seen:
                    global_stats["duplicate_chunk_ids"] += 1
                else:
                    global_chunk_id_seen.add(chunk_id)

                # Null metadata
                if not title:    y_stats["null_title"]   += 1; global_stats["null_title"]   += 1
                if not journal:  y_stats["null_journal"] += 1; global_stats["null_journal"] += 1
                if not pmcid:    y_stats["null_pmcid"]   += 1; global_stats["null_pmcid"]   += 1

                # PMCID tracking
                if pmcid:
                    y_chunks_per_pmcid[pmcid] += 1
                    y_chunks_per_pmcid_by_source[src][pmcid] += 1
                    y_pmcid_source_types[pmcid].add(src)
                    global_chunks_per_pmcid[pmcid] += 1
                    global_pmcid_source_types[pmcid].add(src)
                    global_pmcid_year_seen[pmcid].add(year)

                # Journal tracking
                if journal:
                    y_journals[journal] += 1
                    global_journals[journal] += 1

                # Length distributions
                words = len(text.split())
                toks  = count_tokens(text)
                y_word_lens.append(words)
                y_token_lens.append(toks)
                global_word_lens.append(words)
                global_token_lens.append(toks)

                wbk = bucket_words(words)
                tbk = bucket_tokens(toks)
                y_stats["word_buckets"][wbk] += 1
                y_stats["token_buckets"][tbk] += 1
                global_stats["word_buckets"][wbk] += 1
                global_stats["token_buckets"][tbk] += 1

                if toks >= EMBED_TOKEN_LIMIT:
                    y_stats["chunks_over_embed_limit"] += 1
                    global_stats["chunks_over_embed_limit"] += 1
                    if len(samples["chunks_over_embed_limit"]) < SAMPLES_PER_CATEGORY:
                        samples["chunks_over_embed_limit"].append({
                            "chunk_id": chunk_id, "tokens": toks, "words": words,
                            "preview": text[:300],
                        })

                if words < MIN_USEFUL_WORDS:
                    y_stats["chunks_under_min_useful"] += 1
                    global_stats["chunks_under_min_useful"] += 1
                    if len(samples["chunks_under_min_useful"]) < SAMPLES_PER_CATEGORY:
                        samples["chunks_under_min_useful"].append({
                            "chunk_id": chunk_id, "words": words, "tokens": toks,
                            "text": text[:400],
                        })

                # Junk pattern hits — only flag short-ish chunks where it's diagnostic
                if words < 250:
                    for label, pat in JUNK_PATTERNS:
                        if pat.search(text):
                            y_stats["junk_pattern_hits"][label] += 1
                            global_stats["junk_pattern_hits"][label] += 1
                            if len(samples[f"junk_{label}"]) < SAMPLES_PER_CATEGORY:
                                samples[f"junk_{label}"].append({
                                    "chunk_id": chunk_id, "words": words,
                                    "text": text[:400],
                                })

        # Same-year fulltext+meta-only collisions
        ft_and_mo = sum(1 for st in y_pmcid_source_types.values() if {"fulltext","metadata_only"}.issubset(st))
        y_stats["pmcid_in_both_ft_and_mo"] = ft_and_mo

        # Compact
        y_stats["unique_pmcids"]   = len(y_chunks_per_pmcid)
        y_stats["unique_journals"] = len(y_journals)
        y_stats["top_journals"]    = y_journals.most_common(20)
        y_stats["chunks_per_pmcid_dist"] = summarize_dist(list(y_chunks_per_pmcid.values()))
        y_stats["chunks_per_article_dist_by_source"] = {
            src: summarize_dist(list(cnt.values())) for src, cnt in y_chunks_per_pmcid_by_source.items()
        }
        y_stats["word_lens"]  = summarize_dist(y_word_lens)
        y_stats["token_lens"] = summarize_dist(y_token_lens)
        # convert Counters to plain dicts for JSON
        y_stats["chunks_by_source"]   = dict(y_stats["chunks_by_source"])
        y_stats["word_buckets"]       = dict(y_stats["word_buckets"])
        y_stats["token_buckets"]      = dict(y_stats["token_buckets"])
        y_stats["junk_pattern_hits"]  = dict(y_stats["junk_pattern_hits"])

        per_year[year] = y_stats

    # Cross-year PMCID duplicates
    pmcid_multi_year = {pid: sorted(list(ys)) for pid, ys in global_pmcid_year_seen.items() if len(ys) > 1}
    cross_year_pmcid_count = len(pmcid_multi_year)
    for pid, ys in list(pmcid_multi_year.items())[:SAMPLES_PER_CATEGORY]:
        samples["pmcid_in_multiple_years"].append({"pmcid": pid, "years": ys})

    # Global PMCID with both FT and MO across whole corpus
    ft_mo_global = [pid for pid, sts in global_pmcid_source_types.items() if {"fulltext","metadata_only"}.issubset(sts)]
    for pid in ft_mo_global[:SAMPLES_PER_CATEGORY]:
        samples["pmcid_in_both_ft_and_mo_global"].append({"pmcid": pid})

    global_stats["chunks_by_source"]   = dict(global_stats["chunks_by_source"])
    global_stats["word_buckets"]       = dict(global_stats["word_buckets"])
    global_stats["token_buckets"]      = dict(global_stats["token_buckets"])
    global_stats["junk_pattern_hits"]  = dict(global_stats["junk_pattern_hits"])
    global_stats["unique_pmcids"]      = len(global_chunks_per_pmcid)
    global_stats["unique_journals"]    = len(global_journals)
    global_stats["top_journals"]       = global_journals.most_common(30)
    global_stats["chunks_per_pmcid_dist"] = summarize_dist(list(global_chunks_per_pmcid.values()))
    global_stats["word_lens"]          = summarize_dist(global_word_lens)
    global_stats["token_lens"]         = summarize_dist(global_token_lens)
    global_stats["pmcid_cross_year_duplicates"] = cross_year_pmcid_count
    global_stats["pmcid_in_both_ft_and_mo_global_count"] = len(ft_mo_global)

    return {"per_year": per_year, "global": global_stats, "samples": dict(samples)}


# -------------------------------------------------------------------
# Cleaned text .txt corpus audit
# -------------------------------------------------------------------
def audit_txt(corpus_root: Path, max_sample_text_bytes: int = 0) -> Dict[str, Any]:
    print(f"\n[2/3] Auditing cleaned text under: {corpus_root}", flush=True)
    per_year: Dict[str, Any] = {}
    global_file_count = 0
    global_byte_total = 0
    global_empty = 0
    global_tiny = 0   # < 1 KB
    global_size_samples: List[int] = []
    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    year_dirs = sorted([
        d for d in corpus_root.iterdir()
        if d.is_dir() and YEAR_RE.match(d.name)
    ], key=lambda p: p.name)

    for year_dir in year_dirs:
        year = year_dir.name
        txt_files = [f for f in year_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
        sizes = []
        empty = 0
        tiny = 0
        for f in txt_files:
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            sizes.append(sz)
            global_byte_total += sz
            if sz == 0:
                empty += 1
                global_empty += 1
                if len(samples["empty_txt_files"]) < SAMPLES_PER_CATEGORY:
                    samples["empty_txt_files"].append({"path": str(f)})
            elif sz < 1024:
                tiny += 1
                global_tiny += 1
                if len(samples["tiny_txt_files"]) < SAMPLES_PER_CATEGORY:
                    samples["tiny_txt_files"].append({"path": str(f), "size_bytes": sz})

        if sizes:
            per_year[year] = {
                "file_count": len(sizes),
                "empty_files": empty,
                "tiny_files_lt_1KB": tiny,
                "byte_size_dist": summarize_dist(sizes),
            }
        global_file_count += len(sizes)
        global_size_samples.extend(sizes)
        print(f"   - {year}: {len(sizes)} txt files", flush=True)

    return {
        "per_year": per_year,
        "global": {
            "file_count": global_file_count,
            "byte_total": global_byte_total,
            "empty_files": global_empty,
            "tiny_files_lt_1KB": global_tiny,
            "byte_size_dist": summarize_dist(global_size_samples),
        },
        "samples": dict(samples),
    }


# -------------------------------------------------------------------
# Metadata audit (light)
# -------------------------------------------------------------------
def audit_metadata(meta_path: Path) -> Dict[str, Any]:
    print(f"\n[3/3] Auditing metadata: {meta_path}", flush=True)
    if not meta_path.exists():
        return {"error": f"metadata file not found: {meta_path}"}

    total = 0
    bad = 0
    by_year: Counter = Counter()
    by_journal: Counter = Counter()
    has_fulltext = 0
    has_oa = 0
    no_pmcid = 0
    duplicate_pmcid_count = 0
    pmcid_seen: set = set()
    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for line_num, obj, err in iter_jsonl(meta_path):
        if obj is None:
            bad += 1
            continue
        total += 1
        pmcid = norm_pmcid(obj.get("pmcid") or obj.get("PMCID") or obj.get("id"))
        if not pmcid:
            no_pmcid += 1
        else:
            if pmcid in pmcid_seen:
                duplicate_pmcid_count += 1
                if len(samples["duplicate_pmcid_in_metadata"]) < SAMPLES_PER_CATEGORY:
                    samples["duplicate_pmcid_in_metadata"].append({"pmcid": pmcid})
            else:
                pmcid_seen.add(pmcid)

        # journal — handle nested or flat
        jt = None
        ji = obj.get("journalInfo")
        if isinstance(ji, dict):
            try:
                jt = ji.get("journal", {}).get("title")
            except Exception:
                pass
        jt = jt or obj.get("journalTitle") or obj.get("journal")
        if jt:
            by_journal[str(jt).strip()] += 1

        # year
        y = obj.get("pubYear") or obj.get("year") or obj.get("publicationYear")
        if not y:
            v = obj.get("firstPublicationDate") or obj.get("pubDate") or obj.get("date")
            if v:
                m = re.search(r"(19|20)\d{2}", str(v))
                if m:
                    y = m.group(0)
        if y:
            by_year[str(y)] += 1

        if obj.get("has_fulltext") or obj.get("has_full_text"):
            has_fulltext += 1
        v = obj.get("isOpenAccess")
        if v and str(v).strip().upper() == "Y":
            has_oa += 1

    return {
        "total_records": total,
        "bad_lines": bad,
        "no_pmcid": no_pmcid,
        "duplicate_pmcid_rows": duplicate_pmcid_count,
        "unique_pmcids": len(pmcid_seen),
        "records_with_fulltext_flag": has_fulltext,
        "records_open_access": has_oa,
        "records_by_year": dict(sorted(by_year.items())),
        "top_journals": by_journal.most_common(30),
        "samples": dict(samples),
    }


# -------------------------------------------------------------------
# Output formatters
# -------------------------------------------------------------------
def write_summary_txt(report: Dict[str, Any], out_path: Path) -> None:
    g = report["chunks"]["global"]
    tx = report["txt"]["global"] if report.get("txt") else None
    m = report["metadata"]

    lines = []
    lines.append("=" * 78)
    lines.append("CHERRYPICKER CORPUS AUDIT — SUMMARY")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Token measurement mode: {report['token_mode']}")
    lines.append("")
    lines.append("--- CHUNKS (GLOBAL) ---")
    lines.append(f"  Files seen:                  {g['files_seen']}")
    lines.append(f"  Total chunks:                {g['total_chunks']:,}")
    lines.append(f"  Bad JSON lines:              {g['bad_lines']}")
    lines.append(f"  Unique PMCIDs:               {g['unique_pmcids']:,}")
    lines.append(f"  Unique journals:             {g['unique_journals']}")
    lines.append(f"  Chunks by source:            {g['chunks_by_source']}")
    lines.append(f"  Null title:                  {g['null_title']:,}")
    lines.append(f"  Null journal:                {g['null_journal']:,}")
    lines.append(f"  Null pmcid:                  {g['null_pmcid']:,}")
    lines.append(f"  Duplicate chunk_ids:         {g['duplicate_chunk_ids']:,}")
    lines.append(f"  PMCID with both FT and MO:   {g.get('pmcid_in_both_ft_and_mo_global_count', '?')}")
    lines.append(f"  PMCID across multiple years: {g.get('pmcid_cross_year_duplicates', '?')}")
    lines.append("")
    lines.append("  Chunk token length distribution:")
    for k, v in (g.get("token_lens") or {}).items():
        lines.append(f"    {k:>5}: {v}")
    lines.append("")
    lines.append("  Token buckets:")
    for k, v in sorted((g.get("token_buckets") or {}).items()):
        lines.append(f"    {k:<32} {v:,}")
    lines.append("")
    lines.append(f"  Chunks over embed limit (>=8192 tokens): {g['chunks_over_embed_limit']:,}")
    lines.append(f"  Chunks under min useful ({MIN_USEFUL_WORDS} words):   {g['chunks_under_min_useful']:,}")
    lines.append("")
    lines.append("  Junk pattern hits (in short chunks):")
    for k, v in sorted((g.get("junk_pattern_hits") or {}).items(), key=lambda kv: -kv[1]):
        lines.append(f"    {k:<35} {v:,}")
    lines.append("")
    lines.append("  Top journals:")
    for j, c in (g.get("top_journals") or [])[:20]:
        lines.append(f"    {c:>8,}  {j}")
    lines.append("")
    if tx:
        lines.append("--- CLEANED TEXT FILES (GLOBAL) ---")
        lines.append(f"  Files:                       {tx['file_count']:,}")
        lines.append(f"  Total bytes:                 {tx['byte_total']:,}")
        lines.append(f"  Empty .txt files:            {tx['empty_files']:,}")
        lines.append(f"  .txt files < 1KB:            {tx['tiny_files_lt_1KB']:,}")
        lines.append(f"  Size distribution (bytes):   {tx['byte_size_dist']}")
        lines.append("")
    if m and not m.get("error"):
        lines.append("--- METADATA ---")
        lines.append(f"  Total records:               {m['total_records']:,}")
        lines.append(f"  Bad lines:                   {m['bad_lines']}")
        lines.append(f"  Records w/o PMCID:           {m['no_pmcid']:,}")
        lines.append(f"  Duplicate PMCID rows:        {m['duplicate_pmcid_rows']:,}")
        lines.append(f"  Unique PMCIDs:               {m['unique_pmcids']:,}")
        lines.append(f"  Has fulltext flag:           {m['records_with_fulltext_flag']:,}")
        lines.append(f"  Open access (Y):             {m['records_open_access']:,}")
        lines.append("")

    lines.append("--- PER-YEAR CHUNK COUNTS ---")
    for year, ys in sorted(report["chunks"]["per_year"].items()):
        lines.append(f"  {year}: chunks={ys['total_chunks']:,} pmcids={ys['unique_pmcids']:,} "
                     f"bad={ys['bad_lines']} short<{MIN_USEFUL_WORDS}w={ys['chunks_under_min_useful']:,} "
                     f"ft_and_mo_collisions={ys.get('pmcid_in_both_ft_and_mo', 0)}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Read-only audit of the CherryPicker corpus.")
    ap.add_argument("--chunks-root", default=DEFAULT_CHUNKS_ROOT)
    ap.add_argument("--corpus-root", default=DEFAULT_CORPUS_ROOT)
    ap.add_argument("--metadata",    default=DEFAULT_METADATA)
    ap.add_argument("--out-dir",     default=".", help="Where to write audit_*.{json,txt}")
    ap.add_argument("--skip-txt",    action="store_true", help="Skip .txt cleaned-text scan (faster)")
    args = ap.parse_args()

    chunks_root = Path(args.chunks_root)
    corpus_root = Path(args.corpus_root)
    meta_path   = Path(args.metadata)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_root.exists():
        print(f"ERROR: chunks_root does not exist: {chunks_root}", file=sys.stderr)
        sys.exit(2)

    print(f"Token mode: {TOKEN_MODE}")
    print(f"Chunks root: {chunks_root}")
    print(f"Corpus root: {corpus_root}")
    print(f"Metadata:    {meta_path}")
    print(f"Out dir:     {out_dir.resolve()}")
    print(f"Skip txt scan: {args.skip_txt}")

    chunks_report = audit_chunks(chunks_root)
    txt_report = None if args.skip_txt else audit_txt(corpus_root)
    meta_report = audit_metadata(meta_path)

    samples_combined: Dict[str, List[Dict[str, Any]]] = {}
    for src in (chunks_report.get("samples"), (txt_report or {}).get("samples"), meta_report.get("samples")):
        if not src: continue
        for k, v in src.items():
            samples_combined.setdefault(k, []).extend(v[:SAMPLES_PER_CATEGORY])

    # strip samples from inside the report objects for cleanliness
    chunks_report.pop("samples", None)
    if txt_report: txt_report.pop("samples", None)
    meta_report.pop("samples", None)

    report = {
        "token_mode": TOKEN_MODE,
        "chunks": chunks_report,
        "txt": txt_report,
        "metadata": meta_report,
    }

    report_path = out_dir / "audit_report.json"
    samples_path = out_dir / "audit_samples.json"
    summary_path = out_dir / "audit_summary.txt"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with samples_path.open("w", encoding="utf-8") as f:
        json.dump(samples_combined, f, ensure_ascii=False, indent=2)
    write_summary_txt(report, summary_path)

    print("\nDONE.")
    print(f"  {report_path}")
    print(f"  {samples_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()