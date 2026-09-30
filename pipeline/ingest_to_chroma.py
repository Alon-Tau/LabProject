#!/usr/bin/env python3
"""
ingest_to_chroma.py
-------------------
Load the OpenAI batch-API embedding results into a persistent ChromaDB
collection, joining each embedding back to its chunk text from the original
batch INPUT files (so retrieval returns real content, not just IDs).

Collection layout: ONE collection named `cherrypicker_chunks`, with `year`,
`source`, `pmcid`, `pmid`, `chunk_id`, `split` stored as metadata so we can
filter at query time (year ranges, source type, etc.).

What it does:
  * Walks data/corpus_embeddings/batch_results/{year}_{batch_id}_results.jsonl
  * For each year, builds a custom_id -> input_text map by scanning the
    matching data/batch_inputs_v2/{year}/batch_*.jsonl files.
  * Parses each result line, joins on custom_id, upserts into Chroma in
    batches (default 1000). Upsert means re-runs are safe and idempotent.

Usage:
  # ingest everything we have results for
  python ingest_to_chroma.py

  # restrict to one or more years (handy for sanity checks)
  python ingest_to_chroma.py --years 2025
  python ingest_to_chroma.py --years 2024,2025

  # dry run — show how many items per year would be ingested, do nothing
  python ingest_to_chroma.py --dry-run

  # use a custom Chroma path or batch size
  python ingest_to_chroma.py --chroma-dir /home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/chroma_db --batch-size 2000

  # fall back to a SQLite-backed text map if a large year (e.g. 2024) OOMs
  python ingest_to_chroma.py --sqlite-cache

Re-running is safe: Chroma upsert replaces by ID, so partial runs resume.
"""

import os
import re
import sys
import json
import glob
import sqlite3
import tempfile
import argparse
from collections import defaultdict, Counter

try:
    import chromadb
except ImportError:
    print("ERROR: chromadb not installed. Run: pip install chromadb", file=sys.stderr)
    sys.exit(1)


# ---- Paths ----
DEFAULT_RESULTS_DIR = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_embeddings/batch_results"
DEFAULT_INPUTS_DIR  = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs_v2"
DEFAULT_CHROMA_DIR  = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/chroma_db"

COLLECTION_NAME = "cherrypicker_chunks"
EMBEDDING_DIM   = 1536    # text-embedding-3-small
DEFAULT_BATCH   = 1000


# ---- custom_id parsing ----
# Format from adjust_to_openai.py:
#   "<year>|<FT|MO>|PMCID:<pmcid>|CH:<chunk_id>"  or "|PMID:..." or "|L:<line>"
#   optionally followed by "|S:<si:04d>" or "|S:<si:04d>.<sj:02d>" for split pieces.
CUSTOM_ID_RE = re.compile(
    r"^(?P<year>\d{4})\|"
    r"(?P<source>FT|MO)\|"
    r"(?:PMCID:(?P<pmcid>PMC\d+)|PMID:(?P<pmid>\d+)|L:(?P<line>\d+))"
    r"(?:\|CH:(?P<chunk_id>[^|]+))?"
    r"(?:\|S:(?P<split>[\d.]+))?$"
)


def parse_custom_id(cid):
    """Return a dict of structured fields parsed from a custom_id."""
    m = CUSTOM_ID_RE.match(cid or "")
    if not m:
        return {"year": None, "source": None, "pmcid": None, "pmid": None,
                "chunk_id": None, "split": None, "raw": cid}
    d = m.groupdict()
    d["raw"] = cid
    return d


class TextMap:
    """
    Key-value store mapping custom_id -> input text for a single year.

    Two backends:
      - in-memory dict (default; fast, fine up to ~1M chunks per year)
      - SQLite on disk (use --sqlite-cache or pass use_sqlite=True; safe at any scale)

    Same .get(cid) interface either way. Used as a context manager so the SQLite
    temp file is cleaned up automatically when the year is done.
    """
    def __init__(self, use_sqlite=False):
        self.use_sqlite = use_sqlite
        self._dict = None
        self._conn = None
        self._tmp_path = None

    def __enter__(self):
        if self.use_sqlite:
            # mkstemp avoids the mktemp race-condition lint warning; we close
            # the FD immediately and let sqlite3 reopen the file by path.
            fd, self._tmp_path = tempfile.mkstemp(suffix=".sqlite",
                                                  prefix="cherrypicker_textmap_")
            os.close(fd)
            self._conn = sqlite3.connect(self._tmp_path)
            self._conn.execute("PRAGMA journal_mode=OFF")
            self._conn.execute("PRAGMA synchronous=OFF")
            self._conn.execute("CREATE TABLE m (cid TEXT PRIMARY KEY, txt TEXT NOT NULL)")
        else:
            self._dict = {}
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                if self._tmp_path and os.path.exists(self._tmp_path):
                    try:
                        os.remove(self._tmp_path)
                    except OSError:
                        pass

    def add(self, cid, text):
        if self.use_sqlite:
            self._conn.execute("INSERT OR REPLACE INTO m (cid, txt) VALUES (?, ?)", (cid, text))
        else:
            self._dict[cid] = text

    def commit(self):
        if self.use_sqlite:
            self._conn.commit()

    def get(self, cid):
        if self.use_sqlite:
            row = self._conn.execute("SELECT txt FROM m WHERE cid = ?", (cid,)).fetchone()
            return row[0] if row else None
        return self._dict.get(cid)

    def __len__(self):
        if self.use_sqlite:
            return self._conn.execute("SELECT COUNT(*) FROM m").fetchone()[0]
        return len(self._dict)


def populate_year_text_map(text_map, inputs_dir, year):
    """
    Scan data/batch_inputs_v2/<year>/batch_*.jsonl and populate `text_map`
    with {custom_id: input_text}. Returns a Counter of QC events:
      - bad_input_json     : lines that didn't parse
      - input_missing_cid  : parsed JSON with no custom_id
      - input_missing_text : parsed JSON with no body.input string

    Each input line looks like:
      {"custom_id": "...", "method": "POST", "url": "/v1/embeddings",
       "body": {"input": "<text>", "model": "text-embedding-3-small"}}
    """
    stats = Counter()
    pattern = os.path.join(inputs_dir, str(year), "batch_*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  ! no input files matched {pattern}")
        return stats

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_input_json"] += 1
                    continue
                cid = obj.get("custom_id")
                if not cid:
                    stats["input_missing_cid"] += 1
                    continue
                body = obj.get("body") or {}
                text = body.get("input")
                if not isinstance(text, str):
                    stats["input_missing_text"] += 1
                    continue
                text_map.add(cid, text)
    text_map.commit()
    return stats


def iter_result_lines(result_path):
    """
    Yield tagged tuples from a batch-results JSONL so the caller can count
    quality-control events instead of silently skipping bad lines.

    Yields one of:
        ("ok",          cid,  vec)    -- usable embedding
        ("bad_result_json", None, None)   -- line failed json.loads
        ("api_error",   cid,  None)   -- OpenAI returned an error for this request
        ("empty_data",  cid,  None)   -- response had no data[] array
        ("missing_vec", cid,  None)   -- data[0] had no usable embedding list
    """
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield ("bad_result_json", None, None)
                continue
            cid = obj.get("custom_id")
            if obj.get("error"):
                yield ("api_error", cid, None)
                continue
            resp = obj.get("response") or {}
            body = resp.get("body") or {}
            data = body.get("data") or []
            if not data:
                yield ("empty_data", cid, None)
                continue
            vec = data[0].get("embedding")
            if not isinstance(vec, list) or not vec:
                yield ("missing_vec", cid, None)
                continue
            yield ("ok", cid, vec)


def group_results_by_year(results_dir):
    """Return {year: [result_file_paths]}."""
    by_year = defaultdict(list)
    pattern = os.path.join(results_dir, "*_results.jsonl")
    for fp in sorted(glob.glob(pattern)):
        name = os.path.basename(fp)
        # Filenames look like: "2025_batch_6a0d..._results.jsonl"
        year = name.split("_", 1)[0]
        if year.isdigit():
            by_year[year].append(fp)
        else:
            by_year["unknown"].append(fp)
    return by_year


def get_or_create_collection(chroma_dir):
    """Open the persistent client + collection. Cosine distance for OpenAI embeddings."""
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine",
                  "description": "CherryPicker biomedical corpus chunks, embedded with text-embedding-3-small (1536d)"},
    )
    return client, coll


def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable (a list/generator)."""
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _build_metadata(parts, text_len):
    """
    Build a Chroma metadata dict, dropping keys whose value is missing/empty.
    Empty strings in Chroma metadata can interfere with $eq filters in some
    versions, so we omit absent fields entirely.
    """
    raw = {
        "year":     int(parts["year"]) if parts["year"] else None,
        "source":   parts["source"],
        "pmcid":    parts["pmcid"],
        "pmid":     parts["pmid"],
        "chunk_id": parts["chunk_id"],
        "split":    parts["split"],
        "n_chars":  text_len,
    }
    return {k: v for k, v in raw.items() if v not in (None, "", 0) or k == "n_chars"}


def _validated_records(result_files, text_map):
    """
    Generator yielding tagged records that the caller routes by tag.

    Tags:
      "ok"               -> upsert to Chroma
      "bad_result_json"  -> a result line failed json parsing
      "api_error"        -> OpenAI marked this request as errored
      "empty_data"       -> response carried no data[]
      "missing_vec"      -> data[0] had no usable embedding list
      "missing_text"     -> no text found for this custom_id (join failed)
      "bad_embedding"    -> embedding wrong length
      "bad_custom_id"    -> custom_id couldn't be parsed (refuse to ingest;
                            otherwise the row would land in Chroma without
                            year/source metadata and break downstream filters)
    """
    for rf in result_files:
        for tag, cid, vec in iter_result_lines(rf):
            if tag != "ok":
                # Already-failed at the result-line level; just forward.
                yield (tag, cid, None, None, None)
                continue
            text = text_map.get(cid)
            if text is None:
                yield ("missing_text", cid, None, None, None)
                continue
            if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
                yield ("bad_embedding", cid, None, None, None)
                continue
            parts = parse_custom_id(cid)
            if parts["year"] is None:
                # Unparseable custom_id -- refuse to ingest. We prefer a hard
                # drop over silently inserting a row with no filterable metadata.
                yield ("bad_custom_id", cid, None, None, None)
                continue
            md = _build_metadata(parts, len(text))
            yield ("ok", cid, vec, text, md)


def ingest_year(coll, year, result_files, inputs_dir, batch_size,
                dry_run=False, use_sqlite=False):
    """
    Ingest all result files for a single year. Returns a Counter of QC stats:
      ingested, missing_text, bad_embedding, bad_custom_id,
      bad_result_json, api_error, empty_data, missing_vec,
      bad_input_json, input_missing_cid, input_missing_text
    """
    print(f"\n[year {year}] reading {len(result_files)} result file(s)")
    backend = "sqlite" if use_sqlite else "in-memory"
    print(f"  building input text map from {inputs_dir}/{year}/  ({backend})")

    stats = Counter()

    with TextMap(use_sqlite=use_sqlite) as text_map:
        input_stats = populate_year_text_map(text_map, inputs_dir, year)
        stats.update(input_stats)
        print(f"  loaded {len(text_map):,} input chunks for this year")

        for batch in chunked(_validated_records(result_files, text_map), batch_size):
            ids, embs, docs, mds = [], [], [], []
            for tag, cid, vec, text, md in batch:
                if tag == "ok":
                    ids.append(cid)
                    embs.append(vec)
                    docs.append(text)
                    mds.append(md)
                else:
                    stats[tag] += 1

            if not ids:
                continue
            if not dry_run:
                coll.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=mds)
            stats["ingested"] += len(ids)
            print(f"    upserted {len(ids):,} -> running total this year: {stats['ingested']:,}")

    # Per-year QC line
    parts = [f"ingested={stats['ingested']:,}"]
    for k in ("missing_text", "bad_embedding", "bad_custom_id",
              "bad_result_json", "api_error", "empty_data", "missing_vec",
              "bad_input_json", "input_missing_cid", "input_missing_text"):
        if stats.get(k):
            parts.append(f"{k}={stats[k]:,}")
    print(f"  year {year} done: " + "  ".join(parts))
    return stats


def main():
    ap = argparse.ArgumentParser(description="Ingest OpenAI batch embedding results into ChromaDB.")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                    help="Where the {year}_{batch_id}_results.jsonl files live")
    ap.add_argument("--inputs-dir", default=DEFAULT_INPUTS_DIR,
                    help="Where the original batch input JSONL files live (for joining text)")
    ap.add_argument("--chroma-dir", default=DEFAULT_CHROMA_DIR,
                    help="Where the persistent Chroma DB lives")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                    help="How many vectors to upsert per Chroma call")
    ap.add_argument("--years", default="",
                    help="Comma-separated list of years to ingest; default = all years found")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show counts per year, do not actually upsert anything")
    ap.add_argument("--sqlite-cache", action="store_true",
                    help="Use a temporary SQLite DB for the per-year text map "
                         "instead of an in-memory dict. Recommended if you hit "
                         "OOM on large years (e.g. 2024/2025).")
    args = ap.parse_args()

    by_year = group_results_by_year(args.results_dir)
    if not by_year:
        raise SystemExit(f"No result files found in {args.results_dir}")

    if args.years.strip():
        wanted = {y.strip() for y in args.years.split(",") if y.strip()}
        by_year = {y: fs for y, fs in by_year.items() if y in wanted}
        if not by_year:
            raise SystemExit(f"None of the requested years {wanted} have result files.")

    print(f"Years to process: {sorted(by_year)}")
    print(f"Total result files: {sum(len(v) for v in by_year.values())}")

    if args.dry_run:
        coll = None
    else:
        _client, coll = get_or_create_collection(args.chroma_dir)
        print(f"Chroma collection '{COLLECTION_NAME}' open at {args.chroma_dir}")
        print(f"  pre-ingest count: {coll.count():,}")

    grand = Counter()
    for year in sorted(by_year):
        year_stats = ingest_year(coll, year, by_year[year],
                                 args.inputs_dir, args.batch_size,
                                 dry_run=args.dry_run,
                                 use_sqlite=args.sqlite_cache)
        grand.update(year_stats)

    # ---- final QC report ----
    print("\n=== Ingest complete ===")
    print(f"  vectors ingested:            {grand['ingested']:,}")
    print()
    print(f"  --- result-side QC ---")
    print(f"  missing_text (join failed):  {grand['missing_text']:,}")
    print(f"  bad_embedding (wrong dim):   {grand['bad_embedding']:,}")
    print(f"  bad_custom_id (refused):     {grand['bad_custom_id']:,}")
    print(f"  bad_result_json:             {grand['bad_result_json']:,}")
    print(f"  api_error (OpenAI errored):  {grand['api_error']:,}")
    print(f"  empty_data:                  {grand['empty_data']:,}")
    print(f"  missing_vec:                 {grand['missing_vec']:,}")
    print()
    print(f"  --- input-side QC ---")
    print(f"  bad_input_json:              {grand['bad_input_json']:,}")
    print(f"  input_missing_cid:           {grand['input_missing_cid']:,}")
    print(f"  input_missing_text:          {grand['input_missing_text']:,}")
    if not args.dry_run:
        print()
        print(f"  collection count now:        {coll.count():,}")
        print(f"  chroma dir:                  {args.chroma_dir}")


if __name__ == "__main__":
    main()