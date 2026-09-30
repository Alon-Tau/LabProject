#!/usr/bin/env python3
"""
build_entity_index.py
---------------------
Scan the full-text corpus once for every entity in the master vocabulary
and write the results to a SQLite database.

Uses Aho-Corasick for multi-pattern matching.
Multi-process over CPU cores. Word-boundary post-filtering eliminates false
positives like matching "glucose" inside "deoxyglucose".

IMPORTANT LOGIC:
    This index is binary presence/absence per article-feature pair.

    If an article contains a feature at least once:
        store one row with present = 1

    If an article does not contain a feature:
        store no row

    Therefore:
        row exists = 1
        row missing = 0

We intentionally do NOT store explicit zero rows because that would make the
database huge.

Output: data/entity_index.sqlite

    CREATE TABLE article_entities (
        pmcid          TEXT,
        category       TEXT,
        canonical_id   TEXT,
        canonical_name TEXT,
        present        INTEGER,
        PRIMARY KEY(pmcid, canonical_id)
    );

Usage:
    python build_entity_index.py \
        --vocab    data/vocab/master_vocab.jsonl \
        --corpus   /home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus \
        --out      data/entity_index.sqlite

Optional:
    --processes N        override number of worker processes
    --limit N            scan only first N files, useful for smoke test
    --min-alias-len K    discard aliases shorter than K chars, default 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import ahocorasick
except ImportError:
    print(
        "ERROR: pyahocorasick not installed. Run: pip install pyahocorasick",
        file=sys.stderr,
    )
    raise

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm not installed. Run: pip install tqdm", file=sys.stderr)
    raise


# Workers pick up a module-level automaton built once per process.
_AUTOMATON = None
_VOCAB_PATH = None
_MIN_ALIAS_LEN = 4


# ---------------------------------------------------------------------------
# Automaton construction
# ---------------------------------------------------------------------------

def build_automaton(vocab_path: Path, min_alias_len: int = 4) -> "ahocorasick.Automaton":
    """
    Construct an Aho-Corasick automaton from master_vocab.jsonl.

    Because multiple distinct entities may share an alias, each automaton entry
    maps one alias to a list of:
        (canonical_id, category, canonical_name)
    """
    alias_to_entities: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    n_entities = 0
    n_alias_mentions = 0

    with vocab_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = e["canonical_id"]
            cat = e["category"]
            name = e["canonical_name"]

            n_entities += 1

            for alias in e.get("aliases", []):
                a = (alias or "").strip().lower()
                if len(a) < min_alias_len:
                    continue

                alias_to_entities[a].append((cid, cat, name))
                n_alias_mentions += 1

    print(
        f"[build] vocab: {n_entities:,} entities, "
        f"{n_alias_mentions:,} alias mentions, "
        f"{len(alias_to_entities):,} unique aliases",
        file=sys.stderr,
    )

    A = ahocorasick.Automaton()

    for alias, entities in alias_to_entities.items():
        # Store alias string too so scan_one_file can calculate start index.
        A.add_word(alias, (alias, entities))

    A.make_automaton()

    print("[build] automaton built", file=sys.stderr)
    return A


# ---------------------------------------------------------------------------
# Per-file scan
# ---------------------------------------------------------------------------

def _init_worker(vocab_path: str, min_alias_len: int):
    """
    Each worker builds its own automaton.

    On Linux, multiprocessing with fork can share memory pages copy-on-write,
    but this initializer is still safe and explicit.
    """
    global _AUTOMATON, _VOCAB_PATH, _MIN_ALIAS_LEN

    _VOCAB_PATH = vocab_path
    _MIN_ALIAS_LEN = min_alias_len
    _AUTOMATON = build_automaton(Path(vocab_path), min_alias_len)


def _word_boundary_ok(text_lower: str, start: int, end: int) -> bool:
    """
    Return True if the match is surrounded by non-alphanumeric boundaries.

    This prevents false positives like:
        glucose inside deoxyglucose
        coli inside coliform
    """
    n = len(text_lower)

    if start > 0 and text_lower[start - 1].isalnum():
        return False

    if end + 1 < n and text_lower[end + 1].isalnum():
        return False

    return True


def scan_one_file(file_path: str) -> Tuple[str, Dict[str, Tuple[str, str, int]]] | None:
    """
    Scan one .txt file.

    Returns:
        (pmcid, {canonical_id: (category, canonical_name, present)})

    Binary logic:
        present = 1 if the article contains this entity at least once.
        Missing row means present = 0.

    Repeated mentions of the same entity in the same article do NOT increase
    the value. It stays present = 1.
    """
    global _AUTOMATON

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return None

    text_lower = text.lower()
    pmcid = Path(file_path).stem

    # canonical_id -> (category, canonical_name, present)
    hits: Dict[str, Tuple[str, str, int]] = {}

    for end_index, payload in _AUTOMATON.iter(text_lower):
        alias, entity_list = payload
        start_index = end_index - len(alias) + 1

        if not _word_boundary_ok(text_lower, start_index, end_index):
            continue

        for cid, cat, name in entity_list:
            # Binary presence:
            # First valid match creates the row.
            # Later matches of the same canonical_id are ignored.
            if cid not in hits:
                hits[cid] = (cat, name, 1)

    return pmcid, hits


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    """
    Initialize a fresh SQLite database.

    This writes to a temporary DB path in main(), so dropping the table here is
    safe. The final DB is replaced only after the full build succeeds.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))

    conn.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous  = OFF;

        DROP TABLE IF EXISTS article_entities;

        CREATE TABLE article_entities (
            pmcid          TEXT NOT NULL,
            category       TEXT NOT NULL,
            canonical_id   TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            present        INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(pmcid, canonical_id),
            CHECK(present IN (0, 1))
        );
    """)

    return conn


def finalize_db(conn: sqlite3.Connection):
    """
    Add indexes after bulk load.

    Creating indexes after inserts is faster than maintaining them during the
    scan.
    """
    print("[build] creating indexes...", file=sys.stderr)

    conn.executescript("""
        CREATE INDEX IF NOT EXISTS ix_cid
            ON article_entities(canonical_id);

        CREATE INDEX IF NOT EXISTS ix_pmcid
            ON article_entities(pmcid);

        CREATE INDEX IF NOT EXISTS ix_category
            ON article_entities(category);

        CREATE INDEX IF NOT EXISTS ix_category_cid
            ON article_entities(category, canonical_id);
    """)

    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--vocab",
        default="data/vocab/master_vocab.jsonl",
        help="Path to master_vocab.jsonl.",
    )

    ap.add_argument(
        "--corpus",
        required=True,
        help="Directory containing .txt files, scanned recursively.",
    )

    ap.add_argument(
        "--out",
        default="data/entity_index.sqlite",
        help="Output SQLite DB path.",
    )

    ap.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Worker process count. Default = CPU count.",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Scan only first N files. 0 = no limit. Useful for smoke tests.",
    )

    ap.add_argument(
        "--min-alias-len",
        type=int,
        default=4,
        help="Drop aliases shorter than this. Default 4.",
    )

    args = ap.parse_args()

    vocab_path = Path(args.vocab)
    corpus_dir = Path(args.corpus)
    final_db_path = Path(args.out)

    # Atomic write:
    # Build into .tmp first, then replace final DB only after success.
    tmp_db_path = final_db_path.with_name(final_db_path.name + ".tmp")

    if not vocab_path.exists():
        raise SystemExit(f"vocab not found: {vocab_path}")

    if not corpus_dir.is_dir():
        raise SystemExit(f"corpus directory not found: {corpus_dir}")

    if args.min_alias_len < 1:
        raise SystemExit("--min-alias-len must be >= 1")

    print(f"[build] gathering .txt files from {corpus_dir} ...", file=sys.stderr)

    files = sorted(str(p) for p in corpus_dir.rglob("*.txt"))

    if args.limit > 0:
        files = files[:args.limit]

    print(f"[build] {len(files):,} files to scan", file=sys.stderr)

    if not files:
        raise SystemExit("No .txt files found.")

    n_proc = args.processes or mp.cpu_count()
    if n_proc < 1:
        raise SystemExit("--processes must be >= 1")

    print(f"[build] using {n_proc} worker processes", file=sys.stderr)

    # Remove old temporary DB if a previous run crashed.
    if tmp_db_path.exists():
        print(f"[build] removing stale temp DB: {tmp_db_path}", file=sys.stderr)
        tmp_db_path.unlink()

    conn = init_db(tmp_db_path)
    cur = conn.cursor()

    insert_sql = (
        "INSERT OR REPLACE INTO article_entities "
        "(pmcid, category, canonical_id, canonical_name, present) "
        "VALUES (?, ?, ?, ?, ?)"
    )

    t0 = time.time()

    batch: list = []
    BATCH_SIZE = 5000

    n_rows = 0
    n_articles_processed = 0
    n_articles_with_hits = 0
    n_failed_files = 0

    # Counts article-feature presence rows per category.
    # This is NOT total mention frequency.
    category_counter = Counter()

    with mp.Pool(
        processes=n_proc,
        initializer=_init_worker,
        initargs=(str(vocab_path), args.min_alias_len),
    ) as pool:

        iterator = pool.imap_unordered(scan_one_file, files, chunksize=8)

        for result in tqdm(iterator, total=len(files), desc="Scanning corpus"):
            n_articles_processed += 1

            if result is None:
                n_failed_files += 1
                continue

            pmcid, hits = result

            if not hits:
                continue

            n_articles_with_hits += 1

            for cid, (cat, name, present) in hits.items():
                batch.append((pmcid, cat, cid, name, present))
                category_counter[cat] += 1

                if len(batch) >= BATCH_SIZE:
                    cur.executemany(insert_sql, batch)
                    n_rows += len(batch)
                    batch.clear()

    if batch:
        cur.executemany(insert_sql, batch)
        n_rows += len(batch)
        batch.clear()

    conn.commit()
    finalize_db(conn)
    conn.close()

    # Atomic replace:
    # Only now does the new DB become the official output.
    tmp_db_path.replace(final_db_path)

    elapsed = time.time() - t0

    print(f"\n[build] DONE in {elapsed / 60:.1f} min", file=sys.stderr)
    print(f"[build] articles processed:       {n_articles_processed:,}", file=sys.stderr)
    print(f"[build] failed files:              {n_failed_files:,}", file=sys.stderr)
    print(f"[build] articles with hits:        {n_articles_with_hits:,}", file=sys.stderr)
    print(f"[build] article-feature rows:      {n_rows:,}", file=sys.stderr)
    print(f"[build] rows per category:", file=sys.stderr)

    for cat, c in category_counter.most_common():
        print(f"           {cat:15s} {c:>12,}", file=sys.stderr)

    print(f"[build] DB at: {final_db_path}", file=sys.stderr)


if __name__ == "__main__":
    main()