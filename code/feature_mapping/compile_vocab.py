#!/usr/bin/env python3
"""
compile_vocab.py
----------------
Orchestrator that builds the master biomedical vocabulary by calling each
KB parser and merging the results into a single JSONL.

Output: data/vocab/master_vocab.jsonl
        One JSON object per line:
        {
          "canonical_id":   "NCBI:txid357276",
          "category":       "bacteria",
          "canonical_name": "Bacteroides dorei",
          "aliases":        ["bacteroides dorei", "b. dorei", ...],
          "source":         "ncbi_taxonomy"
        }

Usage:
    # All sources (default)
    python compile_vocab.py \
        --ncbi-dir   /path/to/taxdump \
        --hmdb-xml   /path/to/hmdb_metabolites.xml \
        --out        data/vocab/master_vocab.jsonl

    # Subset
    python compile_vocab.py --sources ncbi,kegg \
        --ncbi-dir /path/to/taxdump \
        --out data/vocab/master_vocab.jsonl

    # KEGG-only (no downloads needed; uses REST API)
    python compile_vocab.py --sources kegg \
        --out data/vocab/master_vocab.jsonl

Downloads you need to obtain manually first:
  - NCBI Taxonomy:  https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz
                    (extract: tar -xzf taxdump.tar.gz -C taxdump_unpacked/)
  - HMDB:           https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip
                    (extract: unzip hmdb_metabolites.zip)
  - KEGG:           fetched automatically via REST API (no download)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
# Make imports work when this script is executed from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vocab import write_jsonl
from vocab import ncbi_taxonomy, kegg, hmdb

DEFAULT_OUT = "data/vocab/master_vocab.jsonl"
ALL_SOURCES = ["ncbi", "kegg", "hmdb"]


def iter_all_sources(args):
    """Stream Entity records from every enabled source in turn."""
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in sources:
        if s == "ncbi":
            if not args.ncbi_dir:
                print(f"[compile] SKIP ncbi: --ncbi-dir not provided", file=sys.stderr)
                continue
            yield from ncbi_taxonomy.parse(args.ncbi_dir,
                                           keep_categories=args.keep_categories)
        elif s == "kegg":
            yield from kegg.parse(dbs=args.kegg_dbs)
        elif s == "hmdb":
            if not args.hmdb_xml:
                print(f"[compile] SKIP hmdb: --hmdb-xml not provided", file=sys.stderr)
                continue
            yield from hmdb.parse(args.hmdb_xml)
        else:
            print(f"[compile] WARNING: unknown source '{s}' skipped", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Compile a unified biomedical vocabulary from canonical KBs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output JSONL path (default: {DEFAULT_OUT})")
    ap.add_argument("--sources", default=",".join(ALL_SOURCES),
                    help=f"Comma-separated subset of {ALL_SOURCES}. Default: all.")

    # NCBI
    ap.add_argument("--ncbi-dir", default=None,
                    help="Path to extracted NCBI taxdump (contains names.dmp, nodes.dmp).")
    ap.add_argument("--keep-categories", nargs="*", default=None,
                    help="Restrict NCBI output to these categories (e.g. bacteria fungi). "
                         "Default: all biological divisions.")

    # KEGG
    ap.add_argument("--kegg-dbs", nargs="*", default=None,
                    help="KEGG databases to fetch (default: all in kegg.KEGG_DBS).")

    # HMDB
    ap.add_argument("--hmdb-xml", default=None,
                    help="Path to extracted hmdb_metabolites.xml")

    args = ap.parse_args()
    if args.keep_categories:
        args.keep_categories = set(args.keep_categories)

    final_out_path = Path(args.out)
    final_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ATOMIC WRITE: Build into .tmp first
    tmp_path = final_out_path.with_name(final_out_path.name + ".tmp")
    
    # Clean up stale temp file if a previous run crashed
    if tmp_path.exists():
        print(f"[compile] removing stale temp file: {tmp_path}", file=sys.stderr)
        tmp_path.unlink()

    print(f"[compile] writing to {tmp_path}", file=sys.stderr)
    n = write_jsonl(iter_all_sources(args), tmp_path)
    
    # ATOMIC REPLACE: Move the temp file to the final destination upon success
    tmp_path.replace(final_out_path)
    print(f"[compile] DONE. wrote {n:,} entities to {final_out_path}", file=sys.stderr)

    # Quick category breakdown (now safely reading from the final file)
    from collections import Counter
    cats = Counter()
    with final_out_path.open("r", encoding="utf-8") as f:
        import json
        for line in f:
            try:
                cats[json.loads(line)["category"]] += 1
            except Exception:
                pass
    print(f"[compile] category breakdown:", file=sys.stderr)
    for cat, c in cats.most_common():
        print(f"           {cat:15s} {c:>10,}", file=sys.stderr)


if __name__ == "__main__":
    main()