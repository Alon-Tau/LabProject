#!/usr/bin/env python3
"""
retrieve_embedding_batches.py — the missing companion to submit_embedding_batches.py.

Reads the manifest of submitted batches, queries OpenAI for current status,
downloads output files for completed batches, and parses the embeddings into
a clean per-year embeddings JSONL.

Three modes:
  --status-only    : Only print per-batch status table; no downloads. Fast diagnostic.
  (default)        : Status + download completed batches + parse embeddings to per-year files.
  --consolidate    : Skip the API and just (re)parse already-downloaded output files.

Outputs:
  <out-dir>/batches_status.json           per-batch state snapshot
  <out-dir>/raw_outputs/<batch_id>.jsonl  raw download from OpenAI per batch
  <out-dir>/errors/<batch_id>.jsonl       error file from OpenAI (if any)
  <out-dir>/embeddings/<year>.jsonl       parsed embeddings (one line per chunk piece):
                                            {"custom_id":..., "chunk_id":..., "year":...,
                                             "source":..., "pmcid":..., "embedding": [...]}

Idempotent: re-running will skip already-downloaded batches and only fetch new ones.

Run:
  python retrieve_embedding_batches.py --status-only
  python retrieve_embedding_batches.py
  python retrieve_embedding_batches.py --consolidate
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai SDK not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

DEFAULT_MANIFEST = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs/batches_manifest.jsonl"
DEFAULT_OUT_DIR = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_outputs"


# Custom ID format from adjust_to_openai.py is:
#   "<year>|<FT|MO>|PMCID:<pmcid>|CH:<chunk_id>"  (or "|PMID:..." or "|L:<line>")
# Optionally followed by "|S:<si:04d>" or "|S:<si:04d>.<sj:02d>" for split pieces.
CUSTOM_ID_RE = re.compile(
    r"^(?P<year>\d{4})\|(?P<source>FT|MO)\|(?:PMCID:(?P<pmcid>PMC\d+)|PMID:(?P<pmid>\d+)|L:(?P<line>\d+))(?:\|CH:(?P<chunk_id>[^|]+))?(?:\|S:(?P<split>[\d.]+))?"
)


def parse_custom_id(cid: str) -> Dict[str, Optional[str]]:
    """Best-effort parse of the custom_id back into structured fields."""
    m = CUSTOM_ID_RE.match(cid or "")
    if not m:
        return {"year": None, "source": None, "pmcid": None, "chunk_id": None, "split": None, "raw": cid}
    return {
        "year": m.group("year"),
        "source": m.group("source"),
        "pmcid": m.group("pmcid"),
        "pmid": m.group("pmid"),
        "chunk_id": m.group("chunk_id"),
        "split": m.group("split"),
        "raw": cid,
    }


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def fetch_status(client: OpenAI, batch_id: str) -> Dict[str, Any]:
    """Return a normalized status dict for one batch."""
    b = client.batches.retrieve(batch_id)
    return {
        "batch_id": b.id,
        "status": b.status,
        "endpoint": b.endpoint,
        "completion_window": b.completion_window,
        "created_at": b.created_at,
        "in_progress_at": getattr(b, "in_progress_at", None),
        "finalizing_at": getattr(b, "finalizing_at", None),
        "completed_at": getattr(b, "completed_at", None),
        "failed_at": getattr(b, "failed_at", None),
        "expired_at": getattr(b, "expired_at", None),
        "cancelled_at": getattr(b, "cancelled_at", None),
        "input_file_id": getattr(b, "input_file_id", None),
        "output_file_id": getattr(b, "output_file_id", None),
        "error_file_id": getattr(b, "error_file_id", None),
        "request_counts": {
            "total": getattr(b.request_counts, "total", None),
            "completed": getattr(b.request_counts, "completed", None),
            "failed": getattr(b.request_counts, "failed", None),
        } if getattr(b, "request_counts", None) else None,
        "errors": [e.__dict__ if hasattr(e, "__dict__") else str(e) for e in (b.errors.data if getattr(b, "errors", None) else [])],
        "metadata": dict(getattr(b, "metadata", {}) or {}),
    }


def print_status_table(statuses: List[Dict[str, Any]]):
    rows = []
    for s in statuses:
        rc = s.get("request_counts") or {}
        rows.append((
            s.get("batch_id", "?")[:30],
            s.get("status", "?"),
            rc.get("total", "?"),
            rc.get("completed", "?"),
            rc.get("failed", "?"),
            s.get("metadata", {}).get("year", "?"),
        ))
    print(f"{'batch_id':<32} {'status':<14} {'total':>8} {'done':>8} {'fail':>8} {'year':>6}")
    print("-" * 80)
    for r in rows:
        print(f"{r[0]:<32} {r[1]:<14} {str(r[2]):>8} {str(r[3]):>8} {str(r[4]):>8} {str(r[5]):>6}")
    counts = Counter([s.get("status", "?") for s in statuses])
    print("-" * 80)
    print("Status summary:", dict(counts))


def download_file(client: OpenAI, file_id: str, out_path: Path) -> int:
    """Download a file from OpenAI and write to out_path. Returns bytes written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path.stat().st_size  # already there, skip
    resp = client.files.content(file_id)
    # The SDK returns a streamable object; .read() or .content depending on version
    if hasattr(resp, "read"):
        data = resp.read()
    elif hasattr(resp, "content"):
        data = resp.content
    else:
        data = bytes(resp)
    out_path.write_bytes(data)
    return len(data)


def parse_output_jsonl_to_embeddings(raw_path: Path, by_year_writers: Dict[str, Any], stats: Counter):
    """
    Parse a downloaded OpenAI batch-output JSONL and route embeddings into per-year files.

    Each line looks like:
      {"id":"batch_req_...", "custom_id":"...", "response":{"status_code":200,"body":{"data":[{"embedding":[...]}], ...}}, "error": null}
    """
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_jsonl_lines"] += 1
                continue

            cid = obj.get("custom_id")
            err = obj.get("error")
            resp = obj.get("response") or {}

            if err:
                stats["response_errors"] += 1
                continue

            body = resp.get("body") or {}
            data = body.get("data") or []
            if not data:
                stats["empty_data"] += 1
                continue

            # Embeddings endpoint always returns one input → one item (we send 1 input per request)
            vec = data[0].get("embedding")
            if not vec or not isinstance(vec, list):
                stats["missing_embedding"] += 1
                continue

            parts = parse_custom_id(cid)
            year = parts["year"] or "UNKNOWN"
            out_line = {
                "custom_id": cid,
                "year": parts["year"],
                "source": parts["source"],
                "pmcid": parts["pmcid"],
                "pmid": parts.get("pmid"),
                "chunk_id": parts["chunk_id"],
                "split": parts["split"],
                "embedding": vec,
            }

            w = by_year_writers.get(year)
            if w is None:
                # opened lazily by caller
                stats["unrouted_year"] += 1
                continue

            w.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            stats["embeddings_written"] += 1


def main():
    ap = argparse.ArgumentParser(description="Retrieve, download and parse OpenAI embedding batches.")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir",  default=DEFAULT_OUT_DIR)
    ap.add_argument("--status-only", action="store_true", help="Only print status; no downloads or parsing.")
    ap.add_argument("--consolidate", action="store_true",
                    help="Skip API; just (re)parse already-downloaded raw_outputs/*.jsonl into per-year embeddings files.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_outputs"
    err_dir = out_dir / "errors"
    emb_dir = out_dir / "embeddings"
    raw_dir.mkdir(exist_ok=True)
    err_dir.mkdir(exist_ok=True)
    emb_dir.mkdir(exist_ok=True)

    manifest = load_manifest(Path(args.manifest))
    if not manifest:
        print("Manifest is empty — no batches to check.")
        return
    print(f"Manifest: {args.manifest}  ({len(manifest)} batches)")

    # ---------- consolidate-only path ----------
    if args.consolidate:
        print("Consolidate mode: parsing already-downloaded raw outputs into per-year embeddings...")
        raw_files = sorted(raw_dir.glob("*.jsonl"))
        if not raw_files:
            print(f"No raw outputs found in {raw_dir}")
            return
        by_year_writers = {}
        stats: Counter = Counter()
        try:
            for rf in raw_files:
                # Detect years we'll write for from filename — we don't know yet, lazy open by year while parsing
                # Pre-open all year files lazily inside the parser by passing a defaultdict-like wrapper
                pass
            # Use a simple lazy-open dict
            class LazyWriters:
                def __init__(self, base): self.base = base; self.handles = {}
                def get(self, year):
                    if year is None:
                        year = "UNKNOWN"
                    h = self.handles.get(year)
                    if h is None:
                        h = (self.base / f"{year}.jsonl").open("a", encoding="utf-8")
                        self.handles[year] = h
                    return h
                def close(self):
                    for h in self.handles.values():
                        try: h.close()
                        except Exception: pass

            lw = LazyWriters(emb_dir)
            for rf in raw_files:
                print(f"  parsing {rf.name}")
                parse_output_jsonl_to_embeddings(rf, lw, stats)
            lw.close()
            print("Consolidate stats:", dict(stats))
        finally:
            pass
        return

    # ---------- API status + (optional) downloads ----------
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY env var is not set")
    client = OpenAI(api_key=api_key)

    statuses = []
    for rec in manifest:
        batch_id = rec.get("batch_id")
        if not batch_id:
            continue
        try:
            s = fetch_status(client, batch_id)
        except Exception as e:
            s = {"batch_id": batch_id, "status": f"FETCH_ERROR: {e}"}
        statuses.append(s)

    print()
    print_status_table(statuses)

    (out_dir / "batches_status.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote status snapshot: {out_dir/'batches_status.json'}")

    if args.status_only:
        return

    # ---- Download outputs for completed batches ----
    print("\nDownloading outputs for completed batches...")
    completed = [s for s in statuses if s.get("status") == "completed"]
    print(f"  completed batches: {len(completed)}")

    class LazyWriters:
        def __init__(self, base): self.base = base; self.handles = {}
        def get(self, year):
            if year is None:
                year = "UNKNOWN"
            h = self.handles.get(year)
            if h is None:
                h = (self.base / f"{year}.jsonl").open("a", encoding="utf-8")
                self.handles[year] = h
            return h
        def close(self):
            for h in self.handles.values():
                try: h.close()
                except Exception: pass

    lw = LazyWriters(emb_dir)
    parse_stats: Counter = Counter()
    download_stats: Counter = Counter()

    try:
        for s in completed:
            bid = s["batch_id"]
            ofid = s.get("output_file_id")
            efid = s.get("error_file_id")
            raw_path = raw_dir / f"{bid}.jsonl"
            err_path = err_dir / f"{bid}.jsonl"

            if ofid:
                if raw_path.exists() and raw_path.stat().st_size > 0:
                    download_stats["skipped_existing"] += 1
                else:
                    try:
                        bytes_n = download_file(client, ofid, raw_path)
                        print(f"  downloaded output {bid} ({bytes_n} bytes)")
                        download_stats["downloaded"] += 1
                    except Exception as e:
                        print(f"  ERROR downloading {bid}: {e}")
                        download_stats["download_errors"] += 1
                        continue

            if efid:
                try:
                    download_file(client, efid, err_path)
                    print(f"  saved error file for {bid}")
                    download_stats["error_files_saved"] += 1
                except Exception as e:
                    print(f"  ERROR downloading error-file for {bid}: {e}")

            # Parse into per-year embeddings
            if raw_path.exists():
                parse_output_jsonl_to_embeddings(raw_path, lw, parse_stats)
    finally:
        lw.close()

    print("\nDownload summary:", dict(download_stats))
    print("Parse summary:   ", dict(parse_stats))
    print(f"\nPer-year embeddings written under: {emb_dir}")
    print("Done.")


if __name__ == "__main__":
    main()