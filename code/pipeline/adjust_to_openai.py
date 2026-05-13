#!/usr/bin/env python3
"""
adjust_to_openai.py

Build OpenAI Batch input JSONL shards for embeddings from your per-year chunk files.

For each year in [--year-min .. --year-max]:
  - reads BOTH:
      <YEAR>_chunks_combined_650w.jsonl
      <YEAR>_chunks_combined_650w_METAONLY.jsonl
    from:  <chunks-root>/<YEAR>/
  - embeds ONLY the `text` field (metadata is used only for custom_id)
  - writes Batch-ready request JSONL shards to:
      <out-root>/<YEAR>/batch_00001.jsonl, batch_00002.jsonl, ...

Important:
  - Any chunk whose text exceeds the embedding token limit (8192) is SPLIT
    into multiple subchunks (default 7600 tokens with 200 token overlap),
    so NOTHING is truncated or skipped due to length.
  - Special tokens like "<|endoftext|>" are treated safely (won't crash tiktoken).

Run:
  python adjust_to_openai.py
  python adjust_to_openai.py --year-min 2025 --year-max 2025
"""

import json
import argparse
from pathlib import Path
import os

import tiktoken  # pip install tiktoken

# -----------------------------
# Defaults / limits
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"

# OpenAI embeddings input limit (tokens per input). Keep hard limit here.
MAX_INPUT_TOKENS = 8192

# We split oversize texts into subchunks with headroom below MAX_INPUT_TOKENS
SUBCHUNK_TOKENS_DEFAULT = 7600
SUBCHUNK_OVERLAP_DEFAULT = 200

# Batch input sharding (file size / request count)
MAX_BATCH_FILE_BYTES_DEFAULT = 90 * 1024 * 1024   # ~90MB headroom
MAX_REQUESTS_PER_BATCH_FILE_DEFAULT = 50_000

ENCODING_NAME = "cl100k_base"


# -----------------------------
# Helpers
# -----------------------------
def safe_text(text: str) -> str:
    """Clean known special tokens and normalize whitespace lightly."""
    if not text:
        return ""
    # remove known special token that crashed your run
    text = text.replace("<|endoftext|>", " ")
    return text.strip()


def infer_source_tag(path: Path) -> str:
    """FT for fulltext-ish file, MO for METAONLY file."""
    name = path.name.upper()
    if "METAONLY" in name:
        return "MO"
    return "FT"


def build_custom_id(obj: dict, year: int, source_tag: str, fallback_idx: int) -> str:
    """
    Build stable ID for mapping embeddings back.
    Includes YEAR + source tag + PMCID/PMID + chunk_id (if present).
    """
    chunk_id = str(obj.get("chunk_id") or obj.get("id") or "").strip()
    pmcid = str(obj.get("pmcid") or "").strip()
    pmid = str(obj.get("pmid") or "").strip()

    parts = [str(year), source_tag]
    if pmcid:
        parts.append(f"PMCID:{pmcid}")
    elif pmid:
        parts.append(f"PMID:{pmid}")

    if chunk_id:
        parts.append(f"CH:{chunk_id}")
    else:
        parts.append(f"L:{fallback_idx}")

    cid = "|".join(parts)
    return cid[:200]


def encode_tokens(enc, text: str) -> list:
    """Encode with special tokens allowed as normal text (no crashes)."""
    return enc.encode(text or "", disallowed_special=())


def count_tokens(enc, text: str) -> int:
    return len(encode_tokens(enc, text))


def split_by_tokens(enc, text: str, max_tokens: int, overlap: int) -> list[str]:
    """
    Split text into subtexts each <= max_tokens tokens, with overlap tokens.
    Guarantees each piece <= max_tokens (which should be <= 8192).
    """
    toks = encode_tokens(enc, text)
    if len(toks) <= max_tokens:
        return [text]

    step = max_tokens - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than max_tokens")

    pieces = []
    start = 0
    while start < len(toks):
        piece_toks = toks[start:start + max_tokens]
        if not piece_toks:
            break
        pieces.append(enc.decode(piece_toks))
        if start + max_tokens >= len(toks):
            break
        start += step

    return pieces


# -----------------------------
# Core processing
# -----------------------------
def process_year(
    year: int,
    chunks_root: Path,
    out_root: Path,
    text_field: str,
    enc,
    subchunk_tokens: int,
    subchunk_overlap: int,
    max_batch_file_bytes: int,
    max_requests_per_batch_file: int,
) -> None:
    year_dir = chunks_root / str(year)
    ft_file = year_dir / f"{year}_chunks_combined_650w.jsonl"
    mo_file = year_dir / f"{year}_chunks_combined_650w_METAONLY.jsonl"

    if not ft_file.exists() or not mo_file.exists():
        print(f"[{year}] Skipping (missing files). Expected:\n  {ft_file}\n  {mo_file}")
        return

    out_dir = out_root / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 1
    req_in_shard = 0
    bytes_in_shard = 0
    out_f = None

    def open_new_shard():
        nonlocal shard_idx, req_in_shard, bytes_in_shard, out_f
        if out_f:
            out_f.close()
        out_path = out_dir / f"batch_{shard_idx:05d}.jsonl"
        out_f = out_path.open("w", encoding="utf-8")
        shard_idx += 1
        req_in_shard = 0
        bytes_in_shard = 0

    open_new_shard()

    chunks_read = 0
    requests_written = 0
    empty_skipped = 0
    oversize_chunks = 0
    split_subrequests = 0

    for path in (ft_file, mo_file):
        source_tag = infer_source_tag(path)

        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                chunks_read += 1

                text = safe_text(obj.get(text_field, ""))
                if not text:
                    empty_skipped += 1
                    continue

                # If within limit, one piece; otherwise split into multiple pieces
                tok_len = count_tokens(enc, text)
                if tok_len > MAX_INPUT_TOKENS:
                    oversize_chunks += 1
                    pieces = split_by_tokens(enc, text, max_tokens=subchunk_tokens, overlap=subchunk_overlap)
                else:
                    pieces = [text]

                base_id = build_custom_id(obj, year, source_tag, fallback_idx=line_idx)

                for si, piece in enumerate(pieces, 1):
                    # Safety: ensure each piece is within MAX_INPUT_TOKENS
                    if count_tokens(enc, piece) > MAX_INPUT_TOKENS:
                        # Extremely rare; if it happens, hard-split again with tighter window
                        tighter = min(subchunk_tokens, 7000)
                        pieces2 = split_by_tokens(enc, piece, max_tokens=tighter, overlap=min(subchunk_overlap, 150))
                        # Replace current single piece with these pieces2 by writing them now
                        for sj, p2 in enumerate(pieces2, 1):
                            cid = f"{base_id}|S:{si:04d}.{sj:02d}"
                            _write_request_line(
                                out_f_ref=lambda: out_f,
                                open_new_shard=open_new_shard,
                                req_state=lambda: (req_in_shard, bytes_in_shard),
                                set_req_state=lambda r, b: _set_req_state(locals(), r, b),
                                model=EMBED_MODEL,
                                custom_id=cid,
                                input_text=p2,
                                max_file_bytes=max_batch_file_bytes,
                                max_reqs=max_requests_per_batch_file,
                            )
                            requests_written += 1
                            split_subrequests += 1
                        continue

                    cid = base_id if len(pieces) == 1 else f"{base_id}|S:{si:04d}"
                    if len(pieces) > 1:
                        split_subrequests += 1

                    # Write request line with sharding
                    req_line = json.dumps(
                        {
                            "custom_id": cid,
                            "method": "POST",
                            "url": "/v1/embeddings",
                            "body": {"model": EMBED_MODEL, "input": piece},
                        },
                        ensure_ascii=False,
                    ) + "\n"

                    req_bytes = len(req_line.encode("utf-8"))

                    # rotate shard if needed
                    if (req_in_shard + 1 > max_requests_per_batch_file) or (bytes_in_shard + req_bytes > max_batch_file_bytes):
                        open_new_shard()
                        req_in_shard = 0
                        bytes_in_shard = 0

                    out_f.write(req_line)
                    req_in_shard += 1
                    bytes_in_shard += req_bytes
                    requests_written += 1

    if out_f:
        out_f.close()

    print(
        f"[{year}] done | chunks read: {chunks_read:,} | requests written: {requests_written:,} "
        f"| oversize chunks split: {oversize_chunks:,} | split subrequests: {split_subrequests:,} "
        f"| empty skipped: {empty_skipped:,} | shards: {shard_idx-1}"
    )


def _set_req_state(locals_dict, r, b):
    # no-op helper to satisfy type/structure; kept for safety in the ultra-rare tighter split branch
    return


def _write_request_line(
    out_f_ref,
    open_new_shard,
    req_state,
    set_req_state,
    model: str,
    custom_id: str,
    input_text: str,
    max_file_bytes: int,
    max_reqs: int,
):
    # This helper is only used in an ultra-rare path; kept minimal.
    out_f = out_f_ref()
    req_in_shard, bytes_in_shard = req_state()

    req_line = json.dumps(
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {"model": model, "input": input_text},
        },
        ensure_ascii=False,
    ) + "\n"
    req_bytes = len(req_line.encode("utf-8"))

    if (req_in_shard + 1 > max_reqs) or (bytes_in_shard + req_bytes > max_file_bytes):
        open_new_shard()
        req_in_shard = 0
        bytes_in_shard = 0

    out_f = out_f_ref()
    out_f.write(req_line)
    req_in_shard += 1
    bytes_in_shard += req_bytes
    set_req_state(req_in_shard, bytes_in_shard)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-min", type=int, default=1990)
    ap.add_argument("--year-max", type=int, default=2025)

    ap.add_argument(
        "--chunks-root",
        default="/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_chunks/new_corpus_chunks",
        help="Root folder containing per-year chunk directories",
    )
    ap.add_argument(
        "--out-root",
        default="/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs",
        help="Root folder to write batch input shards",
    )

    ap.add_argument("--text-field", default="text", help="Field in chunk JSONL containing the text to embed")

    ap.add_argument("--subchunk-tokens", type=int, default=SUBCHUNK_TOKENS_DEFAULT,
                    help="Token window size for splitting oversize chunks (<=8192)")
    ap.add_argument("--subchunk-overlap", type=int, default=SUBCHUNK_OVERLAP_DEFAULT,
                    help="Token overlap when splitting oversize chunks")

    ap.add_argument("--max-batch-file-mb", type=int, default=90,
                    help="Target max size per batch input file (MB)")
    ap.add_argument("--max-requests-per-file", type=int, default=MAX_REQUESTS_PER_BATCH_FILE_DEFAULT,
                    help="Max requests per batch input file")

    args = ap.parse_args()

    if args.subchunk_tokens > MAX_INPUT_TOKENS:
        raise SystemExit(f"--subchunk-tokens must be <= {MAX_INPUT_TOKENS}")

    chunks_root = Path(args.chunks_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(ENCODING_NAME)

    max_batch_file_bytes = int(args.max_batch_file_mb * 1024 * 1024)

    for year in range(args.year_min, args.year_max + 1):
        process_year(
            year=year,
            chunks_root=chunks_root,
            out_root=out_root,
            text_field=args.text_field,
            enc=enc,
            subchunk_tokens=args.subchunk_tokens,
            subchunk_overlap=args.subchunk_overlap,
            max_batch_file_bytes=max_batch_file_bytes,
            max_requests_per_batch_file=args.max_requests_per_file,
        )

    print("All years finished.")


if __name__ == "__main__":
    main()
