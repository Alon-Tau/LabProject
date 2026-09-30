"""
master.py
---------
Common Entity model + utilities used by every KB parser.

Each parser produces a stream of `Entity` records, which compile_vocab.py
serializes to JSONL (one entity per line) for downstream consumption by
build_entity_index.py.
"""

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, List, Set


# --- Categories we care about. Add more as needed. -------------------------
CATEGORIES = {
    "bacteria",
    "archaea",
    "fungi",
    "virus",
    "eukaryote",          # catch-all for non-bacterial/fungal taxa we still track
    "metabolite",
    "compound",
    "pathway",
    "module",
    "enzyme",
    "drug",
    "disease",
    "protein",
    "gene",
}


@dataclass
class Entity:
    canonical_id:   str              # globally unique, e.g. "NCBI:txid357276"
    category:       str              # one of CATEGORIES
    canonical_name: str              # primary display name
    aliases:        List[str]        # all match-strings; lowercased, deduped
    source:         str              # "ncbi_taxonomy" | "kegg_compound" | ...

    def to_dict(self) -> dict:
        return asdict(self)


# --- Alias cleaning ---------------------------------------------------------

# Aliases shorter than this many characters are dropped UNLESS whitelisted
# (e.g. some valid 2-3 char names exist; nothing currently whitelisted).
MIN_ALIAS_LENGTH = 4

# These patterns are too generic to safely scan for; they cause false positives.
GENERIC_BLOCKLIST = {
    "cell", "cells", "human", "humans", "animal", "animals",
    "rat", "mouse", "mice", "patient", "patients", "control", "controls",
    "study", "studies", "method", "methods", "data", "test", "tests",
    "model", "models", "group", "groups", "sample", "samples",
    "level", "levels", "type", "types", "subject", "subjects",
    "disease", "diseases",  # keep "Crohn's disease" etc. via canonical_name only
    "acid",  # alone is meaningless
    "protein", "proteins", "gene", "genes",
    "factor", "factors", "complex", "system",
    "subspecies", "strain", "strains",
    "alpha", "beta", "gamma", "delta",  # alone, ambiguous; allowed in compounds via length
}


def clean_aliases(raw_aliases: Iterable[str], min_length: int = MIN_ALIAS_LENGTH) -> List[str]:
    """
    Normalize + filter a raw alias list.

      - lowercased
      - whitespace collapsed
      - deduplicated (order-preserving)
      - drop aliases shorter than min_length
      - drop generic blocklist words
      - drop pure-numeric strings (e.g. "12345")
      - drop strings containing no letters (e.g. "1.2.3.4")
    """
    seen: Set[str] = set()
    out: List[str] = []
    for raw in raw_aliases:
        if not raw:
            continue
        a = re.sub(r"\s+", " ", str(raw)).strip().lower()
        if not a:
            continue
        if len(a) < min_length:
            continue
        if a in GENERIC_BLOCKLIST:
            continue
        if not any(c.isalpha() for c in a):
            continue
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out


# --- JSONL I/O -------------------------------------------------------------

def write_jsonl(entities: Iterable[Entity], out_path: Path) -> int:
    """Write an iterable of Entity objects to a JSONL file. Returns count written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for e in entities:
            if e.category not in CATEGORIES:
                # Skip entities with unknown categories rather than fail.
                continue
            if not e.aliases:
                continue
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(in_path: Path) -> Iterable[Entity]:
    """Stream Entity objects back from a JSONL file."""
    in_path = Path(in_path)
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield Entity(
                canonical_id   = d["canonical_id"],
                category       = d["category"],
                canonical_name = d["canonical_name"],
                aliases        = list(d.get("aliases", [])),
                source         = d.get("source", "unknown"),
            )
