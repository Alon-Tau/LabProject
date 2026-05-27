"""
ncbi_taxonomy.py
----------------
Parse the NCBI Taxonomy dump into Entity records.

Source files (from taxdump.tar.gz, https://ftp.ncbi.nih.gov/pub/taxonomy/):
    names.dmp     - one row per (tax_id, name_text, unique_name, name_class)
    nodes.dmp     - one row per (tax_id, parent_tax_id, rank, division_id, ...)
    division.dmp  - division_id -> division_name (e.g. "Bacteria", "Mammals")

We:
    1. Read division.dmp to get id -> division mapping.
    2. Read nodes.dmp to get tax_id -> (rank, division_id).
    3. Read names.dmp to gather all name variants per tax_id.
    4. Filter to entries at species/strain/subspecies/no-rank levels with a
       biologically interesting division (bacteria/archaea/fungi/virus/etc.).
    5. Emit one Entity per tax_id, with canonical name = "scientific name"
       and all other name variants as aliases.

Run via compile_vocab.py --sources ncbi
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Tuple

from .master import Entity, clean_aliases


# Ranks we keep. Higher ranks (genus, family, ...) tend to be referenced often
# but cause too many false positives when we filter by per-paper feature
# overlap. Adjust if you want broader/narrower coverage.
RANKS_TO_KEEP = {
    "species", "subspecies", "strain", "varietas", "forma",
    "species group", "species subgroup", "serogroup", "serotype",
    "no rank",   # NCBI uses "no rank" for many taxa; keep so we don't lose strains
}

# Division ID -> our category bucket
DIVISION_TO_CATEGORY = {
    "Bacteria":               "bacteria",
    "Archaea":                "archaea",      # Not in default divisions; rare anyway
    "Viruses":                "virus",
    "Phages":                 "virus",
    "Plants and Fungi":       "fungi",        # We refine via lineage if possible
    "Invertebrates":          "eukaryote",
    "Mammals":                "eukaryote",
    "Primates":               "eukaryote",
    "Rodents":                "eukaryote",
    "Vertebrates":            "eukaryote",
    "Environmental samples":  "bacteria",     # mostly metagenomic bacteria
    "Synthetic and Chimeric": None,           # skip
    "Unassigned":             None,
}

# Name classes from names.dmp that are useful as aliases.
USEFUL_NAME_CLASSES = {
    "scientific name", "synonym", "equivalent name", "genbank synonym",
    "genbank common name", "common name", "blast name", "anamorph",
    "teleomorph", "includes", "in-part", "acronym", "genbank acronym",
    "genbank anamorph", "type material",
}


def _parse_dmp(path: Path, n_cols: int) -> Iterator[Tuple[str, ...]]:
    """NCBI's .dmp files are tab|tab-separated with row terminator '|\\n'."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\t|")
            parts = [p.strip() for p in line.split("\t|\t")]
            if len(parts) < n_cols:
                continue
            yield tuple(parts[:n_cols])


def _read_divisions(taxdump_dir: Path) -> Dict[str, str]:
    """division_id -> division name (e.g. '0' -> 'Bacteria')."""
    div = {}
    p = taxdump_dir / "division.dmp"
    if not p.exists():
        return div
    for division_id, _div_code, name, _comments in _parse_dmp(p, 4):
        div[division_id] = name
    return div


def _read_nodes(taxdump_dir: Path) -> Dict[str, Tuple[str, str]]:
    """tax_id -> (rank, division_id)."""
    out = {}
    p = taxdump_dir / "nodes.dmp"
    if not p.exists():
        raise FileNotFoundError(f"nodes.dmp not found in {taxdump_dir}")
    # nodes.dmp columns: tax_id | parent | rank | embl | division_id | ...
    for row in _parse_dmp(p, 13):
        tax_id, _parent, rank, _embl, division_id = row[:5]
        out[tax_id] = (rank, division_id)
    return out


def _read_names(taxdump_dir: Path) -> Dict[str, Dict[str, list]]:
    """tax_id -> {name_class: [name, ...]}."""
    out: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    p = taxdump_dir / "names.dmp"
    if not p.exists():
        raise FileNotFoundError(f"names.dmp not found in {taxdump_dir}")
    # names.dmp columns: tax_id | name_text | unique_name | name_class
    for row in _parse_dmp(p, 4):
        tax_id, name_text, _unique, name_class = row
        if name_class in USEFUL_NAME_CLASSES:
            out[tax_id][name_class].append(name_text)
    return out


def parse(taxdump_dir: str | Path,
          keep_categories: set = None) -> Iterator[Entity]:
    """
    Stream Entity records from an unpacked NCBI taxdump directory.

    `taxdump_dir` should contain at minimum names.dmp and nodes.dmp.
    Optionally division.dmp for better category labeling.

    `keep_categories` lets you restrict output (e.g. {"bacteria", "fungi"});
    pass None to emit everything.
    """
    taxdump_dir = Path(taxdump_dir)
    if not taxdump_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {taxdump_dir}")

    print(f"[ncbi] reading division.dmp ...", file=sys.stderr)
    divisions = _read_divisions(taxdump_dir)
    print(f"[ncbi]   {len(divisions)} divisions", file=sys.stderr)

    print(f"[ncbi] reading nodes.dmp ...", file=sys.stderr)
    nodes = _read_nodes(taxdump_dir)
    print(f"[ncbi]   {len(nodes):,} nodes", file=sys.stderr)

    print(f"[ncbi] reading names.dmp ...", file=sys.stderr)
    names = _read_names(taxdump_dir)
    print(f"[ncbi]   {len(names):,} tax_ids have at least one usable name", file=sys.stderr)

    emitted = 0
    skipped_rank = 0
    skipped_div = 0
    skipped_no_sci = 0

    print(f"[ncbi] emitting entities ...", file=sys.stderr)
    for tax_id, name_groups in names.items():
        rank, division_id = nodes.get(tax_id, ("no rank", ""))

        if rank not in RANKS_TO_KEEP:
            skipped_rank += 1
            continue

        division = divisions.get(division_id, "Unassigned")
        category = DIVISION_TO_CATEGORY.get(division)
        if category is None:
            skipped_div += 1
            continue
        if keep_categories and category not in keep_categories:
            continue

        # Canonical name = scientific name; otherwise first available
        sci_names = name_groups.get("scientific name", [])
        if not sci_names:
            skipped_no_sci += 1
            continue
        canonical_name = sci_names[0]

        # All aliases (including the scientific name; clean_aliases will dedupe lowercased)
        raw_aliases = []
        for cls, lst in name_groups.items():
            raw_aliases.extend(lst)
        # Also generate a common abbreviation for binomials: "Bacteroides dorei" -> "B. dorei"
        if rank == "species" and " " in canonical_name:
            genus, *rest = canonical_name.split(" ", 1)
            if len(genus) > 1 and rest:
                raw_aliases.append(f"{genus[0]}. {rest[0]}")

        aliases = clean_aliases(raw_aliases)
        if not aliases:
            continue

        emitted += 1
        yield Entity(
            canonical_id   = f"NCBI:txid{tax_id}",
            category       = category,
            canonical_name = canonical_name,
            aliases        = aliases,
            source         = "ncbi_taxonomy",
        )

    print(f"[ncbi] done. emitted={emitted:,} "
          f"skipped_rank={skipped_rank:,} "
          f"skipped_division={skipped_div:,} "
          f"skipped_no_sci_name={skipped_no_sci:,}", file=sys.stderr)
