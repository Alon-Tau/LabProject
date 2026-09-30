"""
kegg.py
-------
Fetch KEGG compounds, pathways, modules, and drugs via the free REST API.

API docs: https://www.kegg.jp/kegg/rest/keggapi.html

Each /list/<db> endpoint returns one tab-separated line per entity:
    cpd:C00022    Pyruvate; Pyruvic acid; 2-Oxopropanoate; ...
    path:map00010 Glycolysis / Gluconeogenesis
    md:M00001    Glycolysis (Embden-Meyerhof pathway)
    dr:D00005    Methylprednisolone (USP) ...

Names are separated by '; '. The first one is canonical.

KEGG asks for ~3 req/sec max; the /list/ endpoint returns the entire database
in one call so we're well within limits.
"""

from __future__ import annotations

import sys
import time
import urllib.request
from typing import Iterator

from .master import Entity, clean_aliases


KEGG_BASE = "https://rest.kegg.jp/list/"

# KEGG database -> (our category, ID prefix to use in canonical_id)
KEGG_DBS = {
    "compound": ("compound", "KEGG:CPD"),
    "pathway":  ("pathway",  "KEGG:PATH"),
    "module":   ("module",   "KEGG:MOD"),
    "drug":     ("drug",     "KEGG:DR"),
    "enzyme":   ("enzyme",   "KEGG:EC"),
}


def _fetch(db: str, timeout: int = 60) -> str:
    """GET https://rest.kegg.jp/list/<db>  and return the text body."""
    url = f"{KEGG_BASE}{db}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "cherrypicker-vocab-builder/1.0"})
    print(f"[kegg] GET {url}", file=sys.stderr)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _parse_list_response(body: str, category: str, id_prefix: str,
                         source_label: str) -> Iterator[Entity]:
    """Parse a /list/<db> tab-separated body into Entity records."""
    for raw in body.splitlines():
        line = raw.strip()
        if not line or "\t" not in line:
            continue
        kid, names_blob = line.split("\t", 1)
        # kid looks like "cpd:C00022" or "C00022" depending on KEGG version
        short_id = kid.split(":", 1)[-1].strip()
        if not short_id:
            continue

        # Names are '; '-separated. The first is canonical.
        names = [n.strip() for n in names_blob.split(";") if n.strip()]
        if not names:
            continue
        canonical_name = names[0]

        aliases = clean_aliases(names)
        if not aliases:
            continue

        yield Entity(
            canonical_id   = f"{id_prefix}:{short_id}",
            category       = category,
            canonical_name = canonical_name,
            aliases        = aliases,
            source         = source_label,
        )


def parse(dbs: list[str] = None, sleep_between: float = 0.5) -> Iterator[Entity]:
    """
    Stream Entity records from KEGG.

    dbs : list of KEGG database names to fetch; default = all in KEGG_DBS.
    sleep_between : politeness sleep (seconds) between database fetches.
    """
    if dbs is None:
        dbs = list(KEGG_DBS.keys())
    for db in dbs:
        if db not in KEGG_DBS:
            print(f"[kegg] WARNING: unknown db '{db}' skipped", file=sys.stderr)
            continue
        category, id_prefix = KEGG_DBS[db]
        try:
            body = _fetch(db)
        except Exception as e:
            print(f"[kegg] ERROR fetching {db}: {e}", file=sys.stderr)
            continue
        n = 0
        for ent in _parse_list_response(body, category, id_prefix,
                                        source_label=f"kegg_{db}"):
            n += 1
            yield ent
        print(f"[kegg]   {db}: emitted {n:,} entities", file=sys.stderr)
        time.sleep(sleep_between)
