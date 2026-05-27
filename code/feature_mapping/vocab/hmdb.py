"""
hmdb.py
-------
Parse the HMDB metabolites XML dump into Entity records.

Source file: hmdb_metabolites.xml  (extracted from hmdb_metabolites.zip,
https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip — academic free).

The full XML is ~6 GB uncompressed with ~220k metabolites. We use streaming
parsing (iterparse with clear()) so memory stays bounded.

Each <metabolite> element contains:
    <accession>          HMDB0000243
    <name>               Pyruvic acid          (canonical)
    <iupac_name>         2-oxopropanoic acid
    <traditional_iupac>  pyruvic acid
    <synonyms>
        <synonym>...</synonym> ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

from .master import Entity, clean_aliases


# HMDB uses an XML namespace; strip it during iteration so tag-name comparisons
# are simple ("metabolite" not "{http://www.hmdb.ca}metabolite").
def _localtag(elem) -> str:
    tag = elem.tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _text(elem, child_name: str) -> str | None:
    """Return the text of the first direct child whose local-name is `child_name`."""
    for c in elem:
        if _localtag(c) == child_name:
            return (c.text or "").strip() or None
    return None


def _synonyms(elem) -> list[str]:
    """Pull all <synonym> texts under <synonyms>."""
    out = []
    for c in elem:
        if _localtag(c) != "synonyms":
            continue
        for s in c:
            if _localtag(s) == "synonym":
                t = (s.text or "").strip()
                if t:
                    out.append(t)
    return out


def parse(xml_path: str | Path) -> Iterator[Entity]:
    """
    Stream Entity records from a hmdb_metabolites.xml file.

    Uses iterparse + elem.clear() so memory stays bounded even on the full
    ~6 GB file.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"HMDB file not found: {xml_path}")

    print(f"[hmdb] streaming {xml_path}  (this can take a while)...", file=sys.stderr)
    emitted = 0
    skipped_no_name = 0

    # iterparse yields ('start', elem) and ('end', elem). We only care about 'end'
    # for <metabolite> elements, at which point all children are populated.
    context = ET.iterparse(str(xml_path), events=("end",))

    for _event, elem in context:
        if _localtag(elem) != "metabolite":
            continue

        accession = _text(elem, "accession")
        name      = _text(elem, "name")

        if not accession or not name:
            skipped_no_name += 1
            elem.clear()
            continue

        # Gather all candidate names
        raw_aliases = [name]
        for fld in ("iupac_name", "traditional_iupac", "chemical_formula"):
            t = _text(elem, fld)
            if t:
                raw_aliases.append(t)
        raw_aliases.extend(_synonyms(elem))

        aliases = clean_aliases(raw_aliases)
        if aliases:
            emitted += 1
            yield Entity(
                canonical_id   = f"HMDB:{accession}",
                category       = "metabolite",
                canonical_name = name,
                aliases        = aliases,
                source         = "hmdb",
            )

        # CRITICAL: clear() drops the element from memory after processing.
        # Without this, the iterparse tree grows to multi-GB.
        elem.clear()

        if emitted % 10000 == 0 and emitted > 0:
            print(f"[hmdb]   ... {emitted:,} metabolites processed", file=sys.stderr)

    print(f"[hmdb] done. emitted={emitted:,}  skipped_no_name={skipped_no_name:,}",
          file=sys.stderr)
