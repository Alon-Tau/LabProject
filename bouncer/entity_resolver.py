"""
entity_resolver.py
------------------
Convert user-supplied feature names (e.g. "B. dorei", "pyruvate", "Escherichia
coli") into canonical IDs (e.g. "NCBI:txid357276", "HMDB:HMDB0000243") using
the master vocabulary built by compile_vocab.py.

Why this exists:
    The SQL layer (feature_filter.py) works on canonical IDs. Users think in
    common names. This module bridges the two.

Lookup is:
    1. Case-insensitive
    2. Whitespace-collapsed
    3. By aliasing only (no fuzzy / typo matching in v1)

Multiple entities may share an alias ("PA" could be Pantothenic acid OR
Pyruvic acid). resolve() returns the FIRST match; resolve_all() returns
every match. Use resolve_all() and disambiguate when ambiguity matters.

Caching:
    The vocab JSONL is loaded once and the alias -> entities map cached at
    module level. First call pays the load cost (~5-15 sec for ~10M aliases);
    every later call is a dict lookup.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


DEFAULT_VOCAB_PATH = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/vocab/master_vocab.jsonl"


@dataclass(frozen=True)
class ResolvedEntity:
    canonical_id:   str
    category:       str
    canonical_name: str
    matched_alias:  str          # the alias string that the user term matched
    source:         str = ""


# (alias_lower -> [ResolvedEntity, ...])
_alias_to_entities: Optional[Dict[str, List[ResolvedEntity]]] = None
_loaded_from: Optional[str] = None


def _normalize(term: str) -> str:
    """Lowercase and collapse whitespace. The match key."""
    return re.sub(r"\s+", " ", (term or "").strip().lower())


def _load_vocab(vocab_path: str) -> Dict[str, List[ResolvedEntity]]:
    """Stream the JSONL and build the alias -> entities map."""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"vocab JSONL not found: {vocab_path}. "
            f"Run compile_vocab.py first."
        )
    table: Dict[str, List[ResolvedEntity]] = defaultdict(list)
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid    = e.get("canonical_id")
            cat    = e.get("category")
            name   = e.get("canonical_name", "")
            source = e.get("source", "")
            if not cid or not cat:
                continue
            for alias in e.get("aliases", []):
                a = _normalize(alias)
                if not a:
                    continue
                table[a].append(ResolvedEntity(
                    canonical_id=cid,
                    category=cat,
                    canonical_name=name,
                    matched_alias=a,
                    source=source,
                ))
    return table


def _ensure_loaded(vocab_path: str) -> Dict[str, List[ResolvedEntity]]:
    global _alias_to_entities, _loaded_from
    if _alias_to_entities is not None and _loaded_from == vocab_path:
        return _alias_to_entities
    print(f"[entity_resolver] loading vocab from {vocab_path} ...", file=sys.stderr)
    _alias_to_entities = _load_vocab(vocab_path)
    _loaded_from = vocab_path
    print(f"[entity_resolver] loaded {len(_alias_to_entities):,} unique aliases",
          file=sys.stderr)
    return _alias_to_entities


def close() -> None:
    """Forget the cached vocab (free memory or reload after a vocab update)."""
    global _alias_to_entities, _loaded_from
    _alias_to_entities = None
    _loaded_from = None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def resolve(term: str,
            vocab_path: str = DEFAULT_VOCAB_PATH,
            prefer_category: Optional[str] = None) -> Optional[ResolvedEntity]:
    """
    Return the first ResolvedEntity matching `term`, or None.

    prefer_category : if multiple entities share this alias and one belongs to
                       this category, prefer it. Use this when you know what
                       kind of entity the user meant (e.g. "B. dorei" should
                       resolve to a bacterium, not a fungus).
    """
    table = _ensure_loaded(vocab_path)
    hits = table.get(_normalize(term))
    if not hits:
        return None
    if prefer_category:
        for h in hits:
            if h.category == prefer_category:
                return h
    return hits[0]


def resolve_all(term: str,
                vocab_path: str = DEFAULT_VOCAB_PATH) -> List[ResolvedEntity]:
    """Return every ResolvedEntity that matches `term`. Empty list if none."""
    table = _ensure_loaded(vocab_path)
    return list(table.get(_normalize(term), []))


def resolve_many(terms: Iterable[str],
                 vocab_path: str = DEFAULT_VOCAB_PATH,
                 prefer_category: Optional[str] = None
                 ) -> Dict[str, Optional[ResolvedEntity]]:
    """
    Resolve a list of terms in one pass. Returns {term: ResolvedEntity or None}.
    Preserves the input strings (NOT normalized) as keys so the caller can
    report exactly what the user typed.
    """
    _ensure_loaded(vocab_path)  # warm
    return {t: resolve(t, vocab_path=vocab_path, prefer_category=prefer_category)
            for t in terms}


def resolve_features(features_by_category: Dict[str, Iterable[str]],
                     vocab_path: str = DEFAULT_VOCAB_PATH
                     ) -> "ResolvedFeatureGroup":
    """
    Resolve a category-grouped feature dict, preferring entities from each
    category bucket when there's ambiguity.

    Returns a ResolvedFeatureGroup with:
      .resolved   : {category: [canonical_id, ...]}
      .unresolved : {category: [user_term that didn't match, ...]}
      .ambiguous  : {category: {user_term: [ResolvedEntity, ...]}}
    """
    _ensure_loaded(vocab_path)
    resolved: Dict[str, List[str]] = {}
    unresolved: Dict[str, List[str]] = {}
    ambiguous: Dict[str, Dict[str, List[ResolvedEntity]]] = {}
    for category, terms in (features_by_category or {}).items():
        resolved.setdefault(category, [])
        unresolved.setdefault(category, [])
        ambiguous.setdefault(category, {})
        for term in terms:
            hits = resolve_all(term, vocab_path=vocab_path)
            if not hits:
                unresolved[category].append(term)
                continue
            preferred = [h for h in hits if h.category == category] or hits
            if len(preferred) > 1:
                ambiguous[category][term] = preferred
            resolved[category].append(preferred[0].canonical_id)
    return ResolvedFeatureGroup(resolved=resolved,
                                unresolved=unresolved,
                                ambiguous=ambiguous)


@dataclass
class ResolvedFeatureGroup:
    resolved:   Dict[str, List[str]]
    unresolved: Dict[str, List[str]]
    ambiguous:  Dict[str, Dict[str, List[ResolvedEntity]]]

    def summary(self) -> str:
        """Short multi-line summary of what resolved, what didn't."""
        lines = ["Resolution summary:"]
        for cat in sorted(set(list(self.resolved) + list(self.unresolved))):
            ok  = len(self.resolved.get(cat, []))
            bad = len(self.unresolved.get(cat, []))
            amb = len(self.ambiguous.get(cat, {}))
            lines.append(f"  {cat:12s} resolved={ok}  unresolved={bad}  ambiguous={amb}")
        if any(self.unresolved.values()):
            lines.append("  Unresolved terms:")
            for cat, terms in self.unresolved.items():
                for t in terms:
                    lines.append(f"    [{cat}] {t!r}")
        return "\n".join(lines)

    def all_resolved(self) -> bool:
        return not any(self.unresolved.values())
