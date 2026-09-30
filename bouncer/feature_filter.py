"""
feature_filter.py
-----------------
Stage-1 filter for the two-stage RAG pipeline: given a group of biomedical
features the user cares about, return the set of PMCIDs whose article-entity
counts satisfy the requested per-category K thresholds.

User never sees the SQL. They call eligible_pmcids() with Python dicts and
get back a set of strings.

Backing store:
    data/entity_index.sqlite, built by build_entity_index.py.
    Schema:
        article_entities(pmcid, category, canonical_id, canonical_name, present|count)
    Both schemas (present-binary and count-integer) are handled.

Caching:
    The SQLite connection is opened lazily and cached at module level.
    Subsequent calls reuse it. Pass force_reopen=True (or call close()) to reset.

Example:
    from rag.feature_filter import eligible_pmcids

    features = {
        "bacteria":   ["NCBI:txid357276", "NCBI:txid853", ...],
        "metabolite": ["HMDB:HMDB0000243", "HMDB:HMDB0000148", ...],
    }
    thresholds = {"bacteria": 4, "metabolite": 4}

    pmcids = eligible_pmcids(features, thresholds)
    # -> {"PMC7129573", "PMC7129952", ...}
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, Iterable, Optional, Set


# Default path matches build_entity_index.py.
DEFAULT_INDEX_PATH = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/entity_index.sqlite"


_conn: Optional[sqlite3.Connection] = None
_conn_path: Optional[str] = None


def _open(index_path: str) -> sqlite3.Connection:
    """Open (and cache) a read-only-style SQLite connection to the entity index."""
    global _conn, _conn_path
    if _conn is not None and _conn_path == index_path:
        return _conn
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"entity index not found: {index_path}. "
            f"Build it with build_entity_index.py first."
        )
    # uri=True so we can open read-only via mode=ro; safer for a shared DB.
    conn = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    _conn = conn
    _conn_path = index_path
    return conn


def close() -> None:
    """Close the cached connection (mostly for tests / reload scenarios)."""
    global _conn, _conn_path
    if _conn is not None:
        try:
            _conn.close()
        finally:
            _conn = None
            _conn_path = None


def _flatten_canonical_ids(features_by_category: Dict[str, Iterable[str]]) -> list:
    """Collect all canonical_ids across categories into one flat list, deduped, order-preserving."""
    seen = set()
    out = []
    for cat, ids in (features_by_category or {}).items():
        for cid in ids:
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out


def _validate_thresholds(features_by_category: Dict[str, Iterable[str]],
                         thresholds_by_category: Dict[str, int]) -> None:
    """Defensive checks before we build the SQL."""
    if not features_by_category:
        raise ValueError("features_by_category is empty.")
    for cat, k in (thresholds_by_category or {}).items():
        if not isinstance(k, int) or k < 0:
            raise ValueError(f"threshold for category {cat!r} must be a non-negative int, got {k!r}.")
        n_avail = len(list(features_by_category.get(cat, ())))
        if k > n_avail:
            raise ValueError(
                f"threshold for category {cat!r} is {k}, but only {n_avail} "
                f"canonical IDs were provided -- impossible to satisfy."
            )


def eligible_pmcids(features_by_category: Dict[str, Iterable[str]],
                    thresholds_by_category: Optional[Dict[str, int]] = None,
                    index_path: str = DEFAULT_INDEX_PATH) -> Set[str]:
    """
    Return the set of PMCIDs whose article_entities satisfy the per-category
    K thresholds against the provided feature group.

    Args:
        features_by_category   : {category: [canonical_id, ...]}.
                                  e.g. {"bacteria": ["NCBI:txid357276", ...],
                                        "metabolite": ["HMDB:HMDB0000243", ...]}
        thresholds_by_category : {category: K}. An article qualifies when, for
                                  EVERY category in this dict, it mentions at
                                  least K distinct canonical IDs from the
                                  provided feature list in that category.
                                  Categories listed in features_by_category but
                                  NOT in thresholds_by_category have no
                                  threshold -- they contribute to the IN list
                                  but don't constrain the HAVING.
                                  If None, defaults to: every category must
                                  have >= 1 match.
        index_path             : path to entity_index.sqlite

    Returns:
        Set of pmcid strings. Empty set means nothing qualified.
    """
    _validate_thresholds(features_by_category, thresholds_by_category or {})

    flat_ids = _flatten_canonical_ids(features_by_category)
    if not flat_ids:
        return set()

    if thresholds_by_category is None:
        # Default: at least 1 of each category that was supplied
        thresholds_by_category = {cat: 1 for cat in features_by_category if features_by_category[cat]}

    # Only enforce thresholds for categories that have any IDs provided AND a
    # threshold > 0.
    effective_thresholds = {
        cat: k for cat, k in thresholds_by_category.items()
        if k > 0 and features_by_category.get(cat)
    }

    conn = _open(index_path)

    # Build the SQL.
    # SUM(CASE WHEN ...) counts the number of DISTINCT canonical_ids per
    # category per article -- but only counts ids that are in our flat_ids
    # list (since the WHERE clause already filtered to those).
    placeholders = ",".join("?" * len(flat_ids))
    having_clauses = []
    having_params = []
    for cat, k in effective_thresholds.items():
        having_clauses.append(
            f"SUM(CASE WHEN category = ? THEN 1 ELSE 0 END) >= ?"
        )
        having_params.extend([cat, k])

    sql = f"SELECT pmcid FROM article_entities WHERE canonical_id IN ({placeholders})"
    sql += " GROUP BY pmcid"
    if having_clauses:
        sql += " HAVING " + " AND ".join(having_clauses)

    cur = conn.execute(sql, list(flat_ids) + having_params)
    return {row["pmcid"] for row in cur}


def per_article_breakdown(pmcids: Iterable[str],
                          features_by_category: Dict[str, Iterable[str]],
                          index_path: str = DEFAULT_INDEX_PATH) -> Dict[str, Dict[str, int]]:
    """
    For a set of PMCIDs and a feature group, return:
        {pmcid: {category: count_of_matched_canonical_ids}}

    Useful for diagnostics: "why did this article qualify?"
    """
    pmcid_list = list(pmcids)
    if not pmcid_list:
        return {}

    flat_ids = _flatten_canonical_ids(features_by_category)
    if not flat_ids:
        return {p: {} for p in pmcid_list}

    conn = _open(index_path)

    pmc_ph = ",".join("?" * len(pmcid_list))
    cid_ph = ",".join("?" * len(flat_ids))

    sql = (
        f"SELECT pmcid, category, COUNT(DISTINCT canonical_id) AS n "
        f"FROM article_entities "
        f"WHERE pmcid IN ({pmc_ph}) AND canonical_id IN ({cid_ph}) "
        f"GROUP BY pmcid, category"
    )

    out: Dict[str, Dict[str, int]] = {p: {} for p in pmcid_list}
    for row in conn.execute(sql, pmcid_list + flat_ids):
        out[row["pmcid"]][row["category"]] = row["n"]
    return out
