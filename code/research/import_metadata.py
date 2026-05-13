#!/usr/bin/env python3
"""
Backfill Europe PMC METADATA ONLY for selected journals (1990–2025).

Enhancements:
- Deduplicates by PMCID → PMID → DOI → EPMC id
- Appends ONLY missing metadata
- Appends keyword-matching records to metadata_kw.jsonl
- Prints PER-YEAR statistics:
    * total metadata records for that year (after run)
    * how many were added in this run
    * how many already existed
"""

import os
import re
import json
import time
import random
from typing import Dict, Any, Optional, Set, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------
# Paths
# -----------------------------
METADATA_ALL_JSONL = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus/metadata_all.jsonl"
METADATA_KW_JSONL  = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus/metadata_kw.jsonl"

EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

START_YEAR = 1990
END_YEAR   = 2025

PAGE_SIZE = 100
SLEEP_RANGE = (0.10, 0.30)

# -----------------------------
# Journals
# -----------------------------
JOURNALS = sorted(set([
    "Microbiome", "Gut", "The ISME Journal", "Nature Microbiology",
    "Cell Host & Microbe", "Cell Systems", "mSystems",
    "Environmental Microbiology", "Environmental Microbiology Reports",
    "Applied and Environmental Microbiology", "BMC Microbiology",
    "BMC Bioinformatics", "Bioinformatics", "PLOS Computational Biology",
    "Nature Communications", "Nature Medicine", "Nature Biotechnology",
    "Nature Methods", "Nature", "Science", "Cell", "Proc Natl Acad Sci U S A",
    "Lancet Digital Health", "npj Digital Medicine",
    "IEEE Journal of Biomedical and Health Informatics",
]))

journal_clause = " OR ".join([f'JOURNAL:"{j}"' for j in JOURNALS])

BASE_FILTER = (
    'PUB_TYPE:"Journal Article" '
    'AND IN_EPMC:Y '
    'AND NOT PREPRINT:Y '
    f'AND ({journal_clause})'
)

# -----------------------------
# Keywords
# -----------------------------
KEYWORD_TERMS = sorted(set([
    "microbiome", "gut microbiota", "metagenomics",
    "metabolomics", "gene expression", "metatranscriptomics",
    "functional profiling", "pathway enrichment",
    "bile acid metabolism", "short-chain fatty acids",
    "immune response", "host-microbiome",
    "metabolic modeling", "multi-omics",
]))

_kw_rx = [re.compile(re.escape(k), re.I) for k in KEYWORD_TERMS]

# -----------------------------
# Network session
# -----------------------------
def get_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

http = get_session()

# -----------------------------
# Helpers
# -----------------------------
def stable_key(rec: Dict[str, Any]) -> Optional[str]:
    pmcid = (rec.get("pmcid") or "").strip().upper()
    if pmcid:
        if not pmcid.startswith("PMC"):
            pmcid = "PMC" + pmcid
        return f"PMCID:{pmcid}"

    pmid = (rec.get("pmid") or "").strip()
    if pmid:
        return f"PMID:{pmid}"

    doi = (rec.get("doi") or "").strip().lower()
    if doi:
        return f"DOI:{doi}"

    eid = (rec.get("id") or "").strip()
    if eid:
        return f"EPMC_ID:{eid}"

    return None

def load_existing_by_year(path: str) -> Dict[int, Set[str]]:
    per_year: Dict[int, Set[str]] = {}
    if not os.path.exists(path):
        return per_year

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue

            year = rec.get("pubYear")
            try:
                year = int(year)
            except Exception:
                continue

            key = stable_key(rec)
            if not key:
                continue

            per_year.setdefault(year, set()).add(key)

    return per_year

def matches_keywords(rec: Dict[str, Any]) -> bool:
    text = ((rec.get("title") or "") + " " + (rec.get("abstractText") or "")).lower()
    return any(rx.search(text) for rx in _kw_rx)

def fetch_page(query: str, cursor: str) -> Dict:
    time.sleep(random.uniform(*SLEEP_RANGE))
    r = http.get(
        EUROPE_PMC_API_URL,
        params={
            "query": query,
            "pageSize": PAGE_SIZE,
            "cursorMark": cursor,
            "resultType": "core",
            "format": "json",
        },
        timeout=45,
    )
    try:
        return r.json()
    except Exception:
        return {}

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading existing metadata indices...")
    existing_by_year = load_existing_by_year(METADATA_ALL_JSONL)
    existing_global = set().union(*existing_by_year.values())

    f_all = open(METADATA_ALL_JSONL, "a", encoding="utf-8")
    f_kw  = open(METADATA_KW_JSONL, "a", encoding="utf-8")

    try:
        for year in range(START_YEAR, END_YEAR + 1):
            print(f"\n=== Year {year} ===")

            existing_this_year = set(existing_by_year.get(year, set()))
            added_this_year = 0

            cursor = "*"
            query = f"{BASE_FILTER} AND PUB_YEAR:{year}"

            while True:
                data = fetch_page(query, cursor)
                results = (data.get("resultList") or {}).get("result") or []
                if not results:
                    break

                for rec in results:
                    key = stable_key(rec)
                    if not key or key in existing_global:
                        continue

                    rec["matches_keywords"] = matches_keywords(rec)

                    f_all.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if rec["matches_keywords"]:
                        f_kw.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    existing_global.add(key)
                    existing_this_year.add(key)
                    added_this_year += 1

                f_all.flush()
                f_kw.flush()

                nxt = data.get("nextCursorMark")
                if not nxt or nxt == cursor:
                    break
                cursor = nxt

            total_now = len(existing_this_year)
            already_existed = total_now - added_this_year

            print(
                f"Year {year} summary:\n"
                f"  total_metadata_now = {total_now:,}\n"
                f"  added_this_run     = {added_this_year:,}\n"
                f"  already_existed    = {already_existed:,}"
            )

    finally:
        f_all.close()
        f_kw.close()

    print("\n=== Backfill complete ===")

if __name__ == "__main__":
    main()
