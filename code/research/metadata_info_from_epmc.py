#!/usr/bin/env python3
"""
Count Europe PMC coverage by year (1990–2025) for a set of journals.

Per year counts:
  - metadata_count: total matching "Journal Article" in EPMC
  - open_access_count: OPEN_ACCESS:Y
  - pmcid_count: PMCID:*
  - has_fulltext_count: HAS_FULLTEXT:Y
  - has_pdf_count: HAS_PDF:Y
  - epmc_fulltext_proxy_count: (OPEN_ACCESS:Y OR HAS_FULLTEXT:Y OR HAS_PDF:Y OR PMCID:*)

Always writes a CSV with all years and totals row.
"""

import csv
import time
import random
import argparse
from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# ---- Journals (merged + de-duplicated) ----
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

    "ACM Transactions on Computing for Healthcare",
    "PLOS Digital Health",
    "Intelligent Medicine",
    "JMIR mHealth and uHealth",
    "Journal of Medical Internet Research",
    "Journal of Medical Systems",
    "Artificial Intelligence in Medicine",

    "Nature Reviews Methods Primers",
    "Nature Computational Science",
    "Science Advances",
    "Scientific Data",
    "National Science Review",
    "Science Bulletin",
    "Journal of Advanced Research",
    "Research",
    "Global Challenges",
    "Fundamental Research",
    "Research Synthesis Methods",
    "Innovation",
    "Exploration",
    "Nature Human Behaviour",

    "Nature Reviews Microbiology",
    "Trends in Microbiology",
    "FEMS Microbiology Reviews",
    "Clinical Microbiology Reviews",
    "Microbiology and Molecular Biology Reviews",
    "Gut Microbes",
    "npj Biofilms and Microbiomes",
    "Environmental Microbiome",
    "ISME Communications",
    "Emerging Microbes & Infections",
    "Virulence",
    "Journal of Oral Microbiology",
    "Annual Review of Microbiology",
    "Current Opinion in Microbiology",
    "Lancet Microbe",
    "Clinical Infectious Diseases",
    "Clinical Microbiology and Infection",
    "Journal of Clinical Microbiology",
    "New Microbes and New Infections",
    "Critical Reviews in Microbiology",
    "iMeta",
    "Current Research in Microbial Sciences",
    "International Journal of Food Microbiology",
    "Microbial Biotechnology",
    "Microbiological Research",
]))

BASE_FILTERS = (
    'PUB_TYPE:"Journal Article" '
    'AND IN_EPMC:Y '
    'AND NOT PREPRINT:Y'
)

def get_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def journal_clause(journals: List[str]) -> str:
    return " OR ".join([f'JOURNAL:"{j}"' for j in journals])

def epmc_hitcount(http: requests.Session, query: str, sleep_min: float, sleep_max: float) -> int:
    # hitCount is returned regardless of pageSize; keep pageSize tiny.
    time.sleep(random.uniform(sleep_min, sleep_max))
    r = http.get(
        EPMC_SEARCH_URL,
        params={
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": 1,
        },
        timeout=45,
    )
    try:
        data = r.json()
    except Exception:
        return 0

    hc = data.get("hitCount")
    try:
        return int(hc)
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-min", type=int, default=1990)
    ap.add_argument("--year-max", type=int, default=2025)
    ap.add_argument("--csv-out", required=True, help="Where to write CSV output")
    ap.add_argument("--sleep-min", type=float, default=0.08)
    ap.add_argument("--sleep-max", type=float, default=0.18)
    ap.add_argument("--quiet", action="store_true", help="Only print totals + CSV path")
    args = ap.parse_args()

    http = get_session()
    j_clause = journal_clause(JOURNALS)

    rows: List[Dict[str, int]] = []

    totals = {
        "metadata_count": 0,
        "open_access_count": 0,
        "pmcid_count": 0,
        "has_fulltext_count": 0,
        "has_pdf_count": 0,
        "epmc_fulltext_proxy_count": 0,
    }

    if not args.quiet:
        print(f"Journals: {len(JOURNALS)}")
        print(f"Years: {args.year_min}..{args.year_max}\n")

    for year in range(args.year_min, args.year_max + 1):
        base_q = f"{BASE_FILTERS} AND ({j_clause}) AND PUB_YEAR:{year}"

        q_open_access = f"{base_q} AND OPEN_ACCESS:Y"
        q_pmcid       = f"{base_q} AND PMCID:*"
        q_has_full    = f"{base_q} AND HAS_FULLTEXT:Y"
        q_has_pdf     = f"{base_q} AND HAS_PDF:Y"

        # Expanded proxy: any of these indicates decent chance of retrievable full text through EPMC
        q_proxy = f"{base_q} AND (OPEN_ACCESS:Y OR HAS_FULLTEXT:Y OR HAS_PDF:Y OR PMCID:*)"

        metadata_count = epmc_hitcount(http, base_q, args.sleep_min, args.sleep_max)
        open_access_count = epmc_hitcount(http, q_open_access, args.sleep_min, args.sleep_max)
        pmcid_count = epmc_hitcount(http, q_pmcid, args.sleep_min, args.sleep_max)
        has_fulltext_count = epmc_hitcount(http, q_has_full, args.sleep_min, args.sleep_max)
        has_pdf_count = epmc_hitcount(http, q_has_pdf, args.sleep_min, args.sleep_max)
        proxy_count = epmc_hitcount(http, q_proxy, args.sleep_min, args.sleep_max)

        rows.append({
            "year": year,
            "metadata_count": metadata_count,
            "open_access_count": open_access_count,
            "pmcid_count": pmcid_count,
            "has_fulltext_count": has_fulltext_count,
            "has_pdf_count": has_pdf_count,
            "epmc_fulltext_proxy_count": proxy_count,
        })

        totals["metadata_count"] += metadata_count
        totals["open_access_count"] += open_access_count
        totals["pmcid_count"] += pmcid_count
        totals["has_fulltext_count"] += has_fulltext_count
        totals["has_pdf_count"] += has_pdf_count
        totals["epmc_fulltext_proxy_count"] += proxy_count

        if not args.quiet:
            print(
                f"{year}: meta={metadata_count:,} | "
                f"OA={open_access_count:,} | PMCID={pmcid_count:,} | "
                f"HAS_FULLTEXT={has_fulltext_count:,} | HAS_PDF={has_pdf_count:,} | "
                f"PROXY={proxy_count:,}"
            )

    # Append totals row
    rows.append({
        "year": -1,  # sentinel
        "metadata_count": totals["metadata_count"],
        "open_access_count": totals["open_access_count"],
        "pmcid_count": totals["pmcid_count"],
        "has_fulltext_count": totals["has_fulltext_count"],
        "has_pdf_count": totals["has_pdf_count"],
        "epmc_fulltext_proxy_count": totals["epmc_fulltext_proxy_count"],
    })

    # Write CSV
    fieldnames = [
        "year",
        "metadata_count",
        "open_access_count",
        "pmcid_count",
        "has_fulltext_count",
        "has_pdf_count",
        "epmc_fulltext_proxy_count",
    ]
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n=== TOTALS (year=-1 in CSV) ===")
    for k, v in totals.items():
        print(f"{k}: {v:,}")
    print(f"\nWrote CSV: {args.csv_out}")

if __name__ == "__main__":
    main()
