#!/usr/bin/env python3
"""
build_corpus.py
----------------
Builds (and incrementally extends) the CherryPicker literature corpus from
Europe PMC: full text + metadata for every article matching a configured
journal set and topic-keyword list, year by year.

This replaces two near-duplicate scripts that used to do this job
separately -- one for the original 22 seed journals, one for a later
49-journal JCR-tier expansion, plus a third one-off script that added a
single journal (PNAS) -- with ONE script driven by a single JOURNAL_SETS
config. Run with --journal-set to build/extend just one named set, or omit
it to run the full combined list.

Safe to re-run: every run loads existing article IDs (PMCID/PMID/EuropePMC
id) from metadata_all.jsonl first and skips anything already present, so
adding a new journal set later never re-downloads or duplicates existing
articles. (Previously this dedup logic existed only in the one-off PNAS
script; it's now applied on every run, which is what actually makes a
single script safe to use for every journal set instead of three.)

Journal matching:
    A journal with one or more ISSNs is matched by ISSN:<issn> (exact, and
    not fooled by name variants -- prefer this whenever the ISSN is known).
    A journal with no ISSN listed falls back to JOURNAL:"<name>" (Europe
    PMC's own journal-name field).

Usage:
    python build_corpus.py                     # full combined list, all years
    python build_corpus.py --journal-set seed
    python build_corpus.py --journal-set jcr_expansion
    python build_corpus.py --journal-set pnas
    python build_corpus.py --year-start 2020 --year-end 2025
"""

import argparse
import os
import time
import json
import requests
import xml.etree.ElementTree as ET
import html
import re
import traceback
import random
import uuid
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------
# Config
# -----------------------------

EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

OUTPUT_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus"
METADATA_ALL_JSONL = os.path.join(OUTPUT_ROOT, "metadata_all.jsonl")
METADATA_KW_JSONL = os.path.join(OUTPUT_ROOT, "metadata_kw.jsonl")

YEAR_START_DEFAULT = 1990
YEAR_END_DEFAULT = 2025

# Threading & Politeness Config
# 4 workers with a per-article sleep is the "safe zone" for Europe PMC
# without an API key.
MAX_WORKERS = 4
PAGE_SIZE = 100

# IMPORTANT: Do NOT set a blanket Accept: application/json session header --
# the fullTextXML endpoint returns 406 if it sees that. Accept headers are
# set per-request below instead.
HEADERS = {
    "User-Agent": "CorpusBuilder/2.6 (mailto:alonr5@your-org.example)",
}

# -----------------------------
# Journal sets
# -----------------------------
# Each journal maps to a (possibly empty) list of ISSNs. An empty list means
# "match by the Europe PMC JOURNAL name field"; a non-empty list means
# "match by ISSN" (more precise -- use this whenever the ISSN is known).

JOURNAL_SETS: Dict[str, Dict[str, List[str]]] = {
    # The original 22 journals the corpus was seeded with.
    "seed": {
        "Microbiome": [], "Gut": [], "The ISME Journal": [], "Nature Microbiology": [],
        "Cell Host & Microbe": [], "Cell Systems": [], "mSystems": [],
        "Environmental Microbiology": [], "Environmental Microbiology Reports": [],
        "Applied and Environmental Microbiology": [], "BMC Microbiology": [],
        "BMC Bioinformatics": [], "Bioinformatics": [], "PLOS Computational Biology": [],
        "Nature Communications": [], "Nature Medicine": [], "Nature Biotechnology": [],
        "Nature Methods": [], "Nature": [], "Science": [], "Cell": [],
        "Proc Natl Acad Sci U S A": [],
    },
    # The later 49-journal expansion across JCR tiers (medical informatics,
    # multidisciplinary sciences, microbiology). See the paper's Appendix A.2.
    "jcr_expansion": {
        # ---- Medical Informatics ----
        "Lancet Digital Health": ["2589-7500"],
        "npj Digital Medicine": ["2398-6352"],
        "ACM Transactions on Computing for Healthcare": ["2691-1957", "2637-8051"],
        "PLOS Digital Health": ["2767-3170"],
        "Intelligent Medicine": ["2667-1026"],
        "IEEE Journal of Biomedical and Health Informatics": ["2168-2194", "2168-2208"],
        "JMIR mHealth and uHealth": ["2291-5222"],
        "Journal of Medical Internet Research": ["1439-4456", "1438-8871"],
        "Journal of Medical Systems": ["0148-5598", "1573-689X"],
        "Artificial Intelligence in Medicine": ["0933-3657", "1873-2860"],
        # ---- Multidisciplinary Sciences ----
        "Nature Reviews Methods Primers": ["2662-8449"],
        "Nature Computational Science": ["2662-8457"],
        "Science Advances": ["2375-2548"],
        "Scientific Data": ["2052-4463"],
        "National Science Review": ["2095-5138", "2053-714X"],
        "Science Bulletin": ["2095-9273", "2095-9281"],
        "Journal of Advanced Research": ["2090-1232", "2090-1224"],
        "Research": ["2096-5168", "2639-5274"],
        "Global Challenges": ["2056-6646"],
        "Fundamental Research": ["2096-9457", "2667-3258"],
        "Research Synthesis Methods": ["1759-2879", "1759-2887"],
        "Innovation": ["2666-6758"],
        "Exploration": ["2766-8509", "2766-2098"],
        "Nature Human Behaviour": ["2397-3374"],
        # ---- Microbiology ----
        "Nature Reviews Microbiology": ["1740-1526", "1740-1534"],
        "Trends in Microbiology": ["0966-842X", "1878-4380"],
        "FEMS Microbiology Reviews": ["0168-6445", "1574-6976"],
        "Clinical Microbiology Reviews": ["0893-8512", "1098-6618"],
        "Microbiology and Molecular Biology Reviews": ["1092-2172", "1098-5557"],
        "Gut Microbes": ["1949-0976", "1949-0984"],
        "npj Biofilms and Microbiomes": ["2055-5008"],
        "Environmental Microbiome": ["2524-6372"],
        "ISME Communications": ["2730-6151"],
        "Emerging Microbes & Infections": ["2222-1751"],
        "Virulence": ["2150-5594", "2150-5608"],
        "Journal of Oral Microbiology": ["2000-2297"],
        "Annual Review of Microbiology": ["0066-4227", "1545-3251"],
        "Current Opinion in Microbiology": ["1369-5274", "1879-0364"],
        "Lancet Microbe": ["2666-5247"],
        "Clinical Infectious Diseases": ["1058-4838", "1537-6591"],
        "Clinical Microbiology and Infection": ["1198-743X", "1469-0691"],
        "Journal of Clinical Microbiology": ["0095-1137", "1098-660X"],
        "New Microbes and New Infections": ["2052-2975"],
        "Critical Reviews in Microbiology": ["1040-841X", "1549-7828"],
        "iMeta": ["2770-5986", "2770-596X"],
        "Current Research in Microbial Sciences": ["2666-5174"],
        "International Journal of Food Microbiology": ["0168-1605", "1879-3460"],
        "Microbial Biotechnology": ["1751-7915"],
        "Microbiological Research": ["0944-5013", "1618-0623"],
    },
    # A one-off targeted addition (kept as its own named set so it can still
    # be run in isolation, e.g. to re-check for new PNAS articles later).
    "pnas": {
        "Proc Natl Acad Sci U S A": [],
    },
}

# Keyword list used to flag "topically relevant" articles into the
# metadata_kw.jsonl subset. This is the superset of terms used across the
# three original scripts.
KEYWORD_TERMS = [
    "microbiome", "gut microbiota", "metagenomics", "microbiome bioinformatics",
    "shotgun metagenomics", "functional metagenomics", "metagenomic sequencing",
    "gene", "genes", "gene expression", "transcriptomics", "metatranscriptomics",
    "functional genes", "gene regulation", "gene pathways", "gene networks",
    "gene ontology", "go enrichment", "go term", "go terms", "go analysis",
    "metabolite", "metabolites", "metabolism", "metabolomics", "metabolic pathway",
    "metabolic network", "metabolomic profiling", "metabolite production",
    "short-chain fatty acids", "scfa", "butyrate", "acetate", "propionate",
    "metabolic reconstruction", "metabolic modeling", "microbiome-metabolome",
    "microbial metabolism", "metabolic capacity", "mimosa", "mimosa2",
    "bacteria", "bacterial", "microbiota", "microbial", "microbial community",
    "microbial species", "taxa", "taxonomy", "otu", "otus", "amplicon",
    "16s", "16s rrna", "16s sequencing", "host gene expression",
    "host transcriptomics", "host response", "host-microbiome", "host-microbiota",
    "host genetics", "tryptophan metabolism", "bile acid metabolism",
    "bile acids", "immune pathways", "immune gene", "immune genes",
    "immune response", "t cell", "t cells", "t-cell", "t-cells",
    "b cell", "b cells", "b-cell", "b-cells", "functional profiling",
    "functional annotation", "pathway enrichment", "gene set enrichment",
    "gsea", "metagenome-scale", "genome-scale metabolic model",
    "gsmm", "constraint-based modeling", "prokaryote", "prokaryotes", "prokaryotic",
    "archaea", "archaeome",
    "microbial ecology", "microbial dysbiosis", "dysbiosis",
    "microbial interactions", "microbe-microbe interactions",
    "microbial diversity", "alpha diversity", "beta diversity",
    "microbial networks",
    "multi-omics", "immune-microbiome interaction",
    "microbiome-immune axis",
    "gut-brain axis", "gut-liver axis", "gut-lung axis",
    "fecal microbiota transplantation", "fmt",
    "probiotics", "prebiotics", "synbiotics", "postbiotics",
    "qiime2", "dada2", "asv", "asvs",
    "humann", "metaphlan",
    "metabolite quantification",
    "tryptamine", "kynurenine pathway",
    "secondary bile acids",
    "branched-chain fatty acids", "amino acid metabolism",
    "trimethylamine", "tma", "tmao",
]

# -----------------------------
# Query construction
# -----------------------------

def _clean_issn(s: str) -> str:
    return str(s).strip().upper()

def build_base_query(journals: Dict[str, List[str]]) -> str:
    """
    Journals with ISSNs -> matched by ISSN:<issn> (OR'd together).
    Journals with no ISSN -> matched by JOURNAL:"<name>" (OR'd together).
    Both clauses are OR'd together into one journal filter.
    """
    issns = sorted({
        _clean_issn(i)
        for issn_list in journals.values()
        for i in (issn_list or [])
        if i and str(i).strip()
    })
    names_without_issn = sorted(
        name for name, issn_list in journals.items() if not issn_list
    )

    clauses = []
    if issns:
        clauses.append("(" + " OR ".join(f"ISSN:{i}" for i in issns) + ")")
    if names_without_issn:
        clauses.append("(" + " OR ".join(f'JOURNAL:"{n}"' for n in names_without_issn) + ")")
    if not clauses:
        raise ValueError("No journals configured.")
    journal_clause = "(" + " OR ".join(clauses) + ")"

    return (
        'PUB_TYPE:"Journal Article" '
        "AND IN_EPMC:Y "
        "AND NOT PREPRINT:Y "
        f"AND {journal_clause}"
    )

def build_query_with_year(base_query: str, year: int) -> str:
    return f"{base_query} AND PUB_YEAR:{year}"

# -----------------------------
# Network Session Setup
# -----------------------------

def get_session() -> requests.Session:
    """Creates a requests Session with automatic retries and connection pooling."""
    session = requests.Session()
    session.headers.update(HEADERS)
    retries = Retry(
        total=5,
        backoff_factor=2,  # Exponential backoff: 2s, 4s, 8s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        raise_on_redirect=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Global session for BOTH metadata and XML calls
http = get_session()

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# -----------------------------
# XML Cleaning Logic
# -----------------------------

def _localname(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

UNWANTED_SECTION_TITLES = {
    "acknowledgements", "acknowledgments", "funding", "author contributions",
    "author contribution", "competing interests", "conflict of interest",
    "conflicts of interest", "ethics", "ethical approval", "data availability",
    "availability of data", "additional information", "supplementary material",
    "supplementary materials", "supplementary information", "extended data",
    "publisher's note", "publisher note", "references", "abbreviations"
}

def _should_skip_section(title_text: str) -> bool:
    title_norm = title_text.strip().lower()
    return any(x in title_norm for x in UNWANTED_SECTION_TITLES)

def _extract_body_from_sec(sec: ET.Element, pieces: list) -> None:
    sec_title = None
    for child in sec:
        if _localname(child.tag) == "title":
            t = _normalize_text("".join(child.itertext()))
            if t:
                sec_title = t
            break

    if sec_title and _should_skip_section(sec_title):
        return
    if sec_title:
        pieces.append(sec_title)

    for child in sec:
        lname = _localname(child.tag)
        if lname == "p":
            text = _normalize_text("".join(child.itertext()))
            if text:
                pieces.append(text)
        elif lname == "list":
            for li in child:
                if _localname(li.tag) in {"list-item", "li"}:
                    li_text = _normalize_text("".join(li.itertext()))
                    if li_text:
                        pieces.append(li_text)
        elif lname == "sec":
            _extract_body_from_sec(child, pieces)

def extract_clean_text_from_xml(xml_text: str) -> Optional[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    pieces: List[str] = []

    # Title (capture the main article title once)
    try:
        for elem in root.iter():
            if _localname(elem.tag) in ("article-title", "title"):
                t = _normalize_text("".join(elem.itertext()))
                if t and len(t) > 5 and t not in pieces:
                    pieces.append(t)
                    break
    except Exception:
        pass

    # Abstract
    try:
        for abs_el in root.iter():
            if _localname(abs_el.tag) == "abstract":
                t = _normalize_text("".join(abs_el.itertext()))
                if t:
                    pieces.append(t)
    except Exception:
        pass

    # Body + Back
    for container_name in ["body", "back"]:
        try:
            for elem in root.iter():
                if _localname(elem.tag) == container_name:
                    for child in elem:
                        lname = _localname(child.tag)
                        if lname == "p":
                            t = _normalize_text("".join(child.itertext()))
                            if t:
                                pieces.append(t)
                        elif lname == "sec":
                            _extract_body_from_sec(child, pieces)
        except Exception:
            traceback.print_exc()
            continue

    if not pieces:
        return None
    return "\n\n".join(pieces)

def article_matches_keywords(article: Dict, clean_text: Optional[str] = None) -> bool:
    """Check keyword matches in title, abstract, and cleaned full text (if available)."""
    title = article.get("title") or ""
    abstract = article.get("abstract") or article.get("abstractText") or ""
    haystack = (title + " " + abstract).lower()
    if clean_text:
        haystack += " " + clean_text.lower()
    return any(kw in haystack for kw in KEYWORD_TERMS)

# -----------------------------
# Fetching
# -----------------------------

def fetch_page(cursor_mark: str, page_size: int, query: str) -> Dict:
    params = {
        "query": query,
        "pageSize": page_size,
        "cursorMark": cursor_mark,
        "resultType": "core",
        "format": "json",
    }
    try:
        resp = http.get(
            EUROPE_PMC_API_URL, params=params, timeout=30,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()
    except json.JSONDecodeError as e:
        print(f"    [API Error Metadata] JSON decode failed (cursor={cursor_mark}): {e}")
        return {}
    except Exception as e:
        print(f"    [API Error Metadata] cursor={cursor_mark}, error={e}")
        return {}

def fetch_fulltext_content(pmcid: str) -> Optional[str]:
    """Fetch the full-text XML for a PMCID. Uses the global session for connection pooling."""
    url = EUROPE_PMC_FULLTEXT_URL.format(pmcid=pmcid)
    try:
        resp = http.get(
            url, timeout=30,
            headers={"Accept": "application/xml, text/xml;q=0.9, */*;q=0.8"},
        )
        if resp.status_code == 200 and (resp.text or "").strip():
            return resp.text
        return None
    except Exception as e:
        print(f"    [Fulltext Error] PMCID={pmcid}, error={e}")
        return None

# -----------------------------
# Dedup (always on -- this is what makes one script safe for every journal
# set instead of needing a separate "adding" script per set)
# -----------------------------

def load_existing_ids(meta_path: str) -> Set[str]:
    ids: Set[str] = set()
    if not os.path.exists(meta_path):
        return ids
    print(f"Loading existing IDs from {meta_path} ...")
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = make_key_from_article(rec)
            if key:
                ids.add(key)
    print(f"Loaded {len(ids):,} existing article IDs.")
    return ids

def make_key_from_article(article: Dict) -> Optional[str]:
    pmcid = article.get("pmcid") or article.get("PMCID")
    pmid = article.get("pmid") or article.get("PMID")
    eid = article.get("id") or article.get("ID")
    if pmcid:
        return f"PMCID:{pmcid}"
    if pmid:
        return f"PMID:{pmid}"
    if eid:
        return f"ID:{eid}"
    return None

# -----------------------------
# Processing
# -----------------------------

# Global counters (overall summary, printed at the end of the run)
OA_ELIGIBLE = 0
FULLTEXT_SUCCESS = 0
FULLTEXT_FAIL = 0

def process_single_article(article: Dict) -> Dict:
    global OA_ELIGIBLE, FULLTEXT_SUCCESS, FULLTEXT_FAIL

    # Politeness: stagger requests across threads
    time.sleep(random.uniform(0.5, 1.0))

    year = article.get("pubYear") or "unknown"
    year_dir = os.path.join(OUTPUT_ROOT, str(year))
    try:
        os.makedirs(year_dir, exist_ok=True)
    except OSError:
        pass

    base_id = article.get("pmcid") or article.get("id") or f"unknown_{uuid.uuid4().hex[:8]}"
    base_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(base_id))
    txt_path = os.path.join(year_dir, f"{base_id}.txt")

    pmcid = article.get("pmcid")
    is_open_access = str(article.get("isOpenAccess") or "").upper() == "Y"
    clean_text = None

    if pmcid and is_open_access:
        OA_ELIGIBLE += 1
        raw_xml = fetch_fulltext_content(pmcid)
        if raw_xml:
            try:
                clean_text = extract_clean_text_from_xml(raw_xml)
            except Exception as e:
                print(f"    [XML Cleaning Error] PMCID={pmcid}, error={e}")
                clean_text = None

            if clean_text:
                try:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(clean_text)
                    FULLTEXT_SUCCESS += 1
                except Exception as e:
                    print(f"    [File Write Error] path={txt_path}, error={e}")
                    clean_text = None
                    FULLTEXT_FAIL += 1
            else:
                FULLTEXT_FAIL += 1
        else:
            FULLTEXT_FAIL += 1

    meta = article.copy()
    meta["txt_path"] = txt_path if clean_text else None
    meta["has_fulltext"] = bool(clean_text)

    try:
        meta["matches_keywords"] = article_matches_keywords(meta, clean_text)
    except Exception as e:
        print(f"    [Keyword Match Error] id={base_id}, error={e}")
        meta["matches_keywords"] = False

    return meta

# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/extend the CherryPicker corpus from Europe PMC.")
    parser.add_argument(
        "--journal-set", choices=list(JOURNAL_SETS.keys()) + ["all"], default="all",
        help="Which configured journal set to build/extend (default: all, i.e. every set combined).",
    )
    parser.add_argument("--year-start", type=int, default=YEAR_START_DEFAULT)
    parser.add_argument("--year-end", type=int, default=YEAR_END_DEFAULT)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.journal_set == "all":
        journals: Dict[str, List[str]] = {}
        for s in JOURNAL_SETS.values():
            journals.update(s)
    else:
        journals = JOURNAL_SETS[args.journal_set]

    base_query = build_base_query(journals)

    ensure_dir(OUTPUT_ROOT)
    existing_ids = load_existing_ids(METADATA_ALL_JSONL)
    print(f"Journal set: {args.journal_set} ({len(journals)} journals). "
          f"{len(existing_ids):,} existing articles loaded for dedup.")

    f_all = open(METADATA_ALL_JSONL, "a", encoding="utf-8")
    f_kw = open(METADATA_KW_JSONL, "a", encoding="utf-8")

    try:
        for year in range(args.year_start, args.year_end + 1):
            print(f"\n=== Processing Year {year} ===")
            year_dir = os.path.join(OUTPUT_ROOT, str(year))
            ensure_dir(year_dir)

            query = build_query_with_year(base_query, year)
            cursor = "*"
            page_count = 0
            year_articles_count = 0

            while True:
                page_count += 1
                data = fetch_page(cursor, PAGE_SIZE, query)

                if not data or "resultList" not in data:
                    print(f"  [Year {year}] No data returned or end of results at page {page_count}.")
                    break

                results = data.get("resultList", {}).get("result", [])
                if not results:
                    print(f"  [Year {year}] Empty result list at page {page_count}.")
                    break

                # Skip anything already in metadata_all.jsonl (dedup, so
                # re-running with a different --journal-set never
                # re-downloads or duplicates existing articles).
                new_results = []
                for art in results:
                    key = make_key_from_article(art)
                    if key and key not in existing_ids:
                        existing_ids.add(key)
                        new_results.append(art)

                if not new_results:
                    print(f"  [Year {year}] Page {page_count}: all {len(results)} articles already present, skipping.")
                else:
                    print(f"  [Year {year}] Page {page_count}: {len(new_results)}/{len(results)} new articles, "
                          f"processing with {MAX_WORKERS} workers...")

                    processed_batch: List[Dict] = []
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = {executor.submit(process_single_article, art): art for art in new_results}
                        for future in as_completed(futures):
                            try:
                                processed_batch.append(future.result())
                            except Exception as e:
                                print(f"  [Error in worker] {e}")
                                traceback.print_exc()

                    for meta in processed_batch:
                        try:
                            json_line = json.dumps(meta, ensure_ascii=False)
                            f_all.write(json_line + "\n")
                            if meta.get("matches_keywords"):
                                f_kw.write(json_line + "\n")
                        except Exception as e:
                            print(f"  [Write JSONL Error] {e}")

                    try:
                        f_all.flush()
                        f_kw.flush()
                    except Exception as e:
                        print(f"  [Flush Error] {e}")

                    year_articles_count += len(processed_batch)

                next_cursor = data.get("nextCursorMark")
                if not next_cursor or next_cursor == cursor:
                    print(f"  [Year {year}] No next cursor, stopping at page {page_count}.")
                    break
                cursor = next_cursor

            print(f"=== Finished {year}. New articles added: {year_articles_count} ===")

    finally:
        try:
            f_all.close()
            f_kw.close()
        except Exception as e:
            print(f"[File Close Error] {e}")

        print("\n============================")
        print("FULLTEXT SUMMARY (overall)")
        print("============================")
        print(f"OA eligible (OA+PMCID): {OA_ELIGIBLE}")
        print(f"Fulltext success:        {FULLTEXT_SUCCESS}")
        print(f"Fulltext fail:           {FULLTEXT_FAIL}")
        print("\nDone. Files closed.")

if __name__ == "__main__":
    main()
