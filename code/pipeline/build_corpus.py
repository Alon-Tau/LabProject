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
from typing import List, Dict, Optional
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

# Years to cover
END_YEAR = 2025
RESUME_FROM_YEAR = 1990  # Change this if you need to restart mid-way

# Threading & Politeness Config
# 4 workers with sleep is the "Safe Zone" for Europe PMC without an API key.
MAX_WORKERS = 4
PAGE_SIZE = 100

JOURNALS = [
    "Microbiome", "Gut", "The ISME Journal", "Nature Microbiology",
    "Cell Host & Microbe", "Cell Systems", "mSystems",
    "Environmental Microbiology", "Environmental Microbiology Reports",
    "Applied and Environmental Microbiology", "BMC Microbiology",
    "BMC Bioinformatics", "Bioinformatics", "PLOS Computational Biology",
    "Nature Communications", "Nature Medicine", "Nature Biotechnology",
    "Nature Methods", "Nature", "Science", "Cell", "Proc Natl Acad Sci U S A",
]

# Construct Journal Query
journal_clause = " OR ".join([f'JOURNAL:"{j}"' for j in JOURNALS])

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
    "gsmm", "constraint-based modeling",
]

# Base Query (only journal + article type filters)
FILTERS = (
    'PUB_TYPE:"Journal Article" '
    'AND IN_EPMC:Y '
    'AND NOT PREPRINT:Y '
    f'AND ({journal_clause})'
)
BASE_QUERY = FILTERS

# -----------------------------
# Network Session Setup
# -----------------------------

def get_session() -> requests.Session:
    """
    Creates a requests Session with automatic retries and connection pooling.
    """
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,  # Exponential backoff: 2s, 4s, 8s...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        raise_on_redirect=False,
    )
    # Increase pool size to accommodate threads
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

def build_query_with_year(base_query: str, year: int) -> str:
    return f"{base_query} AND PUB_YEAR:{year}"

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
            t = "".join(child.itertext())
            t = _normalize_text(t)
            if t:
                sec_title = t
            break

    # Skip whole section if it matches unwanted titles
    if sec_title and _should_skip_section(sec_title):
        return

    if sec_title:
        pieces.append(sec_title)

    for child in sec:
        lname = _localname(child.tag)
        if lname == "p":
            text = "".join(child.itertext())
            text = _normalize_text(text)
            if text:
                pieces.append(text)
        elif lname == "list":
            for li in child:
                if _localname(li.tag) in {"list-item", "li"}:
                    li_text = "".join(li.itertext())
                    li_text = _normalize_text(li_text)
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

    # Title (try to capture main article title once)
    try:
        for elem in root.iter():
            if _localname(elem.tag) in ("article-title", "title"):
                t = "".join(elem.itertext())
                t = _normalize_text(t)
                if t and len(t) > 5 and t not in pieces:
                    pieces.append(t)
                    break
    except Exception:
        pass

    # Abstract
    try:
        for abs_el in root.iter():
            if _localname(abs_el.tag) == "abstract":
                t = "".join(abs_el.itertext())
                t = _normalize_text(t)
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
                            t = "".join(child.itertext())
                            t = _normalize_text(t)
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
    """
    Check keyword matches in:
      - title
      - abstract or abstractText (Europe PMC field)
      - cleaned full text if available
    """
    title = article.get("title") or ""
    abstract = article.get("abstract") or article.get("abstractText") or ""

    haystack = (title + " " + abstract).lower()
    if clean_text:
        haystack += " " + clean_text.lower()

    for kw in KEYWORD_TERMS:
        if kw in haystack:
            return True
    return False

# -----------------------------
# Fetching & Processing
# -----------------------------

def fetch_page(cursor_mark: str, page_size: int, query: str) -> Dict:
    """
    Fetch a single page of metadata.
    """
    params = {
        "query": query,
        "pageSize": page_size,
        "cursorMark": cursor_mark,
        "resultType": "core",
        "format": "json",
    }
    try:
        # Uses global session
        resp = http.get(EUROPE_PMC_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            print(f"    [API Error Metadata] JSON decode failed (cursor={cursor_mark}): {e}")
            return {}
        return data
    except Exception as e:
        print(f"    [API Error Metadata] cursor={cursor_mark}, error={e}")
        return {}

def fetch_fulltext_content(pmcid: str) -> Optional[str]:
    """
    Fetch the full-text XML for a PMCID.
    OPTIMIZATION: Uses the global 'http' session for connection pooling.
    """
    url = EUROPE_PMC_FULLTEXT_URL.format(pmcid=pmcid)
    try:
        # Uses global session (Keep-Alive)
        resp = http.get(url, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text
        else:
            return None
    except Exception as e:
        print(f"    [Fulltext Error] PMCID={pmcid}, error={e}")
        return None

def process_single_article(article: Dict) -> Dict:
    """
    Worker function for a single article.
    """
    # --- POLITENESS: Sleep 0.5 to 1.0 second per thread ---
    time.sleep(random.uniform(0.5, 1.0))

    # Init metadata paths
    year = article.get("pubYear") or "unknown"
    year_dir = os.path.join(OUTPUT_ROOT, str(year))

    # Ensure directory exists (cheap check)
    try:
        os.makedirs(year_dir, exist_ok=True)
    except OSError:
        pass

    # Determine Base ID safely
    base_id = article.get("pmcid") or article.get("id")
    if not base_id:
        base_id = f"unknown_{uuid.uuid4().hex[:8]}"

    # Sanitize filename
    base_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(base_id))

    txt_path = os.path.join(year_dir, f"{base_id}.txt")

    pmcid = article.get("pmcid")
    is_open_access = str(article.get("isOpenAccess") or "").upper() == "Y"

    clean_text = None

    # Fetch XML if OA + PMCID
    if pmcid and is_open_access:
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
                except Exception as e:
                    print(f"    [File Write Error] path={txt_path}, error={e}")
                    clean_text = None

    # Enrich Metadata
    meta = article.copy()
    meta["txt_path"] = txt_path if clean_text else None
    meta["has_fulltext"] = bool(clean_text)

    # Keyword matching
    try:
        meta["matches_keywords"] = article_matches_keywords(meta, clean_text)
    except Exception as e:
        print(f"    [Keyword Match Error] id={base_id}, error={e}")
        meta["matches_keywords"] = False

    return meta

# -----------------------------
# Main Loop
# -----------------------------

def main():
    ensure_dir(OUTPUT_ROOT)

    print(f"Opening output files in {OUTPUT_ROOT}...")
    f_all = open(METADATA_ALL_JSONL, "a", encoding="utf-8")
    f_kw = open(METADATA_KW_JSONL, "a", encoding="utf-8")

    try:
        for year in range(RESUME_FROM_YEAR, END_YEAR + 1):
            print(f"\n=== Processing Year {year} ===")

            # Create directory once per year (main thread)
            year_dir = os.path.join(OUTPUT_ROOT, str(year))
            ensure_dir(year_dir)

            query = build_query_with_year(BASE_QUERY, year)
            cursor = "*"
            page_count = 0
            year_articles_count = 0

            while True:
                page_count += 1

                # 1. Fetch Page of Metadata
                data = fetch_page(cursor, PAGE_SIZE, query)

                if not data or "resultList" not in data:
                    print(f"  [Year {year}] No data returned or end of results at page {page_count}.")
                    break

                results = data.get("resultList", {}).get("result", [])
                if not results:
                    print(f"  [Year {year}] Empty result list at page {page_count}.")
                    break

                print(f"  [Year {year}] Processing page {page_count} ({len(results)} articles) with {MAX_WORKERS} workers...")

                # 2. Process Full Texts in Parallel
                processed_batch: List[Dict] = []
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(process_single_article, art): art for art in results}

                    for future in as_completed(futures):
                        try:
                            res = future.result()
                            processed_batch.append(res)
                        except Exception as e:
                            print(f"  [Error in worker] {e}")
                            traceback.print_exc()

                # 3. Write Metadata immediately to disk
                for meta in processed_batch:
                    try:
                        json_line = json.dumps(meta, ensure_ascii=False)
                        f_all.write(json_line + "\n")
                        if meta.get("matches_keywords"):
                            f_kw.write(json_line + "\n")
                    except Exception as e:
                        print(f"  [Write JSONL Error] {e}")

                # Flush to ensure data is saved
                try:
                    f_all.flush()
                    f_kw.flush()
                except Exception as e:
                    print(f"  [Flush Error] {e}")

                year_articles_count += len(processed_batch)

                # Update Cursor
                next_cursor = data.get("nextCursorMark")
                if not next_cursor or next_cursor == cursor:
                    print(f"  [Year {year}] No next cursor, stopping at page {page_count}.")
                    break
                cursor = next_cursor

            print(f"=== Finished {year}. Total articles processed: {year_articles_count} ===")

    finally:
        try:
            f_all.close()
            f_kw.close()
        except Exception as e:
            print(f"[File Close Error] {e}")
        print("\nDone. Files closed.")

if __name__ == "__main__":
    main()