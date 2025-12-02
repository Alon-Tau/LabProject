import os
import time
import json
from typing import List, Dict, Optional
import requests
import xml.etree.ElementTree as ET
import html
import re
import traceback  # for nicer error logs

# -----------------------------
# Config
# -----------------------------

EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

JOURNALS = [
    "Microbiome",
    "Gut",
    "The ISME Journal",
    "Nature Microbiology",
    "Cell Host & Microbe",
    "Cell Systems",
    "mSystems",
    "Environmental Microbiology",
    "Environmental Microbiology Reports",
    "Applied and Environmental Microbiology",
    "BMC Microbiology",
    "BMC Bioinformatics",
    "Bioinformatics",
    "PLOS Computational Biology",
    "Nature Communications",
    "Nature Medicine",
    "Nature Biotechnology",
    "Nature Methods",
    "Nature",
    "Science",
    "Cell",
    "Proc Natl Acad Sci U S A",
]

journal_clause = " OR ".join([f'JOURNAL:\"{j}\"' for j in JOURNALS])

# ---- keyword terms (for the keyword-filtered corpus) ----
KEYWORD_TERMS = [
    # Core microbiome terms
    "microbiome",
    "gut microbiota",
    "metagenomics",
    "microbiome bioinformatics",
    "shotgun metagenomics",
    "functional metagenomics",
    "metagenomic sequencing",
    # Gene-related terms
    "gene",
    "genes",
    "gene expression",
    "transcriptomics",
    "metatranscriptomics",
    "functional genes",
    "gene regulation",
    "gene ontology",
    "kegg",
    "host gene expression",
    "microbiome-gene interactions",
    "gene pathways",
    # Metabolite / metabolism terms
    "metabolite",
    "metabolites",
    "metabolism",
    "metabolomics",
    "metabolic pathway",
    "metabolic network",
    "metabolomic profiling",
    "metabolite production",
    "short-chain fatty acids",
    "scfa",
    "butyrate",
    "acetate",
    "propionate",
    "metabolic reconstruction",
    "metabolic modeling",
    "microbiome-metabolome",
    "microbial metabolism",
    "metabolic capacity",
    "mimosa",
    "mimosa2",
    # Bacteria / taxa terms
    "bacteria",
    "bacterial",
    "microbiota",
    "microbial",
    "microbial community",
    "microbial species",
    "gut flora",
    "gut bacteria",
    "16s rna",
    "16s rrna",
    "taxonomy",
    "firmicutes",
    "bacteroidetes",
    "proteobacteria",
    "actinobacteria",
    "escherichia",
    "bifidobacterium",
    "lactobacillus",
    "akkermansia",
    "prevotella",
]

KEYWORD_BLOCK = "(" + " OR ".join(
    [f'\"{kw}\"' if " " in kw and not kw.startswith("16s") else kw for kw in KEYWORD_TERMS]
) + ")"

FILTERS = (
    'PUB_TYPE:"Journal Article" '
    'AND IN_EPMC:Y '
    'AND NOT PREPRINT:Y '
    f'AND ({journal_clause})'
)

BASE_QUERY = FILTERS

PAGE_SIZE = 200
MAX_RECORDS_PER_YEAR = 20000

OUTPUT_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_europepmc"

METADATA_ALL_JSONL = os.path.join(OUTPUT_ROOT, "metadata_all_1990_2025.jsonl")
METADATA_KW_JSONL  = os.path.join(OUTPUT_ROOT, "metadata_keywords_1990_2025.jsonl")

START_YEAR = 1990
END_YEAR = 2025

# For a fresh full run: RESUME_FROM_YEAR = START_YEAR
# To resume from 2020: set RESUME_FROM_YEAR = 2020
RESUME_FROM_YEAR = 2020

# -----------------------------
# Helper functions
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_query_with_year(base_query: str, year_range: str) -> str:
    start_year, end_year = year_range.split(":")
    start_year = start_year.strip()
    end_year = end_year.strip()

    if start_year == end_year:
        return f"{base_query} AND PUB_YEAR:{start_year}"
    else:
        return f"{base_query} AND PUB_YEAR:[{start_year} TO {end_year}]"


def fetch_page(cursor_mark: str, page_size: int, query: str, result_type: str = "core") -> Dict:
    """
    Fetch one page from Europe PMC.
    Retries on temporary server errors (503 etc.) and never crashes the whole script.
    On repeated failure, returns an empty result so the caller can stop the year gracefully.
    """
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "cursorMark": cursor_mark,
        "resultType": result_type,
    }

    max_retries = 5
    backoff_sec = 10  # wait time between retries; you can tweak this

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(EUROPE_PMC_API_URL, params=params, timeout=30)

            if resp.status_code == 503:
                print(f"  EuropePMC {result_type} page got 503 (attempt {attempt}/{max_retries}). "
                      f"Sleeping {backoff_sec} sec and retrying...")
                time.sleep(backoff_sec)
                continue

            if resp.status_code != 200:
                print(f"  EuropePMC {result_type} page HTTP {resp.status_code} "
                      f"(attempt {attempt}/{max_retries}).")
                time.sleep(backoff_sec)
                continue

            return resp.json()

        except requests.RequestException as e:
            print(f"  Error fetching EuropePMC {result_type} page "
                  f"(attempt {attempt}/{max_retries}): {e}")
            time.sleep(backoff_sec)

    # All retries failed
    print("  Giving up on this page after repeated errors. "
          "Returning empty result for this year-page.")
    return {
        "resultList": {"result": []},
        "hitCount": 0,
        "nextCursorMark": None,
    }


# ---------- XML cleaning helpers ----------

def _localname(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[0-9,\-\– ]+\]", "", text)
    text = text.strip()
    text = re.sub(r"^([A-Za-z]+)([A-Z])", r"\1 \2", text)
    text = re.sub(r"([.!?])([A-Z]{2,})([A-Z][a-z])", r"\1 \2 \3", text)
    return text


def _get_front(root: ET.Element) -> ET.Element:
    for child in root:
        if _localname(child.tag) == "front":
            return child
    return root


def _extract_titles(root: ET.Element) -> str:
    front = _get_front(root)
    titles = []
    for elem in front.findall(".//{*}article-title"):
        t = "".join(elem.itertext())
        t = _normalize_text(t)
        if t:
            titles.append(t)
            break
    return "\n".join(titles)


def _extract_abstracts(root: ET.Element) -> str:
    front = _get_front(root)
    abstracts = []
    for elem in front.findall(".//{*}abstract"):
        t = "".join(elem.itertext())
        t = _normalize_text(t)
        if t:
            abstracts.append(t)
    return "\n\n".join(abstracts)


UNWANTED_SECTION_TITLES = {
    "acknowledgements",
    "acknowledgments",
    "funding",
    "author contributions",
    "authors contributions",
    "authors’ contributions",
    "authors' contributions",
    "data availability",
    "availability of data",
    "supplementary material",
    "supplementary materials",
    "competing interests",
    "conflict of interest",
    "publishers note",
    "publisher's note",
    "publisher’s note",
    "references",
}


def _should_skip_section(title_text: str) -> bool:
    t = title_text.strip().lower()
    t_simple = re.sub(r"[^a-z\s]", "", t)
    return t_simple in UNWANTED_SECTION_TITLES


def _extract_body_from_sec(sec: ET.Element, pieces: list) -> None:
    title_elem = None
    for child in sec:
        if _localname(child.tag) == "title":
            title_elem = child
            break

    if title_elem is not None:
        title_text = "".join(title_elem.itertext())
        title_text = _normalize_text(title_text)
        if title_text and _should_skip_section(title_text):
            return
        if title_text:
            pieces.append(title_text)

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
        elif lname in {"fig", "figure", "table-wrap", "caption", "tbl"}:
            continue


def extract_clean_text_from_xml(xml_text: str, pmcid: Optional[str] = None) -> Optional[str]:
    """Parse and clean full text from an XML string. Returns None on parse failure."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"    WARNING: failed to parse XML for {pmcid or 'unknown PMCID'}: {e}. Skipping full text.")
        return None

    pieces = []

    title_block = _extract_titles(root)
    if title_block:
        pieces.append(title_block)

    abstract_block = _extract_abstracts(root)
    if abstract_block:
        pieces.append(abstract_block)

    body = None
    for elem in root.iter():
        if _localname(elem.tag) == "body":
            body = elem
            break

    if body is not None:
        for child in body:
            if _localname(child.tag) == "sec":
                _extract_body_from_sec(child, pieces)

    return "\n\n".join(pieces)


def extract_journal_from_xml(xml_text: str, pmcid: Optional[str] = None) -> Optional[str]:
    """Extract journal title from XML. Returns None on parse failure."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"    WARNING: failed to parse XML for {pmcid or 'unknown PMCID'} when extracting journal: {e}")
        return None

    for elem in root.iter():
        if _localname(elem.tag) == "journal-title":
            t = "".join(elem.itertext())
            t = _normalize_text(t)
            if t:
                return t
    return None


# ---------- Keyword matching ----------

def article_matches_keywords(article: Dict, clean_text: Optional[str] = None) -> bool:
    title = article.get("title") or ""
    abstract = article.get("abstract") or ""
    haystack = (title + " " + abstract).lower()

    if clean_text:
        haystack += " " + clean_text.lower()

    for kw in KEYWORD_TERMS:
        if kw in haystack:
            return True
    return False


# ---------- Europe PMC full text ----------

def fetch_fulltext_xml(pmcid: str) -> Optional[str]:
    if not pmcid:
        return None
    url = EUROPE_PMC_FULLTEXT_URL.format(pmcid=pmcid)
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            print(f"    Full text not available for {pmcid} (status {resp.status_code})")
            return None

        text = resp.text
        if not text or not text.strip():
            print(f"    Empty full text response for {pmcid}")
            return None

        # Basic check for HTML error pages instead of XML
        head = text.lstrip()[:200].lower()
        if head.startswith("<!doctype html") or head.startswith("<html"):
            print(f"    Got HTML instead of XML for {pmcid}, skipping.")
            return None

        return text

    except requests.RequestException as e:
        print(f"    Error fetching full text for {pmcid}: {e}")
        return None


# ---------- Parsing search results ----------

def parse_articles_from_response(
    data: Dict,
    lite_map: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    results = data.get("resultList", {}).get("result", [])
    articles: List[Dict] = []

    for r in results:
        art_id = r.get("id")
        lite_rec = lite_map.get(art_id) if lite_map else None

        journal = r.get("journalTitle") or (lite_rec.get("journalTitle") if lite_rec else None)
        pub_type = r.get("pubType") or (lite_rec.get("pubType") if lite_rec else None)

        article = {
            "id": art_id,
            "source": r.get("source"),
            "pmid": r.get("pmid"),
            "pmcid": r.get("pmcid"),
            "doi": r.get("doi"),
            "title": r.get("title"),
            "abstract": r.get("abstractText"),
            "journal": journal,
            "pub_year": r.get("pubYear"),
            "author_string": r.get("authorString"),
            "is_open_access": r.get("isOpenAccess") == "Y",
            "pub_type": pub_type,
            "is_preprint": r.get("pubType") == "preprint" or r.get("isPreprint") == "Y",
            "europe_pmc_url": (
                f"https://europepmc.org/article/{r.get('source')}/{art_id}"
                if r.get("source") and art_id else None
            ),
        }
        articles.append(article)

    return articles


def fetch_articles_for_year(
    base_query: str,
    year: int,
    max_records: int,
    page_size: int = 200,
    sleep_sec: float = 0.2,
):
    year_range = f"{year}:{year}"
    final_query = build_query_with_year(base_query, year_range)
    print(f"\n=== Year {year} ===")
    print(f"Query: {final_query}")

    cursor_mark = "*"
    total_fetched = 0

    while True:
        print(f"  Requesting page with cursorMark={cursor_mark}, total so far={total_fetched}")

        core_data = fetch_page(
            cursor_mark=cursor_mark,
            page_size=page_size,
            query=final_query,
            result_type="core",
        )

        try:
            lite_data = fetch_page(
                cursor_mark=cursor_mark,
                page_size=page_size,
                query=final_query,
                result_type="lite",
            )
            lite_results = lite_data.get("resultList", {}).get("result", [])
            lite_map = {r.get("id"): r for r in lite_results}
        except Exception as e:
            print(f"  Warning: failed to fetch lite data for this page: {e}")
            lite_map = {}

        next_cursor = core_data.get("nextCursorMark")
        hit_count = int(core_data.get("hitCount", 0))

        remaining = max_records - total_fetched
        if remaining <= 0:
            break

        page_articles = parse_articles_from_response(core_data, lite_map=lite_map)
        if not page_articles:
            print("  No more articles in this page. Stopping year.")
            break

        page_articles = page_articles[:remaining]

        for art in page_articles:
            yield art

        total_fetched += len(page_articles)
        print(f"  Fetched {len(page_articles)} articles; total {total_fetched}/{hit_count}")

        if total_fetched >= max_records or total_fetched >= hit_count:
            print("  Reached max_records or hitCount. Stopping year.")
            break

        if not next_cursor or next_cursor == cursor_mark:
            print("  No nextCursorMark (or unchanged). Stopping year.")
            break

        cursor_mark = next_cursor
        time.sleep(sleep_sec)


# ---------- Per-article processing ----------

def process_article_to_corpus(
    article: Dict,
    output_root: str,
    sleep_sec_xml: float = 0.2,
) -> (Dict, bool):
    """
    For a single article:
      - if OA + PMCID -> fetch XML, clean text, save ONLY the cleaned TXT
      - always return a metadata dict including file paths (if any)
      - also return: matches_keywords (bool)
    """
    year = article.get("pub_year") or "unknown"
    year_dir = os.path.join(output_root, str(year))
    ensure_dir(year_dir)

    base_id = article.get("pmcid") or (
        f"PMID_{article['pmid']}" if article.get("pmid") else f"ID_{article['id']}"
    )

    xml_path = None
    text_path = None
    has_full_text = False
    clean_text_for_match = None

    if article.get("is_open_access") and article.get("pmcid"):
        pmcid = article["pmcid"]
        print(f"    OA article, fetching full text for {pmcid} ...")
        try:
            xml_text = fetch_fulltext_xml(pmcid)
            if xml_text:
                if not article.get("journal"):
                    j_from_xml = extract_journal_from_xml(xml_text, pmcid=pmcid)
                    if j_from_xml:
                        article["journal"] = j_from_xml

                clean_text = extract_clean_text_from_xml(xml_text, pmcid=pmcid)
                if clean_text:
                    clean_text_for_match = clean_text
                    text_path = os.path.join(year_dir, f"{base_id}.txt")
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(clean_text)
                    has_full_text = True
                else:
                    print(f"    WARNING: no clean text extracted for {pmcid}, skipping TXT save.")
            else:
                print(f"    Could not fetch usable XML for {pmcid}, keeping metadata only.")

            time.sleep(sleep_sec_xml)

        except Exception as e:
            print(f"    ERROR while processing full text for {pmcid}: {e}")
            traceback.print_exc()

    matches_kw = article_matches_keywords(article, clean_text_for_match)

    meta_rec = {
        "id": article.get("id"),
        "source": article.get("source"),
        "pmid": article.get("pmid"),
        "pmcid": article.get("pmcid"),
        "doi": article.get("doi"),
        "title": article.get("title"),
        "abstract": article.get("abstract"),
        "journal": article.get("journal"),
        "pub_year": article.get("pub_year"),
        "author_string": article.get("author_string"),
        "is_open_access": article.get("is_open_access"),
        "pub_type": article.get("pub_type"),
        "is_preprint": article.get("is_preprint"),
        "europe_pmc_url": article.get("europe_pmc_url"),
        "has_full_text": has_full_text,
        "xml_path": xml_path,
        "text_path": text_path,
    }

    return meta_rec, matches_kw


# -----------------------------
# Main driver
# -----------------------------

def main():
    ensure_dir(OUTPUT_ROOT)

    # Global counters for final summary
    total_processed = 0           # metadata lines written
    total_kw_matches = 0
    total_with_fulltext = 0       # has_full_text == True
    total_skipped = 0             # articles that raised exception in main loop

    # If resuming from the very beginning, overwrite files.
    # If starting mid-range (e.g. 2020), append to existing metadata.
    mode = "w" if RESUME_FROM_YEAR == START_YEAR else "a"

    with open(METADATA_ALL_JSONL, mode, encoding="utf-8") as meta_all, \
         open(METADATA_KW_JSONL, mode, encoding="utf-8") as meta_kw:

        for year in range(RESUME_FROM_YEAR, END_YEAR + 1):
            print(f"\n=== Processing year {year} ===")
            count_year = 0
            count_kw_year = 0

            for art in fetch_articles_for_year(
                base_query=BASE_QUERY,
                year=year,
                max_records=MAX_RECORDS_PER_YEAR,
                page_size=PAGE_SIZE,
                sleep_sec=0.2,
            ):
                try:
                    meta_rec, matches_kw = process_article_to_corpus(
                        art,
                        output_root=OUTPUT_ROOT,
                        sleep_sec_xml=0.2,
                    )
                except Exception as e:
                    # Extra safety: never let a single bad article kill the run
                    print(f"  ERROR processing article {art.get('id')} ({art.get('pmcid')}): {e}")
                    traceback.print_exc()
                    total_skipped += 1
                    continue

                # Update counts
                count_year += 1
                total_processed += 1

                if meta_rec.get("has_full_text"):
                    total_with_fulltext += 1

                # Write metadata (all articles)
                meta_all.write(json.dumps(meta_rec, ensure_ascii=False) + "\n")

                # Keyword-positive subset
                if matches_kw:
                    meta_kw.write(json.dumps(meta_rec, ensure_ascii=False) + "\n")
                    count_kw_year += 1
                    total_kw_matches += 1

                if count_year % 1000 == 0:
                    meta_all.flush()
                    meta_kw.flush()
                    print(f"  Flushed metadata after {count_year} articles for {year} "
                          f"(keyword matches so far this year: {count_kw_year})")

            print(f"Year {year}: processed {count_year} articles, "
                  f"{count_kw_year} matched keywords.\n")

        meta_all.flush()
        meta_kw.flush()

    print(f"All metadata written to:\n  {METADATA_ALL_JSONL}\n  {METADATA_KW_JSONL}")

    # -------- Final global summary --------
    print("\n==================== FINAL SUMMARY ====================")
    print(f"Total processed articles (metadata written): {total_processed}")
    print(f"Total articles with full text saved (.txt):  {total_with_fulltext}")
    print(f"Total keyword-matching articles:             {total_kw_matches}")
    print(f"Total articles skipped due to errors:        {total_skipped}")
    print("=======================================================")


if __name__ == "__main__":
    main()
