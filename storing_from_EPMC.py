import requests
import time
import json
from typing import List, Dict

# -----------------------------
# Config
# -----------------------------

# Europe PMC Articles API base URL
EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Query for microbiome-related, peer-reviewed, open access journal articles
# You can tweak this query later.
QUERY = (
    '(microbiome OR "gut microbiota" OR metagenomics OR "microbiome bioinformatics" OR metabolomics OR metagenomic) '
    'AND OPEN_ACCESS:Y '
    'AND PUB_TYPE:"Journal Article" '
    'AND NOT PREPRINT:Y '
    'AND ('
    'JOURNAL:"Microbiome" OR '
    'JOURNAL:"Gut" OR '
    'JOURNAL:"The ISME Journal" OR '
    'JOURNAL:"Nature Microbiology" OR '
    'JOURNAL:"Cell Host & Microbe" OR '
    'JOURNAL:"Frontiers in Microbiology" OR '
    'JOURNAL:"Frontiers in Cellular and Infection Microbiology" OR '
    'JOURNAL:"Frontiers in Microbiomes" OR '
    'JOURNAL:"Environmental Microbiology" OR '
    'JOURNAL:"Environmental Microbiology Reports" OR '
    'JOURNAL:"Applied and Environmental Microbiology" OR '
    'JOURNAL:"BMC Microbiology" OR '
    'JOURNAL:"BMC Bioinformatics" OR '
    'JOURNAL:"Bioinformatics" OR '
    'JOURNAL:"PLOS Computational Biology" OR '
    'JOURNAL:"PLOS ONE" OR '
    'JOURNAL:"Nature Communications" OR '
    'JOURNAL:"Scientific Reports" '
    ')'
)


# Year filter example: 2023–2025 (you can change this)
YEAR_FILTER = "2020:2020"

# Maximum number of records you want to fetch in total
MAX_RECORDS = 200

# Page size (Europe PMC allows up to ~1000 per page, 100–500 זה נוח)
PAGE_SIZE = 200

# Output file
OUTPUT_JSON_PATH = "europe_pmc_microbiome_articles.json"


# -----------------------------
# Helper functions
# -----------------------------

def build_query_with_year(base_query: str, year_range: str) -> str:
    """
    Temporarily: use a single year to confirm PUB_YEAR works.
    Example: year_range="2023:2023" -> PUB_YEAR:2023
    """
    start_year, end_year = year_range.split(":")
    if start_year == end_year:
        return f'{base_query} AND PUB_YEAR:{start_year}'
    else:
        # For now, just use the start year to debug
        return f'{base_query} AND PUB_YEAR:{start_year}'


def fetch_page(cursor_mark: str, page_size: int, query: str, result_type: str = "core") -> Dict:
    """
    Fetch a single page of results from Europe PMC using the cursorMark mechanism.
    result_type: "core" (with abstractText) or "lite" (with journalTitle, pubType, etc.).
    """
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "cursorMark": cursor_mark,  # "*" for first page
        "resultType": result_type,
    }
    response = requests.get(EUROPE_PMC_API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()



from typing import Optional

def parse_articles_from_response(
    data: Dict,
    lite_map: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """
    Extract a clean list of articles from the Europe PMC JSON response.
    Optionally enrich with journal/pub_type from a lite result map.
    """
    results = data.get("resultList", {}).get("result", [])
    articles: List[Dict] = []

    for r in results:
        art_id = r.get("id")

        lite_rec = lite_map.get(art_id) if lite_map else None

        # Prefer core, fall back to lite
        journal = r.get("journalTitle") or (lite_rec.get("journalTitle") if lite_rec else None)
        pub_type = r.get("pubType") or (lite_rec.get("pubType") if lite_rec else None)

        article = {
            "id": art_id,                  # Europe PMC internal id
            "source": r.get("source"),     # e.g. "MED" / "PMC"
            "pmid": r.get("pmid"),
            "pmcid": r.get("pmcid"),
            "doi": r.get("doi"),
            "title": r.get("title"),

            # abstract from core
            "abstract": r.get("abstractText"),

            # enriched fields
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


# -----------------------------
# Main fetch logic
# -----------------------------

def fetch_microbiome_articles(
    base_query: str,
    year_range: str,
    max_records: int,
    page_size: int = 200,
    sleep_sec: float = 0.2,
) -> List[Dict]:
    """
    Fetch microbiome-related, peer-reviewed, open access articles from Europe PMC.
    Uses cursor-based pagination to retrieve up to max_records results.
    For each page, fetches core (with abstract) and lite (with journal/pub_type)
    and merges them.
    """
    final_query = build_query_with_year(base_query, year_range)
    print(f"Final Europe PMC query:\n  {final_query}\n")

    cursor_mark = "*"
    all_articles: List[Dict] = []
    total_fetched = 0

    while True:
        print(f"Requesting page with cursorMark={cursor_mark}, "
              f"current total={total_fetched}...")

        # 1) Core results (include abstractText)
        core_data = fetch_page(
            cursor_mark=cursor_mark,
            page_size=page_size,
            query=final_query,
            result_type="core",
        )

        # 2) Lite results (journalTitle, pubType, etc.) for same page
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
            print(f"Warning: failed to fetch lite data for this page: {e}")
            lite_map = {}

        # Cursor/total from core
        next_cursor = core_data.get("nextCursorMark")
        hit_count = int(core_data.get("hitCount", 0))

        # How many we still need
        remaining = max_records - total_fetched
        if remaining <= 0:
            break

        # 3) Parse & enrich
        page_articles = parse_articles_from_response(core_data, lite_map=lite_map)

        if not page_articles:
            print("No more articles in this page. Stopping.")
            break

        # Don't exceed max_records
        page_articles = page_articles[:remaining]

        all_articles.extend(page_articles)
        total_fetched += len(page_articles)

        print(f"Fetched {len(page_articles)} articles, total so far: {total_fetched}/{hit_count}")

        if total_fetched >= max_records or total_fetched >= hit_count:
            print("Reached max_records or hitCount limit. Stopping.")
            break

        if not next_cursor or next_cursor == cursor_mark:
            print("No nextCursorMark (or same as current). Stopping.")
            break

        cursor_mark = next_cursor
        time.sleep(sleep_sec)

    return all_articles[:max_records]


def save_articles_to_json(articles: List[Dict], path: str) -> None:
    """
    Save the list of articles as a JSON file.
    """
    print(f"Saving {len(articles)} articles to {path} ...")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print("Done!")

def get_hit_count(full_query: str) -> int:
    """
    Send a single request to Europe PMC and return how many articles
    match this query (hitCount).
    """
    params = {
        "query": full_query,
        "format": "json",
        "pageSize": 1,   # we only need 1 record, hitCount is global
    }
    response = requests.get(EUROPE_PMC_API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return int(data.get("hitCount", 0))

# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    # 1) Build the final query (with year filter for 2020)
    full_query = build_query_with_year(QUERY, YEAR_FILTER)
    print("Final Europe PMC query:")
    print("  ", full_query)

    # 2) (Optional) Ask Europe PMC how many articles match – just to know the scale
    total_hits = get_hit_count(full_query)
    print(f"\nTotal matching articles for 2020: {total_hits}")

    # 3) Decide how many to actually fetch
    to_fetch = min(total_hits, MAX_RECORDS)
    pages = (to_fetch + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"With MAX_RECORDS={MAX_RECORDS} and PAGE_SIZE={PAGE_SIZE}:")
    print(f"  You will fetch up to {to_fetch} articles in ~{pages} pages.\n")

    # 4) Fetch the actual articles
    articles_2020 = fetch_microbiome_articles(
        base_query=QUERY,
        year_range=YEAR_FILTER,   # "2020:2020"
        max_records=MAX_RECORDS,
        page_size=PAGE_SIZE,
    )

    # 5) Save them to JSON (separate file for 2020)
    OUTPUT_JSON_PATH_2020 = "europe_pmc_microbiome_articles_2020.json"
    save_articles_to_json(articles_2020, OUTPUT_JSON_PATH_2020)
