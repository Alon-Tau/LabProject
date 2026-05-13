import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any

# ============ CONFIG ============
print(">>> pilot_chunking.py STARTED")
CORPUS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus"

METADATA_PATH = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus/metadata_all.jsonl"

KW_METADATA_PATH = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus/metadata_kw.jsonl"

CHUNKS_ROOT = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_chunks"

YEAR = 2019           # year to process
MAX_ARTICLES = 100  # first N .txt files (for fulltext; metadata-only is not limited)

# "Medium" chunk size
TARGET_WORDS = 650
OVERLAP_WORDS = 150   # not used in paragraph mode, but kept for flexibility

# Paragraph-based chunking mode ON
CHUNK_BY_PARAGRAPH = True

# ------------ JOURNAL TIERS ------------

# Tier 1 = core microbiome / microbial ecology
# Tier 2 = methods / bioinformatics / focused biomedical
# Tier 3 = very broad general journals (kw-filtered only)
JOURNAL_TIER: Dict[str, int] = {
    # Tier 1 – core microbiome / host-microbe / microbial ecology
    "Microbiome": 1,
    "Gut": 1,
    "The ISME Journal": 1,
    "Nature Microbiology": 1,
    "Cell Host & Microbe": 1,
    "Cell Systems": 1,
    "mSystems": 1,
    "Environmental Microbiology": 1,
    "Environmental Microbiology Reports": 1,
    "Applied and Environmental Microbiology": 1,
    "BMC Microbiology": 1,

    # Tier 2 – bioinformatics & focused biomedical / methods
    "BMC Bioinformatics": 2,
    "Bioinformatics": 2,
    "PLOS Computational Biology": 2,
    "PLoS Computational Biology": 2,
    "Nature Medicine": 2,
    "Nature Biotechnology": 2,
    "Nature Methods": 2,

    # Tier 3 – very broad general journals
    "Nature": 3,
    "Nature Communications": 3,
    "Science": 3,
    "Cell": 3,
    "Proc Natl Acad Sci U S A": 3,
    "PNAS": 3,
}

DEFAULT_TIER = 3  # everything not in JOURNAL_TIER

# =======================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def norm_pmcid(raw: str) -> str:
    """
    Normalize PMCID so filename and metadata match more easily.
    """
    if not raw:
        return ""
    x = raw.strip().upper()
    if x.endswith(".TXT"):
        x = x[:-4]
    if x.startswith("PMC"):
        return x
    if x.isdigit():
        return "PMC" + x
    return x


def load_metadata_index(metadata_path: str) -> Dict[str, Dict]:
    """
    Load big metadata JSONL and index by normalized PMCID.
    """
    index: Dict[str, Dict] = {}
    total = 0

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Sometimes metadata is nested, e.g. {"record": {...}}
            if isinstance(obj, dict) and len(obj) == 1 and isinstance(list(obj.values())[0], dict):
                obj = list(obj.values())[0]

            pmcid_raw = (
                obj.get("pmcid")
                or obj.get("PMCID")
                or obj.get("pmcid_upper")
                or obj.get("pmcid_lower")
            )
            if not pmcid_raw:
                continue

            pmcid = norm_pmcid(pmcid_raw)
            index[pmcid] = obj
            total += 1

    print(f"Loaded metadata for {total} articles into index.")
    sample_keys = list(index.keys())[:5]
    print("Example PMCID keys in index:", sample_keys)
    return index


def load_kw_relevant_ids(path: str) -> set:
    """
    Load PMCID set from the 'keyword relevant' metadata JSONL.
    If the file is missing, returns an empty set.
    """
    ids = set()
    if not os.path.exists(path):
        print(f"⚠️ Keyword metadata file not found: {path}")
        print("   Tier 3 will effectively never be selected by keyword.")
        return ids

    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(rec, dict) and len(rec) == 1 and isinstance(list(rec.values())[0], dict):
                rec = list(rec.values())[0]

            pmcid_raw = (
                rec.get("pmcid")
                or rec.get("PMCID")
                or rec.get("pmcid_upper")
                or rec.get("pmcid_lower")
            )
            if not pmcid_raw:
                continue
            pmcid = norm_pmcid(pmcid_raw)
            ids.add(pmcid)
            total += 1

    print(f"Loaded {total} keyword-relevant article IDs.")
    return ids


def load_text(txt_path: str) -> str:
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"text file not found: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------- Chunking helpers ----------

def split_into_chunks_words(text: str, target_words: int, overlap_words: int) -> List[str]:
    """
    Simple word-based sliding window with overlap.
    """
    words = text.split()
    n = len(words)
    if n == 0:
        return []

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + target_words, n)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == n:
            break
        start = max(0, end - overlap_words)
    return chunks


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs using blank lines.
    """
    paras = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in paras if p.strip()]


def split_into_chunks_paragraphs(text: str, target_words: int) -> List[str]:
    """
    Chunk by grouping paragraphs until we reach ~target_words.
    No overlap here to keep logic simple and avoid duplicating big paragraphs.
    """
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for para in paragraphs:
        pw = len(para.split())

        # If a single paragraph is huge, just cut it with the word-based splitter
        if pw > target_words * 1.5:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0

            long_chunks = split_into_chunks_words(para, target_words, 0)
            chunks.extend(long_chunks)
            continue

        if current_words + pw <= target_words:
            current.append(para)
            current_words += pw
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [para]
            current_words = pw

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def extract_journal(meta: Dict) -> Optional[str]:
    return (
        meta.get("journal")
        or meta.get("journal_title")
        or meta.get("journalTitle")
        or meta.get("journal_name")
        or meta.get("source")
    )


def extract_title(meta: Dict) -> Optional[str]:
    return (
        meta.get("title")
        or meta.get("article_title")
        or meta.get("full_title")
        or meta.get("title_full")
    )


def extract_year(meta: Dict) -> Optional[int]:
    candidates = [
        meta.get("year"),
        meta.get("pubYear"),
        meta.get("publication_year"),
        meta.get("pub_year"),
    ]
    for c in candidates:
        if c is None:
            continue
        try:
            return int(c)
        except (TypeError, ValueError):
            continue
    return None


def extract_abstract(meta: Dict) -> Optional[str]:
    return (
        meta.get("abstractText")
        or meta.get("abstract")
        or meta.get("abstract_text")
    )


def extract_keywords(meta: Dict) -> Optional[str]:
    # keywords could be in many formats: list of dicts, list of strings, etc.
    kw = meta.get("keywords") or meta.get("keyword")
    if not kw:
        return None
    if isinstance(kw, str):
        return kw
    if isinstance(kw, list):
        # flatten list items to strings
        parts = []
        for item in kw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for k in ("term", "keyword", "kw"):
                    if k in item and isinstance(item[k], str):
                        parts.append(item[k])
                        break
        return "; ".join(parts) if parts else None
    return None


def build_metadata_text(meta: Dict) -> str:
    """
    Build a synthetic "text" from metadata for non-open-access articles.
    """
    parts: List[str] = []

    title = extract_title(meta)
    if title:
        parts.append(f"Title: {title}")

    journal = extract_journal(meta)
    year = extract_year(meta)
    journal_bits = []
    if journal:
        journal_bits.append(journal)
    if year is not None:
        journal_bits.append(str(year))
    if journal_bits:
        parts.append("Journal / Year: " + ", ".join(journal_bits))

    abstract = extract_abstract(meta)
    if abstract:
        parts.append("Abstract: " + abstract)

    kw_text = extract_keywords(meta)
    if kw_text:
        parts.append("Keywords: " + kw_text)

    return "\n\n".join(parts).strip()


# ---------- Relevance / tier logic ----------

def get_journal_tier(journal: Optional[str]) -> int:
    if not journal:
        return DEFAULT_TIER
    j = journal.strip()
    return JOURNAL_TIER.get(j, DEFAULT_TIER)


def should_chunk_article(
    pmcid: str,
    journal: Optional[str],
    kw_relevant_ids: set
) -> Tuple[bool, int, bool]:
    """
    Decide if an article should be chunked at all.

    Returns (should_chunk, tier, keyword_relevant_flag)
    """
    tier = get_journal_tier(journal)
    keyword_relevant = pmcid in kw_relevant_ids if pmcid else False

    # Tier 1 & 2: always chunk
    if tier in (1, 2):
        return True, tier, keyword_relevant

    # Tier 3: only if keyword-relevant
    if tier == 3:
        if keyword_relevant:
            return True, tier, keyword_relevant
        else:
            return False, tier, keyword_relevant

    return False, tier, keyword_relevant


def chunk_text_for_article(
    year: int,
    pmcid: str,
    journal: Optional[str],
    title: Optional[str],
    text: str,
    tier: int,
    keyword_relevant: bool,
    source_type: str,
) -> List[Dict[str, Any]]:
    """
    Common function that creates chunk objects from text.
    """
    if not text.strip():
        return []

    if CHUNK_BY_PARAGRAPH:
        raw_chunks = split_into_chunks_paragraphs(text, TARGET_WORDS)
    else:
        raw_chunks = split_into_chunks_words(text, TARGET_WORDS, OVERLAP_WORDS)

    chunk_objs: List[Dict[str, Any]] = []

    for idx, ch_text in enumerate(raw_chunks, start=1):
        chunk_id = f"{year}_{pmcid}_{idx:04d}"
        chunk = {
            "chunk_id": chunk_id,
            "pmcid": pmcid,
            "year": year,
            "journal": journal,
            "title": title,
            "chunk_index": idx,
            "text": ch_text,
            "journal_tier": tier,
            "keyword_relevant": keyword_relevant,
            "source_type": source_type,  # "fulltext" or "metadata_only"
        }
        chunk_objs.append(chunk)

    return chunk_objs


# ---------- Main article-level functions ----------

def chunk_fulltext_article(
    year: int,
    txt_path: str,
    metadata_index: Dict[str, Dict],
    kw_relevant_ids: set
) -> List[Dict]:
    """
    Chunk an article that has a .txt fulltext file.
    Applies tier + keyword filtering.
    """
    filename = os.path.basename(txt_path)
    pmcid = norm_pmcid(os.path.splitext(filename)[0])

    meta = metadata_index.get(pmcid, {})
    journal = extract_journal(meta)
    title = extract_title(meta)

    should, tier, kw_rel = should_chunk_article(pmcid, journal, kw_relevant_ids)
    if not should:
        return []

    text = load_text(txt_path)

    chunks = chunk_text_for_article(
        year=year,
        pmcid=pmcid,
        journal=journal,
        title=title,
        text=text,
        tier=tier,
        keyword_relevant=kw_rel,
        source_type="fulltext",
    )
    return chunks


def chunk_metadata_only_article(
    year: int,
    pmcid: str,
    meta: Dict,
    kw_relevant_ids: set
) -> List[Dict]:
    """
    Chunk an article **only from metadata** (non-open-access / no fulltext .txt).
    Applies tier + keyword filtering.
    """
    journal = extract_journal(meta)
    title = extract_title(meta)

    should, tier, kw_rel = should_chunk_article(pmcid, journal, kw_relevant_ids)
    if not should:
        return []

    text = build_metadata_text(meta)
    if not text:
        return []

    chunks = chunk_text_for_article(
        year=year,
        pmcid=pmcid,
        journal=journal,
        title=title,
        text=text,
        tier=tier,
        keyword_relevant=kw_rel,
        source_type="metadata_only",
    )
    return chunks


# ---------- Orchestration ----------

def pilot_chunk_year(year: int, max_articles: Optional[int]):
    year_dir = os.path.join(CORPUS_ROOT, str(year))
    if not os.path.isdir(year_dir):
        raise FileNotFoundError(f"Year directory not found: {year_dir}")

    ensure_dir(CHUNKS_ROOT)
    out_path = os.path.join(CHUNKS_ROOT, f"{year}_pilot.jsonl")

    metadata_index = load_metadata_index(METADATA_PATH)
    kw_relevant_ids = load_kw_relevant_ids(KW_METADATA_PATH)

    all_files = [f for f in os.listdir(year_dir) if f.lower().endswith(".txt")]
    all_files.sort()
    if max_articles is not None:
        all_files = all_files[:max_articles]

    print(f"Year dir: {year_dir}")
    print(f"Found {len(all_files)} article text files (after MAX_ARTICLES filter).")
    if not all_files:
        print("No .txt files found – will only try metadata-only chunking.")

    total_chunks = 0
    total_articles_fulltext = 0
    total_articles_metadata_only = 0

    processed_pmcids = set()

    with open(out_path, "w", encoding="utf-8") as out_f:
        # ---- 1) Fulltext articles ----
        for filename in all_files:
            txt_path = os.path.join(year_dir, filename)
            pmcid = norm_pmcid(os.path.splitext(filename)[0])
            processed_pmcids.add(pmcid)

            chunks = chunk_fulltext_article(year, txt_path, metadata_index, kw_relevant_ids)

            if not chunks:
                print(f"Skipping fulltext {filename}: filtered out or empty / no chunks.")
                continue

            for ch in chunks:
                out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)
            total_articles_fulltext += 1
            print(f"Chunked fulltext {filename}: {len(chunks)} chunks")

        # ---- 2) Metadata-only articles (no .txt file) ----
        for pmcid, meta in metadata_index.items():
            if pmcid in processed_pmcids:
                continue

            meta_year = extract_year(meta)
            if meta_year != year:
                continue

            chunks = chunk_metadata_only_article(year, pmcid, meta, kw_relevant_ids)
            if not chunks:
                continue

            for ch in chunks:
                out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")

            total_chunks += len(chunks)
            total_articles_metadata_only += 1
            print(f"Chunked metadata-only {pmcid}: {len(chunks)} chunks")

    print(f"\nPilot finished for year {year}.")
    print(f"Fulltext articles processed: {total_articles_fulltext}")
    print(f"Metadata-only articles processed: {total_articles_metadata_only}")
    print(f"Total chunks written: {total_chunks}")
    print(f"Output file: {out_path}")


if __name__ == "__main__":
    pilot_chunk_year(YEAR, MAX_ARTICLES)
