#!/bin/bash
# ============================================================================
# build_entity_pipeline.sh
# ----------------------------------------------------------------------------
# End-to-end orchestrator for the microbiome-features-mapping pipeline.
# Fire once in tmux, come back to a finished entity_index.sqlite.
#
# Steps (each one is skipped if its output already exists):
#   1. pip install pyahocorasick + tqdm (idempotent)
#   2. download NCBI taxdump.tar.gz  (~50 MB compressed)
#   3. extract taxdump
#   4. download HMDB hmdb_metabolites.zip  (~1 GB compressed)
#   5. extract HMDB
#   6. compile_vocab.py  ->  data/vocab/master_vocab.jsonl
#   7. build_entity_index.py  ->  data/entity_index.sqlite
#   8. verification — prints row counts per category
#
# Usage:
#   # one-time:
#   chmod +x code/feature_mapping/build_entity_pipeline.sh
#
#   # fire-and-forget:
#   conda activate guylu_base
#   tmux new -s entity_pipeline
#   bash code/feature_mapping/build_entity_pipeline.sh
#   # Ctrl-b d  to detach
#
# Output:
#   - data/vocab/master_vocab.jsonl
#   - data/entity_index.sqlite
#   - data/entity_pipeline.log    (full log)
#
# Re-running is safe and fast: completed steps are skipped automatically.
# Force a step to re-run by deleting its output file.
# ============================================================================

set -euo pipefail

# ----- Config (edit if your paths differ) -----------------------------------
PROJECT_DIR="/home/elhanan/PROJECTS/CHERRY_PICKER_AR"
CORPUS_DIR="${PROJECT_DIR}/new_corpus"
PIPELINE_DIR="${PROJECT_DIR}/code/feature_mapping"

RAW_KBS_DIR="${PROJECT_DIR}/data/raw_kbs"
TAXDUMP_DIR="${RAW_KBS_DIR}/taxdump"
HMDB_XML="${RAW_KBS_DIR}/hmdb_metabolites.xml"

VOCAB_OUT="${PROJECT_DIR}/data/vocab/master_vocab.jsonl"
INDEX_OUT="${PROJECT_DIR}/data/entity_index.sqlite"

LOG_FILE="${PROJECT_DIR}/data/entity_pipeline.log"

# ----- Plumbing -------------------------------------------------------------
mkdir -p "${RAW_KBS_DIR}" "$(dirname "${VOCAB_OUT}")" "$(dirname "${INDEX_OUT}")" "$(dirname "${LOG_FILE}")"

# Tee everything to the log so we can come back later and see what happened
exec > >(tee -a "${LOG_FILE}") 2>&1

banner() {
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $1"
    echo "============================================================"
}

t_start=$(date +%s)

banner "build_entity_pipeline.sh — starting"
echo "Project root : ${PROJECT_DIR}"
echo "Corpus       : ${CORPUS_DIR}"
echo "Python       : $(which python)"
echo "Log file     : ${LOG_FILE}"

# ----- Sanity check: scripts exist ------------------------------------------
for f in "${PIPELINE_DIR}/compile_vocab.py" "${PIPELINE_DIR}/build_entity_index.py"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required script not found: ${f}" >&2
        exit 1
    fi
done

# ----- Step 1: install python deps ------------------------------------------
banner "STEP 1 / 8  —  pip install deps (idempotent)"
pip install pyahocorasick tqdm --break-system-packages

# ----- Step 2: download NCBI taxdump ----------------------------------------
banner "STEP 2 / 8  —  download NCBI taxdump"
if [ -f "${TAXDUMP_DIR}/names.dmp" ]; then
    echo "  taxdump already extracted, skipping download + extract"
else
    cd "${RAW_KBS_DIR}"
    if [ ! -f "taxdump.tar.gz" ]; then
        echo "  downloading taxdump.tar.gz..."
        wget --continue https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz
    else
        echo "  taxdump.tar.gz already on disk, skipping download"
    fi
fi

# ----- Step 3: extract NCBI taxdump -----------------------------------------
banner "STEP 3 / 8  —  extract NCBI taxdump"
if [ -f "${TAXDUMP_DIR}/names.dmp" ]; then
    echo "  already extracted, skipping"
else
    mkdir -p "${TAXDUMP_DIR}"
    tar -xzf "${RAW_KBS_DIR}/taxdump.tar.gz" -C "${TAXDUMP_DIR}"
    echo "  extracted to ${TAXDUMP_DIR}"
fi

# ----- Step 4: download HMDB ------------------------------------------------
banner "STEP 4 / 8  —  download HMDB metabolites (~1 GB, may take a while)"
if [ -f "${HMDB_XML}" ]; then
    echo "  hmdb_metabolites.xml already extracted, skipping download + extract"
else
    cd "${RAW_KBS_DIR}"
    if [ ! -f "hmdb_metabolites.zip" ]; then
        echo "  downloading hmdb_metabolites.zip..."
        wget --continue https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip
    else
        echo "  hmdb_metabolites.zip already on disk, skipping download"
    fi
fi

# ----- Step 5: extract HMDB -------------------------------------------------
banner "STEP 5 / 8  —  extract HMDB"
if [ -f "${HMDB_XML}" ]; then
    echo "  already extracted, skipping"
else
    cd "${RAW_KBS_DIR}"
    unzip -o hmdb_metabolites.zip
    echo "  extracted to ${HMDB_XML}"
fi

# ----- Step 6: compile master vocabulary ------------------------------------
banner "STEP 6 / 8  —  compile master vocabulary (NCBI + KEGG + HMDB)"
if [ -f "${VOCAB_OUT}" ]; then
    echo "  master_vocab.jsonl already exists at ${VOCAB_OUT}"
    echo "  skipping. Delete it to force a rebuild."
else
    cd "${PROJECT_DIR}"
    python "${PIPELINE_DIR}/compile_vocab.py" \
        --ncbi-dir "${TAXDUMP_DIR}" \
        --hmdb-xml "${HMDB_XML}" \
        --out      "${VOCAB_OUT}"
fi

# ----- Step 7: build entity index (corpus scan) -----------------------------
banner "STEP 7 / 8  —  build entity index over corpus (longest step)"
if [ -f "${INDEX_OUT}" ]; then
    echo "  entity_index.sqlite already exists at ${INDEX_OUT}"
    echo "  skipping. Delete it to force a rebuild."
else
    cd "${PROJECT_DIR}"
    python "${PIPELINE_DIR}/build_entity_index.py" \
        --vocab  "${VOCAB_OUT}" \
        --corpus "${CORPUS_DIR}" \
        --out    "${INDEX_OUT}"
fi

# ----- Step 8: quick verification -------------------------------------------
banner "STEP 8 / 8  —  verification"
echo "  category breakdown in ${INDEX_OUT}:"
sqlite3 "${INDEX_OUT}" \
    "SELECT category, COUNT(*) AS rows, COUNT(DISTINCT pmcid) AS articles
     FROM article_entities GROUP BY category ORDER BY rows DESC;"

echo ""
echo "  total unique articles with at least one entity:"
sqlite3 "${INDEX_OUT}" \
    "SELECT COUNT(DISTINCT pmcid) FROM article_entities;"

echo ""
echo "  total unique canonical_ids referenced:"
sqlite3 "${INDEX_OUT}" \
    "SELECT COUNT(DISTINCT canonical_id) FROM article_entities;"

# ----- Done -----------------------------------------------------------------
t_end=$(date +%s)
elapsed=$((t_end - t_start))
hh=$((elapsed / 3600))
mm=$(( (elapsed % 3600) / 60 ))
ss=$((elapsed % 60))

banner "PIPELINE DONE in ${hh}h ${mm}m ${ss}s"
echo "Outputs:"
echo "  vocab : ${VOCAB_OUT}"
echo "  index : ${INDEX_OUT}"
echo "  log   : ${LOG_FILE}"
echo ""
echo "Next code tasks (small):"
echo "  - rag/feature_filter.py        eligible_pmcids(features, thresholds)"
echo "  - rag/entity_resolver.py       user term -> canonical_id"
echo "  - rag/restrict_to_pmcids in retrieve.search()"
echo "  - rag/ask_group.py             CLI front-end"
