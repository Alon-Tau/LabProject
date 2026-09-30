# CherryPicker

A retrieval-augmented generation (RAG) system for querying a large biomedical
literature corpus, built around a two-stage "bouncer + RAG" architecture and
evaluated against a plain (no-retrieval) LLM baseline on a hand-curated
holdout question set. See the accompanying paper for the full method and
results.

This repo holds the core system and the eval harness that produced the
paper's results. It intentionally does not include: the corpus, vector
store, or entity index (data, not code); the eval question set, ground
truth, or scored run outputs (see the paper's data statement); or the
project's exploratory/diagnostic/analysis scripts, which exist but aren't
part of the reproducible pipeline.

## Layout

- `pipeline/` — corpus construction and the OpenAI Batch embedding pipeline,
  run in this order:
  1. `build_corpus.py` — pulls full text from Europe PMC for a configured
     journal set (`--journal-set seed|jcr_expansion|pnas|all`). Safe to
     re-run: it dedupes against already-downloaded articles, so extending
     the corpus with a new journal set never re-downloads or duplicates
     what's already there.
  2. `paragraph_chunking.py` — splits corpus articles into paragraph-based
     chunks.
  3. `adjust_to_openai.py` → `submit_embedding_batches.py` →
     `retrieve_embedding_batches.py` — embed chunks via the OpenAI Batch
     API.
  4. `ingest_to_chroma.py` — loads the embeddings into the Chroma vector
     store.
- `bouncer/` — the Stage-1 feature-based filter and everything that backs
  it: `bouncer.py` (a standalone brute-force feature scanner) and
  `feature_filter.py` (the SQL-indexed lookup `rag/ask_group.py` actually
  calls at query time) are two independent implementations of the same
  idea, not one calling the other. `entity_resolver.py` resolves
  user-typed feature names to canonical IDs. `compile_vocab.py` +
  `build_entity_index.py` (orchestrated by `build_entity_pipeline.sh`)
  build the SQLite entity index from NCBI Taxonomy / KEGG / HMDB
  (`vocab/`).
- `rag/` — the query engine (`ask.py`, `ask_group.py`, `answer.py`,
  `retrieve.py`, `rerank.py`, `prompts.py`, `llm.py`, `embeddings.py`) and
  the eval harness that produced the paper's results (`eval_all.py`,
  `eval_run.py`, `judge.py`, `run_eval_pipeline.py`, `mmr_ab_test.py`,
  `summarize.py`, `report.py`, `cost_tracker.py`).

## Pipeline order

1. `pipeline/build_corpus.py` (repeat with each `--journal-set` you need).
2. `pipeline/paragraph_chunking.py`.
3. `pipeline/adjust_to_openai.py` → `pipeline/submit_embedding_batches.py`
   → `pipeline/retrieve_embedding_batches.py`.
4. `pipeline/ingest_to_chroma.py`.
5. `bouncer/build_entity_pipeline.sh`.
6. `rag/ask.py` (single query), `rag/ask_group.py` (bouncer-filtered
   query), or `rag/run_eval_pipeline.py` (full eval run).

## Setup

```
pip install -r requirements.txt
```

Requires `OPENAI_API_KEY` (embeddings, GPT models) and `ANTHROPIC_API_KEY`
(Claude models) as environment variables. `bouncer/` additionally needs a
local NCBI Taxonomy dump and HMDB metabolite XML (see
`build_entity_pipeline.sh`).
