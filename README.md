# CHERRY_PICKER_AR
An advanced Retrieval-Augmented Generation (RAG) system designed to process, embed, and analyze large-scale scientific corpora.

## 🚀 Project Overview
This system manages the end-to-end pipeline for scientific data, from initial chunking to large-scale embedding using the OpenAI Batch API. It is specifically optimized to handle high-volume data within strict API rate limits.

## 🛠️ Core Pipeline
* **`build_corpus.py`**: The entry point for aggregating raw scientific journals into a unified structure.
* **`adjust_to_openai.py`**: Formats the corpus into JSONL shards ready for the OpenAI Batch API, including token-based splitting for long documents.
* **`submit_embedding_batches.py`**: Manages the upload and initiation of embedding tasks on OpenAI's servers.
* **`check_and_download_batches.py`**: Monitors batch status and automatically retrieves completed embeddings to the local server.

## ⚖️ Handling API Constraints (Tier 2 Optimization)
This project encountered a significant bottleneck with the OpenAI Tier 2 limit (20M enqueued tokens). To solve this, I developed a custom resharding workflow:
* **`reshard_existing.py`**: A non-destructive tool that splits large shards (~21M+ tokens) into smaller, safe units (~8.5M tokens).
* **`pre_flight_script.py`**: A rigorous validation tool that verifies JSON integrity, ensures unique IDs, and confirms exact token counts using `tiktoken` before submission.

## 📊 Analysis & Utility
* **`sainity_check.py`**: Quick validation of data consistency across different stages of the pipeline.
* **`fulltext_coverage_by_journal.py`**: Generates metrics on the distribution of data across various scientific publications.
* **`count_tokens.py`**: Provides precise token usage estimates for budgeting and limit management.
