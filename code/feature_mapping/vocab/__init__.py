"""
vocab/  — Biomedical vocabulary compilation from canonical KBs.

Modules:
    master           - Entity dataclass + utilities, alias cleaning
    ncbi_taxonomy    - NCBI Taxonomy dump parser (species names + synonyms)
    kegg             - KEGG REST API fetcher (compounds, pathways, modules)
    hmdb             - HMDB XML parser (metabolites with extensive synonyms)

Run via compile_vocab.py (one directory up).
"""

from .master import Entity, write_jsonl, read_jsonl, clean_aliases

__all__ = ["Entity", "write_jsonl", "read_jsonl", "clean_aliases"]
