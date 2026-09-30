"""
bouncer/  -- CherryPicker's Stage-1 feature filter and the entity-index
subsystem that backs it.

Modules:
    bouncer          - standalone brute-force feature scanner over raw
                        corpus text (early implementation of Stage 1)
    feature_filter   - SQL lookup against the built entity index; this is
                        what rag/ask_group.py actually calls at query time
    entity_resolver  - resolves user-typed feature names to canonical IDs
                        (bridges user vocabulary -> feature_filter's IDs)
    compile_vocab     - builds the master vocabulary from the KB parsers
                         under vocab/ (NCBI Taxonomy, KEGG, HMDB)
    build_entity_index - scans the corpus and writes the SQLite entity
                          index that feature_filter.py queries

Run build_entity_pipeline.sh once to go from raw KB downloads to a built
entity_index.sqlite.
"""

from .feature_filter import eligible_pmcids, per_article_breakdown
from .entity_resolver import resolve_features, DEFAULT_VOCAB_PATH

__all__ = ["eligible_pmcids", "per_article_breakdown", "resolve_features", "DEFAULT_VOCAB_PATH"]
