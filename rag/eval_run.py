#!/usr/bin/env python3
"""
eval_run.py
-----------
Run ONE holdout eval question through either:
    mode "rag"     the CherryPicker RAG WITHOUT the feature filter (direct ask),
    mode "plain"   a no-retrieval LLM baseline (the model's own knowledge),

and return / save the model's structured answer as a single JSON object.

The model is asked to return exactly:
    {"components": {<heading>: [items...], ...}, "free_text": "..."}

We wrap that with run metadata and the retrieved chunks (empty for plain mode)
so a human can score `components` against the ground-truth key, using
`free_text` to adjudicate borderline cases.

Reuses the REAL pipeline (retrieve.search, prompts.build_rag_prompt,
llm.generate); the only addition is a JSON-format instruction appended to the
prompt, so the RAG answer stays authentic to the system.

CLI usage (from project root):
    python -m rag.eval_run --question code/rag/eval/q1_endometriosis.json \
        --mode rag --model gpt-4o-mini --k 8 --out code/rag/eval/runs/q1/rag.json

    Add --reranker to rescore the k retrieved chunks with a cross-encoder
    (rerank.rerank_chunks) and drop clear per-query outliers (rerank.filter_by_gap)
    before building the prompt -- see rerank.py for the rationale. No effect in
    "plain" mode (there are no chunks to rerank).

Programmatic usage (see eval_all.py):
    from rag.eval_run import run_question
    result = run_question(cfg, mode="rag", model="gpt-4o-mini", k=8)
    result = run_question(cfg, mode="rag", model="gpt-4o-mini", k=8, use_reranker=True)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from .retrieve import search
from .prompts  import build_rag_prompt
from .llm      import generate


def _skeleton(components):
    return {"components": {c: [] for c in components}, "free_text": ""}


def _format_instruction(components):
    skel = json.dumps(_skeleton(components), ensure_ascii=False, indent=2)
    return (
        "Structure your ENTIRE answer as a single JSON object exactly matching "
        "this schema (same keys, same order):\n\n" + skel + "\n\n"
        "Before composing your answer: if numbered sources are provided above, go "
        "through them ONE BY ONE, in order, and note every specific entity each "
        "individual source names that fits one of the component categories above "
        "(species, enzymes, metabolites, receptors, pathways, etc.). Do "
        "not stop after finding a few strong candidates, and do not skip a source "
        "just because an earlier source already covered similar ground -- every "
        "source gets its own pass. Then compose the components lists from that full "
        "review, so no individual source's named features are missed. (If no "
        "sources are provided and you are answering from your own knowledge, skip "
        "this step.)\n\n"
        "For each component, fill the list with the specific named entities you "
        "assert (species, enzymes, metabolites, receptors, pathways, etc.) -- but "
        "ONLY if that category is actually relevant to your answer, and ONLY if a "
        "source excerpt actually names the entity (never add a plausible-sounding "
        "entity no source named, even if it commonly appears in this context). "
        "Each list entry must be the entity NAME ONLY, plus its citation tag (see "
        "below) -- do NOT attach direction-of-change or any other qualifier to it. "
        "Write \"Lactobacillus [42]\", never \"Lactobacillus -- reduced [42]\" or "
        "\"Lactobacillus (decreased) [42]\". Direction, context, and mechanism belong "
        "exclusively in free_text -- see the CRITICAL rule below. These headings "
        "are not a mandatory checklist: do not force an entry into a component just "
        "to avoid leaving it empty, and do not pad a category with a plausible-sounding "
        "but unsupported entity. An empty list is the correct and expected output "
        "for a component your answer genuinely has nothing specific to report under, "
        "after you've completed the source-by-source review above. "
        "If numbered sources are provided above, tag EVERY entity in every component "
        "list with the source number(s) that support it, in the same [N] format used "
        "in free_text -- appended right after the entity itself, e.g. "
        "\"Faecalibacterium [3]\" or \"indoxyl sulfate [4, 11]\" if more than one source "
        "names it. An entity with no citable source number should not be in the list "
        "at all. (If no sources are provided and you are answering from your own "
        "knowledge, omit the [N] tags -- there is nothing to cite.)\n\n"
        "Put your full explanation in \"free_text\", including [N] citation markers "
        "where you relied on a numbered source. Write free_text as a direct answer "
        "to the question above -- not as a general summary of the topic or of the "
        "sources. Structure it around the question's own parts: which microbiota "
        "and/or metabolites have been reported, how each one is altered (direction "
        "of change), and what mechanism links it to the relevant biological pathway. "
        "A reader should be able to tell you are answering exactly what was asked, "
        "not describing the source article in general terms.\n\n"
        "CRITICAL: every entity that appears in a components list MUST also appear "
        "by name in free_text -- naming an entity in a components list but leaving "
        "it out of free_text counts as an incomplete answer for that entity. Before "
        "finalizing your answer, check the two against each other: every entity name "
        "in components should have a matching mention in free_text. Do not silently "
        "drop any of them.\n\n"
        "Because components entries never carry direction (see above), free_text is "
        "the only place a direction of change can be recorded. Stating it explicitly "
        "-- in plain words such as \"increased\", \"decreased\", \"reduced\", "
        "\"elevated\", or \"no significant change\" -- is strongly encouraged whenever "
        "the source material supports it, along with the context/mechanism that "
        "connects the entity to the pathway: a specific, directional claim is more "
        "useful and more informative than naming the entity alone. This is not a "
        "strict requirement for every single entity (some source material may not "
        "state a clear direction), but it should be your default whenever the "
        "evidence allows it.\n\n"
        "free_text should not be a short summary that only hits a few highlights -- "
        "it should read as a complete narrative covering every feature (species, "
        "enzyme, metabolite, receptor, pathway, etc.) you identified as relevant "
        "during the source-by-source review above, tying each one into the "
        "mechanism rather than just listing it. The components lists and free_text "
        "should therefore cover the same set of entities -- components as a bare, "
        "uncontextualized name list, free_text as the full prose account with "
        "direction and mechanism attached to each one. Output ONLY the JSON "
        "object, with no prose before or after it."
    )


def _extract_json(text):
    """Pull the first {...} JSON object out of a model response, tolerating ``` fences."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError("no JSON object found in model output")
    return json.loads(t[i:j + 1])


def run_question(cfg, mode, model="gpt-4o-mini", k=8,
                 temperature=0.0, max_tokens=8000,
                 use_reranker=False, use_mmr=True, lambda_mmr=0.5):
    """
    Run one question config in one mode and return the result dict.

    cfg keys used: question, components, exclude_pmcids (optional),
                   question_id, topic, source_pmid, source_pmcid (optional).

    use_reranker:
        If True (mode="rag" only), the k retrieved chunks are rescored with
        a cross-encoder (rerank.rerank_chunks) and clear per-query outliers
        are dropped (rerank.filter_by_gap, min_keep=25/max_keep=50 defaults)
        before the prompt is built. Citation numbers [N] in the answer and
        in the saved "chunks" list both refer to this POST-rerank order, not
        the original bi-encoder order -- report.py / judge.py don't need to
        know the difference, they just read whatever chunks[] says.
        sentence-transformers is only imported when this is True, so plain
        (non-reranked) runs have no new dependency.

    use_mmr / lambda_mmr:
        Forwarded to retrieve.search() (mode="rag" only). use_mmr=True
        (default) matches production behavior: fetch a larger candidate
        pool and MMR-rerank down to k for diversity. Set use_mmr=False to
        take the raw top-k by similarity instead -- useful for an A/B test
        of MMR's effect on coverage/citation scores. Point --out-dir (CLI)
        or a fresh output folder (programmatic) at a new location when
        doing this so the non-MMR run never overwrites an existing one.
    """
    question   = cfg["question"]
    components = cfg["components"]
    exclude    = cfg.get("exclude_pmcids") or None

    fmt = _format_instruction(components)
    chunks_out = []

    if mode == "rag":
        print(f"[eval_run] retrieving (k={k}, exclude={exclude}) ...", file=sys.stderr)
        chunks = search(question, k=k, exclude_pmcids=exclude,
                        use_mmr=use_mmr, lambda_mmr=lambda_mmr)

        if use_reranker:
            from .rerank import rerank_and_filter  # lazy: only needed here
            n_before = len(chunks)
            chunks = rerank_and_filter(question, chunks)
            print(f"[eval_run] reranked: kept {len(chunks)}/{n_before} chunks "
                  f"after cross-encoder + gap filter", file=sys.stderr)

        system, user = build_rag_prompt(question, chunks)
        user = user + "\n\n" + fmt
        chunks_out = [
            {"rank": i + 1, "chunk_id": c.chunk_id, "pmcid": c.pmcid,
             "score": round(float(c.score), 4),
             "rerank_score": (round(float(c.rerank_score), 4)
                             if c.rerank_score is not None else None),
             "year": c.year, "source": c.source, "text": c.text}
            for i, c in enumerate(chunks)
        ]
    elif mode == "plain":
        system = ("You are a biomedical research assistant. Answer using only your "
                  "own knowledge; no sources are provided.")
        user = question + "\n\n" + fmt
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'rag' or 'plain')")

    print(f"[eval_run] calling {model} (mode={mode}) ...", file=sys.stderr)
    raw = generate(system=system, user=user, model=model,
                   temperature=temperature, max_tokens=max_tokens,
                   label=f"eval_run:q{cfg.get('question_id', '?')}:{mode}")

    try:
        parsed = _extract_json(raw)
        components_out = parsed.get("components", {})
        free_text_out = parsed.get("free_text", "")
    except Exception as e:  # noqa: BLE001
        print(f"[eval_run] WARNING: could not parse JSON ({e}); raw output saved "
              f"in free_text.", file=sys.stderr)
        components_out = {c: [] for c in components}
        free_text_out = raw

    return {
        "question_id":  cfg.get("question_id"),
        "topic":        cfg.get("topic"),
        "source_pmid":  cfg.get("source_pmid"),
        "source_pmcid": cfg.get("source_pmcid"),
        "model":        model,
        "mode":         "rag_no_bouncer" if mode == "rag" else "no_retrieval",
        "reranker":     bool(use_reranker) if mode == "rag" else False,
        "mmr":          bool(use_mmr) if mode == "rag" else False,
        "components":   components_out,
        "free_text":    free_text_out,
        "chunks":       chunks_out,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eval_run")
    p.add_argument("--question", required=True, help="Path to a question config JSON.")
    p.add_argument("--mode", required=True, choices=["rag", "plain"])
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--k", type=int, default=8, help="Chunks to retrieve (rag mode).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Cap on the answering model's generated JSON "
                        "(components + free_text). Raised 2500 -> 8000: the "
                        "systematic per-source review instruction in "
                        "_format_instruction() pushes toward longer, more "
                        "exhaustive components/free_text output, and 2500 was "
                        "already tight before that change (see judge.py's "
                        "own 4000->8000 bump for the same reason).")
    p.add_argument("--reranker", action="store_true",
                   help="Rescore retrieved chunks with a cross-encoder and drop "
                        "per-query outliers before building the prompt (rag mode "
                        "only; see rerank.py). Default off, so existing runs are "
                        "unaffected unless you opt in.")
    p.add_argument("--no-mmr", action="store_true",
                   help="Disable MMR diversity re-ranking in retrieval (rag mode "
                        "only; see retrieve.py). Default: MMR ON, matching "
                        "production behavior. Use a fresh --out for an A/B "
                        "comparison against an MMR-on run.")
    p.add_argument("--lambda-mmr", type=float, default=0.5,
                   help="MMR relevance/diversity tradeoff, 1.0=relevance only, "
                        "0.0=diversity only. Only used when MMR is on (default 0.5).")
    p.add_argument("--out", required=True, help="Where to write the result JSON.")
    args = p.parse_args(argv)

    with open(args.question, encoding="utf-8") as f:
        cfg = json.load(f)

    result = run_question(cfg, mode=args.mode, model=args.model, k=args.k,
                          temperature=args.temperature, max_tokens=args.max_tokens,
                          use_reranker=args.reranker,
                          use_mmr=not args.no_mmr, lambda_mmr=args.lambda_mmr)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[eval_run] wrote {args.out}  (mode={result['mode']}, "
          f"reranker={result['reranker']}, mmr={result['mmr']}, "
          f"chunks={len(result['chunks'])})", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
