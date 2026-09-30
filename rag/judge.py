#!/usr/bin/env python3
"""
judge.py
--------
LLM-as-a-judge for the CherryPicker holdout eval.

For one question folder (produced by eval_all.py) containing:
    gt.json      ground-truth key (rubric: components -> items, per-item weights)
    rag.json     RAG (no-bouncer) structured answer  (+ chunks)
    plain.json   no-retrieval structured answer       (chunks: [])

it scores each answer against the GT on three axes:

  1. coverage  -- which GT items the answer named (semantic match, direction-
                  aware), which are MISSING, proportional per-component score.
  2. citation  -- (RAG only) do the cited chunks EXIST and actually SUPPORT the
                  claims? Returns per-claim supported/unsupported + a grounding
                  score. For plain.json this is null ("no citations available").
  3. free_text -- a PAIRWISE quality comparison between the two answers'
                  explanations: the better one gets 1, the worse 0, judged in
                  BOTH orders to neutralise position bias (flip -> 0.5 tie).

Design (per the LLM-judge literature, arXiv:2606.19544):
  * coverage + citation are scored INDEPENDENTLY per answer, BLIND to mode
    (we never tell the judge which answer is the RAG) -> no position bias.
  * only the subjective free-text axis is pairwise, and it swaps order.
  * hallucinated / GT-contradicting features are FLAGGED (not deducted).

TWO SEPARATE JUDGE MODELS (v3, this phase):
  * `--judge` (default gemini-3.1-pro-preview) does the two tasks that need
    real reasoning quality: coverage/hallucination scoring against the GT
    rubric+narrative, and the pairwise free-text winner call. Both benefit
    from the stronger, more expensive model and per arXiv:2606.19544 the
    judge should be cross-family from the answering model (default gpt-4o-mini)
    to avoid self-preference bias.
  * `--citation-judge` (default gpt-4o-mini, cheap) does ONLY citation
    faithfulness: given the candidate's claims and the FULL text of every
    retrieved chunk, does each cited chunk actually exist and support the
    claim? This is a narrow, mostly-extractive grounding check (does this
    span of text support this claim, yes/no) rather than an open-ended
    judgment call, so a cheap model is adequate and much less costly across
    a 64-question x 2-mode run. Splitting it out from the coverage call also
    means an expensive-judge failure/timeout on one axis doesn't block the
    other.
  * Citation grounding is now checked against the FULL text of EVERY
    retrieved chunk, not just the ones the answer's free_text happened to
    cite. Previously only chunks matched by a `[N]` regex scan of free_text
    got full text (others were truncated to a 200-char preview); once the
    answering-model prompt started tagging every components entity with its
    own `[N]` citations too (not just free_text), that regex -- which only
    ever scanned free_text -- silently missed citations that appeared ONLY
    in components, so those chunks were still rendered as truncated
    previews and the judge had no way to verify support, producing false
    "unsupported"/unclear citation verdicts. Since the citation judge is now
    a cheap model, there's no longer a cost reason to truncate anything --
    every chunk is now rendered in full, unconditionally, and the citation
    task explicitly tells the judge to scan for `[N]` markers in BOTH the
    free-text explanation AND the structured components lists.

Coverage rubric (v2): only microbiota / metabolites / mechanisms are SCORED
(see eval_all.py SCORED_COMPONENTS). conflicting_or_alternative_findings is
carried through as unscored_components -- visible to the judge for
hallucination-checking, but never contributes to coverage_score, and the
answering model is never required to fill it. gt.json also carries
ground_truth_free_text, a narrative synthesis used as background context for
both the coverage judgment and the pairwise free-text comparison.

Usage (from project root):
    python -m code.rag.judge --folder code/rag/eval/runs/q1
    python -m code.rag.judge --folder code/rag/eval/runs/q1 \\
        --judge gemini-3.1-pro-preview --citation-judge gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from .llm import generate
from .eval_run import _extract_json
from . import cost_tracker


JUDGE_SYS = (
    "You are a meticulous biomedical evaluation judge. You compare a candidate "
    "answer against a fixed ground-truth key and report structured scores. You "
    "do the semantic matching a domain expert would (e.g. 'E. coli' == "
    "'Escherichia coli'), you respect the required DIRECTION of a finding "
    "(enriched vs depleted, increase vs decrease), and you never reward a "
    "feature that is absent or contradicts the key. Output ONLY the requested "
    "JSON object."
)

CITATION_JUDGE_SYS = (
    "You are a careful fact-checker. You are given a candidate answer's claims "
    "(each tagged with the source chunk number(s) it cites, e.g. [3] or "
    "[4, 11]) and the FULL text of every numbered source chunk. For each "
    "claim, decide only whether the chunk(s) it cites actually exist and "
    "whether their text genuinely supports that specific claim -- do not use "
    "outside biomedical knowledge to judge correctness, only whether the "
    "cited text supports what is claimed. Output ONLY the requested JSON "
    "object."
)

COVERAGE_JUDGE_SYS = (
    "You are a meticulous biomedical evaluation judge. You compare a "
    "candidate's STRUCTURED answer (a list of named entities the candidate "
    "asserts, grouped by component) against a fixed ground-truth key and "
    "report which key entities were named. You do the semantic matching a "
    "domain expert would (e.g. 'E. coli' == 'Escherichia coli'). This is a "
    "pure entity-recall check on NAMES ONLY: each ground-truth item's text "
    "after '--' is informational context (direction of change, disease "
    "context, etc.) and must be IGNORED for matching purposes -- a "
    "candidate entity matches a ground-truth item if the named entity "
    "matches, regardless of whether any direction or qualifier is stated "
    "anywhere in the candidate answer. Do not reward an entity that was "
    "named without real support just to appear thorough, and do not "
    "penalize beyond a simple non-match. Output ONLY the requested JSON "
    "object."
)


def _generate_json_retry(system, user, model, max_tokens, label="", retries=1):
    """generate() + _extract_json(), retrying with a DOUBLED max_tokens if the
    response has no parseable JSON in it.

    Why this is needed: Gemini 3.x "thinking" models spend part of their
    max_output_tokens budget on internal reasoning before emitting the
    visible answer, even at thinking_level=LOW. For most questions the fixed
    budgets below (8000 coverage / 4000 citation / 1000 pairwise) are
    generous enough headroom. But rubric size varies a lot across the
    64-question set -- e.g. Q1's combined 37+13+19=69-item rubric is far
    larger than a typical question's -- and a bigger rubric means a bigger,
    harder-to-reason-about prompt, which can push a "thinking" model to burn
    its ENTIRE output budget on reasoning and emit nothing (or a truncated
    fragment with no closing brace) as the visible answer. That surfaces as
    "no JSON object found in model output" from _extract_json, and previously
    that exception propagated all the way up through judge_folder() and
    aborted scoring for the WHOLE question -- discarding every other judge
    call that had already succeeded (coverage, citation, structural
    citation, the first pairwise order...) since nothing gets written to
    disk until the very end. Retrying once with double the token budget
    directly targets the actual cause (starved output budget) rather than
    just hoping a second roll of the dice behaves differently.
    """
    attempt_tokens = max_tokens
    last_err = None
    for attempt in range(retries + 1):
        raw = generate(system=system, user=user, model=model, temperature=0.0,
                       max_tokens=attempt_tokens, label=label)
        try:
            return _extract_json(raw)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                attempt_tokens = attempt_tokens * 2
                print(f"[judge] WARNING: {label or 'judge call'} produced no "
                      f"parseable JSON (attempt {attempt + 1}/{retries + 1}, "
                      f"max_tokens={max_tokens if attempt == 0 else attempt_tokens // 2}) "
                      f"-- retrying with max_tokens={attempt_tokens} ...",
                      file=sys.stderr)
    raise last_err


# --------------------------------------------------------------------------
# 1. Coverage + hallucination scoring, blind to mode (no chunks needed --
#    this task only compares the answer against the GT rubric/narrative)
# --------------------------------------------------------------------------

def _coverage_prompt(gt, answer):
    rubric = gt.get("rubric", [])
    unscored = gt.get("unscored_components", [])
    rubric_txt = json.dumps(rubric, ensure_ascii=False, indent=2)
    unscored_txt = (json.dumps(unscored, ensure_ascii=False, indent=2)
                   if unscored else "(none)")
    comps = json.dumps(answer.get("components", {}), ensure_ascii=False, indent=2)

    schema = {
        "coverage": {
            "components": [
                {"component": "str", "matched": ["gt items the answer named"],
                 "missing": ["gt items NOT named"],
                 "awarded": 0.0, "max": 1.0}
            ],
            "coverage_score": 0.0, "max_score": len(rubric)
        },
        "hallucinated_features": ["features asserted that are fabricated or "
                                  "contradict the ground truth"],
    }

    return (
        "GROUND-TRUTH KEY -- SCORED RUBRIC. Each component is worth 1.0, split "
        "evenly among its items (per_item). Award (matched / n_items) per "
        "component. These items are what the SOURCE ARTICLE reported -- they are "
        "NOT a checklist the candidate was required to exhaustively reproduce. A "
        "well-scoped answer may reasonably choose not to touch a category it "
        "judged tangential to the question; score coverage strictly as "
        "matched-vs-missing against the items below (that already produces a "
        "lower score for an answer that omits a lot), but do not apply any "
        "additional penalty.\n\n"
        "IMPORTANT -- this is a NAME-ONLY match. Each ground-truth item below is "
        "written as \"Entity -- direction/context\"; for matching purposes, "
        "IGNORE everything after the \"--\" and match only on the entity name "
        "itself (semantic match, e.g. 'E. coli' == 'Escherichia coli'). Do NOT "
        "check or require direction of change, magnitude, or any other "
        "qualifier -- the candidate's structured components list below never "
        "states direction by design, so direction is not part of this check. "
        "If two ground-truth items share the same entity name (e.g. the same "
        "taxon reported with different directions across different studies), a "
        "single mention of that entity name in the candidate answer matches "
        "BOTH items.\n\n"
        f"{rubric_txt}\n\n"
        "UNSCORED CONTEXT (e.g. conflicting/alternative findings noted in the "
        "source article). This does NOT count toward coverage_score -- it is "
        "provided only so you can recognize when the candidate hallucinates or "
        "contradicts this material:\n\n"
        f"{unscored_txt}\n\n"
        "CANDIDATE ANSWER -- structured components (this is the ONLY thing you "
        "are matching against; the candidate's free-text explanation is "
        "deliberately NOT shown to you for this task):\n"
        f"{comps}\n\n"
        "TASKS:\n"
        "1) COVERAGE: for each SCORED rubric component ONLY, list which "
        "ground-truth items the candidate named (by entity name only, per the "
        "IMPORTANT note above), which are missing, and award = matched/n_items. "
        "Do not create a coverage entry for the unscored context.\n"
        "2) HALLUCINATED_FEATURES: list any named entity (in the scored "
        "components or the unscored context) that is fabricated or does not "
        "correspond to anything in the key (flag only; do not subtract).\n\n"
        "Return ONLY this JSON shape (fill real values):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def _score_coverage(gt, answer, judge_model, label=""):
    # max_tokens raised 2000 -> 4000 -> 8000: Gemini 3.x "thinking" models
    # spend part of max_output_tokens on internal reasoning before the
    # visible answer, even at thinking_level=LOW (set in llm.py). 2000 was
    # too tight and produced truncated/empty JSON; 4000 was still tight for
    # a large-rubric question. Raised to 8000 with headroom.
    try:
        return _generate_json_retry(
            system=COVERAGE_JUDGE_SYS, user=_coverage_prompt(gt, answer),
            model=judge_model, max_tokens=8000, label=label or "judge:coverage")
    except Exception as e:  # noqa: BLE001
        print(f"[judge] ERROR: {label or 'coverage'} failed after retry: "
              f"{type(e).__name__}: {e} -- recording as error, continuing.",
              file=sys.stderr)
        return {
            "coverage": {"components": [], "coverage_score": None,
                        "max_score": len(gt.get("rubric", []))},
            "hallucinated_features": [],
            "error": f"{type(e).__name__}: {e}",
        }


# --------------------------------------------------------------------------
# 2. Citation faithfulness (RAG only), cheap model, ALL chunks in full text
# --------------------------------------------------------------------------

def _citation_prompt(answer):
    comps = json.dumps(answer.get("components", {}), ensure_ascii=False, indent=2)
    free_text = answer.get("free_text", "")
    chunks = answer.get("chunks", [])

    # Every chunk gets its FULL text, unconditionally -- no cited/uncited
    # split, no truncation. See the module docstring for why: the previous
    # "cited chunks full text, uncited chunks 200-char preview" scheme only
    # detected citations via a regex scan of free_text, so any chunk cited
    # ONLY inside a components entity's [N] tag was wrongly treated as
    # "uncited" and truncated, hiding the very evidence needed to verify it
    # -- a systematic false-negative in faithfulness. Now that this call
    # uses a cheap model, there's no cost reason to truncate anything.
    def _render_chunk(c):
        rank = c.get("rank")
        text = c.get("text", "") or ""
        return f"[{rank}] {c.get('pmcid')}: {text}"

    chunk_txt = "\n\n".join(_render_chunk(c) for c in chunks) \
        or "(no chunks -- this answer had no retrieval)"

    schema = {
        "applicable": True,
        "cited_exist": True,
        "claims": [{"claim": "str", "cited_chunks": [1],
                    "supported": True, "note": "str"}],
        "n_supported": 0, "n_unsupported": 0, "faithfulness": 0.0
    }

    return (
        "The candidate answer below may cite numbered source chunks with `[N]` "
        "markers (or `[N, M]` for more than one source) -- these markers can "
        "appear in the free-text explanation AND/OR next to individual entities "
        "inside the structured components lists. Scan BOTH for citation markers; "
        "do not only look at the free-text.\n\n"
        "CANDIDATE ANSWER -- structured components:\n"
        f"{comps}\n\n"
        "CANDIDATE ANSWER -- free-text explanation:\n"
        f"{free_text}\n\n"
        "RETRIEVED CHUNKS the answer could cite (numbered, FULL TEXT):\n"
        f"{chunk_txt}\n\n"
        "TASK -- CITATION: for each distinct factual claim that carries an `[N]` "
        "citation (whether in components or free-text), list the chunk number(s) "
        "it cited, whether that chunk's text actually supports the specific "
        "claim (not just the general topic), and a short note explaining why. "
        "Set cited_exist=true only if every cited number actually exists in the "
        "chunk list above. Set faithfulness = n_supported / "
        "(n_supported + n_unsupported). If the answer has no chunks at all "
        "(no retrieval), set applicable=false and leave everything else at its "
        "default/empty value.\n\n"
        "Return ONLY this JSON shape (fill real values):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def _score_citation(answer, citation_judge_model, label=""):
    chunks = answer.get("chunks", [])
    if not chunks:
        # No retrieval at all (plain mode) -- nothing to check, no need to
        # spend an API call on it.
        return {"applicable": False}
    try:
        return _generate_json_retry(
            system=CITATION_JUDGE_SYS, user=_citation_prompt(answer),
            model=citation_judge_model, max_tokens=4000,
            label=label or "judge:citation")
    except Exception as e:  # noqa: BLE001
        print(f"[judge] ERROR: {label or 'citation'} failed after retry: "
              f"{type(e).__name__}: {e} -- recording as error, continuing.",
              file=sys.stderr)
        return {"applicable": False, "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# 2b. STRUCTURAL citation check -- like coverage, this is a components-ONLY
#     variant that never looks at free_text. Each components entity may
#     carry an [N] citation tag (e.g. "Lactobacillus [42]"); this checks
#     whether that specific tagged entity is actually grounded in its cited
#     chunk(s), independent of anything the free-text says or claims.
# --------------------------------------------------------------------------

def _structural_citation_prompt(answer):
    comps = json.dumps(answer.get("components", {}), ensure_ascii=False, indent=2)
    chunks = answer.get("chunks", [])

    def _render_chunk(c):
        rank = c.get("rank")
        text = c.get("text", "") or ""
        return f"[{rank}] {c.get('pmcid')}: {text}"

    chunk_txt = "\n\n".join(_render_chunk(c) for c in chunks) \
        or "(no chunks -- this answer had no retrieval)"

    schema = {
        "applicable": True,
        "cited_exist": True,
        "entities": [{"component": "str", "entity": "str", "cited_chunks": [1],
                      "supported": True, "note": "str"}],
        "n_supported": 0, "n_unsupported": 0, "faithfulness": 0.0
    }

    return (
        "The candidate answer\'s STRUCTURED components lists below tag each "
        "named entity with the source chunk number(s) that support it, e.g. "
        "\"Lactobacillus [42]\" or \"indoxyl sulfate [4, 11]\". This is a "
        "STRUCTURAL check ONLY -- the candidate\'s free-text explanation is "
        "deliberately NOT shown to you; you are checking whether each "
        "individually-tagged entity is actually grounded in its cited "
        "chunk(s), independent of anything the free-text separately claims.\n\n"
        "CANDIDATE ANSWER -- structured components (with citation tags):\n"
        f"{comps}\n\n"
        "RETRIEVED CHUNKS the answer could cite (numbered, FULL TEXT):\n"
        f"{chunk_txt}\n\n"
        "TASK -- STRUCTURAL CITATION: for each entity that carries an `[N]` "
        "citation tag, report the component it belongs to, the entity name, "
        "the chunk number(s) it cited, whether those chunks\' text actually "
        "names/supports that specific entity as relevant to the question\'s "
        "topic (not just an unrelated passing mention), and a short note. An "
        "entity with NO citation tag should be skipped entirely -- do not flag "
        "it as unsupported, it is simply outside this check. Set "
        "cited_exist=true only if every cited number actually exists in the "
        "chunk list above. Set faithfulness = n_supported / "
        "(n_supported + n_unsupported). If the answer has no chunks at all "
        "(no retrieval) or no entity carries any citation tag, set "
        "applicable=false and leave everything else at its default/empty "
        "value.\n\n"
        "Return ONLY this JSON shape (fill real values):\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def _score_structural_citation(answer, citation_judge_model, label=""):
    chunks = answer.get("chunks", [])
    if not chunks:
        # No retrieval at all (plain mode) -- nothing to check, no need to
        # spend an API call on it.
        return {"applicable": False}
    try:
        return _generate_json_retry(
            system=CITATION_JUDGE_SYS, user=_structural_citation_prompt(answer),
            model=citation_judge_model, max_tokens=4000,
            label=label or "judge:structural_citation")
    except Exception as e:  # noqa: BLE001
        print(f"[judge] ERROR: {label or 'structural_citation'} failed after "
              f"retry: {type(e).__name__}: {e} -- recording as error, continuing.",
              file=sys.stderr)
        return {"applicable": False, "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# 3. Pairwise free-text comparison with order swap (expensive judge)
# --------------------------------------------------------------------------

def _pairwise_prompt(gt, free_a, free_b):
    return (
        "Two candidate explanations answer the same biomedical question. Using "
        "the ground-truth key and narrative below, decide which explanation is "
        "more accurate, mechanistically correct, and complete. Judge CONTENT "
        "only; ignore length and style. The itemized key lists discrete facts; "
        "the narrative gives the fuller picture (including nuance, direction, "
        "and context) so you can judge claims the itemized list alone might miss "
        "or judge out of context.\n\n"
        "GROUND-TRUTH KEY (itemized, scored components):\n" +
        json.dumps(gt.get("rubric", []), ensure_ascii=False) +
        "\n\nGROUND-TRUTH NARRATIVE (background synthesis, not itself a checklist):\n" +
        (gt.get("ground_truth_free_text", "") or "(none)") +
        "\n\nEXPLANATION A:\n" + (free_a or "(empty)") +
        "\n\nEXPLANATION B:\n" + (free_b or "(empty)") +
        "\n\nReturn ONLY: {\"winner\": \"A\" | \"B\" | \"tie\", \"reason\": \"str\"}"
    )


def _pairwise_once(gt, free_a, free_b, judge_model, label=""):
    # same reasoning-token headroom issue as _score_coverage; 400 was too
    # tight for a thinking model even for this short a schema, and even
    # 1000 was not always enough (see _generate_json_retry).
    try:
        parsed = _generate_json_retry(
            system=JUDGE_SYS, user=_pairwise_prompt(gt, free_a, free_b),
            model=judge_model, max_tokens=1000, label=label or "judge:pairwise")
        return parsed.get("winner", "tie"), parsed.get("reason", ""), None
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        print(f"[judge] ERROR: {label or 'pairwise'} failed after retry: {err} "
              f"-- treating this order as a tie, continuing.", file=sys.stderr)
        return "tie", "", err


def _freetext_pairwise(gt, rag, plain, judge_model):
    """Run both orders; award 1/0 only if the winner is consistent, else 0.5 tie."""
    rag_ft = rag.get("free_text", "")
    plain_ft = plain.get("free_text", "")
    qtag = gt.get("question_id", "?")

    # order 1: A=rag, B=plain
    w1, reason1, err1 = _pairwise_once(gt, rag_ft, plain_ft, judge_model,
                                       label=f"judge:q{qtag}:pairwise:order1")
    # order 2: A=plain, B=rag
    w2, reason2, err2 = _pairwise_once(gt, plain_ft, rag_ft, judge_model,
                                       label=f"judge:q{qtag}:pairwise:order2")

    # map each verdict to which mode it favored
    fav1 = "rag" if w1 == "A" else ("plain" if w1 == "B" else "tie")
    fav2 = "plain" if w2 == "A" else ("rag" if w2 == "B" else "tie")

    if fav1 == fav2 and fav1 in ("rag", "plain"):
        rag_win = 1.0 if fav1 == "rag" else 0.0
    else:
        rag_win = 0.5  # inconsistent across orders (position bias) or tie
    result = {
        "rag_win": rag_win,
        "plain_win": round(1.0 - rag_win, 3),
        "order1_favored": fav1,
        "order2_favored": fav2,
        "order1_reason": reason1,
        "order2_reason": reason2,
        "consistent": fav1 == fav2 and fav1 in ("rag", "plain"),
    }
    if err1 or err2:
        result["error"] = {"order1": err1, "order2": err2}
    return result


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def judge_folder(folder, judge_model="gemini-3.1-pro-preview",
                 citation_judge_model="gpt-4o-mini"):
    def _load(name):
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            return json.load(f)

    gt    = _load("gt.json")
    rag   = _load("rag.json")
    plain = _load("plain.json")

    qtag = gt.get("question_id", "?")
    print(f"[judge] scoring coverage (rag, judge={judge_model}) ...", file=sys.stderr)
    cov_rag = _score_coverage(gt, rag, judge_model, label=f"judge:q{qtag}:coverage:rag")
    print(f"[judge] scoring coverage (plain, judge={judge_model}) ...", file=sys.stderr)
    cov_plain = _score_coverage(gt, plain, judge_model, label=f"judge:q{qtag}:coverage:plain")
    print(f"[judge] scoring citation (rag, judge={citation_judge_model}) ...", file=sys.stderr)
    cit_rag = _score_citation(rag, citation_judge_model, label=f"judge:q{qtag}:citation:rag")
    print(f"[judge] scoring citation (plain, judge={citation_judge_model}) ...", file=sys.stderr)
    cit_plain = _score_citation(plain, citation_judge_model, label=f"judge:q{qtag}:citation:plain")
    print(f"[judge] scoring structural citation (rag, judge={citation_judge_model}) ...", file=sys.stderr)
    struct_cit_rag = _score_structural_citation(rag, citation_judge_model, label=f"judge:q{qtag}:structural_citation:rag")
    print(f"[judge] scoring structural citation (plain, judge={citation_judge_model}) ...", file=sys.stderr)
    struct_cit_plain = _score_structural_citation(plain, citation_judge_model, label=f"judge:q{qtag}:structural_citation:plain")
    print(f"[judge] pairwise free-text (both orders, judge={judge_model}) ...", file=sys.stderr)
    ft = _freetext_pairwise(gt, rag, plain, judge_model)

    # Save the exact prompts sent to each judge for later manual review. This
    # rebuilds the same strings via the same (pure, no-API-call) prompt
    # builders used above -- zero added cost, just so the raw input each
    # judge actually saw is on disk alongside its verdict.
    rag_ft_text = rag.get("free_text", "")
    plain_ft_text = plain.get("free_text", "")
    judge_prompts = {
        "judge_model": judge_model,
        "citation_judge_model": citation_judge_model,
        "coverage_system_prompt": COVERAGE_JUDGE_SYS,
        "citation_system_prompt": CITATION_JUDGE_SYS,
        "coverage_rag_prompt": _coverage_prompt(gt, rag),
        "coverage_plain_prompt": _coverage_prompt(gt, plain),
        "citation_rag_prompt": _citation_prompt(rag),
        "citation_plain_prompt": _citation_prompt(plain),
        "structural_citation_rag_prompt": _structural_citation_prompt(rag),
        "structural_citation_plain_prompt": _structural_citation_prompt(plain),
        "pairwise_order1_prompt": _pairwise_prompt(gt, rag_ft_text, plain_ft_text),
        "pairwise_order2_prompt": _pairwise_prompt(gt, plain_ft_text, rag_ft_text),
    }
    with open(os.path.join(folder, "judge_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(judge_prompts, f, ensure_ascii=False, indent=2)

    scored_rag = {
        "question_id": gt.get("question_id"), "topic": gt.get("topic"),
        "mode": "rag_no_bouncer",
        "judge": judge_model, "citation_judge": citation_judge_model,
        "coverage": cov_rag.get("coverage"),
        "hallucinated_features": cov_rag.get("hallucinated_features", []),
        "citation": cit_rag,
        "structural_citation": struct_cit_rag,
        "free_text_win": ft["rag_win"], "free_text_detail": ft,
    }
    scored_plain = {
        "question_id": gt.get("question_id"), "topic": gt.get("topic"),
        "mode": "no_retrieval",
        "judge": judge_model, "citation_judge": citation_judge_model,
        "coverage": cov_plain.get("coverage"),
        "hallucinated_features": cov_plain.get("hallucinated_features", []),
        "citation": cit_plain,
        "structural_citation": struct_cit_plain,
        "free_text_win": ft["plain_win"], "free_text_detail": ft,
    }

    for name, obj in (("scored_rag.json", scored_rag),
                      ("scored_plain.json", scored_plain)):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    cov_r = (scored_rag["coverage"] or {}).get("coverage_score")
    cov_p = (scored_plain["coverage"] or {}).get("coverage_score")
    print(f"[judge] {folder}: coverage rag={cov_r} plain={cov_p}  "
          f"free-text rag_win={ft['rag_win']} (consistent={ft['consistent']})",
          file=sys.stderr)
    return scored_rag, scored_plain


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="judge")
    p.add_argument("--folder", required=True,
                   help="A qN/ folder with gt.json, rag.json, plain.json.")
    p.add_argument("--judge", default="gemini-3.1-pro-preview",
                   help="Model for coverage/hallucination scoring and the "
                        "pairwise free-text winner call. Should stay a strong, "
                        "cross-family (from the answering model) judge.")
    p.add_argument("--citation-judge", default="gpt-4o-mini",
                   help="Model for citation faithfulness checking only "
                        "(does each [N]-cited chunk exist and support the "
                        "claim?). This is a narrow grounding check against "
                        "chunk text the judge is given in full, so a cheap "
                        "model is adequate -- default gpt-4o-mini.")
    args = p.parse_args(argv)
    try:
        judge_folder(args.folder, judge_model=args.judge,
                    citation_judge_model=args.citation_judge)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        # NOTE: each `judge` CLI invocation is its own process, so this total
        # covers only the ONE folder just judged, not a running total across
        # a tcsh `foreach` loop over all 100 q*/ folders. Sum the per-folder
        # totals yourself, or ask for a single-process driver script that
        # loops in-process if you want one cumulative number.
        cost_tracker.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
