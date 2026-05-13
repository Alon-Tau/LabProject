#!/usr/bin/env python3
"""
fulltext_coverage_by_journal.py — READ-ONLY analysis.

Goal: for each journal, compute the OA fulltext extraction success rate.
       i.e., of the OA papers we tried to fetch from a given journal,
       what fraction ended up with `has_fulltext=True` (a saved .txt file)?

This tells us whether the ~71k OA papers that failed fulltext extraction
are concentrated in specific journals (suggesting a structural cleaning bug)
or spread evenly (suggesting random network/parser failures).

Inputs:
  --metadata    metadata_all.jsonl (default: standard path)
Outputs:
  --out         JSON report with per-journal stats

Run:
  python fulltext_coverage_by_journal.py
  python fulltext_coverage_by_journal.py --out journal_coverage.json
"""

import os
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path

DEFAULT_METADATA = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus/metadata_all.jsonl"


def norm_pmcid(raw):
    if raw is None: return ""
    x = str(raw).strip().upper()
    if not x: return ""
    if x.endswith(".TXT"): x = x[:-4]
    if x.isdigit(): return "PMC" + x
    m = re.search(r"(PMC\d+)", x)
    if m: return m.group(1)
    return x


def get_journal(meta):
    ji = meta.get("journalInfo")
    if isinstance(ji, dict):
        try:
            jt = ji.get("journal", {}).get("title")
            if jt: return str(jt).strip()
        except Exception:
            pass
    for k in ("journalTitle", "journal", "source"):
        v = meta.get(k)
        if v: return str(v).strip()
    return "(unknown)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=DEFAULT_METADATA)
    ap.add_argument("--out", default="journal_coverage.json")
    args = ap.parse_args()

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        raise SystemExit(f"Metadata file not found: {meta_path}")

    # First pass: dedupe by PMCID (metadata has duplicate rows)
    # For each unique PMCID, store: journal, is_oa (Y/N), has_fulltext (bool)
    print(f"Reading: {meta_path}")
    pmcid_info = {}  # pmcid -> dict
    raw_rows = 0
    no_pmcid = 0
    with meta_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_rows += 1
            pmcid = norm_pmcid(obj.get("pmcid") or obj.get("PMCID") or obj.get("id"))
            if not pmcid:
                no_pmcid += 1
                continue
            # If we've seen this PMCID, only update if new info is more complete
            is_oa = str(obj.get("isOpenAccess") or "").strip().upper() == "Y"
            has_ft = bool(obj.get("has_fulltext") or obj.get("has_full_text"))
            journal = get_journal(obj)
            prev = pmcid_info.get(pmcid)
            if prev is None:
                pmcid_info[pmcid] = {"journal": journal, "is_oa": is_oa, "has_ft": has_ft}
            else:
                # Merge: prefer OA=True if either says so, prefer has_ft=True if either says so
                if is_oa: prev["is_oa"] = True
                if has_ft: prev["has_ft"] = True
                # Keep first non-unknown journal
                if prev["journal"] in ("(unknown)", "") and journal not in ("(unknown)", ""):
                    prev["journal"] = journal

    print(f"Raw metadata rows: {raw_rows:,}")
    print(f"Rows without PMCID: {no_pmcid:,}")
    print(f"Unique PMCIDs: {len(pmcid_info):,}")

    # Per-journal aggregation
    j_stats = defaultdict(lambda: {"total": 0, "oa": 0, "oa_with_ft": 0, "non_oa": 0})
    for pid, info in pmcid_info.items():
        j = info["journal"] or "(unknown)"
        s = j_stats[j]
        s["total"] += 1
        if info["is_oa"]:
            s["oa"] += 1
            if info["has_ft"]:
                s["oa_with_ft"] += 1
        else:
            s["non_oa"] += 1

    # Compute success rate and sort
    rows = []
    for j, s in j_stats.items():
        oa = s["oa"]
        oa_ft = s["oa_with_ft"]
        rate = (oa_ft / oa * 100) if oa else None
        rows.append({
            "journal": j,
            "total_papers": s["total"],
            "non_oa": s["non_oa"],
            "oa": oa,
            "oa_with_fulltext": oa_ft,
            "oa_missing_fulltext": oa - oa_ft,
            "oa_fulltext_success_pct": round(rate, 1) if rate is not None else None,
        })
    # Sort: largest journals first, with sortable success rate
    rows.sort(key=lambda r: (-r["total_papers"]))

    # Also produce a "worst offenders" view: journals with >=200 OA papers and <80% success
    worst = sorted(
        [r for r in rows if r["oa"] >= 200 and r["oa_fulltext_success_pct"] is not None and r["oa_fulltext_success_pct"] < 80],
        key=lambda r: r["oa_fulltext_success_pct"]
    )

    # Best performers (high volume + high success) for comparison
    best = sorted(
        [r for r in rows if r["oa"] >= 200 and r["oa_fulltext_success_pct"] is not None],
        key=lambda r: -r["oa_fulltext_success_pct"]
    )[:15]

    # Global summary
    total_oa = sum(r["oa"] for r in rows)
    total_oa_ft = sum(r["oa_with_fulltext"] for r in rows)
    global_rate = (total_oa_ft / total_oa * 100) if total_oa else 0

    report = {
        "summary": {
            "raw_metadata_rows": raw_rows,
            "unique_pmcids": len(pmcid_info),
            "total_oa": total_oa,
            "total_oa_with_fulltext": total_oa_ft,
            "total_oa_missing_fulltext": total_oa - total_oa_ft,
            "global_oa_fulltext_success_pct": round(global_rate, 2),
        },
        "worst_offenders_oa_ge_200_success_lt_80pct": worst,
        "best_performers_top_15": best,
        "all_journals_sorted_by_volume": rows,
    }

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Pretty print to terminal
    print("\n" + "=" * 70)
    print("PER-JOURNAL OA FULLTEXT COVERAGE")
    print("=" * 70)
    s = report["summary"]
    print(f"Unique PMCIDs:                 {s['unique_pmcids']:,}")
    print(f"Total OA papers:               {s['total_oa']:,}")
    print(f"  with fulltext:               {s['total_oa_with_fulltext']:,}")
    print(f"  missing fulltext:            {s['total_oa_missing_fulltext']:,}")
    print(f"Global OA fulltext success:    {s['global_oa_fulltext_success_pct']}%")

    print(f"\n--- WORST OFFENDERS (OA>=200, success<80%) — {len(worst)} journals ---")
    print(f"{'success%':>9} {'oa':>8} {'missing':>8}  journal")
    for r in worst[:30]:
        print(f"{r['oa_fulltext_success_pct']:>9.1f} {r['oa']:>8,} {r['oa_missing_fulltext']:>8,}  {r['journal'][:65]}")

    print(f"\n--- BEST PERFORMERS (top 15 by success rate) ---")
    print(f"{'success%':>9} {'oa':>8}  journal")
    for r in best:
        print(f"{r['oa_fulltext_success_pct']:>9.1f} {r['oa']:>8,}  {r['journal'][:65]}")

    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()