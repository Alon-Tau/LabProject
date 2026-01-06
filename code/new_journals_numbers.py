import time
import requests
import csv
from collections import defaultdict

EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

YEAR_START = 1990
YEAR_END = 2025  # inclusive

# -----------------------------
# Filters
# -----------------------------
BASE_FILTER = (
    'PUB_TYPE:"Journal Article" '
    'AND IN_EPMC:Y '
    'AND NOT PREPRINT:Y'
)
OA_FILTER = "OPEN_ACCESS:Y"


# -------------------------------------------------------------------
# ✅ Journal -> ISSNs (robust Europe PMC querying)
# -------------------------------------------------------------------
JOURNAL_ISSNS = {
    # ---- Medical Informatics ----
    "Lancet Digital Health": ["2589-7500"],
    "npj Digital Medicine": ["2398-6352"],
    "ACM Transactions on Computing for Healthcare": ["2691-1957", "2637-8051"],
    "PLOS Digital Health": ["2767-3170"],
    "Intelligent Medicine": ["2667-1026"],
    "IEEE Journal of Biomedical and Health Informatics": ["2168-2194", "2168-2208"],
    "JMIR mHealth and uHealth": ["2291-5222"],
    "Journal of Medical Internet Research": ["1439-4456", "1438-8871"],
    "Journal of Medical Systems": ["0148-5598", "1573-689X"],
    "Artificial Intelligence in Medicine": ["0933-3657", "1873-2860"],

    # ---- Multidisciplinary Sciences ----
    "Nature Reviews Methods Primers": ["2662-8449"],
    "Nature Computational Science": ["2662-8457"],
    "Science Advances": ["2375-2548"],
    "Scientific Data": ["2052-4463"],
    "National Science Review": ["2095-5138", "2053-714X"],
    "Science Bulletin": ["2095-9273", "2095-9281"],
    "Journal of Advanced Research": ["2090-1232", "2090-1224"],
    "Research": ["2096-5168", "2639-5274"],
    "Global Challenges": ["2056-6646"],
    "Fundamental Research": ["2096-9457", "2667-3258"],
    "Research Synthesis Methods": ["1759-2879", "1759-2887"],
    "Innovation": ["2666-6758"],
    "Exploration": ["2766-8509", "2766-2098"],
    "Nature Human Behaviour": ["2397-3374"],

    # ---- Microbiology ----
    "Nature Reviews Microbiology": ["1740-1526", "1740-1534"],
    "Trends in Microbiology": ["0966-842X", "1878-4380"],
    "FEMS Microbiology Reviews": ["0168-6445", "1574-6976"],
    "Clinical Microbiology Reviews": ["0893-8512", "1098-6618"],
    "Microbiology and Molecular Biology Reviews": ["1092-2172", "1098-5557"],
    "Gut Microbes": ["1949-0976", "1949-0984"],
    "npj Biofilms and Microbiomes": ["2055-5008"],
    "Environmental Microbiome": ["2524-6372"],
    "ISME Communications": ["2730-6151"],
    "Emerging Microbes & Infections": ["2222-1751"],
    "Virulence": ["2150-5594", "2150-5608"],
    "Journal of Oral Microbiology": ["2000-2297"],
    "Annual Review of Microbiology": ["0066-4227", "1545-3251"],
    "Current Opinion in Microbiology": ["1369-5274", "1879-0364"],
    "Lancet Microbe": ["2666-5247"],
    "Clinical Infectious Diseases": ["1058-4838", "1537-6591"],
    "Clinical Microbiology and Infection": ["1198-743X", "1469-0691"],
    "Journal of Clinical Microbiology": ["0095-1137", "1098-660X"],
    "New Microbes and New Infections": ["2052-2975"],
    "Critical Reviews in Microbiology": ["1040-841X", "1549-7828"],
    "iMeta": ["2770-5986", "2770-596X"],
    "Current Research in Microbial Sciences": ["2666-5174"],
    "International Journal of Food Microbiology": ["0168-1605", "1879-3460"],
    "Microbial Biotechnology": ["1751-7915"],
    "Microbiological Research": ["0944-5013", "1618-0623"],
}

# -------------------------------------------------------------------
# ✅ Journal -> Tier mapping (your agreed tiers)
# If a journal is missing here, it will be labeled "UNKNOWN"
# -------------------------------------------------------------------
JOURNAL_TIER = {
    # Tier 1
    "Nature Reviews Microbiology": "Tier 1",
    "Trends in Microbiology": "Tier 1",
    "FEMS Microbiology Reviews": "Tier 1",
    "Clinical Microbiology Reviews": "Tier 1",
    "Microbiology and Molecular Biology Reviews": "Tier 1",
    "Gut Microbes": "Tier 1",
    "npj Biofilms and Microbiomes": "Tier 1",
    "Nature Reviews Methods Primers": "Tier 1",
    "Nature Computational Science": "Tier 1",
    "Science Advances": "Tier 1",
    "Research Synthesis Methods": "Tier 1",
    "npj Digital Medicine": "Tier 1",
    "Artificial Intelligence in Medicine": "Tier 1",
    "IEEE Journal of Biomedical and Health Informatics": "Tier 1",

    # Tier 2
    "Environmental Microbiome": "Tier 2",
    "ISME Communications": "Tier 2",
    "Emerging Microbes & Infections": "Tier 2",
    "Virulence": "Tier 2",
    "Journal of Oral Microbiology": "Tier 2",
    "Annual Review of Microbiology": "Tier 2",
    "Current Opinion in Microbiology": "Tier 2",
    "National Science Review": "Tier 2",
    "Science Bulletin": "Tier 2",
    "Journal of Advanced Research": "Tier 2",
    "Research": "Tier 2",
    "Lancet Digital Health": "Tier 2",
    "PLOS Digital Health": "Tier 2",
    "Journal of Medical Systems": "Tier 2",
    "Microbial Biotechnology": "Tier 2",

    # Tier 3
    "Lancet Microbe": "Tier 3",
    "Clinical Infectious Diseases": "Tier 3",
    "Clinical Microbiology and Infection": "Tier 3",
    "Journal of Clinical Microbiology": "Tier 3",
    "New Microbes and New Infections": "Tier 3",
    "Critical Reviews in Microbiology": "Tier 3",
    "iMeta": "Tier 3",
    "Current Research in Microbial Sciences": "Tier 3",
    "Nature Human Behaviour": "Tier 3",
    "Scientific Data": "Tier 3",
    "Global Challenges": "Tier 3",
    "Innovation": "Tier 3",
    "Exploration": "Tier 3",
    "Fundamental Research": "Tier 3",
    "ACM Transactions on Computing for Healthcare": "Tier 3",
    "Intelligent Medicine": "Tier 3",
    "JMIR mHealth and uHealth": "Tier 3",
    "Journal of Medical Internet Research": "Tier 3",
    "International Journal of Food Microbiology": "Tier 3",
    "Microbiological Research": "Tier 3",
}

# -----------------------------
# Helpers
# -----------------------------
def build_issn_clause(issns):
    cleaned = [x.strip() for x in issns if x and x.strip()]
    if not cleaned:
        return None
    return "(" + " OR ".join(f"ISSN:{i}" for i in cleaned) + ")"


def get_hit_count(query, retries=5, sleep_sec=1.0):
    params = {
        "query": query,
        "format": "json",
        "pageSize": 1,
        "resultType": "lite",
        "synonym": "false",
    }
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(EUROPE_PMC_API_URL, params=params, timeout=30)
            r.raise_for_status()
            return int(r.json().get("hitCount", 0))
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    print(f"[WARN] Failed query: {query}\nLast error: {last_err}")
    return 0


def collect_counts_and_tier_sums(sleep_between=0.25):
    year_range = f"[{YEAR_START} TO {YEAR_END}]"

    # per-journal rows
    rows = []

    # per-tier sums
    tier_totals = defaultdict(int)
    tier_oa_totals = defaultdict(int)

    for journal, issns in JOURNAL_ISSNS.items():
        tier = JOURNAL_TIER.get(journal, "UNKNOWN")
        issn_clause = build_issn_clause(issns)

        if issn_clause is None:
            total = 0
            oa = 0
            oa_pct = 0.0
        else:
            base_q = f"{issn_clause} AND {BASE_FILTER} AND PUB_YEAR:{year_range}"
            oa_q = f"{base_q} AND {OA_FILTER}"

            total = get_hit_count(base_q)
            oa = get_hit_count(oa_q)
            oa_pct = (oa / total * 100.0) if total > 0 else 0.0

        tier_totals[tier] += total
        tier_oa_totals[tier] += oa

        rows.append({
            "journal": journal,
            "tier": tier,
            "issns": ";".join(issns),
            "total_epmc": total,
            "oa_epmc": oa,
            "oa_percent": round(oa_pct, 2),
        })

        print(f"{journal:55s}  [{tier:7s}]  total={total:6d}  oa={oa:6d}  oa%={oa_pct:6.2f}")
        time.sleep(sleep_between)

    # Sort journals: Tier (1->2->3->UNKNOWN), then OA%, then OA count
    tier_order = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "UNKNOWN": 9}
    rows_sorted = sorted(
        rows,
        key=lambda r: (tier_order.get(r["tier"], 9), -r["oa_percent"], -r["oa_epmc"], -r["total_epmc"])
    )

    # Build tier summary rows
    summary_rows = []
    for t in ["Tier 1", "Tier 2", "Tier 3", "UNKNOWN"]:
        total = tier_totals.get(t, 0)
        oa = tier_oa_totals.get(t, 0)
        oa_pct = (oa / total * 100.0) if total > 0 else 0.0
        summary_rows.append({
            "tier": t,
            "total_epmc": total,
            "oa_epmc": oa,
            "oa_percent": round(oa_pct, 2),
        })

    # Also compute grand totals (across all tiers)
    grand_total = sum(tier_totals.values())
    grand_oa = sum(tier_oa_totals.values())
    grand_pct = (grand_oa / grand_total * 100.0) if grand_total > 0 else 0.0
    summary_rows.append({
        "tier": "ALL",
        "total_epmc": grand_total,
        "oa_epmc": grand_oa,
        "oa_percent": round(grand_pct, 2),
    })

    return rows_sorted, summary_rows


def write_csv(rows, filename, fieldnames):
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_journal_table(rows, max_name_len=55):
    headers = ["tier", "journal", "issns", "total_epmc", "oa_epmc", "oa_percent"]
    print("\n" + " | ".join(headers))
    print("-" * 140)
    for r in rows:
        j = r["journal"]
        j_disp = (j[: max_name_len - 1] + "…") if len(j) > max_name_len else j
        print(
            f'{r["tier"]:<7} | {j_disp:<{max_name_len}} | {r["issns"]:<25} | '
            f'{r["total_epmc"]:9d} | {r["oa_epmc"]:6d} | {r["oa_percent"]:9.2f}'
        )


def print_tier_summary(summary_rows):
    print("\n=== Tier totals (Europe PMC) ===")
    print("tier    | total_epmc | oa_epmc | oa_percent")
    print("-" * 45)
    for r in summary_rows:
        print(f'{r["tier"]:<7} | {r["total_epmc"]:9d} | {r["oa_epmc"]:6d} | {r["oa_percent"]:9.2f}')


if __name__ == "__main__":
    journal_rows, tier_summary = collect_counts_and_tier_sums()

    print_journal_table(journal_rows)
    print_tier_summary(tier_summary)

    # Save outputs
    out_journals_csv = f"epmc_counts_by_journal_with_tier_{YEAR_START}_{YEAR_END}.csv"
    out_tiers_csv = f"epmc_counts_by_tier_{YEAR_START}_{YEAR_END}.csv"

    write_csv(
        journal_rows,
        out_journals_csv,
        fieldnames=["tier", "journal", "issns", "total_epmc", "oa_epmc", "oa_percent"],
    )
    write_csv(
        tier_summary,
        out_tiers_csv,
        fieldnames=["tier", "total_epmc", "oa_epmc", "oa_percent"],
    )

    print(f"\n[SAVED] {out_journals_csv}")
    print(f"[SAVED] {out_tiers_csv}")
