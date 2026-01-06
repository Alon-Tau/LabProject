import os
import time
import requests
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for server use
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------------
# Config
# ----------------------------------

EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

YEAR_START = 1990
YEAR_END = 2025  # inclusive

JOURNALS = [
    "Microbiome",
    "Gut",
    "The ISME Journal",
    "Nature Microbiology",
    "Cell Host & Microbe",
    "Cell Systems",
    "mSystems",
    "Environmental Microbiology",
    "Environmental Microbiology Reports",
    "Applied and Environmental Microbiology",
    "BMC Microbiology",
    "BMC Bioinformatics",
    "Bioinformatics",
    "PLOS Computational Biology",
    "Nature Communications",
    "Nature Medicine",
    "Nature Biotechnology",
    "Nature Methods",
    "Nature",
    "Science",
    "Cell",
    "Proc Natl Acad Sci U S A",
]

# Folder where graphs will be saved
OUTPUT_DIR = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/analysis_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------
# Global filters: only OA journal articles in EPMC, no preprints
# ----------------------------------

COMMON_FILTER = (
    'PUB_TYPE:"Journal Article" '
    'AND IN_EPMC:Y '
    'AND NOT PREPRINT:Y '
    'AND OPEN_ACCESS:Y'
)

# -----------------------------
# Keyword sets
# -----------------------------

KW_SET_1 = [
    # Core microbiome terms
    "microbiome", "gut microbiota", "metagenomics", "microbiome bioinformatics",
    "shotgun metagenomics", "functional metagenomics", "metagenomic sequencing",
    # Gene-related terms
    "gene", "genes", "gene expression", "transcriptomics", "metatranscriptomics",
    "functional genes", "gene regulation", "gene pathways", "gene networks",
    "gene ontology", "go enrichment", "go term", "go terms", "go analysis",
    # Metabolite / metabolism
    "metabolite", "metabolites", "metabolism", "metabolomics", "metabolic pathway",
    "metabolic network", "metabolomic profiling", "metabolite production",
    "short-chain fatty acids", "scfa", "butyrate", "acetate", "propionate",
    "metabolic reconstruction", "metabolic modeling", "microbiome-metabolome",
    "microbial metabolism", "metabolic capacity", "mimosa", "mimosa2",
    # Bacteria / taxa
    "bacteria", "bacterial", "microbiota", "microbial", "microbial community",
    "microbial species", "taxa", "taxonomy", "otu", "otus", "amplicon",
    "16s", "16s rrna", "16s sequencing",
    # Host terms
    "host gene expression", "host transcriptomics", "host response",
    "host-microbiome", "host-microbiota", "host genetics",
    # Pathway-specific / immune
    "tryptophan metabolism", "bile acid metabolism", "bile acids",
    "immune pathways", "immune gene", "immune genes", "immune response",
    "t cell", "t cells", "t-cell", "t-cells",
    "b cell", "b cells", "b-cell", "b-cells",
    # Analysis / tools
    "functional profiling", "functional annotation", "pathway enrichment",
    "gene set enrichment", "gsea", "metagenome-scale",
    "genome-scale metabolic model", "gsmm", "constraint-based modeling",
]

KW_SET_2_EXTRA = [
    "prokaryote", "prokaryotes", "prokaryotic",
    "archaea", "archaeome",
    "microbial ecology", "microbial dysbiosis", "dysbiosis",
    "microbial interactions", "microbe-microbe interactions",
    "microbial diversity", "alpha diversity", "beta diversity",
    "microbial networks",
    "multi-omics", "immune–microbiome interaction",
    "microbiome–immune axis",
    "gut–brain axis", "gut–liver axis", "gut–lung axis",
    "fecal microbiota transplantation", "fmt",
    "probiotics", "prebiotics", "synbiotics", "postbiotics",
    "qiime2", "dada2", "asv", "asvs",
    "humann", "metaphlan",
    "metabolite quantification",
    "tryptamine", "kynurenine pathway",
    "secondary bile acids",
    "branched-chain fatty acids", "amino acid metabolism",
    "trimethylamine", "tma", "tmao",
]

KW_SET_2 = KW_SET_1 + KW_SET_2_EXTRA

CATEGORIES = {
    "all": None,     # no keyword filter
    "kw1": KW_SET_1,
    "kw2": KW_SET_2,
}

# ----------------------------------
# Helper functions
# ----------------------------------

def build_journal_clause(journals):
    return "(" + " OR ".join(f'JOURNAL:"{j}"' for j in journals) + ")"


def build_kw_clause(kw_list):
    return "(" + " OR ".join(f'"{kw}"' for kw in kw_list) + ")"


def get_hit_count(query, max_retries=5, sleep_sec=1.0):
    params = {
        "query": query,
        "format": "json",
        "pageSize": 1,
        "resultType": "lite",
        "synonym": "false",
    }
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(EUROPE_PMC_API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("hitCount", 0))
        except Exception as e:
            last_exc = e
            time.sleep(sleep_sec)

    print(f"[WARN] Failed to get hitCount for query: {query}\nLast error: {last_exc}")
    return 0


# ----------------------------------
# Main data collection (per year)
# ----------------------------------

def collect_stats():
    journal_clause = build_journal_clause(JOURNALS)

    stats = {
        cat: {
            "total": defaultdict(int),
            "oa": defaultdict(int),
        }
        for cat in CATEGORIES.keys()
    }

    years = list(range(YEAR_START, YEAR_END + 1))

    for year in years:
        print(f"=== Year {year} ===")
        # All queries restricted to OA journal articles in our journals
        base_query = (
            f"{journal_clause} AND {COMMON_FILTER} AND PUB_YEAR:{year}"
        )

        # All (no KW filter) – this is already OA-only
        q_all = base_query
        total_all = get_hit_count(q_all)
        stats["all"]["total"][year] = total_all
        # OA == total, because COMMON_FILTER already has OPEN_ACCESS:Y
        stats["all"]["oa"][year] = total_all

        # KW1 / KW2 (still OA-only)
        for cat, kw_list in CATEGORIES.items():
            if kw_list is None:
                continue
            kw_clause = build_kw_clause(kw_list)

            q_kw_total = f"{base_query} AND {kw_clause}"
            total_kw = get_hit_count(q_kw_total)

            stats[cat]["total"][year] = total_kw
            stats[cat]["oa"][year] = total_kw  # OA == total here as well

        time.sleep(0.3)  # be nice to the API

    return years, stats


# ----------------------------------
# Additional data collection (per journal)
# ----------------------------------

def collect_journal_oa_stats():
    """
    For each journal, aggregate over YEAR_START–YEAR_END:
      total OA journal articles (with COMMON_FILTER).
    """
    journal_stats = {}
    year_range_str = f"[{YEAR_START} TO {YEAR_END}]"

    for journal in JOURNALS:
        print(f"=== Journal {journal} (all years, OA stats) ===")
        base = (
            f'JOURNAL:"{journal}" AND {COMMON_FILTER} '
            f'AND PUB_YEAR:{year_range_str}'
        )
        total = get_hit_count(base)  # OA-only total
        journal_stats[journal] = {"total": total, "oa": total}  # OA == total
        print(f"  total={total}, oa={total}")
        time.sleep(0.3)

    return journal_stats


def collect_journal_kw_stats():
    """
    For each journal, aggregate over YEAR_START–YEAR_END:
      total OA journal articles (all), OA+KW1, OA+KW2.
    """
    journal_stats = {}
    year_range_str = f"[{YEAR_START} TO {YEAR_END}]"

    for journal in JOURNALS:
        print(f"=== Journal {journal} (all years, KW stats) ===")
        base = (
            f'JOURNAL:"{journal}" AND {COMMON_FILTER} '
            f'AND PUB_YEAR:{year_range_str}'
        )

        all_total = get_hit_count(base)

        kw1_clause = build_kw_clause(KW_SET_1)
        kw2_clause = build_kw_clause(KW_SET_2)

        kw1_total = get_hit_count(f"{base} AND {kw1_clause}")
        kw2_total = get_hit_count(f"{base} AND {kw2_clause}")

        journal_stats[journal] = {
            "all": all_total,
            "kw1": kw1_total,
            "kw2": kw2_total,
        }

        print(f"  all={all_total}, kw1={kw1_total}, kw2={kw2_total}")
        time.sleep(0.3)

    return journal_stats


# ----------------------------------
# Plotting helpers
# ----------------------------------

def _add_value_labels(bars, values=None, denom_values=None, fontsize=6):
    """
    Write values (and optional percentage) on top of bars.

    If denom_values is provided, the label is:
        "<value>\n(XX.X%)"
    where percent = value / denom * 100.
    """
    if values is None:
        values = [bar.get_height() for bar in bars]

    for i, bar in enumerate(bars):
        height = bar.get_height()
        val = values[i]

        if denom_values is not None:
            denom = denom_values[i]
            pct = (val / denom * 100.0) if denom > 0 else 0.0
            label = f"{val}\n({pct:.1f}%)"
        else:
            label = str(val)

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
        )


# ----------------------------------
# Plotting
# ----------------------------------

def plot_triple_bar_years(years, stats):
    """
    Graph 1:
    Per year, 3 bars:
      - all (OA, no keyword filter)
      - kw1
      - kw2
    Labels on KW1/KW2 show coverage (% of all OA in that year).
    """
    totals_all = [stats["all"]["total"][y] for y in years]
    totals_kw1 = [stats["kw1"]["total"][y] for y in years]
    totals_kw2 = [stats["kw2"]["total"][y] for y in years]

    x = list(range(len(years)))
    width = 0.25

    plt.figure(figsize=(16, 6))
    bars_all = plt.bar([i - width for i in x], totals_all, width, label="No filter (all OA)")
    bars_kw1 = plt.bar(x, totals_kw1, width, label="KW1 (medicinal focus, OA)")
    bars_kw2 = plt.bar([i + width for i in x], totals_kw2, width, label="KW2 (extended microbiome, OA)")

    plt.xticks(x, years, rotation=90)
    plt.xlabel("Publication Year")
    plt.ylabel("Number of matching OA articles")
    plt.title("OA journal articles per year (Europe PMC, selected journals)")
    plt.legend()

    _add_value_labels(bars_all, totals_all)  # just counts
    _add_value_labels(bars_kw1, totals_kw1, denom_values=totals_all)  # counts + %
    _add_value_labels(bars_kw2, totals_kw2, denom_values=totals_all)  # counts + %

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "graph1_triple_bars_years_OA_only.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_oa_ratio_years(years, stats):
    """
    Graph 2:
    Now OA == total (we filtered OA already),
    so the ratio will just be 1.0 for all years.
    You can comment this out if not useful.
    """
    plt.figure(figsize=(16, 6))

    for cat, label in [
        ("all", "No filter (all OA)"),
        ("kw1", "KW1 (OA)"),
        ("kw2", "KW2 (OA)"),
    ]:
        ratios = []
        for y in years:
            tot = stats[cat]["total"][y]
            oa = stats[cat]["oa"][y]
            ratio = oa / tot if tot > 0 else 0
            ratios.append(ratio)

        plt.plot(years, ratios, marker="o", label=label)

    plt.xticks(years, rotation=90)
    plt.xlabel("Publication Year")
    plt.ylabel("OA fraction (OA / total)")
    plt.ylim(0, 1.05)
    plt.title("Open Access ratio per year (trivial = 1.0 because OA filtered)")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "graph2_oa_ratio_OA_only.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_oa_vs_all_per_journal(journal_stats):
    """
    Graph 3:
    Here 'all' == 'OA' because of COMMON_FILTER.
    Still shows counts per journal.
    """
    journals = list(journal_stats.keys())
    totals = [journal_stats[j]["total"] for j in journals]
    oa_vals = [journal_stats[j]["oa"] for j in journals]

    ratios = []
    for tot, oa in zip(totals, oa_vals):
        if tot > 0:
            ratios.append(oa / tot * 100.0)
        else:
            ratios.append(0.0)

    x = range(len(journals))
    width = 0.4

    plt.figure(figsize=(16, 7))

    bars_all = plt.bar(
        [i - width / 2 for i in x],
        totals,
        width=width,
        label="All OA Articles",
    )
    bars_oa = plt.bar(
        [i + width / 2 for i in x],
        oa_vals,
        width=width,
        label="OA Articles (same)",
    )

    xtick_labels = [
        f"{journals[i]}\n({ratios[i]:.1f}% OA)"
        for i in range(len(journals))
    ]
    plt.xticks(x, xtick_labels, rotation=55, ha="right")

    plt.ylabel(f"Number of OA Articles ({YEAR_START}–{YEAR_END})")
    plt.title("OA Articles per Journal (Europe PMC, selected journals)")
    plt.legend()

    _add_value_labels(bars_all, fontsize=6)
    _add_value_labels(bars_oa, fontsize=6)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "graph3_all_vs_oa_per_journal_OA_only.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")

    print("\n=== OA ratios per journal (trivial = 100%) ===")
    for j, tot, oa, r in zip(journals, totals, oa_vals, ratios):
        print(f"{j}: total={tot}, OA={oa}, OA fraction={r:.2f}%")


def plot_kw_coverage_per_journal(journal_stats):
    """
    Graph 4:
    x-axis = journal
    Three bars per journal:
      - all OA (no KW filter)
      - kw1 (OA + KW_SET_1)
      - kw2 (OA + KW_SET_2)
    """
    journals = list(journal_stats.keys())
    totals_all = [journal_stats[j]["all"] for j in journals]
    totals_kw1 = [journal_stats[j]["kw1"] for j in journals]
    totals_kw2 = [journal_stats[j]["kw2"] for j in journals]

    x = range(len(journals))
    width = 0.25

    plt.figure(figsize=(18, 7))

    bars_all = plt.bar(
        [i - width for i in x],
        totals_all,
        width=width,
        label="No filter (all OA)",
    )
    bars_kw1 = plt.bar(
        x,
        totals_kw1,
        width=width,
        label="KW1 (OA, medicinal focus)",
    )
    bars_kw2 = plt.bar(
        [i + width for i in x],
        totals_kw2,
        width=width,
        label="KW2 (OA, extended microbiome)",
    )

    plt.xticks(x, journals, rotation=55, ha="right")
    plt.ylabel(f"Number of OA Articles ({YEAR_START}–{YEAR_END})")
    plt.title("Coverage of Keyword Filters per Journal (OA only)")
    plt.legend()

    _add_value_labels(bars_all, totals_all, fontsize=6)
    _add_value_labels(bars_kw1, totals_kw1, denom_values=totals_all, fontsize=6)
    _add_value_labels(bars_kw2, totals_kw2, denom_values=totals_all, fontsize=6)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "graph4_kw_coverage_per_journal_OA_only.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")

    print("\n=== KW coverage per journal (percent of all OA) ===")
    for j, a, k1, k2 in zip(journals, totals_all, totals_kw1, totals_kw2):
        p1 = (k1 / a * 100.0) if a > 0 else 0.0
        p2 = (k2 / a * 100.0) if a > 0 else 0.0
        print(f"{j}: all={a}, kw1={k1} ({p1:.2f}%), kw2={k2} ({p2:.2f}%)")


# ----------------------------------
# Run
# ----------------------------------

if __name__ == "__main__":
    years, stats = collect_stats()
    plot_triple_bar_years(years, stats)
    plot_oa_ratio_years(years, stats)          # optional now
    journal_oa_stats = collect_journal_oa_stats()
    plot_oa_vs_all_per_journal(journal_oa_stats)
    journal_kw_stats = collect_journal_kw_stats()
    plot_kw_coverage_per_journal(journal_kw_stats)
    print(f"\nAll graphs saved in: {OUTPUT_DIR}")
