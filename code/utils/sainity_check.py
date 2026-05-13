import requests
import time

ISSN_MAP = {
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

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

def hitcount(query):
    r = requests.get(BASE_URL, params={
        "query": query,
        "format": "json",
        "pageSize": 1
    }, timeout=30)
    r.raise_for_status()
    return int(r.json()["hitCount"])

TOTAL_ALL = 0
OA_ALL = 0

for journal, issns in ISSN_MAP.items():
    total_j = 0
    oa_j = 0

    for issn in issns:
        q_total = f'ISSN:"{issn}" AND PUB_YEAR:2025'
        q_oa = f'ISSN:"{issn}" AND PUB_YEAR:2025 AND OPEN_ACCESS:y'

        total = hitcount(q_total)
        oa = hitcount(q_oa)

        total_j += total
        oa_j += oa

        time.sleep(0.1)  # be polite

    non_oa_j = total_j - oa_j
    TOTAL_ALL += total_j
    OA_ALL += oa_j

    print(f"{journal:45s}  total={total_j:5d}  OA={oa_j:5d}  non-OA={non_oa_j:5d}")

print("\n==============================")
print(f"TOTAL (all journals) : {TOTAL_ALL}")
print(f"OA (2025)            : {OA_ALL}")
print(f"NON-OA (2025)        : {TOTAL_ALL - OA_ALL}")
print("==============================")
