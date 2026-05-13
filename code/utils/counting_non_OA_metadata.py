import json, re, argparse

def norm_pmcid(raw):
    if raw is None:
        return ""
    x = str(raw).strip().upper()
    if not x:
        return ""
    if x.endswith(".TXT"):
        x = x[:-4]
    # digits only
    if x.isdigit():
        return "PMC" + x
    # already PMC\d+
    m = re.search(r"(PMC\\d+)", x)
    if m:
        return m.group(1)
    # 'PMC' + digits-ish
    if x.startswith("PMC") and x[3:].isdigit():
        return x
    return ""

def meta_year(meta):
    for k in ("year","pubYear","publicationYear","firstPublicationDate","pubDate","date"):
        v = meta.get(k)
        if not v:
            continue
        m = re.search(r"(19|20)\\d{2}", str(v))
        if m:
            return int(m.group(0))
    try:
        y = meta.get("journalInfo", {}).get("yearOfPublication")
        if y:
            return int(y)
    except Exception:
        pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-jsonl", default="/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus/metadata_all.jsonl")
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()

    total = 0
    with_real_pmcid = 0

    with open(args.metadata_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                meta = json.loads(line)
            except Exception:
                continue

            if meta_year(meta) != args.year:
                continue

            total += 1
            raw = meta.get("pmcid") or meta.get("PMCID") or meta.get("id")
            pmcid = norm_pmcid(raw)
            if pmcid.startswith("PMC") and pmcid[3:].isdigit():
                with_real_pmcid += 1

    print(f"Metadata records parsed as {args.year}: {total}")
    print(f"...of those with a real PMCID (pmcid/PMCID/id normalized): {with_real_pmcid}")

if __name__ == "__main__":
    main()

