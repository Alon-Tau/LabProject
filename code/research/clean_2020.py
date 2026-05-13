import json

def filter_file(src, dest):
    kept = removed = 0
    with open(src, encoding="utf-8") as fin, open(dest, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # If a line is corrupted or not JSON, keep it instead of losing metadata
                fout.write(line + "\n")
                continue

            # Remove only 2020
            if str(obj.get("pub_year")) == "2020":
                removed += 1
                continue

            # Keep all other years
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"{src}: kept {kept} lines, removed {removed} (pub_year=2020)")


# Clean both metadata files
filter_file("metadata_all_1990_2025.jsonl", "metadata_all_1990_2025.cleaned.jsonl")
filter_file("metadata_keywords_1990_2025.jsonl", "metadata_keywords_1990_2025.cleaned.jsonl")
