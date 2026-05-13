import os
import json

# Define the base directory
base_path = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/corpus_chunks/new_corpus_chunks/"

total_fulltext_tokens = 0
total_meta_tokens = 0
years_processed = 0

# Iterate through each year folder in the directory
for year_folder in sorted(os.listdir(base_path)):
    folder_path = os.path.join(base_path, year_folder)
    
    if os.path.isdir(folder_path):
        # Construct the expected filename pattern
        filename = f"{year_folder}_chunking_stats_650w.json"
        file_path = os.path.join(folder_path, filename)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Summing the specific token keys
                total_fulltext_tokens += data.get("tokens_fulltext_total", 0)
                total_meta_tokens += data.get("tokens_meta_only_total", 0)
                years_processed += 1

# Results
grand_total = total_fulltext_tokens + total_meta_tokens

print(f"Processed {years_processed} years.")
print(f"Total Fulltext Tokens: {total_fulltext_tokens:,}")
print(f"Total Meta Tokens:     {total_meta_tokens:,}")
print(f"---")
print(f"Grand Total Tokens:    {grand_total:,}")