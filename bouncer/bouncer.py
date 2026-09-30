import os
import json
import time
import multiprocessing
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm via terminal: pip install tqdm")
    exit()

# ==========================================
# Configuration
# ==========================================
CORPUS_DIR = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus"
OUTPUT_FILE = "stage_1_passed_documents.json"

FEATURES_BY_CATEGORY = {
    "Bacteria": [
        ["Bacteroides dorei", "Bacteroides vulgatus", "B. dorei", "B. vulgatus"],
        ["Bilophila wadsworthia", "B. wadsworthia"],
        ["Clostridium bolteae", "C. bolteae"],
        ["Parabacteroides merdae", "P. merdae"],
        ["Pseudoflavonifractor capillosus", "P. capillosus"],
        ["Ruminococcus sp. 5_1_39BFAA", "Ruminococcus sp."], 
        ["motu_linkage_group_349"],
        ["motu_linkage_group_456"]
    ],
    "Fungi": [],
    "Specific_Genes": [],
    "Metabolites": [
        ["Pyruvic acid", "Pyruvate"],
        ["alpha-ketoglutaric acid", "alpha-ketoglutarate", "a-ketoglutarate"],
        ["D-mannose", "mannose"],
        ["D-glucose", "glucose"],
        ["Isoleucine"],
        ["Glutamic acid", "Glutamate"],
        ["2-oleoylglycerol"],
        ["Lactamide"]
    ],
    "Pathways": [
        ["Keratan sulfate degradation"],
        ["KinABCDE-Spo0FA sporulation control", "sporulation control"],
        ["3-Hydroxypropionate bi-cycle", "3-Hydroxypropionate"],
        ["Multidrug resistance, efflux pump BpeEF-OprC", "efflux pump BpeEF-OprC"]
    ],
    "Host_Biomarkers": []
}

THRESHOLDS = {
    "Bacteria": 6,          # Must find at least 6 bacteria
    "Fungi": 0,             # Optional category
    "Specific_Genes": 0,    # Optional category
    "Metabolites": 6,       # Must find at least 6 metabolites
    "Pathways": 2,          # Optional category
    "Host_Biomarkers": 0    # Set to 0 to prevent automatic failure since the list above is empty
}

# ==========================================
# Pre-processing
# ==========================================
# 1. Lowercase all aliases inside the category dictionary
FEATURES_LOWER = {}
for category, groups in FEATURES_BY_CATEGORY.items():
    FEATURES_LOWER[category] = [[alias.lower() for alias in group] for group in groups]

# ==========================================
# Worker Function (Runs on multiple cores)
# ==========================================
def scan_single_file(file_path):
    """
    Scans a single text file checking for categorized feature thresholds.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().lower()
            
        found_features_dict = {}
        passes_all_thresholds = True
        
        # Iterate through each category
        for category, alias_groups in FEATURES_LOWER.items():
            category_found = []
            
            # Check for each specific feature group in this category
            for alias_group in alias_groups:
                if any(alias in text for alias in alias_group):
                    # Save the primary name (first item) and count it once
                    primary_name = alias_group[0] 
                    category_found.append(primary_name)
            
            # Store what we found for this category
            found_features_dict[category] = category_found
            
            # THE BOUNCER CHECK: Does this category meet the required threshold?
            required_k = THRESHOLDS.get(category, 0)
            if len(category_found) < required_k:
                passes_all_thresholds = False
                break # Short-circuit: The paper failed, stop checking
                
        # If the document survived all category thresholds, it passes!
        if passes_all_thresholds:
            # Calculate total features found across all categories
            total_features = sum(len(items) for items in found_features_dict.values())
            
            return {
                "file_path": str(file_path),
                "total_feature_count": total_features,
                "found_features": found_features_dict
            }
            
    except Exception as e:
        pass # Suppressed so it doesn't break the visual progress bar
        
    return None

# ==========================================
# Main Orchestrator
# ==========================================
def main():
    print(f"Starting Stage 1: The Bouncer (Categorized Mode)")
    print("Required Thresholds:")
    for cat, thresh in THRESHOLDS.items():
        if thresh > 0:
            print(f"  - {cat}: {thresh} required")
    
    start_time = time.time()
    
    # Gather all text files
    path = Path(CORPUS_DIR)
    all_files = list(path.rglob("*.txt"))
    
    print(f"\nFound {len(all_files)} files in the corpus.")
    if not all_files:
        print("No files found. Please check the directory path and file extensions.")
        return

    # Parallel Processing
    num_cores = multiprocessing.cpu_count()
    print(f"Distributing workload across {num_cores} CPU cores...\n")
    
    passed_documents = []
    
    # Map the worker function to all files using a multiprocessing pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Wrap pool.imap with tqdm to get a beautiful progress bar
        results = list(tqdm(pool.imap(scan_single_file, all_files), total=len(all_files), desc="Scanning Corpus"))
        
    # Filter out the failures (the None values)
    passed_documents = [res for res in results if res is not None]
    
    # Save the manifest for Stage 2
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(passed_documents, f, indent=4)
        
    end_time = time.time()
    
    print("\n==========================================")
    print("SCAN COMPLETE")
    print("==========================================")
    print(f"Total time   : {round(end_time - start_time, 2)} seconds")
    print(f"Total files  : {len(all_files)}")
    print(f"Passed files : {len(passed_documents)}")
    print(f"Manifest saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()