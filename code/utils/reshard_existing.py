import os

# Configuration
INPUT_DIR = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs"
OUTPUT_DIR = "/home/elhanan/PROJECTS/CHERRY_PICKER_AR/data/batch_inputs_v2"
LINES_PER_FILE = 12000 

def split_files():
    print(f"Starting split from {INPUT_DIR} to {OUTPUT_DIR}...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if not file.endswith(".jsonl"): continue
            
            # Skip the years you already successfully processed
            if any(yr in file for yr in ["1990", "1991", "2014"]):
                print(f"--- Skipping already completed file: {file}")
                continue

            input_path = os.path.join(root, file)
            year = root.split('/')[-1]
            target_dir = os.path.join(OUTPUT_DIR, year)
            os.makedirs(target_dir, exist_ok=True)

            print(f"Processing: {file}")
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            total_lines = len(lines)
            for i in range(0, total_lines, LINES_PER_FILE):
                part_num = (i // LINES_PER_FILE) + 1
                out_name = file.replace(".jsonl", f"_p{part_num:02d}.jsonl")
                with open(os.path.join(target_dir, out_name), 'w') as out_f:
                    out_f.writelines(lines[i:i + LINES_PER_FILE])
            
            print(f"  -> Successfully split into {part_num} parts.")

if __name__ == "__main__":
    split_files()