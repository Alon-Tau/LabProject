import os
import matplotlib.pyplot as plt

def analyze_paragraphs(root_dir, bin_size=50, max_bin=1000):
    n_bins = max_bin // bin_size
    bins = [0] * n_bins
    over_max = 0
    total_paragraphs = 0
    total_files = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            total_files += 1
            path = os.path.join(dirpath, filename)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")
                continue

            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            for p in paragraphs:
                wc = len(p.split())
                total_paragraphs += 1
                if wc > max_bin:
                    over_max += 1
                elif wc > 0:
                    idx = (wc - 1) // bin_size
                    bins[idx] += 1
                else:
                    bins[0] += 1

    labels = [f"{i*bin_size+1}-{(i+1)*bin_size}" for i in range(n_bins)] + [f">{max_bin}"]
    values = bins + [over_max]

    plt.figure(figsize=(16, 8)) # Increased width for better label spacing
    bars = plt.bar(labels, values, color='steelblue', edgecolor='black', alpha=0.8)
    
    # --- ADDING THE NUMBERS ON TOP ---
    for bar in bars:
        height = bar.get_height()
        # Format the number for readability (e.g., 5.7M or 150K)
        if height >= 1_000_000:
            label_text = f'{height/1_000_000:.2f}M'
        elif height >= 1_000:
            label_text = f'{height/1_000:.1f}K'
        else:
            label_text = f'{int(height)}'
            
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 label_text,
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    # ---------------------------------

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Paragraph Word Count", fontweight='bold')
    plt.ylabel("Number of Paragraphs", fontweight='bold')
    plt.title(f"Distribution of Paragraph Lengths\n(Files: {total_files:,} | Total Paragraphs: {total_paragraphs:,})", 
              fontsize=14, pad=20)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Example:
analyze_paragraphs("/home/elhanan/PROJECTS/CHERRY_PICKER_AR/new_corpus")