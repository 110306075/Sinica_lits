import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os

def compute_tumor_percentage(mask_path):
    mask_np = np.array(Image.open(mask_path))
    total_pixels = mask_np.size
    tumor_pixels = np.sum(mask_np == 2)  # assuming tumor is labeled with '2'
    percentage = 100 * tumor_pixels / total_pixels
    return percentage

def get_bin(percentage):
    if percentage < 1:
        return '<1%'  # Less than 1%
    elif 1 <= percentage < 5:
        return '1%-5%'
    elif 5 <= percentage < 10:
        return '5%-10%'
    else:
        return '>10%'  # Above 10%

def main():
    # Load the poor cases CSV
    df = pd.read_csv('poor_cases.csv')

    os.makedirs('fold_tumor_stats', exist_ok=True)

    for fold_name in df.columns:
        print(f"\nAnalyzing tumor percentages for {fold_name}...")

        tumor_percentages = []

        for image_name in df[fold_name].dropna():
            mask_path = Path('train_masks') / f"{image_name}_mask.png"
            if mask_path.exists():
                tumor_percentage = compute_tumor_percentage(mask_path)
                tumor_percentages.append(tumor_percentage)
            else:
                print(f"Mask not found: {mask_path}")

        if tumor_percentages:
            # Categorize tumor percentages into bins
            bins = ['<1%', '1%-5%', '5%-10%', '>10%']
            bin_counts = {bin_label: 0 for bin_label in bins}

            # Sorting percentages into bins
            for percentage in tumor_percentages:
                bin_label = get_bin(percentage)
                bin_counts[bin_label] += 1

            # Plotting
            plt.figure(figsize=(7,5))
            plt.bar(bin_counts.keys(), bin_counts.values(), color='blue')
            plt.xlabel('Tumor Pixel Percentage Range')
            plt.ylabel('Number of Images')
            plt.title(f'Tumor Pixel Percentage Distribution in {fold_name}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'fold_tumor_stats/{fold_name}_tumor_distribution.png')
            plt.close()
        else:
            print(f"No masks analyzed for {fold_name}.")

if __name__ == "__main__":
    main()
