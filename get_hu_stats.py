import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from liver_preprocessing import read_nii

def compute_hu_stats(ct_data, seg_data, label_val):
    
    mask = (seg_data == label_val)
    if not np.any(mask):
        return {
            "mean": np.nan,
            "std":  np.nan,
            "min":  np.nan,
            "max":  np.nan
        }

    vals = ct_data[mask]
    return {
        "mean": float(np.mean(vals)),
        "std":  float(np.std(vals)),
        "min":  float(np.min(vals)),
        "max":  float(np.max(vals))
    }

def build_file_pairs(dataset_path):
   
    files = os.listdir(dataset_path)
    volume_files = sorted([f for f in files if f.startswith("volume-") and f.endswith(".nii")])
    mask_files   = sorted([f for f in files if f.startswith("segmentation-") and f.endswith(".nii")])

    assert len(volume_files) == len(mask_files), "Mismatch between volume and mask files"

    data = []
    for vol, seg in zip(volume_files, mask_files):
        data.append({
            "filename": vol,
            "mask_filename": seg,
            "dirname": dataset_path,
            "mask_dirname": dataset_path
        })

    return pd.DataFrame(data)


def process_lits_dataset(dataset_path, save_csv_path=None):
    
    df_files = build_file_pairs(dataset_path)
    records = []

    for idx, row in tqdm(df_files.iterrows(), total=len(df_files)):
        ct_path = os.path.join(row["dirname"], row["filename"])
        seg_path = os.path.join(row["mask_dirname"], row["mask_filename"])

        ct_data = read_nii(ct_path)
        seg_data = read_nii(seg_path)

        liver_stats = compute_hu_stats(ct_data, seg_data, label_val=1)
        tumor_stats = compute_hu_stats(ct_data, seg_data, label_val=2)

        records.append({
            "VolumeFile": row["filename"],
            "MaskFile":   row["mask_filename"],
            "LiverMean":  liver_stats["mean"],
            "LiverStd":   liver_stats["std"],
            "LiverMin":   liver_stats["min"],
            "LiverMax":   liver_stats["max"],
            "TumorMean":  tumor_stats["mean"],
            "TumorStd":   tumor_stats["std"],
            "TumorMin":   tumor_stats["min"],
            "TumorMax":   tumor_stats["max"]
        })

    results_df = pd.DataFrame(records)

    if save_csv_path:
        results_df.to_csv(save_csv_path, index=False)
        print(f"Saved HU stats to: {save_csv_path}")

    return results_df

dataset_path = '../../Downloads/archive'
df = process_lits_dataset(dataset_path, save_csv_path='./hu_stats.csv')
print(df.head())