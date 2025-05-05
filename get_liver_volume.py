import nibabel as nib
import numpy as np
import pandas as pd
import os


def get_liver_volume(dataset_path):
    records = []
    df = pd.DataFrame()
    all_files = os.listdir(dataset_path)
    mask_files = sorted([f for f in all_files if f.startswith("segmentation-") and f.endswith(".nii")])

    for mask_path in mask_files:
        mask_nii = nib.load(dataset_path+'/'+mask_path)
        mask_data = mask_nii.get_fdata()
        spacing = mask_nii.header.get_zooms() 

        liver_voxels = np.sum(mask_data == 1)
        voxel_volume = spacing[0] * spacing[1] * spacing[2] 
        liver_volume_mm3 = liver_voxels * voxel_volume

        records.append({
            "filename": mask_path,
            "voxels": liver_voxels,
            "volume_mm3": liver_volume_mm3,
            "volume_cm3": liver_volume_mm3 / 1000,
            "volume_liters": liver_volume_mm3 / 1_000_000
        })
    df = pd.DataFrame(records)
    df.to_csv("liver_volumes.csv", index=False)
    return df

df = get_liver_volume("../../Downloads/archive")
print(df.head())
