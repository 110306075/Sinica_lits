import os
import numpy as np
import nibabel as nib
from PIL import Image

def count_slice_voxels(slice_dir):

    total = 0
    for fname in os.listdir(slice_dir):
       
        path = os.path.join(slice_dir, fname)
        
        mask = np.array(Image.open(path))
        total += np.sum(mask == 1)
    return int(total)

def count_nii_voxels(nii_path):

    img = nib.load(nii_path)
    data = img.get_fdata()
    return int(np.sum(data == 1))

if __name__ == '__main__':
    slice_mask_dir = '../../Downloads/mask0-test'
    slice_voxels = count_slice_voxels(slice_mask_dir)
    print(f"Total voxels from slice masks: {slice_voxels}")

    nii_mask_file = '../../Downloads/archive/segmentation-0.nii'  # or .nii.gz
    nii_voxels = count_nii_voxels(nii_mask_file)
    print(f"Total voxels from NIfTI mask: {nii_voxels}")


