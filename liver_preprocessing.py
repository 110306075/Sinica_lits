# liver_preprocessing.py

import os
import glob
from pathlib import Path
import types

import cv2
import imageio
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from matplotlib.pyplot import figure

import torch
from nibabel.orientations import aff2axcodes
from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *

target_axcodes = ('L', 'A', 'S')

# Utility: Load .nii file

def read_nii(filepath):
    ct_scan = nib.load(filepath)
    if aff2axcodes(ct_scan.affine) != target_axcodes:
          ct_scan = nib.as_closest_canonical(ct_scan)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array


# Window settings

dicom_windows = types.SimpleNamespace(
    brain=(80, 40),
    subdural=(254, 100),
    stroke=(8, 32),
    brain_bone=(2800, 600),
    brain_soft=(375, 40),
    lungs=(1500, -600),
    mediastinum=(350, 50),
    abdomen_soft=(400, 50),
    liver=(76, 94),
    spine_soft=(250, 50),
    spine_bone=(1800, 400),
    custom=(200, 60)
)


# Patches for fastai TensorImageBW
class TensorCTScan(TensorImageBW):
    _show_args = {'cmap': 'bone'}


@patch
def windowed(self: Tensor, w, l):
    px = self.clone()
    px_min = l - w // 2
    px_max = l + w // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px - px_min) / (px_max - px_min)


@patch
def freqhist_bins(self: Tensor, n_bins=100):
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
                   tensor([0.999])])
    t = (len(imsd) * t).long()
    return imsd[t].unique()


@patch
def hist_scaled(self: Tensor, brks=None):
    if self.device.type == 'cuda': return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0., 1., len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0., 1.)


@patch
def to_nchan(x: Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins, int) or bins != 0:
        res.append(x.hist_scaled(bins).clamp(0, 1))
    dim = [0, 1][x.dim() == 3]
    return TensorCTScan(torch.stack(res, dim=dim))


@patch
def save_jpg(x: Tensor, path, wins, bins=None, quality=120, combined_mask=None):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins) * 255).byte()
    im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=['RGB', 'CMYK'][x.shape[0] == 4])
    im.save(fn, quality=quality)


# Mask preprocessing

def preprocess_mask(mask_array):
    # mask_array[mask_array == 1] = 0
    # mask_array[mask_array == 2] = 1
    return Image.fromarray(mask_array.astype(np.uint8))


# Placeholder (You should define this properly based on your liver segmentation logic)
def black_outside_liver(image, mask):
    image = np.array(image, dtype=np.float32)
    return np.where(mask > 0, image, 0)


# Main preprocessing function

def preprocess_liver_dataset(dataset_path, output_image_dir='train_images', output_mask_dir='train_masks', train_ratio=0.8):
    all_files = os.listdir(dataset_path)
    volume_files = sorted([f for f in all_files if f.startswith("volume-") and f.endswith(".nii")])
    mask_files = sorted([f for f in all_files if f.startswith("segmentation-") and f.endswith(".nii")])
    assert len(volume_files) == len(mask_files), "Mismatch between volume and segmentation files!"

    df_files = pd.DataFrame({
        "dirname": dataset_path,
        "filename": volume_files,
        "mask_dirname": dataset_path,
        "mask_filename": mask_files
    })
    df_train = df_files
    # df_files = df_files.sample(frac=1, random_state=42).reset_index(drop=True)
    # train_size = int(train_ratio * len(df_files))
    # df_train = df_files.iloc[:train_size].reset_index(drop=True)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    for idx in tqdm(range(len(df_train))):
        row = df_train.loc[idx]
        ct_path = str(Path(row["dirname"]) / row["filename"])
        mask_path = str(Path(row["mask_dirname"]) / row["mask_filename"])
        curr_ct = read_nii(ct_path)
        curr_mask = read_nii(mask_path)
        curr_file_name = row["filename"].split('.')[0]
        curr_dim = curr_ct.shape[2]

        valid_slices = []
        for curr_slice in range(curr_dim):
            mask_slice = curr_mask[..., curr_slice].astype('uint8')
            unique_vals = np.unique(mask_slice)

            if len(unique_vals) == 1 and unique_vals[0] == 0:
                continue  
            valid_slices.append(curr_slice)

        
        for final_slice_idx in valid_slices:
            blacked_data = black_outside_liver(curr_ct[..., final_slice_idx], curr_mask[..., final_slice_idx])
            data = tensor(blacked_data.astype(np.float32))
            
            mask_slice = curr_mask[..., final_slice_idx].astype('uint8')
            processed_mask = preprocess_mask(mask_slice)

            image_filename = f"{curr_file_name}_slice_{final_slice_idx}.jpg"
            mask_filename  = f"{curr_file_name}_slice_{final_slice_idx}_mask.png"

            data.save_jpg(Path(output_image_dir) / image_filename,
                          [dicom_windows.liver, dicom_windows.custom])
            processed_mask.save(Path(output_mask_dir) / mask_filename)

    print("✅ Preprocessing complete.")


if __name__ == '__main__':
    dataset_path = "/mnt/c/Users/Elaine/Downloads/archive" # Modify this path
    preprocess_liver_dataset(dataset_path)
