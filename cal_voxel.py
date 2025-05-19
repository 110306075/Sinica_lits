import os
import re
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import nibabel as nib
import torch
from fastai.vision.all import load_learner, DataBlock, ImageBlock, ColReader, Resize, IntToFloatTensor, Normalize
# ──────────────── CONFIG ────────────────
MODEL_PKL       = Path("models/large_tumor_and_liver_bycase_omitsmallslice_fold3.pkl")
JPEG_DIR        = Path("../../Downloads/voxel_test/volume-76")
SLICE_MASK_DIR  = Path("../../Downloads/voxel_test/volume-76mask")
NIFTI_DIR       = Path("../../Downloads/archive")
PRED_CLASSES    = [1, 2]       # 1=liver, 2=tumor
BATCH_SIZE      = 16
IMG_SIZE        = 128
# ─────────────────────────────────────────

def dice(pred, target,class_idx, eps=1e-6):
    pred_bin = (pred == class_idx).astype(np.uint8)
    target_bin = (target == class_idx).astype(np.uint8)

    inter = (pred_bin & target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    return (2*inter + eps) / (union + eps)


def foreground_acc(inp, targ, bkg_idx=0, axis=1):  # exclude a background from metric
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean()

def cust_foreground_acc(inp, targ):  # # include a background into the metric
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1)
# --- helper functions ---
def count_slice_voxels_by_class(slice_dir, classes):
    counts = {cls:0 for cls in classes}
    for fname in os.listdir(slice_dir):
        mask = np.array(Image.open(slice_dir/fname))
        for cls in classes:
            counts[cls] += int((mask == cls).sum())
    return counts

def count_nii_voxels_by_class(nii_path, classes):
    img = nib.load(str(nii_path))
    data = img.get_fdata()
    counts = {cls:int((data == cls).sum()) for cls in classes}
    return counts

# --- 1) Load learner & make sure it’s on GPU ---
learn = load_learner(MODEL_PKL)
learn.model = learn.model.cuda()
print(f"Model device: {next(learn.model.parameters()).device}")

# --- 2) Build test DataLoader over your JPEGs ---
jpgs = sorted(JPEG_DIR.glob("volume-*_slice_*.jpg"))
df_test = pd.DataFrame({'image': jpgs})

dblock = DataBlock(
    blocks    =(ImageBlock(),),
    get_x     =ColReader('image'),
    item_tfms =Resize(IMG_SIZE),
    batch_tfms=[IntToFloatTensor(), Normalize()]
)
test_dl = learn.dls.test_dl(df_test['image'], bs=BATCH_SIZE, device=torch.device('cuda'))

# --- 3) Run inference ---
print("Running batch prediction...")
preds, _ = learn.get_preds(dl=test_dl)
# preds shape is (N, C, H, W); get pixel‐wise class
pred_classes = preds.argmax(1).cpu().numpy().ravel()

mask_files = sorted(Path(SLICE_MASK_DIR).glob("volume-*_slice_*.png")) 
# stack them into a single array of shape (N, H, W)
gt_stack = np.stack([np.array(Image.open(p)) for p in mask_files], axis=0)


# ──────────────── COMPUTE DICE PER SLICE & CLASS ────────────────
classes = [1, 2]  # liver, tumor
dice_scores = {cls: [] for cls in classes}

# iterate over slices
for i in range(pred_classes.shape[0]):
    pred_slice = pred_classes[i]
    gt_slice   = gt_stack[i]

    if pred_slice.shape != gt_slice.shape:
        # cv2.resize needs (width, height), so reverse the shape
        gt_slice = cv2.resize(gt_slice,
                              dsize=pred_slice.shape[::-1],
                              interpolation=cv2.INTER_NEAREST)
    
    # compute for each class
    for cls in classes:
        score = dice(pred_slice, gt_slice, cls)
        dice_scores[cls].append(score)

# ──────────────── SUMMARIZE ────────────────
for cls in classes:
    arr = np.array(dice_scores[cls])
    print(f"\nClass {cls} Dice scores over {len(arr)} slices:")
    print(f"  • Average: {arr.mean():.4f}")
    print(f"  • Maximum: {arr.max():.4f}")
    print(f"  • Minimum: {arr.min():.4f}")


# --- 4) Count predicted voxels for each class ---
unique, counts = np.unique(pred_classes, return_counts=True)
pred_counts = dict(zip(unique.tolist(), counts.tolist()))
for cls in PRED_CLASSES:
    print(f"Predicted voxels for class {cls}: {pred_counts.get(cls,0)}")

# --- 5) Count ground‐truth voxels in your slice masks ---
slice_counts = count_slice_voxels_by_class(SLICE_MASK_DIR, PRED_CLASSES)
for cls, c in slice_counts.items():
    print(f"Slice‐mask voxels for class {cls}: {c}")

# --- 6) Count ground‐truth voxels in the NIfTI file(s) ---
# (example for a single file; loop over several if you have more)
nii_file = NIFTI_DIR/"segmentation-76.nii"  # adjust filename as needed
nii_counts = count_nii_voxels_by_class(nii_file, PRED_CLASSES)
for cls, c in nii_counts.items():
    print(f"NIfTI voxels for class {cls}: {c}")
