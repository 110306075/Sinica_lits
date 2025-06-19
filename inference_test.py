from fastai.vision.all import *
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# ---- Configuration ----
MODEL_NAME = "0601_large_tumor_and_liver_bycase_new_save_best_model_Reload_best_fold4.pkl"
MODEL_PKL = f"models/{MODEL_NAME}"
# INPUT_IMG = "train_images_v2Window/volume-27_slice_551.jpg"        # input CT slice
# GT_MASK   = "train_masks_v2Window/volume-27_slice_551_mask.png"   # ground-truth mask
IMG_DIR     = Path("train_images_v2Window")
MASK_DIR    = Path("train_masks_v2Window")
# VOLUME_ID   = 27
LIVER_CLASS_IDX = 1  
TUMOR_CLASS_INDX = 2                           
TEST_INDEX = [51,88,109,101,50,75,64,108,57,100]



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



# ---- Load model ----
learn = load_learner(MODEL_PKL)
learn.model = learn.model.cuda()

summaries = []

for i in TEST_INDEX:

    pattern   = f"volume-{i}_slice_*.jpg"
    img_paths = sorted(IMG_DIR.glob(pattern))

    dice_scores_liver  = []
    pred_voxels_liver  = []
    gt_voxels_liver    = []
    dice_scores_tumor  = []
    pred_voxels_tumor  = []
    gt_voxels_tumor  = []

    for img_path in img_paths:
        stem      = img_path.stem                  # e.g. "volume-27_slice_551"
        slice_id  = stem.split('_')[-1]

        mask_path = MASK_DIR / f"{stem}_mask.png"
        if not mask_path.exists():
            print(f"Missing GT mask for {stem}, skipping")
            continue

        # a) Predict
        img       = PILImage.create(img_path)
        pred_mask = learn.predict(img)[0].numpy()  # (H, W)

        # b) Load GT and resize if needed
        gt_mask = np.array(Image.open(mask_path), dtype=np.uint8)
        if pred_mask.shape != gt_mask.shape:
            gt_mask = cv2.resize(
                gt_mask,
                dsize=pred_mask.shape[::-1],
                interpolation=cv2.INTER_NEAREST
            )

        # c) Dice
        d_liver = dice(pred_mask, gt_mask, LIVER_CLASS_IDX)
        d_tumor = dice(pred_mask, gt_mask, TUMOR_CLASS_INDX)
        dice_scores_liver.append(d_liver)
        dice_scores_tumor.append(d_tumor)

        # d) Voxel counts
        pv = int((pred_mask == LIVER_CLASS_IDX).sum())
        gv = int((gt_mask   == LIVER_CLASS_IDX).sum())
        pred_voxels_liver.append(pv)
        gt_voxels_liver.append(gv)

        pv_t = int((pred_mask == TUMOR_CLASS_INDX).sum())
        gv_t = int((gt_mask   == TUMOR_CLASS_INDX).sum())
        pred_voxels_tumor.append(pv_t)
        gt_voxels_tumor.append(gv_t)

        # print(f"Slice {slice_id}: Dice={d:.4f}, PredVoxels={pv}, GTVoxels={gv}")

    avg_liver_dice   = np.mean(dice_scores_liver)  if dice_scores_liver else 0
    avg_tumor_dice   = np.mean(dice_scores_tumor)  if dice_scores_tumor else 0
    total_pred_l     = sum(pred_voxels_liver)
    total_gt_l       = sum(gt_voxels_liver)
    total_pred_t     = sum(pred_voxels_tumor)
    total_gt_t       = sum(gt_voxels_tumor)

    # append a dict for this case
    summaries.append({
        'case_id':             f"volume-{i}",
        'Avg_liver_dice':      avg_liver_dice,
        'Avg_tumor_dice':      avg_tumor_dice,
        'pred_liver_voxel':    total_pred_l,
        'gt_liver_voxel':      total_gt_l,
        'liver_diff_abs':      abs(total_pred_l - total_gt_l),
        'pred_tumor_voxel':    total_pred_t,
        'gt_tumor_voxel':      total_gt_t,
        'tumor_diff_abs':      abs(total_pred_t - total_gt_t),
    })

    # optional: still print to console
    print(f"volume-{i}  liver Dice avg {avg_liver_dice:.4f}, tumor Dice avg {avg_tumor_dice:.4f}")

# build DataFrame and export
df = pd.DataFrame(summaries)
df.to_csv(f"./global_test_result/{MODEL_NAME}_segmentation_test_summary.csv", index=False)
print("Wrote segmentation_summary.csv with", len(df), "rows.")

    # 4) Summary
    # dice_arr_l = np.array(dice_scores_liver)
    # dice_arr_t = np.array(dice_scores_tumor)
    # print(f"\n=== Summary for volume-{i} ===")
    # print("for liver: ")
    # print(f"Dice   → avg: {dice_arr_l.mean():.4f}, max: {dice_arr_l.max():.4f}, min: {dice_arr_l.min():.4f}")
    # print(f"Total Predicted Voxels (class {LIVER_CLASS_IDX}): {sum(pred_voxels_liver)}")
    # print(f"Total GT Voxels       (class {LIVER_CLASS_IDX}): {sum(gt_voxels_liver)}")
    # print("for tumor: ")
    # print(f"Dice   → avg: {dice_arr_t.mean():.4f}, max: {dice_arr_t.max():.4f}, min: {dice_arr_t.min():.4f}")
    # print(f"Total Predicted Voxels (class {TUMOR_CLASS_INDX}): {sum(pred_voxels_tumor)}")
    # print(f"Total GT Voxels       (class {TUMOR_CLASS_INDX}): {sum(gt_voxels_tumor)}")


#visualization

# input_img_np = np.array(img)

# plt.figure(figsize=(12, 4))

# # 1. Original image
# plt.subplot(1, 4, 1)
# plt.imshow(input_img_np, cmap='gray')
# plt.title("Original Slice")
# plt.axis('off')

# # 2. Predicted liver mask
# plt.subplot(1, 4, 2)
# plt.imshow(pred_mask == CLASS_IDX, cmap='Reds')
# plt.title("Predicted Liver")
# plt.axis('off')

# # 3. Ground Truth mask
# plt.subplot(1, 4, 3)
# plt.imshow(gt_mask == CLASS_IDX, cmap='Greens')
# plt.title("Ground Truth Liver")
# plt.axis('off')

# # 4. Overlay
# plt.subplot(1, 4, 4)
# plt.imshow(input_img_np, cmap='gray')
# plt.imshow(gt_mask == CLASS_IDX, cmap='Greens', alpha=0.4)
# plt.imshow(pred_mask == CLASS_IDX, cmap='Reds', alpha=0.4)
# plt.title("Overlay (GT=Green, Pred=Red)")
# plt.axis('off')

# plt.suptitle(f"Liver Dice Score: {dice_score:.4f}", fontsize=14)
# plt.tight_layout()
# plt.show()
