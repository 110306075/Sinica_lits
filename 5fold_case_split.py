import numpy as np
import pandas as pd
import re
from pathlib import Path
from PIL import Image
import cv2
from sklearn.model_selection import GroupKFold
from fastai.vision.all import *
from test_case_split import extract_case_ids

N_FOLDS = 5


def build_metadata(img_dir='./train_images', mask_dir='./train_masks',
                   slice_list_txt=None, rm_specific_slice=None, n_folds=N_FOLDS, seed=42):
    """
    Build a DataFrame of image-mask pairs, filter by case_ids from slice_list_txt,
    and assign fold indices using GroupKFold by case_id.
    """
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    imgs = sorted(img_dir.glob('*.jpg'))

    invalid_slices = set()
    if rm_specific_slice:
        with open(rm_specific_slice, 'r') as f:
            invalid_slices = {line.strip() for line in f}

    if slice_list_txt:
        valid_cases, test_sets = extract_case_ids(slice_list_txt)
        # print("the train case is:",valid_cases)
        print("the test set is :", test_sets)
        def get_case_id(path):
            m = re.search(r'volume-(\d+)_slice_\d+', path.stem)
            return m.group(1) if m else None
        imgs = [p for p in imgs if get_case_id(p) in valid_cases and p.stem not in invalid_slices]
        print(len(imgs))


    masks = [mask_dir/f'{p.stem}_mask.png' for p in imgs]
    case_ids = [re.search(r'volume-(\d+)_slice_\d+', p.stem).group(1) for p in imgs]

    df = pd.DataFrame({
        'image': imgs,
        'mask': masks,
        'case_id': case_ids
    })
    print(f"Total slices after filtering: {len(df)}")

    # Assign folds by grouping on case_id
    gkf = GroupKFold(n_splits=n_folds)
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df['case_id'])):
        df.loc[val_idx, 'fold'] = fold

    return df


def get_dls(df, fold, bs=16, sz=128):
    valid_idx = df[df.fold == fold].index.tolist()
    dblock = DataBlock(
        blocks=(ImageBlock(), MaskBlock(codes=np.array(['background','liver','tumor']))),
        get_x=ColReader('image'),
        get_y=ColReader('mask'),
        splitter=IndexSplitter(valid_idx),
        item_tfms=Resize(sz),
        batch_tfms=[IntToFloatTensor(), Normalize()]
    )
    return dblock.dataloaders(df, bs=bs, num_workers=0)


def dice(pred, target, class_idx, eps=1e-6):
    pred_bin = (pred == class_idx).astype(np.uint8)
    target_bin = (target == class_idx).astype(np.uint8)
    inter = (pred_bin & target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    return (2*inter + eps) / (union + eps)


def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean()


def cust_foreground_acc(inp, targ):
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1)


def train_and_eval_all_folds(n_folds=N_FOLDS, case='liver_seg',
                             slice_list_txt=None,
                             rm_specific_slice=None,
                             img_dir='train_images', mask_dir='train_masks'):
    df = build_metadata(img_dir=img_dir, mask_dir=mask_dir,
                        slice_list_txt=slice_list_txt, rm_specific_slice=rm_specific_slice, n_folds=n_folds)
    results = []

    for fold in range(n_folds):
        print(f"\nFold {fold}")
        dls = get_dls(df, fold)
        learn = unet_learner(
            dls, resnet50,
            loss_func=CrossEntropyLossFlat(axis=1),
            metrics=[foreground_acc, cust_foreground_acc]
        )
        learn.fine_tune(
            100, wd=0.1,
            cbs=[SaveModelCallback(fname=f'best_fold{fold}'),
                 EarlyStoppingCallback(patience=8)]
        )
        learn.export(f'models/{case}_fold{fold}.pkl')

        preds, _ = learn.get_preds(dl=dls.valid)
        preds = preds.argmax(1).cpu().numpy()
        valid_rows = dls.valid_ds.items.reset_index(drop=True)

        fold_liver_dices, fold_tumor_dices = [], []
        for i, row in valid_rows.iterrows():
            img_name = Path(row['image']).name
            gt = np.array(Image.open(row['mask']), np.uint8)
            pr = preds[i].astype(np.uint8)
            if pr.shape != gt.shape:
                gt = cv2.resize(gt, pr.shape[::-1], interpolation=cv2.INTER_NEAREST)
            liver_dice = dice(pr, gt, class_idx=1)
            tumor_dice = dice(pr, gt, class_idx=2)
            fold_liver_dices.append(liver_dice)
            fold_tumor_dices.append(tumor_dice)
            results.append({'fold': fold, 'image': img_name,
                            'liver_dice': liver_dice, 'tumor_dice': tumor_dice})

        print(f"Fold {fold} – Avg Liver Dice: {np.mean(fold_liver_dices):.4f}"
              f"  Avg Tumor Dice: {np.mean(fold_tumor_dices):.4f}")

    result_df = pd.DataFrame(results)
    print('\n=== Dice scores by image ===')
    print(result_df.head())
    result_df.to_csv(f'{case}_dice_results.csv', index=False)
    return result_df

if __name__ == '__main__':
    df = train_and_eval_all_folds(
        case='large_tumor_and_liver_bycase_omitsmallslice',
        slice_list_txt='./large_tumor_slices.txt',
        img_dir='train_images_v2Window',
        mask_dir='train_masks_v2Window',
        rm_specific_slice = 'small_tumor_slices.txt'
    )
