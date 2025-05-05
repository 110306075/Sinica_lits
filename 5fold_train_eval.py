import numpy as np, pandas as pd, cv2
from pathlib import Path
from PIL import Image
from sklearn.model_selection import KFold
from fastai.vision.all import *


N_FOLDS = 5

def build_metadata(img_dir='./train_images', mask_dir='./train_masks',
                   n_folds=N_FOLDS, seed=42,filter_tumor_size_path=None):
    
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    imgs  = sorted(img_dir.glob('*.jpg'))


    if filter_tumor_size_path:
        with open(filter_tumor_size_path, 'r') as f:
            valid_keywords = set(line.strip() for line in f)
        print(len(valid_keywords))
        imgs = [img for img in imgs if img.stem in valid_keywords]


    # img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    # imgs  = sorted(img_dir.glob('*.jpg'))
    masks = [mask_dir/f'{p.stem}_mask.png' for p in imgs]

    df = pd.DataFrame({'image': imgs, 'mask': masks})
    print("shape of training data",df.shape)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_col = np.zeros(len(df), dtype=int)
    for fold, (_, val_idx) in enumerate(kf.split(df)):
        fold_col[val_idx] = fold
    df['fold'] = fold_col
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
    return dblock.dataloaders(df, bs=bs,num_workers=0)


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
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1) # 3 is a dummy value to include the background which is 0

def train_and_eval_all_folds(n_folds=N_FOLDS,case = 'liver_seg',filter_tumor_size_path=None,img_dir='train_images',mask_dir='train_masks'):
    df = build_metadata(n_folds=n_folds,filter_tumor_size_path=filter_tumor_size_path,img_dir=img_dir,mask_dir=mask_dir)
    results = []

    for fold in range(n_folds):
        print(f'\n Fold {fold}')
        dls = get_dls(df, fold)

        learn = unet_learner(
            dls, resnet50,
            loss_func=CrossEntropyLossFlat(axis=1),
            metrics=[foreground_acc, cust_foreground_acc]
        )
        learn.fine_tune(
            100, wd=0.1,
            cbs=[SaveModelCallback(fname=f'best_fold{fold}'),
                 EarlyStoppingCallback(patience=5)]
        )
        learn.export(f'models/{case}model_fold{fold}.pkl')


        preds, _ = learn.get_preds(dl=dls.valid)
        preds = preds.argmax(1).cpu().numpy()

        valid_rows = dls.valid_ds.items.reset_index(drop=True)


        fold_liver_dices = []
        fold_tumor_dices = []
        for i, row in valid_rows.iterrows():
            img_name = Path(row['image']).name
            gt  = np.array(Image.open(row['mask']), np.uint8)
            pr  = preds[i].astype(np.uint8)

            if pr.shape != gt.shape:
                gt = cv2.resize(gt, pr.shape[::-1], interpolation=cv2.INTER_NEAREST)

            liver_dice = dice(pr, gt, class_idx=1)  
            tumor_dice = dice(pr, gt, class_idx=2)  

            fold_liver_dices.append(liver_dice)
            fold_tumor_dices.append(tumor_dice)

            results.append({
                'fold': fold,
                'image': img_name,
                'liver_dice': liver_dice,
                'tumor_dice': tumor_dice
            })


        print(f'Fold {fold} – Avg Liver Dice: {np.mean(fold_liver_dices):.4f}  Avg Tumor Dice: {np.mean(fold_tumor_dices):.4f}')

    result_df = pd.DataFrame(results)
    print('\n=== Dice scores by image ===')
    print(result_df.head())
    result_df.to_csv(f'{case}_dice_results.csv', index=False)
    return result_df

# --------------------------------------------------
if __name__ == '__main__':
    result_df = train_and_eval_all_folds(case='large_tumor',filter_tumor_size_path='./large_tumor_slices.txt',img_dir='train_images_v2Window',mask_dir='train_masks_v2Window')
