import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

def plot_images_mask(images_path='train_images',mask_path='train_masks',index=None,slice_num=None):
    # Set your image and mask path
    img_path = Path(f'{images_path}/volume-{index}_slice_{slice_num}.jpg')
    mask_path = Path(f'{mask_path}/volume-{index}_slice_{slice_num}_mask.png')

    # Open image and mask
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    mask_np = np.array(mask)

    total_pixels = mask_np.size
    labels, counts = np.unique(mask_np, return_counts=True)

    print(f"Label distribution in mask:")
    for label, count in zip(labels, counts):
        percentage = 100 * count / total_pixels
        print(f"  Label {label}: {count} pixels ({percentage:.2f}%)")
    # Plot them side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img)
    axs[0].set_title('CT Image')
    axs[0].axis('off')

    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
 plot_images_mask(index=0,slice_num=46)