import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

# Set your image and mask path
img_path = Path('train_images/volume-100_slice_408.jpg')
mask_path = Path('train_masks/volume-100_slice_408_mask.png')

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
