import os
import nibabel as nib
import numpy as np

def find_large_tumor_slices(seg_dir, threshold_cm=2.0):
    threshold_mm = threshold_cm * 10
    results = set()

    for filename in os.listdir(seg_dir):
        if not filename.startswith("segmentation-"):
            continue

        filepath = os.path.join(seg_dir, filename)
        mask = nib.load(filepath)
        mask_data = mask.get_fdata()
        spacing = mask.header.get_zooms() 

        pixel_area_mm2 = spacing[0] * spacing[1]
        seg_number = filename.split("-")[1].split(".")[0]

        for i in range(mask_data.shape[2]):
            # mask_slice = curr_mask[..., curr_slice]
            slice_mask = mask_data[..., i]
            tumor_pixels = (slice_mask == 2)

            if np.sum(tumor_pixels) == 0:
                continue

            tumor_area_mm2 = np.sum(tumor_pixels) * pixel_area_mm2
            tumor_diameter_mm = np.sqrt(4 * tumor_area_mm2 / np.pi)

            if tumor_diameter_mm > threshold_mm:
                identifier = f"volume-{seg_number}_slice_{i}"
                results.add(identifier)

    return results


def find_small_tumor_slices(seg_dir, threshold_cm=2.0):
    threshold_mm = threshold_cm * 10
    results = set()

    for filename in os.listdir(seg_dir):
        if not filename.startswith("segmentation-"):
            continue

        filepath = os.path.join(seg_dir, filename)
        mask = nib.load(filepath)
        mask_data = mask.get_fdata()
        spacing = mask.header.get_zooms() 

        pixel_area_mm2 = spacing[0] * spacing[1]
        seg_number = filename.split("-")[1].split(".")[0]

        for i in range(mask_data.shape[2]):
            # mask_slice = curr_mask[..., curr_slice]
            slice_mask = mask_data[..., i]
            tumor_pixels = (slice_mask == 2)

            if np.sum(tumor_pixels) == 0:
                continue

            tumor_area_mm2 = np.sum(tumor_pixels) * pixel_area_mm2
            tumor_diameter_mm = np.sqrt(4 * tumor_area_mm2 / np.pi)

            if tumor_diameter_mm <= threshold_mm:
                identifier = f"volume-{seg_number}_slice_{i}"
                results.add(identifier)

    return results


segmentation_dir = "../../Downloads/archive"
tumor_slices = find_small_tumor_slices(segmentation_dir, threshold_cm=2.0)
print(len(tumor_slices))

with open("small_tumor_slices.txt", "w") as f:
    for item in sorted(tumor_slices):
        f.write(f"{item}\n")


# # Print result
# for item in sorted(tumor_slices):
#     print(item)
