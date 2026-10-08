import os
import numpy as np
import pandas as pd
import rasterio

S1_DIR = "data/raw/sen1floods11/S1Hand"
LABEL_DIR = "data/raw/sen1floods11/LabelHand"
SPLIT_FILE = "data/raw/sen1floods11/splits/flood_train_data.csv"

df = pd.read_csv(SPLIT_FILE, header=None)

sum_values = np.zeros(2, dtype=np.float64)
sum_squares = np.zeros(2, dtype=np.float64)
pixel_count = np.zeros(2, dtype=np.int64)

global_min = np.full(2, np.inf)
global_max = np.full(2, -np.inf)

skipped_chips = 0
invalid_pixels = 0

for _, row in df.iterrows():
    image_name = row[0]
    label_name = row[1]

    image_path = os.path.join(S1_DIR, image_name)
    label_path = os.path.join(LABEL_DIR, label_name)

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float64)

    with rasterio.open(label_path) as src:
        label = src.read(1)

    # Valid ground-truth pixels
    label_valid = label != -1

    if not np.any(label_valid):
        skipped_chips += 1
        continue

    for band in range(2):
        # Pixel must have a valid label AND finite SAR value
        valid = label_valid & np.isfinite(image[band])

        values = image[band][valid]

        invalid_pixels += np.sum(label_valid & ~np.isfinite(image[band]))

        if values.size == 0:
            continue

        sum_values[band] += values.sum()
        sum_squares[band] += np.square(values).sum()
        pixel_count[band] += values.size

        global_min[band] = min(global_min[band], values.min())
        global_max[band] = max(global_max[band], values.max())

mean = sum_values / pixel_count
variance = (sum_squares / pixel_count) - np.square(mean)
variance = np.maximum(variance, 0)
std = np.sqrt(variance)

print("\nTraining-set Sentinel-1 statistics")
print("-----------------------------------")

for band in range(2):
    print(f"\nBand {band + 1}")
    print(f"Pixels : {pixel_count[band]:,}")
    print(f"Min    : {global_min[band]:.4f}")
    print(f"Max    : {global_max[band]:.4f}")
    print(f"Mean   : {mean[band]:.4f}")
    print(f"Std    : {std[band]:.4f}")

print(f"\nCompletely unlabeled chips skipped : {skipped_chips}")
print(f"Non-finite SAR pixels ignored     : {invalid_pixels:,}")