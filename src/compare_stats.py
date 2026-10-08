import os
import numpy as np
import pandas as pd
import rasterio

S1_DIR = "data/raw/sen1floods11/S1Hand"
LABEL_DIR = "data/raw/sen1floods11/LabelHand"
SPLIT_FILE = "data/raw/sen1floods11/splits/flood_train_data.csv"

df = pd.read_csv(SPLIT_FILE, header=None)

all_sum = np.zeros(2, dtype=np.float64)
all_sq = np.zeros(2, dtype=np.float64)
all_count = np.zeros(2, dtype=np.int64)

label_sum = np.zeros(2, dtype=np.float64)
label_sq = np.zeros(2, dtype=np.float64)
label_count = np.zeros(2, dtype=np.int64)

all_min = np.full(2, np.inf)
all_max = np.full(2, -np.inf)

label_min = np.full(2, np.inf)
label_max = np.full(2, -np.inf)

for _, row in df.iterrows():

    image_path = os.path.join(S1_DIR, row[0])
    label_path = os.path.join(LABEL_DIR, row[1])

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float64)

    with rasterio.open(label_path) as src:
        label = src.read(1)

    for band in range(2):

        # A: all finite SAR pixels
        finite = np.isfinite(image[band])
        values = image[band][finite]

        if values.size:
            all_sum[band] += values.sum()
            all_sq[band] += np.square(values).sum()
            all_count[band] += values.size

            all_min[band] = min(all_min[band], values.min())
            all_max[band] = max(all_max[band], values.max())

        # B: finite SAR + valid label
        valid = finite & (label != -1)
        values = image[band][valid]

        if values.size:
            label_sum[band] += values.sum()
            label_sq[band] += np.square(values).sum()
            label_count[band] += values.size

            label_min[band] = min(label_min[band], values.min())
            label_max[band] = max(label_max[band], values.max())


all_mean = all_sum / all_count
all_var = (all_sq / all_count) - all_mean**2
all_std = np.sqrt(np.maximum(all_var, 0))

label_mean = label_sum / label_count
label_var = (label_sq / label_count) - label_mean**2
label_std = np.sqrt(np.maximum(label_var, 0))

print("\nALL FINITE SAR PIXELS")
print("=====================")

for b in range(2):
    print(f"\nBand {b+1}")
    print(f"Count : {all_count[b]:,}")
    print(f"Min   : {all_min[b]:.4f}")
    print(f"Max   : {all_max[b]:.4f}")
    print(f"Mean  : {all_mean[b]:.4f}")
    print(f"Std   : {all_std[b]:.4f}")


print("\nVALID-LABEL FINITE SAR PIXELS")
print("=============================")

for b in range(2):
    print(f"\nBand {b+1}")
    print(f"Count : {label_count[b]:,}")
    print(f"Min   : {label_min[b]:.4f}")
    print(f"Max   : {label_max[b]:.4f}")
    print(f"Mean  : {label_mean[b]:.4f}")
    print(f"Std   : {label_std[b]:.4f}")