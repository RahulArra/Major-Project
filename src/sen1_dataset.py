import os
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class Sen1FloodsDataset(Dataset):

    MEAN = np.array([-10.3929, -17.2411], dtype=np.float32)
    STD = np.array([4.0388, 4.7540], dtype=np.float32)

    def __init__(self, split_file):
        self.s1_dir = "data/raw/sen1floods11/S1Hand"
        self.label_dir = "data/raw/sen1floods11/LabelHand"

        self.df = pd.read_csv(split_file, header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image_name = self.df.iloc[index, 0]
        label_name = self.df.iloc[index, 1]

        image_path = os.path.join(self.s1_dir, image_name)
        label_path = os.path.join(self.label_dir, label_name)

        # Read Sentinel-1 image
        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        # Read ground-truth label
        with rasterio.open(label_path) as src:
            label = src.read(1)

        # Replace non-finite SAR values
        for band in range(2):
            invalid = ~np.isfinite(image[band])
            image[band][invalid] = self.MEAN[band]

        # Normalize each Sentinel-1 band
        image = (image - self.MEAN[:, None, None]) / self.STD[:, None, None]

        # Create ignore mask
        ignore_mask = label == -1

        # Convert labels:
        # 0 = background
        # 1 = flood
        label = np.where(label == 1, 1, 0).astype(np.float32)

        # Convert ignored pixels to 0 in target.
        # The ignore mask tells the loss to exclude them later.
        label[ignore_mask] = 0

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        ignore_mask = torch.from_numpy(ignore_mask)

        return image, label, ignore_mask


if __name__ == "__main__":

    dataset = Sen1FloodsDataset(
        "data/raw/sen1floods11/splits/flood_train_data.csv"
    )

    print("Dataset size:", len(dataset))

    image, label, ignore_mask = dataset[0]

    print("\nImage")
    print("Shape:", image.shape)
    print("Dtype:", image.dtype)
    print("Min:", image.min().item())
    print("Max:", image.max().item())

    print("\nLabel")
    print("Shape:", label.shape)
    print("Dtype:", label.dtype)
    print("Unique:", torch.unique(label))

    print("\nIgnore mask")
    print("Shape:", ignore_mask.shape)
    print("Ignored pixels:", ignore_mask.sum().item())