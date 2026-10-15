import os
import random

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class Sen1FloodsDataset(Dataset):

    MEAN = np.array(
        [-10.3929, -17.2411],
        dtype=np.float32
    )

    STD = np.array(
        [4.0388, 4.7540],
        dtype=np.float32
    )

    def __init__(
        self,
        split_file,
        augment=False
    ):

        self.s1_dir = (
            "data/raw/sen1floods11/S1Hand"
        )

        self.label_dir = (
            "data/raw/sen1floods11/LabelHand"
        )

        self.df = pd.read_csv(
            split_file,
            header=None
        )

        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image_name = self.df.iloc[index, 0]
        label_name = self.df.iloc[index, 1]

        image_path = os.path.join(
            self.s1_dir,
            image_name
        )

        label_path = os.path.join(
            self.label_dir,
            label_name
        )

        # -------------------------
        # Read image
        # -------------------------

        with rasterio.open(image_path) as src:
            image = src.read().astype(
                np.float32
            )

        # -------------------------
        # Read label
        # -------------------------

        with rasterio.open(label_path) as src:
            label = src.read(1)

        # -------------------------
        # Invalid SAR values
        # -------------------------

        for band in range(2):

            invalid = ~np.isfinite(
                image[band]
            )

            image[band][invalid] = (
                self.MEAN[band]
            )

        # -------------------------
        # Normalization
        # -------------------------

        image = (
            image
            - self.MEAN[:, None, None]
        ) / self.STD[:, None, None]

        # -------------------------
        # Ignore mask
        # -------------------------

        ignore_mask = label == -1

        # Flood = 1
        # Everything else = 0
        label = np.where(
            label == 1,
            1,
            0
        ).astype(np.float32)

        label[ignore_mask] = 0

        # -------------------------
        # Data augmentation
        # -------------------------

        if self.augment:

            # Horizontal flip
            if random.random() < 0.5:

                image = np.flip(
                    image,
                    axis=2
                ).copy()

                label = np.flip(
                    label,
                    axis=1
                ).copy()

                ignore_mask = np.flip(
                    ignore_mask,
                    axis=1
                ).copy()

            # Vertical flip
            if random.random() < 0.5:

                image = np.flip(
                    image,
                    axis=1
                ).copy()

                label = np.flip(
                    label,
                    axis=0
                ).copy()

                ignore_mask = np.flip(
                    ignore_mask,
                    axis=0
                ).copy()

            # 90-degree rotation
            if random.random() < 0.5:

                k = random.randint(
                    1,
                    3
                )

                image = np.rot90(
                    image,
                    k=k,
                    axes=(1, 2)
                ).copy()

                label = np.rot90(
                    label,
                    k=k
                ).copy()

                ignore_mask = np.rot90(
                    ignore_mask,
                    k=k
                ).copy()

        # -------------------------
        # Convert to tensors
        # -------------------------

        image = torch.from_numpy(
            image
        )

        label = torch.from_numpy(
            label
        )

        ignore_mask = torch.from_numpy(
            ignore_mask
        )

        return (
            image,
            label,
            ignore_mask
        )