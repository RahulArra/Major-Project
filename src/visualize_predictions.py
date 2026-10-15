import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sen1_dataset import Sen1FloodsDataset
from unet import UNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_SPLIT = (
    "data/raw/sen1floods11/splits/flood_test_data.csv"
)

CHECKPOINT = "checkpoints/best_unet.pth"

OUTPUT_DIR = "results/predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# Dataset
# -------------------------

dataset = Sen1FloodsDataset(TEST_SPLIT)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# -------------------------
# Model
# -------------------------

model = UNet(
    in_channels=2,
    out_channels=1
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -------------------------
# Select examples
# -------------------------

examples = [0, 10, 20, 30, 40]


with torch.no_grad():

    for sample_index, (image, label, ignore_mask) in enumerate(loader):

        if sample_index not in examples:
            continue

        image_gpu = image.to(DEVICE)

        logits = model(image_gpu)

        probability = torch.sigmoid(logits)

        prediction = (
            probability >= 0.5
        ).float()

        prediction = prediction.squeeze().cpu().numpy()
        label = label.squeeze().numpy()
        ignore = ignore_mask.squeeze().numpy()
        image = image.squeeze().numpy()

        # Sentinel-1 VV channel
        sar = image[0]

        # Mask ignored pixels
        ground_truth = label.copy()
        ground_truth[ignore] = np.nan

        predicted = prediction.copy()
        predicted[ignore] = np.nan


        # -------------------------
        # Plot
        # -------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4)
        )

        axes[0].imshow(
            sar,
            cmap="gray"
        )
        axes[0].set_title("Sentinel-1 SAR")

        axes[1].imshow(
            ground_truth,
            cmap="gray"
        )
        axes[1].set_title("Ground Truth")

        axes[2].imshow(
            predicted,
            cmap="gray"
        )
        axes[2].set_title("U-Net Prediction")

        axes[3].imshow(
            sar,
            cmap="gray"
        )
        axes[3].imshow(
            predicted,
            cmap="Reds",
            alpha=0.45
        )
        axes[3].set_title("Prediction Overlay")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"sample_{sample_index}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

print("\nVisualization completed.")