import torch
from torch.utils.data import DataLoader

from sen1_dataset import Sen1FloodsDataset
from unet import UNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

VAL_SPLIT = (
    "data/raw/sen1floods11/splits/flood_valid_data.csv"
)

CHECKPOINT = "tversky_checkpoints/T2_alpha07_beta03/best_model.pth"
BATCH_SIZE = 2


# -------------------------
# Dataset
# -------------------------

dataset = Sen1FloodsDataset(VAL_SPLIT)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
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
# Collect probabilities
# -------------------------

all_probabilities = []
all_targets = []


with torch.no_grad():

    for images, labels, ignore_masks in loader:

        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.squeeze(1)

        valid = ~ignore_masks

        all_probabilities.append(
            probabilities[valid].cpu()
        )

        all_targets.append(
            labels[valid].cpu()
        )


probabilities = torch.cat(
    all_probabilities
)

targets = torch.cat(
    all_targets
)


# -------------------------
# Search thresholds
# -------------------------

best_threshold = 0.5
best_iou = 0.0

print("\nValidation threshold search")
print("--------------------------------")

for threshold in torch.arange(
    0.40,
    0.851,
    0.01
):

    predictions = (
        probabilities >= threshold
    ).float()

    tp = (
        (predictions == 1) &
        (targets == 1)
    ).sum().item()

    fp = (
        (predictions == 1) &
        (targets == 0)
    ).sum().item()

    fn = (
        (predictions == 0) &
        (targets == 1)
    ).sum().item()

    iou = tp / (
        tp + fp + fn + 1e-8
    )

    dice = 2 * tp / (
        2 * tp + fp + fn + 1e-8
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"IoU {iou:.4f} | "
        f"Dice {dice:.4f}"
    )

    if iou > best_iou:

        best_iou = iou
        best_threshold = threshold.item()


print("--------------------------------")
print(
    f"Best threshold: {best_threshold:.2f}"
)

print(
    f"Best validation IoU: {best_iou:.4f}"
)