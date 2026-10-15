import torch
from torch.utils.data import DataLoader

from sen1_dataset import Sen1FloodsDataset
from unet import UNet


# -------------------------
# Configuration
# -------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_SPLIT = (
    "data/raw/sen1floods11/splits/flood_test_data.csv"
)

CHECKPOINT = (
    "tversky_checkpoints/"
    "T2_alpha07_beta03/"
    "best_model.pth"
)

BATCH_SIZE = 2

THRESHOLD = 0.50

# -------------------------
# Dataset
# -------------------------

test_dataset = Sen1FloodsDataset(
    TEST_SPLIT
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


print("Device:", DEVICE)
print("Test samples:", len(test_dataset))


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

print(
    "Loaded checkpoint from epoch:",
    checkpoint["epoch"]
)


# -------------------------
# Metric accumulators
# -------------------------

TP = 0
TN = 0
FP = 0
FN = 0


# -------------------------
# Evaluation
# -------------------------

with torch.no_grad():

    for images, labels, ignore_masks in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        ignore_masks = ignore_masks.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        # Convert dimensions:
        # [B, 1, H, W] → [B, H, W]

        predictions = predictions.squeeze(1)

        # Valid pixels only
        valid = ~ignore_masks

        pred = predictions[valid]
        target = labels[valid]

        TP += (
            (pred == 1) &
            (target == 1)
        ).sum().item()

        TN += (
            (pred == 0) &
            (target == 0)
        ).sum().item()

        FP += (
            (pred == 1) &
            (target == 0)
        ).sum().item()

        FN += (
            (pred == 0) &
            (target == 1)
        ).sum().item()


# -------------------------
# Metrics
# -------------------------

epsilon = 1e-8

precision = TP / (
    TP + FP + epsilon
)

recall = TP / (
    TP + FN + epsilon
)

f1 = 2 * precision * recall / (
    precision + recall + epsilon
)

iou = TP / (
    TP + FP + FN + epsilon
)

dice = 2 * TP / (
    2 * TP + FP + FN + epsilon
)

accuracy = (
    TP + TN
) / (
    TP + TN + FP + FN + epsilon
)


# -------------------------
# Results
# -------------------------

print("\n========== TEST RESULTS ==========")

print(f"TP: {TP}")
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")

print()

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"IoU      : {iou:.4f}")
print(f"Dice     : {dice:.4f}")

print("==================================")