import os
import csv
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from sen1_dataset import Sen1FloodsDataset
from unet import UNet

from losses import (
    BCEDiceLoss,
    FocalDiceLoss,
    TverskyDiceLoss
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


TRAIN_SPLIT = (
    "data/raw/sen1floods11/splits/flood_train_data.csv"
)

VAL_SPLIT = (
    "data/raw/sen1floods11/splits/flood_valid_data.csv"
)


BATCH_SIZE = 2
EPOCHS = 20

RESULT_FILE = "experiment_results.csv"


# =========================================================
# Experiments
# =========================================================

EXPERIMENTS = [

    {
        "name": "E1_baseline",
        "augment": False,
        "loss": "bce_dice",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "pos_weight": 3.0,
        "dice_weight": 0.5,
    },

    {
        "name": "E2_augmented",
        "augment": True,
        "loss": "bce_dice",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "pos_weight": 3.0,
        "dice_weight": 0.5,
    },

    {
        "name": "E3_lr5e4",
        "augment": True,
        "loss": "bce_dice",
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "pos_weight": 3.0,
        "dice_weight": 0.5,
    },

    {
        "name": "E4_focal",
        "augment": True,
        "loss": "focal_dice",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "pos_weight": 3.0,
        "dice_weight": 0.5,
    },

    {
        "name": "E5_tversky",
        "augment": True,
        "loss": "tversky",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "pos_weight": 3.0,
        "dice_weight": 0.5,
    },

    {
        "name": "E6_posweight2",
        "augment": True,
        "loss": "bce_dice",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "pos_weight": 2.0,
        "dice_weight": 0.5,
    },

    {
        "name": "E7_posweight4",
        "augment": True,
        "loss": "bce_dice",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "pos_weight": 4.0,
        "dice_weight": 0.5,
    },

]


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Loss factory
# =========================================================

def create_loss(config):

    if config["loss"] == "bce_dice":

        return BCEDiceLoss(
            pos_weight=config["pos_weight"],
            dice_weight=config["dice_weight"]
        )

    if config["loss"] == "focal_dice":

        return FocalDiceLoss(
            pos_weight=config["pos_weight"],
            dice_weight=config["dice_weight"]
        )

    if config["loss"] == "tversky":

        return TverskyDiceLoss(
            pos_weight=config["pos_weight"],
            dice_weight=config["dice_weight"]
        )

    raise ValueError(
        f"Unknown loss: {config['loss']}"
    )


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(
    model,
    loader
):

    model.eval()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():

        for images, labels, ignore_masks in loader:

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

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):

                logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            predictions = predictions.squeeze(1)

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

    eps = 1e-8

    precision = TP / (
        TP + FP + eps
    )

    recall = TP / (
        TP + FN + eps
    )

    f1 = (
        2 * precision * recall
        / (precision + recall + eps)
    )

    iou = TP / (
        TP + FP + FN + eps
    )

    dice = 2 * TP / (
        2 * TP + FP + FN + eps
    )

    accuracy = (
        TP + TN
    ) / (
        TP + TN + FP + FN + eps
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice
    }


# =========================================================
# Train one experiment
# =========================================================

def run_experiment(config):

    print("\n")
    print("=" * 70)
    print(
        f"STARTING {config['name']}"
    )
    print("=" * 70)

    print(config)

    set_seed(42)

    train_dataset = Sen1FloodsDataset(
        TRAIN_SPLIT,
        augment=config["augment"]
    )

    val_dataset = Sen1FloodsDataset(
        VAL_SPLIT,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = UNet(
        in_channels=2,
        out_channels=1
    ).to(DEVICE)

    criterion = create_loss(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    best_iou = -1

    best_metrics = None
    best_epoch = 0

    checkpoint_dir = os.path.join(
        "experiment_checkpoints",
        config["name"]
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for images, labels, ignore_masks in train_loader:

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

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    labels,
                    ignore_masks
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        metrics = calculate_metrics(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | "
            f"IoU {metrics['iou']:.4f} | "
            f"Dice {metrics['dice']:.4f} | "
            f"F1 {metrics['f1']:.4f}"
        )

        if metrics["iou"] > best_iou:

            best_iou = metrics["iou"]

            best_metrics = metrics

            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "config": config,
                    "metrics": metrics
                },
                os.path.join(
                    checkpoint_dir,
                    "best_model.pth"
                )
            )

    result = {
        "experiment": config["name"],
        "loss": config["loss"],
        "augment": config["augment"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "pos_weight": config["pos_weight"],
        "dice_weight": config["dice_weight"],
        "best_epoch": best_epoch,
        "val_accuracy": best_metrics["accuracy"],
        "val_precision": best_metrics["precision"],
        "val_recall": best_metrics["recall"],
        "val_f1": best_metrics["f1"],
        "val_iou": best_metrics["iou"],
        "val_dice": best_metrics["dice"]
    }

    print("\nBEST RESULT")
    print(result)

    return result


# =========================================================
# Main
# =========================================================

def main():

    print("Device:", DEVICE)

    results = []

    for config in EXPERIMENTS:

        result = run_experiment(
            config
        )

        results.append(result)

        with open(
            RESULT_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=results[0].keys()
            )

            writer.writeheader()

            writer.writerows(results)

    # Sort by IoU

    results.sort(
        key=lambda x: x["val_iou"],
        reverse=True
    )

    print("\n")
    print("=" * 70)
    print("FINAL EXPERIMENT RANKING")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['experiment']} | "
            f"IoU={result['val_iou']:.4f} | "
            f"Dice={result['val_dice']:.4f} | "
            f"F1={result['val_f1']:.4f}"
        )

    print("\nResults saved to:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()