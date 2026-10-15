# @'
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sen1_dataset import Sen1FloodsDataset
from deeplabv3 import DeepLabV3Plus
from losses import TverskyDiceLoss


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_SPLIT = "data/raw/sen1floods11/splits/flood_train_data.csv"
VAL_SPLIT = "data/raw/sen1floods11/splits/flood_valid_data.csv"

CHECKPOINT_DIR = Path("checkpoints_deeplab_lr0005")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = CHECKPOINT_DIR / "best_deeplab.pth"
HISTORY_FILE = CHECKPOINT_DIR / "training_history.csv"

BATCH_SIZE = 1
MAX_EPOCHS = 30
PATIENCE = 7

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

POS_WEIGHT = 3.0
DICE_WEIGHT = 0.5
TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3

THRESHOLD = 0.50
NUM_WORKERS = 0

USE_AMP = torch.cuda.is_available()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(tp, tn, fp, fn):
    eps = 1e-8

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    tp = tn = fp = fn = 0

    for images, targets, ignore_mask in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        ignore_mask = ignore_mask.to(device, non_blocking=True)
        if not (~ignore_mask.bool()).any().item():
            continue
        # Skip batches with no labelled pixels.
        if ignore_mask.all():
            continue
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            logits = model(images)

        # Evaluate the loss in float32 for numerical stability.
        loss = criterion(
            logits.float(),
            targets,
            ignore_mask,
        )

        total_loss += loss.item()
        total_samples += images.size(0)

        probabilities = torch.sigmoid(logits.float())
        predictions = probabilities[:, 0] >= THRESHOLD

        valid = ~ignore_mask

        tp += int(((predictions == 1) & (targets == 1) & valid).sum().item())
        tn += int(((predictions == 0) & (targets == 0) & valid).sum().item())
        fp += int(((predictions == 1) & (targets == 0) & valid).sum().item())
        fn += int(((predictions == 0) & (targets == 1) & valid).sum().item())

    metrics = calculate_metrics(tp, tn, fp, fn)
    metrics["loss"] = total_loss / max(total_samples, 1)

    return metrics


# ============================================================
# TRAINING
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DeepLabV3+ FLOOD SEGMENTATION")
    print("=" * 60)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    train_dataset = Sen1FloodsDataset(
        TRAIN_SPLIT,
        augment=True,
    )

    val_dataset = Sen1FloodsDataset(
        VAL_SPLIT,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", MAX_EPOCHS)
    print("Mixed precision:", USE_AMP)

    model = DeepLabV3Plus(
        in_channels=2,
        num_classes=1,
        pretrained=False,
    ).to(device)

    criterion = TverskyDiceLoss(
        pos_weight=POS_WEIGHT,
        dice_weight=DICE_WEIGHT,
        alpha=TVERSKY_ALPHA,
        beta=TVERSKY_BETA,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    best_val_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history_exists = HISTORY_FILE.exists()

    with HISTORY_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_iou",
            "val_dice",
            "learning_rate",
            "epoch_seconds",
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        if not history_exists:
            writer.writeheader()

        for epoch in range(1, MAX_EPOCHS + 1):
            start_time = time.time()

            model.train()

            running_loss = 0.0
            samples_seen = 0

            for images, targets, ignore_mask in train_loader:
                images = images.to(
                    device,
                    non_blocking=True,
                )

                targets = targets.to(
                    device,
                    non_blocking=True,
                )

                ignore_mask = ignore_mask.to(
                    device,
                    non_blocking=True,
                )
                if not (~ignore_mask.bool()).any().item():
                    continue
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=USE_AMP,
                ):
                    logits = model(images)

                # Check model output before calculating loss.
                if not torch.isfinite(logits).all():
                    print(f"\nNon-finite logits at epoch {epoch}")
                    print("Image min:", images.min().item())
                    print("Image max:", images.max().item())
                    print("Logit min:", logits.nan_to_num().min().item())
                    print("Logit max:", logits.nan_to_num().max().item())
                    raise RuntimeError("Model produced NaN/Inf logits.")

                # Calculate the loss in float32.
                loss = criterion(
                    logits.float(),
                    targets,
                    ignore_mask,
                )

                if not torch.isfinite(loss):
                    print(f"\nNon-finite loss at epoch {epoch}")
                    print("Logit min:", logits.float().min().item())
                    print("Logit max:", logits.float().max().item())
                    print("Target values:", torch.unique(targets).tolist())
                    print("Ignored pixels:", ignore_mask.sum().item())
                    raise RuntimeError("Tversky loss produced NaN/Inf.")

                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                samples_seen += images.size(0)

            train_loss = running_loss / max(samples_seen, 1)

            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            scheduler.step(val_metrics["iou"])

            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f} | "
                f"Val Precision: {val_metrics['precision']:.4f} | "
                f"Val Recall: {val_metrics['recall']:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
                "learning_rate": current_lr,
                "epoch_seconds": elapsed,
            })

            csvfile.flush()

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_iou": best_val_iou,
                        "config": {
                            "seed": SEED,
                            "batch_size": BATCH_SIZE,
                            "learning_rate": LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                            "pos_weight": POS_WEIGHT,
                            "dice_weight": DICE_WEIGHT,
                            "tversky_alpha": TVERSKY_ALPHA,
                            "tversky_beta": TVERSKY_BETA,
                            "threshold": THRESHOLD,
                            "pretrained": False,
                        },
                    },
                    BEST_CHECKPOINT,
                )

                print(
                    f"  Saved best checkpoint: "
                    f"validation IoU = {best_val_iou:.4f}"
                )

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(
                    f"Early stopping after epoch {epoch}; "
                    f"best epoch was {best_epoch}."
                )
                break

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Best epoch:", best_epoch)
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print("Checkpoint:", BEST_CHECKPOINT)
    print("History:", HISTORY_FILE)

    if device.type == "cuda":
        print(
            "Peak GPU memory (MB):",
            round(
                torch.cuda.max_memory_allocated() / 1024**2,
                1,
            ),
        )


if __name__ == "__main__":
    main()
# '@ | Set-Content -Encoding utf8 src\train_deeplab.py
