import os
import torch
from torch.utils.data import DataLoader

from sen1_dataset import Sen1FloodsDataset
from unet import UNet
from losses import MaskedBCEDiceLoss


# -------------------------
# Configuration
# -------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 2
EPOCHS = 20
LEARNING_RATE = 1e-3

TRAIN_SPLIT = (
    "data/raw/sen1floods11/splits/flood_train_data.csv"
)

VAL_SPLIT = (
    "data/raw/sen1floods11/splits/flood_valid_data.csv"
)

CHECKPOINT_DIR = "checkpoints_augmented"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -------------------------
# Dataset
# -------------------------

train_dataset = Sen1FloodsDataset(
    TRAIN_SPLIT,
    augment=True
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


print("Device:", DEVICE)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# -------------------------
# Model
# -------------------------

model = UNet(
    in_channels=2,
    out_channels=1
).to(DEVICE)


# -------------------------
# Loss
# -------------------------

criterion = MaskedBCEDiceLoss(
    pos_weight=3.0,
    dice_weight=0.5
)


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# -------------------------
# Mixed precision
# -------------------------

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# -------------------------
# Training
# -------------------------

best_val_loss = float("inf")


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

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, labels, ignore_masks in val_loader:

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

                loss = criterion(
                    logits,
                    labels,
                    ignore_masks
                )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # -------------------------
    # Save best model
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "best_unet.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            checkpoint_path
        )

        print(
            f"Saved best model → {checkpoint_path}"
        )


print("\nTraining completed.")
print("Best validation loss:", best_val_loss)