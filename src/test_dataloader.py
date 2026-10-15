from sen1_dataset import Sen1FloodsDataset
from torch.utils.data import DataLoader
import torch


dataset = Sen1FloodsDataset(
    "data/raw/sen1floods11/splits/flood_train_data.csv"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

images, labels, ignore_masks = next(iter(loader))

print("Batch test")
print("----------")

print("Images:")
print("  Shape:", images.shape)
print("  Dtype:", images.dtype)

print("\nLabels:")
print("  Shape:", labels.shape)
print("  Dtype:", labels.dtype)

print("\nIgnore masks:")
print("  Shape:", ignore_masks.shape)
print("  Dtype:", ignore_masks.dtype)

print("\nFinite image values:", torch.isfinite(images).all().item())

# GPU test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images = images.to(device)
labels = labels.to(device)
ignore_masks = ignore_masks.to(device)

print("\nDevice:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("Images on:", images.device)
print("Labels on:", labels.device)
print("Ignore masks on:", ignore_masks.device)

print("\nDataLoader + GPU test PASSED.")