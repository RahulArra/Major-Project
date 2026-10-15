import torch
from unet import UNet
from sen1_dataset import Sen1FloodsDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# Load one sample
dataset = Sen1FloodsDataset(
    "data/raw/sen1floods11/splits/flood_train_data.csv"
)

image, label, ignore_mask = dataset[0]

# Add batch dimension
image = image.unsqueeze(0).to(DEVICE)

print("Input shape:", image.shape)

# Create model
model = UNet(in_channels=2, out_channels=1).to(DEVICE)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

# Forward pass
with torch.no_grad():
    output = model(image)

print("Output shape:", output.shape)
print("Output dtype:", output.dtype)
print("Output finite:", torch.isfinite(output).all().item())

# GPU memory
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    print(f"GPU memory allocated: {allocated:.2f} MB")
    print(f"GPU memory reserved:  {reserved:.2f} MB")

print("\nU-Net forward test PASSED.")