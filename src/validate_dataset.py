from sen1_dataset import Sen1FloodsDataset
import torch

dataset = Sen1FloodsDataset(
    "data/raw/sen1floods11/splits/flood_train_data.csv"
)

print("Checking", len(dataset), "training samples...")

for i in range(len(dataset)):
    image, label, ignore_mask = dataset[i]

    assert image.shape == (2, 512, 512)
    assert label.shape == (512, 512)
    assert ignore_mask.shape == (512, 512)

    assert torch.isfinite(image).all()
    assert torch.isfinite(label).all()

    unique_labels = torch.unique(label)

    assert all(v.item() in [0.0, 1.0] for v in unique_labels)

    if (i + 1) % 25 == 0:
        print(f"Checked {i + 1}/{len(dataset)}")

print("\nDataset validation PASSED.")