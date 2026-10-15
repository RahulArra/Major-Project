# @'
import torch
from losses import TverskyDiceLoss


def check_case(name, logits, targets, ignore_mask, criterion):
    logits.requires_grad_(True)

    loss = criterion(logits, targets, ignore_mask)

    assert torch.isfinite(loss).item(), f"{name}: loss is not finite"

    loss.backward()

    assert logits.grad is not None, f"{name}: gradients are missing"
    assert torch.isfinite(logits.grad).all().item(), (
        f"{name}: gradients contain NaN or Inf"
    )

    print(f"{name}: PASSED | loss = {loss.item():.6f}")


def main():
    torch.manual_seed(42)

    criterion = TverskyDiceLoss(
        pos_weight=3.0,
        dice_weight=0.5,
        alpha=0.7,
        beta=0.3,
    )

    # Case 1: Fully labelled batch.
    logits1 = torch.randn(1, 1, 32, 32)
    targets1 = torch.randint(0, 2, (1, 32, 32)).float()
    ignore1 = torch.zeros(1, 32, 32, dtype=torch.bool)

    check_case(
        "Ordinary labelled batch",
        logits1,
        targets1,
        ignore1,
        criterion,
    )

    # Case 2: Some pixels are ignored.
    logits2 = torch.randn(1, 1, 32, 32)
    targets2 = torch.randint(0, 2, (1, 32, 32)).float()
    ignore2 = torch.zeros(1, 32, 32, dtype=torch.bool)
    ignore2[:, :16, :16] = True

    check_case(
        "Partially labelled batch",
        logits2,
        targets2,
        ignore2,
        criterion,
    )

    # Case 3: Every pixel is ignored.
    logits3 = torch.randn(1, 1, 32, 32)
    targets3 = torch.zeros(1, 32, 32)
    ignore3 = torch.ones(1, 32, 32, dtype=torch.bool)

    check_case(
        "Fully unlabelled batch",
        logits3,
        targets3,
        ignore3,
        criterion,
    )

    print("\\nAll loss tests passed.")


if __name__ == "__main__":
    main()
# '@ | Set-Content -Encoding UTF8 src\test_loss.py