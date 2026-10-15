import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_binary_inputs(logits, targets, ignore_mask):
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape [B, 1, H, W]")
    expected = (logits.shape[0], *logits.shape[2:])
    if targets.shape != expected:
        raise ValueError(f"targets must have shape {expected}, got {tuple(targets.shape)}")
    if ignore_mask.shape != expected:
        raise ValueError(f"ignore_mask must have shape {expected}, got {tuple(ignore_mask.shape)}")
    logits = logits.float()
    targets = targets.unsqueeze(1).float()
    ignore_mask = ignore_mask.unsqueeze(1).bool()
    return logits, targets, ignore_mask


class TverskyDiceLoss(nn.Module):
    def __init__(
        self,
        pos_weight=3.0,
        dice_weight=0.5,
        alpha=0.7,
        beta=0.3,
        smooth=1.0,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

    def forward(self, logits, targets, ignore_mask):
        # Calculate loss in float32 for numerical stability.
        logits, targets, ignore_mask = _prepare_binary_inputs(
            logits, targets, ignore_mask
        )

        valid_mask = ~ignore_mask

        # A completely unlabelled batch must not produce NaN.
        if not valid_mask.any().item():
            return logits.sum() * 0.0

        probabilities = torch.sigmoid(logits)

        pos_weight = torch.tensor(
            self.pos_weight,
            device=logits.device,
            dtype=torch.float32,
        )

        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )

        bce = bce_map[valid_mask].mean()

        p = probabilities[valid_mask]
        t = targets[valid_mask]

        tp = (p * t).sum()
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()

        tversky_index = (
            (tp + self.smooth)
            / (
                tp
                + self.alpha * fp
                + self.beta * fn
                + self.smooth
            )
        )

        tversky_loss = 1.0 - tversky_index

        return (
            (1.0 - self.dice_weight) * bce
            + self.dice_weight * tversky_loss
        )


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        dice_weight=0.5,
        smooth=1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def forward(self, logits, targets, ignore_mask):
        logits, targets, ignore_mask = _prepare_binary_inputs(
            logits, targets, ignore_mask
        )

        valid_mask = ~ignore_mask

        if not valid_mask.any().item():
            return logits.sum() * 0.0

        probabilities = torch.sigmoid(logits)

        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        pt = torch.exp(-bce_map)
        alpha_factor = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        focal_map = alpha_factor * (1.0 - pt).pow(self.gamma) * bce_map
        focal_loss = focal_map[valid_mask].mean()

        p = probabilities[valid_mask]
        t = targets[valid_mask]

        intersection = (p * t).sum()
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (p.sum() + t.sum() + self.smooth)
        )

        return (
            (1.0 - self.dice_weight) * focal_loss
            + self.dice_weight * dice_loss
        )
class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        pos_weight=3.0,
        dice_weight=0.5,
        smooth=1.0,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def forward(self, logits, targets, ignore_mask):
        logits, targets, ignore_mask = _prepare_binary_inputs(
            logits, targets, ignore_mask
        )

        valid_mask = ~ignore_mask

        if not valid_mask.any().item():
            return logits.sum() * 0.0

        probabilities = torch.sigmoid(logits)

        pos_weight = torch.tensor(
            self.pos_weight,
            device=logits.device,
            dtype=torch.float32,
        )

        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        bce = bce_map[valid_mask].mean()

        p = probabilities[valid_mask]
        t = targets[valid_mask]

        intersection = (p * t).sum()

        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth)
            / (p.sum() + t.sum() + self.smooth)
        )

        return (
            (1.0 - self.dice_weight) * bce
            + self.dice_weight * dice_loss
        )
