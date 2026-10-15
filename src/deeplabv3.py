
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights


class ConvBNReLU(nn.Sequential):
    """Convolution followed by normalization and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    """
    Global-context branch of ASPP.

    GroupNorm avoids the batch-size-one limitation of BatchNorm
    on a pooled feature map with spatial dimensions 1x1.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        spatial_size = x.shape[-2:]

        x = self.pool(x)
        x = self.projection(x)

        return F.interpolate(
            x,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Expects encoder features with output stride 16.
    """

    def __init__(
        self,
        in_channels=2048,
        out_channels=128,
        rates=(6, 12, 18),
    ):
        super().__init__()

        branches = [
            ConvBNReLU(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        ]

        for rate in rates:
            branches.append(
                ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                )
            )

        branches.append(
            ASPPPooling(
                in_channels,
                out_channels,
            )
        )

        self.branches = nn.ModuleList(branches)

        self.project = nn.Sequential(
            nn.Conv2d(
                len(branches) * out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        features = [
            branch(x)
            for branch in self.branches
        ]

        x = torch.cat(features, dim=1)

        return self.project(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+-style binary segmentation model for Sentinel-1.

    Input:
        [B, 2, H, W] — normalized VV/VH SAR bands

    Output:
        [B, 1, H, W] — raw flood logits

    The output must be passed through sigmoid for probabilities.
    Do not apply sigmoid before BCEWithLogitsLoss.
    """

    def __init__(
        self,
        in_channels=2,
        num_classes=1,
        pretrained=False,
        decoder_channels=128,
    ):
        super().__init__()

        if in_channels != 2:
            raise ValueError(
                "This experiment expects two Sentinel-1 input bands."
            )

        if num_classes != 1:
            raise ValueError(
                "Binary flood segmentation requires one output channel."
            )

        weights = (
            ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet50(
            weights=weights,
            replace_stride_with_dilation=[
                False,
                False,
                True,
            ],
        )

        # Adapt the RGB stem to the two SAR input bands.
        original_conv = backbone.conv1

        sar_conv = nn.Conv2d(
            in_channels=2,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if pretrained:
                # Initialize both SAR channels from the RGB kernel mean.
                mean_kernel = original_conv.weight.mean(
                    dim=1,
                    keepdim=True,
                )

                sar_conv.weight.copy_(
                    mean_kernel.repeat(1, 2, 1, 1) * 1.5
                )

        backbone.conv1 = sar_conv

        # ResNet-50 encoder.
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Deep features: 2048 channels, output stride 16.
        self.aspp = ASPP(
            in_channels=2048,
            out_channels=decoder_channels,
        )

        # Shallow features: 256 channels, output stride 4.
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(
                256,
                48,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            ConvBNReLU(
                decoder_channels + 48,
                decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            ConvBNReLU(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Dropout2d(0.1),
            nn.Conv2d(
                decoder_channels,
                num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder.
        x = self.stem(x)

        low_level = self.layer1(x)

        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)

        # Multi-scale context.
        x = self.aspp(x)

        # Restore spatial resolution and fuse low-level features.
        x = F.interpolate(
            x,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        low_level = self.low_level_projection(low_level)

        x = torch.cat(
            [x, low_level],
            dim=1,
        )

        # Decoder.
        x = self.decoder(x)

        # Restore original image resolution.
        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return x


def run_smoke_test():
    """
    Test forward pass, loss, backward pass and GPU memory.

    Uses batch size 1 to check whether the model can train
    within the available GPU memory.
    """

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = DeepLabV3Plus(
        pretrained=False,
    ).to(device)

    model.train()

    # Clear GPU peak-memory statistics.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    images = torch.randn(
        1,
        2,
        512,
        512,
        device=device,
    )

    targets = torch.randint(
        0,
        2,
        (1, 512, 512),
        device=device,
    ).float()

    valid_mask = torch.ones_like(
        targets,
        dtype=torch.bool,
    )

    model.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.bfloat16
        ),
        enabled=(device.type == "cuda"),
    ):
        logits = model(images)

        # Smoke-test loss only.
        # Actual training will use the project's masked
        # Tversky loss and the dataset ignore mask.
        loss = F.binary_cross_entropy_with_logits(
            logits[:, 0][valid_mask],
            targets[valid_mask],
        )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Non-finite loss detected."
        )

    loss.backward()

    if not all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in model.parameters()
    ):
        raise RuntimeError(
            "Non-finite gradients detected."
        )

    print("\n===== DEEPLABV3+ SMOKE TEST =====")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(logits.shape))
    print("Loss:", round(loss.item(), 6))
    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    if device.type == "cuda":
        print(
            "Peak allocated GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / 1024**2,
                1,
            ),
            "MB",
        )

        print(
            "Peak reserved GPU memory:",
            round(
                torch.cuda.max_memory_reserved() / 1024**2,
                1,
            ),
            "MB",
        )

    assert logits.shape == (1, 1, 512, 512)
    assert torch.isfinite(logits).all()

    print("Forward pass: PASSED")
    print("Backward pass: PASSED")
    print("Gradient check: PASSED")
    print("DeepLabV3+ smoke test: PASSED")


if __name__ == "__main__":
    run_smoke_test()
