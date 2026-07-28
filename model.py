from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)


CLASS_NAMES = (
    "Bone",
    "Fibrocartilage",
    "Cartilage",
    "Muscle",
    "Marrow",
    "Background",
)

COARSE_CLASS_NAMES = (
    "Bone",
    "Fibrocartilage_or_Muscle",
    "Cartilage",
    "Marrow",
    "Background",
)


@dataclass(frozen=True)
class ModelConfig:
    context_scale: int = 4
    pretrained: bool = True
    decoder_channels: int = 64


def _kernel(values: list[list[float]], divisor: float = 1.0) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, 1, 3, 3) / divisor


class HandcraftedFeatures(nn.Module):
    """Nine fixed feature maps calculated from native-resolution luminance."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer(
            "sobel_x", _kernel([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 8)
        )
        self.register_buffer(
            "sobel_y", _kernel([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 8)
        )
        self.register_buffer(
            "laplace", _kernel([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        )
        self.register_buffer(
            "gaussian", _kernel([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 16)
        )
        self.register_buffer(
            "dxx", _kernel([[1, -2, 1], [2, -4, 2], [1, -2, 1]], 4)
        )
        self.register_buffer(
            "dyy", _kernel([[1, 2, 1], [-2, -4, -2], [1, 2, 1]], 4)
        )
        self.register_buffer(
            "dxy", _kernel([[1, 0, -1], [0, 0, 0], [-1, 0, 1]], 4)
        )
        # Fixed scaling avoids the tile-dependent mean/std normalization that
        # created visible seams. BatchNorm in the feature stem learns global
        # training-set statistics and freezes them for inference.
        self.register_buffer(
            "scales",
            torch.tensor(
                [0.25, 0.10, 0.10, 0.05, 0.05, 1.0, 0.05, 0.25, 0.25],
                dtype=torch.float32,
            ).view(1, 9, 1, 1),
        )

    @staticmethod
    def _conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected [N, 3, H, W], received {tuple(rgb.shape)}")

        gray = (
            0.2126 * rgb[:, 0:1]
            + 0.7152 * rgb[:, 1:2]
            + 0.0722 * rgb[:, 2:3]
        )
        gx = self._conv(gray, self.sobel_x)
        gy = self._conv(gray, self.sobel_y)
        laplace = self._conv(gray, self.laplace)
        gradient_magnitude = torch.sqrt(gx.square() + gy.square() + self.eps)

        local_mean = self._conv(gray, self.gaussian)
        local_mean_sq = self._conv(gray.square(), self.gaussian)
        weighted_deviation = torch.sqrt(
            torch.clamp(local_mean_sq - local_mean.square(), min=0) + self.eps
        )

        jxx = self._conv(gx.square(), self.gaussian)
        jyy = self._conv(gy.square(), self.gaussian)
        jxy = self._conv(gx * gy, self.gaussian)
        j_trace = jxx + jyy
        j_root = torch.sqrt((jxx - jyy).square() + 4 * jxy.square() + self.eps)
        structure_max = 0.5 * (j_trace + j_root)
        structure_min = 0.5 * (j_trace - j_root)
        coherence = (structure_max - structure_min) / (
            structure_max + structure_min + self.eps
        )

        hxx = self._conv(gray, self.dxx)
        hyy = self._conv(gray, self.dyy)
        hxy = self._conv(gray, self.dxy)
        hessian_det = hxx * hyy - hxy.square()
        h_trace = hxx + hyy
        h_root = torch.sqrt((hxx - hyy).square() + 4 * hxy.square() + self.eps)
        hessian_max = 0.5 * (h_trace + h_root)
        hessian_min = 0.5 * (h_trace - h_root)

        raw = torch.cat(
            [
                laplace,
                weighted_deviation,
                gradient_magnitude,
                structure_max,
                structure_min,
                coherence,
                hessian_det,
                hessian_max,
                hessian_min,
            ],
            dim=1,
        )
        return torch.asinh(raw / self.scales).clamp(-8, 8)


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.residual(x)


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.refine = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.project(x)
        return self.refine(torch.cat([x, skip], dim=1))


class AccurateTissueNet(nn.Module):
    """
    Accuracy-first dual-scale segmentation model.

    local_rgb contains a native-resolution patch. context_rgb contains the
    corresponding context_scale-times-wider field of view resized to the same
    tensor dimensions. Muscle is modeled conditionally inside the combined
    Fibrocartilage-or-Muscle parent class.
    """

    def __init__(
        self,
        context_scale: int = 4,
        pretrained: bool = True,
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()
        if context_scale < 1:
            raise ValueError("context_scale must be at least 1")
        self.config = ModelConfig(
            context_scale=context_scale,
            pretrained=pretrained,
            decoder_channels=decoder_channels,
        )
        self.context_scale = context_scale

        local_weights = ResNet50_Weights.DEFAULT if pretrained else None
        context_weights = ResNet18_Weights.DEFAULT if pretrained else None
        local_encoder = resnet50(weights=local_weights)
        context_encoder = resnet18(weights=context_weights)

        self.rgb_conv1 = local_encoder.conv1
        self.rgb_bn1 = local_encoder.bn1
        self.rgb_relu = local_encoder.relu
        self.local_maxpool = local_encoder.maxpool
        self.local_layer1 = local_encoder.layer1
        self.local_layer2 = local_encoder.layer2
        self.local_layer3 = local_encoder.layer3
        self.local_layer4 = local_encoder.layer4

        self.context_conv1 = context_encoder.conv1
        self.context_bn1 = context_encoder.bn1
        self.context_relu = context_encoder.relu
        self.context_maxpool = context_encoder.maxpool
        self.context_layer1 = context_encoder.layer1
        self.context_layer2 = context_encoder.layer2
        self.context_layer3 = context_encoder.layer3
        self.context_layer4 = context_encoder.layer4

        self.handcrafted = HandcraftedFeatures()
        self.feature_stem = nn.Sequential(
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 32, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.stem_fusion = ConvBlock(128, 64)

        self.local_projection = ConvBlock(2048, 512)
        self.context_projection = ConvBlock(512, 256)
        self.context_film = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
        )
        self.bottleneck_fusion = ConvBlock(768, 512)

        d = decoder_channels
        self.decode4 = DecoderBlock(512, 1024, d * 8)
        self.decode3 = DecoderBlock(d * 8, 512, d * 4)
        self.decode2 = DecoderBlock(d * 4, 256, d * 2)
        self.decode1 = DecoderBlock(d * 2, 64, d)
        self.final_refine = ConvBlock(d, d)

        self.coarse_classifier = nn.Conv2d(d, 5, 1)
        self.muscle_classifier = nn.Conv2d(d, 1, 1)
        self.boundary_classifier = nn.Conv2d(d, 1, 1)

        self.muscle_presence = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )
        self.presence_scale = nn.Parameter(torch.tensor(0.25))

        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def model_config(self) -> dict:
        config = asdict(self.config)
        # A checkpoint load never needs to download pretrained weights.
        config["pretrained"] = False
        return config

    def _normalize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        return (rgb - self.rgb_mean) / self.rgb_std

    def _context_center_features(
        self, context_features: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        height, width = context_features.shape[-2:]
        crop_height = max(1, round(height / self.context_scale))
        crop_width = max(1, round(width / self.context_scale))
        top = max(0, (height - crop_height) // 2)
        left = max(0, (width - crop_width) // 2)
        center = context_features[
            :, :, top : top + crop_height, left : left + crop_width
        ]
        return F.interpolate(
            center, size=output_size, mode="bilinear", align_corners=False
        )

    def forward(
        self, local_rgb: torch.Tensor, context_rgb: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if local_rgb.shape != context_rgb.shape:
            raise ValueError(
                "Local and resized context tensors must have identical shapes; "
                f"received {tuple(local_rgb.shape)} and {tuple(context_rgb.shape)}"
            )

        local_normalized = self._normalize_rgb(local_rgb)
        rgb_stem = self.rgb_relu(self.rgb_bn1(self.rgb_conv1(local_normalized)))
        feature_stem = self.feature_stem(self.handcrafted(local_rgb))
        stem = self.stem_fusion(torch.cat([rgb_stem, feature_stem], dim=1))

        local1 = self.local_layer1(self.local_maxpool(stem))
        local2 = self.local_layer2(local1)
        local3 = self.local_layer3(local2)
        local4 = self.local_layer4(local3)
        local_bottleneck = self.local_projection(local4)

        context_normalized = self._normalize_rgb(context_rgb)
        context = self.context_relu(
            self.context_bn1(self.context_conv1(context_normalized))
        )
        context = self.context_layer1(self.context_maxpool(context))
        context = self.context_layer2(context)
        context = self.context_layer3(context)
        context = self.context_layer4(context)

        context_global = F.adaptive_avg_pool2d(context, 1).flatten(1)
        gamma, beta = self.context_film(context_global).chunk(2, dim=1)
        gamma = 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = 0.1 * beta.unsqueeze(-1).unsqueeze(-1)
        local_bottleneck = local_bottleneck * (1 + gamma) + beta

        context_center = self._context_center_features(
            context, local_bottleneck.shape[-2:]
        )
        context_center = self.context_projection(context_center)
        bottleneck = self.bottleneck_fusion(
            torch.cat([local_bottleneck, context_center], dim=1)
        )

        decoded = self.decode4(bottleneck, local3)
        decoded = self.decode3(decoded, local2)
        decoded = self.decode2(decoded, local1)
        decoded = self.decode1(decoded, stem)
        decoded = F.interpolate(
            decoded,
            size=local_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.final_refine(decoded)

        pooled_local = F.adaptive_avg_pool2d(local_bottleneck, 1).flatten(1)
        presence_logit = self.muscle_presence(
            torch.cat([pooled_local, context_global], dim=1)
        )
        muscle_logit = self.muscle_classifier(decoded)
        muscle_logit = muscle_logit + (
            self.presence_scale * presence_logit[:, :, None, None]
        )

        return {
            "coarse_logits": self.coarse_classifier(decoded),
            "muscle_logits": muscle_logit,
            "boundary_logits": self.boundary_classifier(decoded),
            "muscle_presence_logits": presence_logit,
        }

    @staticmethod
    def probabilities_from_outputs(
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        coarse = outputs["coarse_logits"].softmax(dim=1)
        muscle_given_parent = outputs["muscle_logits"].sigmoid()
        parent = coarse[:, 1:2]
        return torch.cat(
            [
                coarse[:, 0:1],
                parent * (1 - muscle_given_parent),
                coarse[:, 2:3],
                parent * muscle_given_parent,
                coarse[:, 3:4],
                coarse[:, 4:5],
            ],
            dim=1,
        )

    def probabilities(
        self, local_rgb: torch.Tensor, context_rgb: torch.Tensor
    ) -> torch.Tensor:
        return self.probabilities_from_outputs(self(local_rgb, context_rgb))


if __name__ == "__main__":
    model = AccurateTissueNet(pretrained=False)
    local = torch.rand(1, 3, 256, 256)
    context = torch.rand(1, 3, 256, 256)
    with torch.inference_mode():
        probabilities = model.probabilities(local, context)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"input={tuple(local.shape)} probabilities={tuple(probabilities.shape)} "
        f"parameters={parameters:,} sum={probabilities.sum(1).mean().item():.6f}"
    )
