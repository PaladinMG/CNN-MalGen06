from __future__ import annotations

import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as f
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from data import (
    SegmentationDataset,
    discover_samples,
    estimate_training_cache_bytes,
)
from model import AccurateTissueNet, CLASS_NAMES


COARSE_MAPPING = (0, 1, 2, 1, 3, 4)


def deterministic_split_indices(
    item_count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return disjoint training/validation IDs with at least one in each."""
    if item_count < 2:
        raise ValueError("A split requires at least two items")
    order: list[int] = torch.randperm(
        item_count,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    validation_count = min(
        item_count - 1,
        max(1, round(validation_fraction * item_count)),
    )
    return order[validation_count:], order[:validation_count]




def coarse_targets(mask: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(COARSE_MAPPING, device=mask.device)
    return mapping[mask]


def multiclass_dice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    classes = len(CLASS_NAMES)
    dimensions = (0, 2, 3)
    target_flat = target.reshape(-1)
    target_probabilities = probabilities.gather(
        1, target[:, None]
    ).reshape(-1)
    intersection = torch.zeros(
        classes,
        dtype=probabilities.dtype,
        device=probabilities.device,
    ).scatter_add_(0, target_flat, target_probabilities)
    target_count = torch.bincount(
        target_flat, minlength=classes
    ).to(probabilities.dtype)
    denominator = probabilities.sum(dim=dimensions) + target_count
    loss_per_class = 1 - (2 * intersection + eps) / (denominator + eps)
    if class_weights is None:
        return loss_per_class.mean()
    weights = class_weights / class_weights.sum()
    return (loss_per_class * weights).sum()


def binary_dice_loss(
    probabilities: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> int | Tensor:
    intersection = (probabilities * target).sum()
    denominator = probabilities.sum() + target.sum()
    return 1 - (2 * intersection + eps) / (denominator + eps)


def masked_muscle_loss(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    valid = (mask == 1) | (mask == 3)
    if not torch.any(valid):
        return logits.sum() * 0
    target = (mask == 3).float()
    selected_logits = logits[:, 0][valid].float()
    selected_target = target[valid]
    positives = selected_target.sum()
    negatives = selected_target.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1)).clamp(0.25, 20)
    return f.binary_cross_entropy_with_logits(
        selected_logits,
        selected_target,
        pos_weight=positive_weight,
    )


def boundary_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    target = target[:, None].float()
    positives = target.sum()
    negatives = target.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1)).clamp(1, 50)
    bce = f.binary_cross_entropy_with_logits(
        logits.float(), target, pos_weight=positive_weight
    )
    dice = binary_dice_loss(logits.float().sigmoid(), target)
    return 0.6 * bce + 0.4 * dice


def complete_loss(
    model: AccurateTissueNet,
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    boundary: torch.Tensor,
    muscle_presence: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    coarse_weight = None
    if class_weights is not None:
        coarse_weight = torch.stack(
            [
                class_weights[0],
                0.5 * (class_weights[1] + class_weights[3]),
                class_weights[2],
                class_weights[4],
                class_weights[5],
            ]
        )
    coarse = f.cross_entropy(
        outputs["coarse_logits"],
        coarse_targets(mask),
        weight=coarse_weight,
    )
    muscle = masked_muscle_loss(outputs["muscle_logits"], mask)
    # Accumulate Dice terms in float32. A 512x512 mixed-precision batch can
    # exceed the largest finite FP16 reduction value.
    probabilities = model.probabilities_from_outputs(outputs).float()
    dice = multiclass_dice_loss(probabilities, mask, class_weights)
    boundary_component = boundary_loss(outputs["boundary_logits"], boundary)
    presence = f.binary_cross_entropy_with_logits(
        outputs["muscle_presence_logits"][:, 0],
        muscle_presence.to(outputs["muscle_presence_logits"].dtype),
    )
    total = (
        0.40 * coarse
        + 0.20 * muscle
        + 0.25 * dice
        + 0.10 * boundary_component
        + 0.05 * presence
    )
    components = {
        "coarse": coarse.detach(),
        "muscle": muscle.detach(),
        "dice": dice.detach(),
        "boundary": boundary_component.detach(),
        "presence": presence.detach(),
    }
    return total, components, probabilities


class Metrics:
    def __init__(self, device: torch.device, classes: int = 6) -> None:
        self.classes = classes
        self.confusion = torch.zeros(
            classes, classes, dtype=torch.int64, device=device
        )
        self.boundary_counts = torch.zeros(
            3, dtype=torch.int64, device=device
        )

    def update(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
        boundary_probability: torch.Tensor,
        boundary_target: torch.Tensor,
    ) -> None:
        prediction = probabilities.argmax(1)
        flat = target.reshape(-1) * self.classes + prediction.reshape(-1)
        matrix = torch.bincount(
            flat, minlength=self.classes * self.classes
        ).reshape(self.classes, self.classes)
        self.confusion += matrix.detach()

        predicted_boundary = boundary_probability[:, 0] >= 0.5
        true_boundary = boundary_target >= 0.5
        self.boundary_counts += torch.stack(
            [
                (predicted_boundary & true_boundary).sum(),
                (predicted_boundary & ~true_boundary).sum(),
                (~predicted_boundary & true_boundary).sum(),
            ]
        )

    def results(self) -> tuple[float, list[float], float]:
        confusion = self.confusion.double()
        intersection = confusion.diag()
        union = (
            confusion.sum(0)
            + confusion.sum(1)
            - intersection
        )
        valid = union > 0
        per_class = torch.full_like(intersection, float("nan"))
        per_class[valid] = intersection[valid] / union[valid]
        mean_iou = float(per_class[valid].mean()) if torch.any(valid) else 0.0
        boundary_tp, boundary_fp, boundary_fn = self.boundary_counts.double()
        boundary_f1 = 2 * boundary_tp / (
            2 * boundary_tp + boundary_fp + boundary_fn
        ).clamp_min(1)
        return (
            float(mean_iou),
            per_class.detach().cpu().tolist(),
            float(boundary_f1),
        )


def device_channels_last_collate(
    samples: list[
        dict[
            str,
            torch.Tensor | str | int | tuple[float, float, float, float],
        ]
    ],
) -> dict[str, torch.Tensor | list[str]]:
    """Collate and augment cached CUDA samples with a bounded launch count."""
    if "geometry_code" in samples[0]:
        # Sorting is harmless because the training loader is already shuffled.
        # It makes each transform group a contiguous batch slice, avoiding
        # hundreds of per-sample index/copy kernels.
        samples = sorted(samples, key=lambda sample: int(sample["geometry_code"]))

    first_local = samples[0]["local"]
    assert isinstance(first_local, torch.Tensor)

    def tensor_batch(key: str) -> torch.Tensor:
        values = [sample[key] for sample in samples]
        assert all(isinstance(value, torch.Tensor) for value in values)
        return torch.stack(values)  # type: ignore[arg-type]

    transform_groups: list[tuple[int, int, int]] = []
    if "geometry_code" in samples[0]:
        start = 0
        while start < len(samples):
            code = int(samples[start]["geometry_code"])
            end = start + 1
            while (
                end < len(samples)
                and int(samples[end]["geometry_code"]) == code
            ):
                end += 1
            transform_groups.append((start, end, code))
            start = end

    def apply_geometry(batch: torch.Tensor) -> torch.Tensor:
        if not transform_groups:
            return batch
        transformed = []
        for start, end, code in transform_groups:
            part = batch[start:end]
            if code & 4:
                part = torch.flip(part, dims=[3])
            if code & 8:
                part = torch.flip(part, dims=[2])
            rotation = code & 3
            if rotation:
                part = torch.rot90(part, rotation, dims=(2, 3))
            transformed.append(part)
        return torch.cat(transformed, dim=0)

    # Local and context share transformations, so process their six channels
    # together. The same applies to the two uint8 target channels.
    rgb = apply_geometry(
        torch.cat(
            [tensor_batch("local"), tensor_batch("context")],
            dim=1,
        )
    )
    local = rgb[:, :3].contiguous(memory_format=torch.channels_last)
    context = rgb[:, 3:].contiguous(memory_format=torch.channels_last)
    raw_mask = torch.stack(
        [sample["mask"] for sample in samples]  # type: ignore[list-item]
    )
    raw_boundary = torch.stack(
        [sample["boundary"] for sample in samples]  # type: ignore[list-item]
    )
    targets = apply_geometry(
        torch.stack([raw_mask, raw_boundary], dim=1)
    )
    mask = targets[:, 0].long()
    boundary = targets[:, 1].float()

    result: dict[str, torch.Tensor | list[str]] = {
        "local": local,
        "context": context,
        "mask": mask,
        "boundary": boundary,
        "muscle_presence": mask.eq(3).flatten(1).any(1).float(),
        "name": [str(sample["name"]) for sample in samples],
    }
    if "handcrafted" in samples[0]:
        result["handcrafted"] = apply_geometry(
            tensor_batch("handcrafted")
        ).contiguous(memory_format=torch.channels_last)
    if "color_jitter" in samples[0]:
        result["color_jitter"] = torch.tensor(
            [sample["color_jitter"] for sample in samples],
            dtype=torch.float32,
            device=first_local.device,
        )
    return result


def batched_color_jitter(
    image: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Apply independent photometric augmentation to an assembled CUDA batch."""
    if image.ndim != 4 or parameters.shape != (image.shape[0], 4):
        raise ValueError(
            "Expected image [N,3,H,W] and jitter parameters [N,4], "
            f"received {tuple(image.shape)} and {tuple(parameters.shape)}"
        )
    brightness, contrast, saturation, gamma = (
        parameters[:, index, None, None, None] for index in range(4)
    )
    image = image.float().div_(255.0).mul_(brightness)
    channel_mean = image.mean(dim=(2, 3), keepdim=True)
    image = (image - channel_mean).mul_(contrast).add_(channel_mean)
    gray = (
        0.2126 * image[:, 0:1]
        + 0.7152 * image[:, 1:2]
        + 0.0722 * image[:, 2:3]
    )
    image = gray + saturation * (image - gray)
    return image.clamp_(0, 1).pow_(gamma).clamp_(0, 1)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    training: bool,
    use_cuda: bool,
    already_on_device: bool = False,
) -> DataLoader:
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": training,
        "num_workers": workers,
        "pin_memory": use_cuda and not already_on_device,
        "persistent_workers": workers > 0,
        "drop_last": training and len(dataset) >= batch_size,
        "collate_fn": (
            device_channels_last_collate if already_on_device else None
        ),
    }

    if workers > 0:
        options["prefetch_factor"] = 2

    return DataLoader(**options)


def run_epoch(
    model: AccurateTissueNet,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    accumulation_steps: int = 1,
    gradient_clip: float = 1.0,
) -> tuple[float, float, list[float], float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    metrics = Metrics(device)
    total_loss = torch.zeros((), device=device)
    components: dict[str, float] = defaultdict(float)
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        local = batch["local"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        color_jitter = batch.get("color_jitter")
        if color_jitter is not None:
            color_jitter = color_jitter.to(device, non_blocking=True)
            local = batched_color_jitter(local, color_jitter)
            context = batched_color_jitter(context, color_jitter)
        if device.type == "cuda":
            local = local.contiguous(memory_format=torch.channels_last)
            context = context.contiguous(memory_format=torch.channels_last)
        mask = batch["mask"].to(device, non_blocking=True)
        boundary = batch["boundary"].to(device, non_blocking=True)
        presence = batch["muscle_presence"].to(device, non_blocking=True)
        handcrafted = batch.get("handcrafted")
        if handcrafted is not None:
            handcrafted = handcrafted.to(device, non_blocking=True)
            if not amp and handcrafted.dtype != local.dtype:
                # Cached maps use FP16 storage. Without CUDA autocast, the
                # first convolution requires their dtype to match the model.
                handcrafted = handcrafted.to(dtype=local.dtype)
            if device.type == "cuda":
                handcrafted = handcrafted.contiguous(
                    memory_format=torch.channels_last
                )

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                outputs = model(local, context, handcrafted)
                loss, batch_components, probabilities = complete_loss(
                    model,
                    outputs,
                    mask,
                    boundary,
                    presence,
                    class_weights,
                )
            if training:
                assert scaler is not None
                scaler.scale(loss / accumulation_steps).backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach()
        for name, value in batch_components.items():
            components[name] += value
        metrics.update(
            probabilities.detach(),
            mask,
            outputs["boundary_logits"].detach().sigmoid(),
            boundary,
        )

    batches = len(loader)
    mean_iou, per_class, boundary_f1 = metrics.results()
    return (
        float((total_loss / batches).detach().cpu()),
        mean_iou,
        per_class,
        boundary_f1,
        {
            name: float((value / batches).detach().cpu())
            for name, value in components.items()
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--val-data",
        type=Path,
        help="Separate specimen-level validation dataset. Recommended.",
    )
    parser.add_argument(
        "--cache-vram-reserve-gb",
        type=float,
        default=7.0,
        help=(
            "VRAM to leave unused by the image cache for the "
            "model, activations, gradients, and temporary allocations."
        ),
    )
    parser.add_argument(
        "--cache-training-images",
        action="store_true",
        help="Decode training images once and cache uint8 images in CUDA VRAM.",
    )
    parser.add_argument(
        "--cache-handcrafted-features",
        action="store_true",
        help=(
            "Precompute the nine structural maps once in FP16 VRAM. Requires "
            "--cache-training-images and about 18 extra bytes/source pixel."
        ),
    )
    parser.add_argument(
        "--cache-feature-tile-size",
        type=int,
        default=1024,
        help="Tile size used to bound temporary VRAM during feature precomputation.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--context-scale", type=int, default=4)
    parser.add_argument("--train-patches-per-image", type=int, default=2)
    parser.add_argument("--val-patches-per-image", type=int, default=4)
    parser.add_argument(
        "--split-unit",
        choices=("patch", "image"),
        default="patch",
        help=(
            "When --val-data is omitted, split deterministic patches by "
            "default or retain the previous whole-image split."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Fraction used for the automatic patch or image validation split.",
    )
    parser.add_argument(
        "--patch-pool-per-image",
        type=int,
        help=(
            "Deterministic candidate patches per source image for a patch "
            "split. Defaults to train-patches + val-patches."
        ),
    )
    parser.add_argument("--boundary-focus", type=float, default=0.45)
    parser.add_argument("--class-focus", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help=(
            "Compile the model with torch.compile. CUDA startup is slower, "
            "but steady-state Python/kernel overhead can be lower."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--no-fused-optimizer",
        action="store_true",
        help="Disable the fused CUDA AdamW implementation.",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TensorFloat-32 for remaining FP32 CUDA operations.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--class-weights", type=float, nargs=6)
    parser.add_argument("--output", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--last-output", type=Path, default=Path("last_model.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.crop_size <= 0 or args.crop_size % 32:
        parser.error("--crop-size must be a positive multiple of 32")
    if args.context_scale < 1:
        parser.error("--context-scale must be at least 1")
    if args.batch_size < 1 or args.accumulation_steps < 1:
        parser.error("Batch and accumulation sizes must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if (
        args.patch_pool_per_image is not None
        and args.patch_pool_per_image < 2
    ):
        parser.error("--patch-pool-per-image must be at least 2")
    if args.boundary_focus + args.class_focus > 1:
        parser.error("--boundary-focus + --class-focus cannot exceed 1")
    if args.cache_vram_reserve_gb < 0:
        parser.error("--cache-vram-reserve-gb cannot be negative")
    if args.cache_handcrafted_features and not args.cache_training_images:
        parser.error(
            "--cache-handcrafted-features requires --cache-training-images"
        )
    if args.cache_feature_tile_size < 64:
        parser.error("--cache-feature-tile-size must be at least 64")
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled "
            "PyTorch build and an NVIDIA driver."
        )
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        tf32_enabled = not args.no_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        torch.set_float32_matmul_precision(
            "high" if tf32_enabled else "highest"
        )
    else:
        if args.compile_model:
            raise RuntimeError("--compile-model is currently supported only on CUDA")
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    print(
        f"Using device={device}, AMP={amp}, "
        f"TF32={device.type == 'cuda' and not args.no_tf32}"
    )
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    resume_checkpoint = None
    runtime_context_scale = args.context_scale
    if args.resume:
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        runtime_context_scale = int(
            resume_checkpoint.get("model_config", {}).get(
                "context_scale", args.context_scale
            )
        )
    train_image_indices: list[int] | None
    val_image_indices: list[int] | None
    train_patch_indices: list[int] | None = None
    val_patch_indices: list[int] | None = None
    patch_pool_per_image: int | None = None

    if args.val_data:
        train_image_indices = None
        val_image_indices = None
        validation_root = args.val_data
    else:
        sample_count = len(discover_samples(args.data))
        validation_root = args.data
        if args.split_unit == "patch":
            train_image_indices = None
            val_image_indices = None
            patch_pool_per_image = (
                args.patch_pool_per_image
                if args.patch_pool_per_image is not None
                else (
                    args.train_patches_per_image
                    + args.val_patches_per_image
                )
            )
            patch_count = sample_count * patch_pool_per_image
            train_patch_indices, val_patch_indices = (
                deterministic_split_indices(
                    patch_count,
                    args.validation_fraction,
                    args.seed,
                )
            )
            print(
                "Warning: using an automatic patch-level split. Training and "
                "validation contain different deterministic patches from the "
                "same source images, so validation is not independent by "
                "patient/specimen."
            )
        else:
            if sample_count < 2:
                raise RuntimeError(
                    "An image-level split requires at least two source images"
                )
            train_image_indices, val_image_indices = (
                deterministic_split_indices(
                    sample_count,
                    args.validation_fraction,
                    args.seed,
                )
            )
            print(
                "Warning: using an automatic image-level split. For final "
                "measurements, pass --val-data split by patient/specimen."
            )

    if args.cache_training_images and device.type == "cuda":
        all_samples = discover_samples(args.data)

        training_samples: list[tuple[Path, Path]] = (
            [all_samples[index] for index in train_image_indices]
            if train_image_indices is not None
            else all_samples
        )

        cache_sizes = estimate_training_cache_bytes(
            training_samples,
            runtime_context_scale,
            args.cache_handcrafted_features,
        )
        cache_bytes = cache_sizes["total"]

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)

        reserve_bytes = int(
            args.cache_vram_reserve_gb * 1024 ** 3
        )

        usable_bytes = max(
            0,
            free_bytes - reserve_bytes,
        )

        print(
            "Training CUDA cache: "
            f"RGB={cache_sizes['rgb'] / 1024 ** 3:.2f} GiB, "
            f"context pyramid={cache_sizes['context'] / 1024 ** 3:.2f} GiB, "
            f"features={cache_sizes['features'] / 1024 ** 3:.2f} GiB, "
            f"targets={cache_sizes['targets'] / 1024 ** 3:.2f} GiB, "
            f"total={cache_bytes / 1024 ** 3:.2f} GiB"
        )

        print(
            f"CUDA memory: "
            f"{free_bytes / 1024 ** 3:.2f} GiB free / "
            f"{total_bytes / 1024 ** 3:.2f} GiB total"
        )

        print(
            f"Reserving "
            f"{args.cache_vram_reserve_gb:.2f} GiB "
            f"for training"
        )
        if args.cache_handcrafted_features:
            print(
                "Note: cached structural maps receive geometric augmentation "
                "but remain invariant to RGB color jitter. Compare validation "
                "area error against an uncached-feature run."
            )

        if cache_bytes > usable_bytes:
            raise RuntimeError(
                "Training image cache is too large for the "
                "configured VRAM reserve. "
                f"Cache requires {cache_bytes / 1024 ** 3:.2f} GiB, "
                f"but only {usable_bytes / 1024 ** 3:.2f} GiB "
                "is available for caching."
            )

    patch_level_split = train_patch_indices is not None
    if patch_level_split:
        assert patch_pool_per_image is not None
    train_base_dataset = SegmentationDataset(
        args.data,
        augment=True,
        crop_size=args.crop_size,
        context_scale=runtime_context_scale,
        boundary_focus_probability=(
            0 if patch_level_split else args.boundary_focus
        ),
        class_focus_probability=(
            0 if patch_level_split else args.class_focus
        ),
        patches_per_image=(
            patch_pool_per_image
            if patch_level_split
            else args.train_patches_per_image
        ),
        sample_indices=train_image_indices,
        cache_device=(
            device
            if args.cache_training_images and device.type == "cuda"
            else None
        ),
        cache_handcrafted_features=args.cache_handcrafted_features,
        cache_feature_tile_size=args.cache_feature_tile_size,
        fixed_patch_centers=patch_level_split,
    )
    val_base_dataset = SegmentationDataset(
        validation_root,
        augment=False,
        crop_size=args.crop_size,
        context_scale=runtime_context_scale,
        boundary_focus_probability=0,
        class_focus_probability=0,
        patches_per_image=(
            patch_pool_per_image
            if patch_level_split
            else args.val_patches_per_image
        ),
        sample_indices=val_image_indices,
        fixed_patch_centers=patch_level_split,
    )
    if patch_level_split:
        assert train_patch_indices is not None
        assert val_patch_indices is not None
        train_dataset: Dataset = Subset(
            train_base_dataset,
            train_patch_indices,
        )
        val_dataset: Dataset = Subset(
            val_base_dataset,
            val_patch_indices,
        )
    else:
        train_dataset = train_base_dataset
        val_dataset = val_base_dataset
    train_workers = (
        0
        if args.cache_training_images and device.type == "cuda"
        else args.workers
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        train_workers,
        True,
        device.type == "cuda",
        already_on_device=(
            args.cache_training_images
            and device.type == "cuda"
        ),
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, args.workers, False, device.type == "cuda"
    )
    if patch_level_split:
        print(
            f"Patch split: {patch_pool_per_image} deterministic patches/image, "
            f"{len(train_dataset):,} training and "
            f"{len(val_dataset):,} validation patches. Boundary/class-focused "
            "center sampling is disabled so patch ownership stays fixed."
        )
    data_split = {
        "unit": (
            "external"
            if args.val_data
            else ("patch" if patch_level_split else "image")
        ),
        "validation_fraction": args.validation_fraction,
        "patch_pool_per_image": patch_pool_per_image,
        "train_indices": (
            train_patch_indices
            if patch_level_split
            else train_image_indices
        ),
        "validation_indices": (
            val_patch_indices
            if patch_level_split
            else val_image_indices
        ),
    }
    optimizer_updates = math.ceil(
        len(train_loader) / args.accumulation_steps
    )
    print(
        f"Training workload: {len(train_dataset):,} patches, "
        f"{len(train_loader):,} batches/epoch, "
        f"batch size={args.batch_size}, "
        f"{optimizer_updates:,} optimizer updates/epoch"
    )
    if optimizer_updates < 25:
        print(
            "Warning: fewer than 25 optimizer updates per epoch. A very large "
            "batch can reduce accuracy unless total updates and the learning-"
            "rate schedule are retuned."
        )

    model_configuration = (
        resume_checkpoint["model_config"]
        if resume_checkpoint is not None
        else {
            "context_scale": runtime_context_scale,
            "pretrained": not args.no_pretrained,
        }
    )
    model = AccurateTissueNet(**model_configuration)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    encoder_prefixes = (
        "rgb_",
        "local_",
        "context_",
    )
    encoder_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        target = (
            encoder_parameters
            if name.startswith(encoder_prefixes)
            else other_parameters
        )
        target.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": other_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
        fused=device.type == "cuda" and not args.no_fused_optimizer,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=6, factor=0.5, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    class_weights = (
        torch.tensor(args.class_weights, device=device)
        if args.class_weights
        else None
    )

    start_epoch, best_iou = 1, -1.0
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_iou = float(resume_checkpoint.get("best_miou", -1))
        print(f"Resumed {args.resume} at epoch {start_epoch}")
        del resume_checkpoint

    if args.compile_model:
        model.compile(mode=args.compile_mode)
        print(
            f"Enabled torch.compile mode={args.compile_mode}; the first "
            "training steps will include compilation time."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.last_output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_started = time.perf_counter()
        train_result = run_epoch(
            model,
            train_loader,
            device,
            class_weights,
            amp,
            optimizer=optimizer,
            scaler=scaler,
            accumulation_steps=args.accumulation_steps,
            gradient_clip=args.gradient_clip,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_started
        patches_per_second = len(train_dataset) / max(train_seconds, 1e-9)
        with torch.inference_mode():
            val_result = run_epoch(
                model,
                val_loader,
                device,
                class_weights,
                amp,
            )
        train_loss, train_iou = train_result[0], train_result[1]
        val_loss, val_iou, class_iou, boundary_f1, components = val_result
        scheduler.step(val_iou)
        class_summary = " ".join(
            f"{name}={score:.3f}"
            for name, score in zip(CLASS_NAMES, class_iou)
        )
        print(
            f"{epoch:03d}/{args.epochs} train loss={train_loss:.4f} "
            f"mIoU={train_iou:.3f} "
            f"speed={patches_per_second:.1f} patches/s | "
            f"val loss={val_loss:.4f} "
            f"mIoU={val_iou:.3f} boundaryF1={boundary_f1:.3f}"
        )
        print(f"  IoU: {class_summary}")
        print(
            "  losses: "
            + " ".join(f"{name}={value:.4f}" for name, value in components.items())
        )

        checkpoint = {
            "model": model.state_dict(),
            "model_config": model.model_config(),
            "classes": CLASS_NAMES,
            "epoch": epoch,
            "best_miou": max(best_iou, val_iou),
            "val_miou": val_iou,
            "val_class_iou": class_iou,
            "val_boundary_f1": boundary_f1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "data_split": data_split,
        }
        torch.save(checkpoint, args.last_output)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(checkpoint, args.output)
            print(f"  saved new best checkpoint: {args.output}")

    print(f"Best validation mIoU={best_iou:.3f}; checkpoint={args.output}")


if __name__ == "__main__":
    main()
