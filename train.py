from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import SegmentationDataset, discover_samples
from model import AccurateTissueNet, CLASS_NAMES


COARSE_MAPPING = (0, 1, 2, 1, 3, 4)


def coarse_targets(mask: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(COARSE_MAPPING, device=mask.device)
    return mapping[mask]


def multiclass_dice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    one_hot = F.one_hot(target, num_classes=len(CLASS_NAMES)).permute(0, 3, 1, 2)
    one_hot = one_hot.to(probabilities.dtype)
    dimensions = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + one_hot.sum(
        dim=dimensions
    )
    loss_per_class = 1 - (2 * intersection + eps) / (denominator + eps)
    if class_weights is None:
        return loss_per_class.mean()
    weights = class_weights / class_weights.sum()
    return (loss_per_class * weights).sum()


def binary_dice_loss(
    probabilities: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
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
    return F.binary_cross_entropy_with_logits(
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
    bce = F.binary_cross_entropy_with_logits(
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
    coarse = F.cross_entropy(
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
    presence = F.binary_cross_entropy_with_logits(
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
        "coarse": coarse.detach().item(),
        "muscle": muscle.detach().item(),
        "dice": dice.detach().item(),
        "boundary": boundary_component.detach().item(),
        "presence": presence.detach().item(),
    }
    return total, components, probabilities


class Metrics:
    def __init__(self, classes: int = 6) -> None:
        self.classes = classes
        self.confusion = torch.zeros(classes, classes, dtype=torch.float64)
        self.boundary_tp = 0
        self.boundary_fp = 0
        self.boundary_fn = 0

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
        self.confusion += matrix.detach().cpu()

        predicted_boundary = boundary_probability[:, 0] >= 0.5
        true_boundary = boundary_target >= 0.5
        self.boundary_tp += int((predicted_boundary & true_boundary).sum())
        self.boundary_fp += int((predicted_boundary & ~true_boundary).sum())
        self.boundary_fn += int((~predicted_boundary & true_boundary).sum())

    def results(self) -> tuple[float, list[float], float]:
        intersection = self.confusion.diag()
        union = (
            self.confusion.sum(0)
            + self.confusion.sum(1)
            - intersection
        )
        valid = union > 0
        per_class = torch.full_like(intersection, float("nan"))
        per_class[valid] = intersection[valid] / union[valid]
        mean_iou = float(per_class[valid].mean()) if torch.any(valid) else 0.0
        boundary_f1 = (
            2 * self.boundary_tp
            / max(
                1,
                2 * self.boundary_tp
                + self.boundary_fp
                + self.boundary_fn,
            )
        )
        return mean_iou, per_class.tolist(), boundary_f1


def make_loader(
    dataset: SegmentationDataset,
    batch_size: int,
    workers: int,
    training: bool,
    use_cuda: bool,
) -> DataLoader:
    options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": training,
        "num_workers": workers,
        "pin_memory": use_cuda,
        "persistent_workers": workers > 0,
        "drop_last": training and len(dataset) >= batch_size,
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
    metrics = Metrics()
    total_loss = 0.0
    components: dict[str, float] = defaultdict(float)
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        local = batch["local"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        if device.type == "cuda":
            local = local.contiguous(memory_format=torch.channels_last)
            context = context.contiguous(memory_format=torch.channels_last)
        mask = batch["mask"].to(device, non_blocking=True)
        boundary = batch["boundary"].to(device, non_blocking=True)
        presence = batch["muscle_presence"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                outputs = model(local, context)
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

        total_loss += loss.detach().item()
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
        total_loss / batches,
        mean_iou,
        per_class,
        boundary_f1,
        {name: value / batches for name, value in components.items()},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--val-data",
        type=Path,
        help="Separate specimen-level validation dataset. Recommended.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--context-scale", type=int, default=4)
    parser.add_argument("--train-patches-per-image", type=int, default=2)
    parser.add_argument("--val-patches-per-image", type=int, default=4)
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
    if args.boundary_focus + args.class_focus > 1:
        parser.error("--boundary-focus + --class-focus cannot exceed 1")
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
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    print(f"Using device={device}, AMP={amp}")
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

    if args.val_data:
        train_indices = None
        val_indices = None
        validation_root = args.val_data
    else:
        sample_count = len(discover_samples(args.data))
        order = torch.randperm(
            sample_count, generator=torch.Generator().manual_seed(args.seed)
        ).tolist()
        validation_count = max(1, round(0.15 * sample_count))
        val_indices = order[:validation_count]
        train_indices = order[validation_count:]
        validation_root = args.data
        print(
            "Warning: using an automatic image-level 85/15 split. For final "
            "measurements, pass --val-data split by patient/specimen."
        )

    train_dataset = SegmentationDataset(
        args.data,
        augment=True,
        crop_size=args.crop_size,
        context_scale=runtime_context_scale,
        boundary_focus_probability=args.boundary_focus,
        class_focus_probability=args.class_focus,
        patches_per_image=args.train_patches_per_image,
        sample_indices=train_indices,
    )
    val_dataset = SegmentationDataset(
        validation_root,
        augment=False,
        crop_size=args.crop_size,
        context_scale=runtime_context_scale,
        boundary_focus_probability=0,
        class_focus_probability=0,
        patches_per_image=args.val_patches_per_image,
        sample_indices=val_indices,
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, args.workers, True, device.type == "cuda"
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, args.workers, False, device.type == "cuda"
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.last_output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
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
            f"mIoU={train_iou:.3f} | val loss={val_loss:.4f} "
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
        }
        torch.save(checkpoint, args.last_output)
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(checkpoint, args.output)
            print(f"  saved new best checkpoint: {args.output}")

    print(f"Best validation mIoU={best_iou:.3f}; checkpoint={args.output}")


if __name__ == "__main__":
    main()
