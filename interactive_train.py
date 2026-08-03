from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tf

from data import (
    BACKGROUND_ID,
    SegmentationDataset,
    _extract_centered,
    bone_fibro_boundary,
    discover_samples,
)
from model import AccurateTissueNet, CLASS_NAMES
from predict import extract_context, open_image_sources, read_padded_region
from train import complete_loss, deterministic_split_indices, make_loader, run_epoch


LABEL_COLORS = {
    1: "#003B73",  # Bone (class ID 0)
    2: "#73C9E6",  # Fibrocartilage (class ID 1)
    3: "#8B1A1A",  # Cartilage (class ID 2)
    4: "#F4A6B8",  # Muscle (class ID 3)
    5: "#7B2D6F",  # Marrow (class ID 4)
    6: "#FFFFFF",  # Background (class ID 5)
}


@dataclass(frozen=True)
class ReviewTile:
    local: np.ndarray
    context: np.ndarray
    target: np.ndarray | None
    source_name: str
    source_path: str
    center_y: int
    center_x: int
    context_scale: int
    source_scene: int | None = None


@dataclass(frozen=True)
class ReviewDecision:
    action: str
    score: int | None = None
    corrected_target: np.ndarray | None = None


class RandomReviewSampler:
    """Read an unaugmented local tile and its exactly aligned context field."""

    def __init__(
        self,
        root: Path,
        crop_size: int,
        context_scale: int,
        sample_indices: list[int] | None = None,
    ) -> None:
        all_samples = discover_samples(root)
        self.samples = (
            [all_samples[index] for index in sample_indices]
            if sample_indices is not None
            else all_samples
        )
        self.crop_size = crop_size
        self.context_scale = context_scale

    def sample(self) -> ReviewTile:
        image_path, mask_path = random.choice(self.samples)
        with Image.open(image_path) as image_file:
            image = np.array(image_file.convert("RGB"), dtype=np.uint8, copy=True)
        with Image.open(mask_path) as mask_file:
            mask = np.array(mask_file, dtype=np.uint8, copy=True)

        if mask.ndim != 2:
            raise ValueError(
                f"{mask_path} must be a single-channel or indexed PNG, "
                f"received shape {mask.shape}"
            )
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image and mask dimensions differ for {image_path.name}: "
                f"{image.shape[:2]} versus {mask.shape}"
            )
        if mask.max() >= len(CLASS_NAMES):
            raise ValueError(f"{mask_path} contains class IDs outside 0..5")

        height, width = mask.shape
        center_y = random.randrange(height)
        center_x = random.randrange(width)
        local = _extract_centered(
            image,
            center_y,
            center_x,
            self.crop_size,
        )
        target = _extract_centered(
            mask,
            center_y,
            center_x,
            self.crop_size,
            constant_value=BACKGROUND_ID,
        )
        context_native = _extract_centered(
            image,
            center_y,
            center_x,
            self.crop_size * self.context_scale,
        )
        context_tensor = torch.from_numpy(context_native).permute(2, 0, 1)
        context = (
            tf.resize(
                context_tensor,
                [self.crop_size, self.crop_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        return ReviewTile(
            local=np.ascontiguousarray(local),
            context=np.ascontiguousarray(context),
            target=np.ascontiguousarray(target),
            source_name=image_path.stem,
            source_path=str(image_path.resolve()),
            center_y=center_y,
            center_x=center_x,
            context_scale=self.context_scale,
        )


class CziReviewSampler:
    """Sample native-resolution local/context pairs without decoding a slide."""

    def __init__(
        self,
        review_data: Path,
        crop_size: int,
        context_scale: int,
        recursive: bool,
        minimum_nonwhite_fraction: float,
        read_attempts: int,
    ) -> None:
        if review_data.is_file():
            if review_data.suffix.lower() != ".czi":
                raise ValueError("--review-data must be a CZI file or directory")
            self.paths = [review_data]
        elif review_data.is_dir():
            iterator = (
                review_data.rglob("*.czi")
                if recursive
                else review_data.glob("*.czi")
            )
            self.paths = sorted(iterator)
        else:
            raise FileNotFoundError(review_data)
        if not self.paths:
            scope = "recursively below" if recursive else "directly under"
            raise RuntimeError(f"No CZI files found {scope} {review_data}")
        self.crop_size = crop_size
        self.context_scale = context_scale
        self.minimum_nonwhite_fraction = minimum_nonwhite_fraction
        self.read_attempts = read_attempts

    @staticmethod
    def _nonwhite_fraction(image: np.ndarray) -> float:
        # Empty CZI mosaic regions are decoded as white. Requiring at least
        # one channel below 245 avoids wasting reviews on nearly blank tiles.
        return float(np.any(image < 245, axis=2).mean())

    def sample(self) -> ReviewTile:
        czi_path = random.choice(self.paths)
        with open_image_sources(czi_path, czi_scene=None) as sources:
            scene_index, source = random.choice(sources)
            best: tuple[float, int, int, np.ndarray] | None = None
            for _ in range(self.read_attempts):
                center_y = random.randrange(source.height)
                center_x = random.randrange(source.width)
                local = read_padded_region(
                    source,
                    center_y - self.crop_size // 2,
                    center_x - self.crop_size // 2,
                    self.crop_size,
                    self.crop_size,
                )
                nonwhite_fraction = self._nonwhite_fraction(local)
                if best is None or nonwhite_fraction > best[0]:
                    best = (nonwhite_fraction, center_y, center_x, local)
                if nonwhite_fraction >= self.minimum_nonwhite_fraction:
                    break
            assert best is not None
            _, center_y, center_x, local = best
            context = extract_context(
                source,
                center_y,
                center_x,
                self.crop_size * self.context_scale,
                self.crop_size,
            )

        scene_suffix = (
            "" if scene_index is None else f"__scene_{scene_index:03d}"
        )
        return ReviewTile(
            local=np.ascontiguousarray(local),
            context=np.ascontiguousarray(context),
            target=None,
            source_name=f"{czi_path.stem}{scene_suffix}",
            source_path=str(czi_path.resolve()),
            center_y=center_y,
            center_x=center_x,
            context_scale=self.context_scale,
            source_scene=scene_index,
        )


class FeedbackStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def paths(self, window: int | None = None) -> list[Path]:
        paths = sorted(self.root.glob("feedback_*.npz"))
        if window is not None and window > 0:
            paths = paths[-window:]
        return paths

    def save(
        self,
        tile: ReviewTile,
        probabilities: np.ndarray,
        corrected_target: np.ndarray,
        score: int,
        epoch: int,
    ) -> Path:
        expected_target_shape = tile.local.shape[:2]
        if corrected_target.shape != expected_target_shape:
            raise ValueError("Corrected target dimensions do not match the local tile")
        if corrected_target.min() < 0 or corrected_target.max() >= len(CLASS_NAMES):
            raise ValueError("Corrected target contains class IDs outside 0..5")
        if probabilities.shape != (
            len(CLASS_NAMES),
            tile.local.shape[0],
            tile.local.shape[1],
        ):
            raise ValueError("Probability dimensions do not match the reviewed tile")

        now = datetime.now(timezone.utc)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tile.source_name).strip("_")
        filename = (
            f"feedback_{now.strftime('%Y%m%dT%H%M%S%fZ')}_"
            f"e{epoch:04d}_{safe_name or 'tile'}.npz"
        )
        destination = self.root / filename
        temporary = destination.with_suffix(".tmp.npz")
        original_target = (
            tile.target
            if tile.target is not None
            else probabilities.argmax(axis=0).astype(np.uint8)
        )
        corrected_pixel_count = int(np.count_nonzero(
            corrected_target != original_target
        ))
        metadata = {
            "format_version": 1,
            "created_utc": now.isoformat(),
            "epoch": epoch,
            "score": score,
            "source_name": tile.source_name,
            "source_path": tile.source_path,
            "source_scene": tile.source_scene,
            "center_y": tile.center_y,
            "center_x": tile.center_x,
            "crop_size": int(tile.local.shape[0]),
            "context_scale": tile.context_scale,
            "target_origin": (
                "annotation" if tile.target is not None else "prediction_seed"
            ),
            "corrected_pixel_count": corrected_pixel_count,
            "corrected_pixel_fraction": (
                corrected_pixel_count / corrected_target.size
            ),
            "classes": list(CLASS_NAMES),
        }
        np.savez_compressed(
            temporary,
            local=tile.local.astype(np.uint8, copy=False),
            context=tile.context.astype(np.uint8, copy=False),
            target=corrected_target.astype(np.uint8, copy=False),
            original_target=original_target.astype(np.uint8, copy=False),
            probabilities=probabilities.astype(np.float16, copy=False),
            metadata=np.asarray(json.dumps(metadata)),
        )
        temporary.replace(destination)
        return destination


class InteractiveFeedbackDataset(Dataset):
    """Replay reviewed local/context pairs as ordinary supervised samples."""

    def __init__(
        self,
        paths: list[Path],
        crop_size: int,
        context_scale: int,
        hard_example_boost: float,
    ) -> None:
        self.paths = paths
        self.crop_size = crop_size
        self.context_scale = context_scale
        self.hard_example_boost = hard_example_boost

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as record:
            local = np.array(record["local"], dtype=np.uint8, copy=True)
            context = np.array(record["context"], dtype=np.uint8, copy=True)
            target = np.array(record["target"], dtype=np.uint8, copy=True)
            metadata = json.loads(str(record["metadata"].item()))

        expected_rgb_shape = (self.crop_size, self.crop_size, 3)
        expected_target_shape = (self.crop_size, self.crop_size)
        if local.shape != expected_rgb_shape or context.shape != expected_rgb_shape:
            raise ValueError(
                f"{path} has incompatible local/context dimensions; expected "
                f"{expected_rgb_shape}, received {local.shape} and {context.shape}"
            )
        if target.shape != expected_target_shape:
            raise ValueError(
                f"{path} has incompatible target dimensions {target.shape}"
            )
        if int(metadata["context_scale"]) != self.context_scale:
            raise ValueError(
                f"{path} was reviewed at context scale "
                f"{metadata['context_scale']}, but the model expects "
                f"{self.context_scale}"
            )

        score = float(metadata["score"])
        feedback_weight = 1.0 + self.hard_example_boost * (1.0 - score / 100.0)
        target_tensor = torch.from_numpy(target).long()
        return {
            "local": torch.from_numpy(local).permute(2, 0, 1).float().div(255.0),
            "context": (
                torch.from_numpy(context).permute(2, 0, 1).float().div(255.0)
            ),
            "mask": target_tensor,
            "boundary": torch.from_numpy(
                bone_fibro_boundary(target, width=3).astype(np.float32)
            ),
            "muscle_presence": target_tensor.eq(3).any().float(),
            "feedback_weight": torch.tensor(feedback_weight, dtype=torch.float32),
            "name": path.stem,
        }


def predict_tile(
    model: AccurateTissueNet,
    tile: ReviewTile,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    local = (
        torch.from_numpy(tile.local)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    context = (
        torch.from_numpy(tile.context)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )
    if device.type == "cuda":
        local = local.contiguous(memory_format=torch.channels_last)
        context = context.contiguous(memory_format=torch.channels_last)

    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp,
    ):
        # The prediction being scored uses the same context tensor that is
        # saved in the feedback record and replayed during training.
        outputs = model(local, context)
        probabilities = model.probabilities_from_outputs(outputs)
    return probabilities[0].cpu().numpy()


def review_with_napari(
    tile: ReviewTile,
    probabilities: np.ndarray,
    epoch: int,
) -> ReviewDecision:
    try:
        import napari
        from qtpy.QtWidgets import (
            QComboBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as error:
        raise RuntimeError(
            "Interactive review requires Napari and a Qt backend. Install "
            "the optional GUI dependencies with: pip install -r "
            "requirements-interactive.txt"
        ) from error

    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    confidence = probabilities.max(axis=0).astype(np.float32)
    uncertainty = 1.0 - confidence
    display_prediction = prediction.astype(np.uint16) + 1
    has_annotation = tile.target is not None
    target_seed = tile.target if has_annotation else prediction
    assert target_seed is not None
    display_target = target_seed.astype(np.uint16) + 1
    stride = tile.local.shape[1] + 32
    decision = ReviewDecision(action="skip")

    viewer = napari.Viewer(
        title=f"Interactive tissue review - epoch {epoch} - {tile.source_name}"
    )
    viewer.add_image(tile.local, rgb=True, name="1 - Original local tile")
    viewer.add_image(
        tile.local,
        rgb=True,
        name="2 - Prediction image",
        translate=(0, stride),
    )
    viewer.add_labels(
        display_prediction,
        name="2 - Model prediction",
        color=LABEL_COLORS,
        opacity=0.58,
        translate=(0, stride),
    )
    viewer.add_image(
        tile.local,
        rgb=True,
        name="3 - Editable target image",
        translate=(0, 2 * stride),
    )
    target_layer = viewer.add_labels(
        display_target,
        name="3 - Training target (editable)",
        color=LABEL_COLORS,
        opacity=0.58,
        translate=(0, 2 * stride),
    )
    viewer.add_image(
        tile.context,
        rgb=True,
        name=f"4 - {tile.context_scale}x context field (resized)",
        translate=(0, 3 * stride),
    )
    viewer.add_image(
        uncertainty,
        name="Prediction uncertainty (hidden)",
        colormap="magma",
        opacity=0.65,
        visible=False,
        translate=(0, stride),
    )
    target_layer.selected_label = 1
    target_layer.brush_size = 20
    target_layer.mode = "paint"

    controls = QWidget()
    layout = QVBoxLayout(controls)
    target_instructions = (
        "The editable target starts from the existing annotation, never "
        "from the prediction. Paint corrections in panel 3 before saving."
        if has_annotation
        else (
            "This CZI tile has no annotation, so the editable target starts "
            "from the prediction. Inspect the entire tile and paint every "
            "needed correction before saving it as supervised training data."
        )
    )
    instructions = QLabel(
        "<b>Panels, left to right</b><br>"
        "Original | Prediction | Editable training target | Context<br><br>"
        "Rate the model prediction from 0 (wrong) to 100 (perfect). "
        f"{target_instructions}<br><br>"
        f"The context panel represents the same {tile.context_scale}x-wider "
        "field used by the context encoder."
    )
    instructions.setWordWrap(True)
    layout.addWidget(instructions)

    form = QFormLayout()
    score_box = QSpinBox()
    score_box.setRange(0, 100)
    score_box.setValue(50)
    score_box.setSuffix(" / 100")
    form.addRow("Prediction score", score_box)

    class_box = QComboBox()
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_box.addItem(f"{class_id}: {class_name}", class_id + 1)

    def select_class(index: int) -> None:
        target_layer.selected_label = int(class_box.itemData(index))

    class_box.currentIndexChanged.connect(select_class)
    form.addRow("Paint class", class_box)

    brush_box = QSpinBox()
    brush_box.setRange(1, max(1, tile.local.shape[0] // 2))
    brush_box.setValue(20)
    brush_box.valueChanged.connect(
        lambda value: setattr(target_layer, "brush_size", value)
    )
    form.addRow("Brush size", brush_box)
    layout.addLayout(form)

    legend = QLabel(
        "<b>Stored class IDs</b><br>"
        + "<br>".join(
            f'<span style="color:{LABEL_COLORS[index + 1]}">&#9632;</span> '
            f"{index}: {name}"
            for index, name in enumerate(CLASS_NAMES)
        )
    )
    layout.addWidget(legend)

    button_row = QHBoxLayout()
    save_button = QPushButton("Save feedback and continue")
    skip_button = QPushButton("Skip tile")
    stop_button = QPushButton("Stop reviews")
    button_row.addWidget(save_button)
    button_row.addWidget(skip_button)
    button_row.addWidget(stop_button)
    layout.addLayout(button_row)

    def save_review() -> None:
        nonlocal decision
        corrected_display = np.asarray(target_layer.data)
        if (
            corrected_display.ndim != 2
            or corrected_display.shape != tile.target.shape
            or corrected_display.min() < 1
            or corrected_display.max() > len(CLASS_NAMES)
        ):
            QMessageBox.critical(
                controls,
                "Invalid training target",
                "Every target pixel must have one of the six displayed classes.",
            )
            return
        corrected = corrected_display.astype(np.uint8) - 1
        if (
            not has_annotation
            and np.array_equal(corrected, prediction)
            and score_box.value() < 100
        ):
            QMessageBox.critical(
                controls,
                "CZI target was not corrected",
                "No label edits were detected. Correct the prediction before "
                "saving, or assign a score of 100 to confirm that every pixel "
                "was reviewed and the prediction is already correct.",
            )
            return
        decision = ReviewDecision(
            action="save",
            score=int(score_box.value()),
            corrected_target=np.ascontiguousarray(corrected),
        )
        viewer.close()

    def skip_review() -> None:
        nonlocal decision
        decision = ReviewDecision(action="skip")
        viewer.close()

    def stop_reviews() -> None:
        nonlocal decision
        decision = ReviewDecision(action="stop")
        viewer.close()

    save_button.clicked.connect(save_review)
    skip_button.clicked.connect(skip_review)
    stop_button.clicked.connect(stop_reviews)
    viewer.window.add_dock_widget(
        controls,
        name="Interactive feedback",
        area="right",
    )
    viewer.reset_view()
    napari.run()
    return decision


def run_feedback_epoch(
    model: AccurateTissueNet,
    paths: list[Path],
    device: torch.device,
    class_weights: torch.Tensor | None,
    amp: bool,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    crop_size: int,
    context_scale: int,
    hard_example_boost: float,
    feedback_loss_scale: float,
    gradient_clip: float,
) -> float | None:
    if not paths:
        return None
    dataset = InteractiveFeedbackDataset(
        paths,
        crop_size=crop_size,
        context_scale=context_scale,
        hard_example_boost=hard_example_boost,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        local = batch["local"].to(device, non_blocking=True)
        context = batch["context"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        boundary = batch["boundary"].to(device, non_blocking=True)
        presence = batch["muscle_presence"].to(device, non_blocking=True)
        feedback_weight = batch["feedback_weight"].to(device, non_blocking=True)
        if device.type == "cuda":
            local = local.contiguous(memory_format=torch.channels_last)
            context = context.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            # Both branches are active during feedback replay. The saved
            # context field is not reconstructed from the 512x512 local tile.
            outputs = model(local, context)
            base_loss, _, _ = complete_loss(
                model,
                outputs,
                mask,
                boundary,
                presence,
                class_weights,
            )
            loss = base_loss * feedback_weight.mean() * feedback_loss_scale

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu())
    return total_loss / len(loader)


def build_optimizer(
    model: AccurateTissueNet,
    encoder_lr: float,
    lr: float,
    weight_decay: float,
    fused: bool,
) -> torch.optim.AdamW:
    encoder_prefixes = ("rgb_", "local_", "context_")
    encoder_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        target = (
            encoder_parameters
            if name.startswith(encoder_prefixes)
            else other_parameters
        )
        target.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": encoder_lr},
            {"params": other_parameters, "lr": lr},
        ],
        weight_decay=weight_decay,
        fused=fused,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an existing dual-scale tissue model with supervised "
            "human feedback collected in Napari before each epoch."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--review-data",
        type=Path,
        help=(
            "CZI file or directory from which interactive tiles are sampled. "
            "Ordinary --data and --val-data remain annotated raster datasets."
        ),
    )
    parser.add_argument(
        "--review-recursive",
        action="store_true",
        help="Find CZI files recursively below a --review-data directory.",
    )
    parser.add_argument(
        "--review-min-nonwhite-fraction",
        type=float,
        default=0.02,
        help="Prefer review tiles with at least this fraction of nonwhite pixels.",
    )
    parser.add_argument(
        "--review-read-attempts",
        type=int,
        default=12,
        help="Random CZI locations tried before accepting the least-blank tile.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of additional interactive epochs to run.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--train-patches-per-image", type=int, default=2)
    parser.add_argument("--val-patches-per-image", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--boundary-focus", type=float, default=0.45)
    parser.add_argument("--class-focus", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--class-weights", type=float, nargs=6)
    parser.add_argument(
        "--reviews-per-epoch",
        type=int,
        default=1,
        help="Random local/context pairs to review before each epoch.",
    )
    parser.add_argument(
        "--review-every",
        type=int,
        default=1,
        help="Open Napari every N epochs.",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Do not open Napari; replay existing feedback records only.",
    )
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        default=Path("interactive_feedback"),
    )
    parser.add_argument(
        "--feedback-window",
        type=int,
        default=32,
        help="Replay only the N most recent records per epoch; 0 replays all.",
    )
    parser.add_argument(
        "--hard-example-boost",
        type=float,
        default=2.0,
        help="Extra loss multiplier at score 0; it decays linearly to 0 at 100.",
    )
    parser.add_argument("--feedback-loss-scale", type=float, default=1.0)
    parser.add_argument(
        "--cache-training-images",
        action="store_true",
        help="Cache the ordinary training dataset as uint8 tensors in CUDA VRAM.",
    )
    parser.add_argument("--cache-handcrafted-features", action="store_true")
    parser.add_argument("--cache-feature-tile-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-fused-optimizer", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument(
        "--fresh-optimizer",
        action="store_true",
        help="Load model weights but not optimizer/scheduler/scaler state.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("interactive_best_model.pt")
    )
    parser.add_argument(
        "--last-output", type=Path, default=Path("interactive_last_model.pt")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.crop_size <= 0 or args.crop_size % 32:
        parser.error("--crop-size must be a positive multiple of 32")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Batch size must be positive and workers cannot be negative")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.boundary_focus + args.class_focus > 1:
        parser.error("--boundary-focus + --class-focus cannot exceed 1")
    if args.review_every < 1 or args.reviews_per_epoch < 0:
        parser.error("Review counts must be non-negative and frequency positive")
    if not 0 <= args.review_min_nonwhite_fraction <= 1:
        parser.error("--review-min-nonwhite-fraction must be between 0 and 1")
    if args.review_read_attempts < 1:
        parser.error("--review-read-attempts must be positive")
    if args.feedback_window < 0:
        parser.error("--feedback-window cannot be negative")
    if args.hard_example_boost < 0 or args.feedback_loss_scale < 0:
        parser.error("Feedback weights cannot be negative")
    if args.cache_handcrafted_features and not args.cache_training_images:
        parser.error("--cache-handcrafted-features requires --cache-training-images")
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
        torch.set_float32_matmul_precision("high" if tf32_enabled else "highest")
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    print(f"Using device={device}, AMP={amp}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_configuration = dict(checkpoint["model_config"])
    context_scale = int(model_configuration["context_scale"])
    model = AccurateTissueNet(**model_configuration)
    model.load_state_dict(checkpoint["model"])
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    if args.val_data:
        train_indices = None
        val_indices = None
        validation_root = args.val_data
    else:
        sample_count = len(discover_samples(args.data))
        if sample_count < 2:
            raise RuntimeError(
                "Automatic validation requires at least two source images; "
                "otherwise pass --val-data"
            )
        train_indices, val_indices = deterministic_split_indices(
            sample_count,
            args.validation_fraction,
            args.seed,
        )
        validation_root = args.data
        print(
            "Warning: interactive training uses an automatic image-level split. "
            "For final measurements, pass --val-data split by patient/specimen."
        )

    cache_device = (
        device
        if args.cache_training_images and device.type == "cuda"
        else None
    )
    if args.cache_training_images and device.type != "cuda":
        print("Warning: --cache-training-images is ignored without CUDA")
    if args.cache_handcrafted_features and cache_device is None:
        raise RuntimeError(
            "--cache-handcrafted-features requires CUDA and a CUDA image cache"
        )
    train_dataset = SegmentationDataset(
        args.data,
        augment=True,
        crop_size=args.crop_size,
        context_scale=context_scale,
        boundary_focus_probability=args.boundary_focus,
        class_focus_probability=args.class_focus,
        patches_per_image=args.train_patches_per_image,
        sample_indices=train_indices,
        cache_device=cache_device,
        cache_handcrafted_features=args.cache_handcrafted_features,
        cache_feature_tile_size=args.cache_feature_tile_size,
    )
    val_dataset = SegmentationDataset(
        validation_root,
        augment=False,
        crop_size=args.crop_size,
        context_scale=context_scale,
        boundary_focus_probability=0,
        class_focus_probability=0,
        patches_per_image=args.val_patches_per_image,
        sample_indices=val_indices,
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        0 if cache_device is not None else args.workers,
        True,
        device.type == "cuda",
        already_on_device=cache_device is not None,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        args.workers,
        False,
        device.type == "cuda",
    )
    review_sampler: RandomReviewSampler | CziReviewSampler | None
    if args.no_review:
        review_sampler = None
    elif args.review_data is not None:
        review_sampler = CziReviewSampler(
            args.review_data,
            crop_size=args.crop_size,
            context_scale=context_scale,
            recursive=args.review_recursive,
            minimum_nonwhite_fraction=args.review_min_nonwhite_fraction,
            read_attempts=args.review_read_attempts,
        )
        print(
            f"Interactive reviews will be sampled from "
            f"{len(review_sampler.paths):,} CZI file(s) under "
            f"{args.review_data}."
        )
    else:
        review_sampler = RandomReviewSampler(
            args.data,
            crop_size=args.crop_size,
            context_scale=context_scale,
            sample_indices=train_indices,
        )
        print("Interactive reviews will be sampled from annotated --data images.")
    feedback_store = FeedbackStore(args.feedback_dir)

    optimizer = build_optimizer(
        model,
        encoder_lr=args.encoder_lr,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda" and not args.no_fused_optimizer,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=6,
        factor=0.5,
        min_lr=1e-7,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if not args.fresh_optimizer:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    checkpoint_class_weights = checkpoint.get("args", {}).get("class_weights")
    effective_class_weights = args.class_weights or checkpoint_class_weights
    class_weights = (
        torch.tensor(effective_class_weights, device=device)
        if effective_class_weights
        else None
    )

    source_epoch = int(checkpoint.get("epoch", 0))
    source_best_iou = float(checkpoint.get("best_miou", -1.0))
    best_interactive_iou = -1.0
    review_enabled = not args.no_review
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.last_output.parent.mkdir(parents=True, exist_ok=True)
    del checkpoint

    for interactive_epoch in range(1, args.epochs + 1):
        epoch = source_epoch + interactive_epoch
        if review_enabled and (interactive_epoch - 1) % args.review_every == 0:
            for review_number in range(args.reviews_per_epoch):
                assert review_sampler is not None
                tile = review_sampler.sample()
                print(
                    f"Review {review_number + 1}/{args.reviews_per_epoch} "
                    f"before epoch {epoch}: {tile.source_name} at "
                    f"({tile.center_y}, {tile.center_x})"
                )
                probabilities = predict_tile(model, tile, device, amp)
                decision = review_with_napari(tile, probabilities, epoch)
                if decision.action == "save":
                    assert decision.score is not None
                    assert decision.corrected_target is not None
                    saved_path = feedback_store.save(
                        tile,
                        probabilities,
                        decision.corrected_target,
                        decision.score,
                        epoch,
                    )
                    print(f"Saved feedback: {saved_path}")
                elif decision.action == "stop":
                    review_enabled = False
                    print("Interactive reviews stopped; training will continue.")
                    break
                else:
                    print("Review skipped.")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        train_result = run_epoch(
            model,
            train_loader,
            device,
            class_weights,
            amp,
            optimizer=optimizer,
            scaler=scaler,
            gradient_clip=args.gradient_clip,
        )
        feedback_paths = feedback_store.paths(
            None if args.feedback_window == 0 else args.feedback_window
        )
        feedback_loss = run_feedback_epoch(
            model,
            feedback_paths,
            device,
            class_weights,
            amp,
            optimizer,
            scaler,
            args.crop_size,
            context_scale,
            args.hard_example_boost,
            args.feedback_loss_scale,
            args.gradient_clip,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - started

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
        feedback_summary = (
            "none"
            if feedback_loss is None
            else f"{feedback_loss:.4f} from {len(feedback_paths)} tile(s)"
        )
        print(
            f"{interactive_epoch:03d}/{args.epochs} (epoch {epoch}) "
            f"train loss={train_loss:.4f} mIoU={train_iou:.3f} "
            f"time={train_seconds:.1f}s | feedback={feedback_summary} | "
            f"val loss={val_loss:.4f} mIoU={val_iou:.3f} "
            f"boundaryF1={boundary_f1:.3f}"
        )
        print(
            "  IoU: "
            + " ".join(
                f"{name}={score:.3f}"
                for name, score in zip(CLASS_NAMES, class_iou)
            )
        )
        print(
            "  losses: "
            + " ".join(f"{name}={value:.4f}" for name, value in components.items())
        )

        saved_checkpoint = {
            "model": model.state_dict(),
            "model_config": model.model_config(),
            "classes": CLASS_NAMES,
            "epoch": epoch,
            "best_miou": max(source_best_iou, best_interactive_iou, val_iou),
            "val_miou": val_iou,
            "val_class_iou": class_iou,
            "val_boundary_f1": boundary_f1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "interactive_feedback": {
                "directory": str(args.feedback_dir.resolve()),
                "record_count": len(feedback_store.paths()),
                "replayed_count": len(feedback_paths),
                "hard_example_boost": args.hard_example_boost,
                "feedback_loss_scale": args.feedback_loss_scale,
                "class_weights": effective_class_weights,
            },
        }
        torch.save(saved_checkpoint, args.last_output)
        if val_iou > best_interactive_iou:
            best_interactive_iou = val_iou
            torch.save(saved_checkpoint, args.output)
            print(f"  saved new best interactive checkpoint: {args.output}")

    print(
        f"Best interactive validation mIoU={best_interactive_iou:.3f}; "
        f"checkpoint={args.output}"
    )


if __name__ == "__main__":
    main()
