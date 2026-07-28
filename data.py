from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


Image.MAX_IMAGE_PIXELS = None
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
BACKGROUND_ID = 5


def discover_samples(root: str | Path) -> list[tuple[Path, Path]]:
    root = Path(root)
    image_dir, mask_dir = root / "images", root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected {image_dir} and {mask_dir}")
    samples: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for {image_path.name}: {mask_path}"
            )
        samples.append((image_path, mask_path))
    if not samples:
        raise RuntimeError(f"No image/mask pairs found under {root}")
    return samples


def _neighbor_any(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def bone_fibro_boundary(mask: np.ndarray, width: int = 2) -> np.ndarray:
    bone = mask == 0
    fibro = mask == 1
    boundary = (bone & _neighbor_any(fibro)) | (fibro & _neighbor_any(bone))
    for _ in range(max(0, width - 1)):
        boundary |= _neighbor_any(boundary)
    return boundary


def _extract_centered(
    array: np.ndarray,
    center_y: int,
    center_x: int,
    size: int,
    constant_value: int | None = None,
) -> np.ndarray:
    height, width = array.shape[:2]
    top = center_y - size // 2
    left = center_x - size // 2
    bottom, right = top + size, left + size
    pad_top, pad_left = max(0, -top), max(0, -left)
    pad_bottom, pad_right = max(0, bottom - height), max(0, right - width)
    top, left = max(0, top), max(0, left)
    bottom, right = min(height, bottom), min(width, right)
    crop = array[top:bottom, left:right]
    if pad_top or pad_bottom or pad_left or pad_right:
        padding = [(pad_top, pad_bottom), (pad_left, pad_right)]
        if array.ndim == 3:
            padding.append((0, 0))
        if constant_value is None:
            crop = np.pad(crop, padding, mode="edge")
        else:
            crop = np.pad(
                crop,
                padding,
                mode="constant",
                constant_values=constant_value,
            )
    return np.ascontiguousarray(crop)


def _color_jitter(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    gamma: float,
) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    image *= brightness
    channel_mean = image.mean(axis=(0, 1), keepdims=True)
    image = (image - channel_mean) * contrast + channel_mean
    gray = (
        0.2126 * image[..., 0:1]
        + 0.7152 * image[..., 1:2]
        + 0.0722 * image[..., 2:3]
    )
    image = gray + saturation * (image - gray)
    image = np.power(np.clip(image, 0, 1), gamma)
    return np.clip(image, 0, 1)


class SegmentationDataset(Dataset):
    """
    Returns aligned native-resolution local crops, 1:context_scale context crops
    resized to the local dimensions, class masks, and Bone/Fibro boundaries.
    """

    def __init__(
        self,
        root: str | Path,
        augment: bool = False,
        crop_size: int = 512,
        context_scale: int = 4,
        boundary_focus_probability: float = 0.45,
        class_focus_probability: float = 0.35,
        patches_per_image: int = 2,
        sample_indices: list[int] | None = None,
    ) -> None:
        if crop_size <= 0 or crop_size % 32:
            raise ValueError("crop_size must be a positive multiple of 32")
        if context_scale < 1:
            raise ValueError("context_scale must be at least 1")
        self.root = Path(root)
        self.augment = augment
        self.crop_size = crop_size
        self.context_scale = context_scale
        self.boundary_focus_probability = boundary_focus_probability
        self.class_focus_probability = class_focus_probability
        self.patches_per_image = max(1, patches_per_image)
        all_samples = discover_samples(root)
        self.samples = (
            [all_samples[index] for index in sample_indices]
            if sample_indices is not None
            else all_samples
        )

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def _select_center(
        self, mask: np.ndarray, patch_number: int
    ) -> tuple[int, int]:
        height, width = mask.shape
        if not self.augment:
            grid = math.ceil(math.sqrt(self.patches_per_image))
            row, column = divmod(patch_number, grid)
            center_y = round((row + 0.5) * height / grid)
            center_x = round((column + 0.5) * width / grid)
            return min(height - 1, center_y), min(width - 1, center_x)

        choice = random.random()
        if choice < self.boundary_focus_probability:
            boundary_y, boundary_x = np.nonzero(
                bone_fibro_boundary(mask, width=1)
            )
            if boundary_y.size:
                selected = random.randrange(boundary_y.size)
                jitter = self.crop_size // 4
                center_y = int(boundary_y[selected]) + random.randint(
                    -jitter, jitter
                )
                center_x = int(boundary_x[selected]) + random.randint(
                    -jitter, jitter
                )
                return (
                    min(height - 1, max(0, center_y)),
                    min(width - 1, max(0, center_x)),
                )

        if choice < (
            self.boundary_focus_probability + self.class_focus_probability
        ):
            present = np.unique(mask)
            foreground = present[present != BACKGROUND_ID]
            selected_class = int(
                random.choice(foreground.tolist() or present.tolist())
            )
            class_y, class_x = np.nonzero(mask == selected_class)
            if class_y.size:
                selected = random.randrange(class_y.size)
                return int(class_y[selected]), int(class_x[selected])

        return random.randrange(height), random.randrange(width)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_index, patch_number = divmod(index, self.patches_per_image)
        image_path, mask_path = self.samples[sample_index]
        with Image.open(image_path) as image_file:
            image = np.array(image_file.convert("RGB"), dtype=np.uint8, copy=True)
        with Image.open(mask_path) as mask_file:
            # Preserve indexed-PNG class IDs. Converting a palette mask to
            # luminance would silently replace indices with palette brightness.
            mask = np.array(mask_file, dtype=np.int64, copy=True)
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
        if mask.min() < 0 or mask.max() > 5:
            raise ValueError(f"{mask_path} contains class IDs outside 0..5")

        center_y, center_x = self._select_center(mask, patch_number)
        local = _extract_centered(
            image, center_y, center_x, self.crop_size
        )
        local_mask = _extract_centered(
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
        context_image = Image.fromarray(context_native)
        context = np.array(
            context_image.resize(
                (self.crop_size, self.crop_size),
                Image.Resampling.LANCZOS,
                reducing_gap=2.0,
            ),
            dtype=np.uint8,
            copy=True,
        )
        context_image.close()

        if self.augment:
            if random.random() < 0.5:
                local = local[:, ::-1].copy()
                context = context[:, ::-1].copy()
                local_mask = local_mask[:, ::-1].copy()
            if random.random() < 0.5:
                local = local[::-1, :].copy()
                context = context[::-1, :].copy()
                local_mask = local_mask[::-1, :].copy()
            rotation = random.randrange(4)
            local = np.rot90(local, rotation).copy()
            context = np.rot90(context, rotation).copy()
            local_mask = np.rot90(local_mask, rotation).copy()

            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.75, 1.25)
            gamma = random.uniform(0.85, 1.15)
            local = _color_jitter(
                local, brightness, contrast, saturation, gamma
            )
            context = _color_jitter(
                context, brightness, contrast, saturation, gamma
            )
        else:
            local = local.astype(np.float32) / 255.0
            context = context.astype(np.float32) / 255.0

        boundary = bone_fibro_boundary(local_mask, width=3).astype(np.float32)
        muscle_presence = np.float32(np.any(local_mask == 3))
        return {
            "local": torch.from_numpy(local).permute(2, 0, 1),
            "context": torch.from_numpy(context).permute(2, 0, 1),
            "mask": torch.from_numpy(local_mask),
            "boundary": torch.from_numpy(boundary),
            "muscle_presence": torch.tensor(muscle_presence),
            "name": image_path.stem,
        }
