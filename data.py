from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from torchvision.transforms.v2 import functional as TF
from torchvision.transforms import InterpolationMode

import torch.nn.functional as F

from collections.abc import Sequence

from model import HandcraftedFeatures

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

def _extract_centered_tensor(
    tensor: torch.Tensor,
    center_y: int,
    center_x: int,
    size: int,
    constant_value: float | None = None,
) -> torch.Tensor:
    _, height, width = tensor.shape

    top = center_y - size // 2
    left = center_x - size // 2
    bottom = top + size
    right = left + size

    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - height)
    pad_right = max(0, right - width)

    top = max(0, top)
    left = max(0, left)
    bottom = min(height, bottom)
    right = min(width, right)

    crop = tensor[:, top:bottom, left:right]

    if pad_top or pad_bottom or pad_left or pad_right:
        padding = (
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
        )
        if constant_value is None:
            crop = F.pad(crop, padding, mode="replicate")
        else:
            crop = F.pad(
                crop,
                padding,
                mode="constant",
                value=constant_value,
            )

    # In-bounds crops remain views. The collate function copies them directly
    # into the final channels-last batch, avoiding a per-sample allocation.
    return crop

def _color_jitter_tensor(
    image: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
    gamma: float,
) -> torch.Tensor:

    image = image.float().div(255.0)

    image = image * brightness

    channel_mean = image.mean(
        dim=(1, 2),
        keepdim=True,
    )

    image = (
        (image - channel_mean) * contrast
        + channel_mean
    )

    gray = (
        0.2126 * image[0:1]
        + 0.7152 * image[1:2]
        + 0.0722 * image[2:3]
    )

    image = gray + saturation * (image - gray)

    image = (
        image.clamp(0, 1)
        .pow(gamma)
        .clamp(0, 1)
    )

    return image

def estimate_rgb_cache_bytes(
    samples: Sequence[tuple[Path, Path]],
) -> int:
    total = 0

    for image_path, _ in samples:
        with Image.open(image_path) as image:
            width, height = image.size

        total += width * height * 3

    return total


def estimate_training_cache_bytes(
    samples: Sequence[tuple[Path, Path]],
    context_scale: int,
    include_handcrafted_features: bool,
) -> dict[str, int]:
    rgb_bytes = 0
    context_bytes = 0
    feature_bytes = 0
    target_bytes = 0
    for image_path, _ in samples:
        with Image.open(image_path) as image:
            width, height = image.size
        pixels = width * height
        rgb_bytes += pixels * 3
        if context_scale > 1:
            context_height = math.ceil(height / context_scale)
            context_width = math.ceil(width / context_scale)
            context_bytes += context_height * context_width * 3
        if include_handcrafted_features:
            feature_bytes += pixels * 9 * 2  # cached as float16
        # One uint8 class mask and one uint8 Bone/Fibro boundary map. The CPU
        # mask remains available for inexpensive class-aware center sampling.
        target_bytes += pixels * 2
    return {
        "rgb": rgb_bytes,
        "context": context_bytes,
        "features": feature_bytes,
        "targets": target_bytes,
        "total": rgb_bytes + context_bytes + feature_bytes + target_bytes,
    }


@torch.no_grad()
def _precompute_handcrafted_features(
    image: torch.Tensor,
    extractor: HandcraftedFeatures,
    tile_size: int,
) -> torch.Tensor:
    """Calculate full-image maps once with a two-pixel receptive-field halo."""
    _, height, width = image.shape
    output = torch.empty(
        (9, height, width),
        dtype=torch.float16,
        device=image.device,
    )
    halo = 2
    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            extended_top = max(0, top - halo)
            extended_bottom = min(height, bottom + halo)
            extended_left = max(0, left - halo)
            extended_right = min(width, right + halo)
            rgb = image[
                :,
                extended_top:extended_bottom,
                extended_left:extended_right,
            ].unsqueeze(0)
            features = extractor(rgb.float().div(255.0))[0]
            source_top = top - extended_top
            source_left = left - extended_left
            output[:, top:bottom, left:right].copy_(
                features[
                    :,
                    source_top : source_top + (bottom - top),
                    source_left : source_left + (right - left),
                ]
            )
    return output

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
        cache_device: torch.device | None = None,
        cache_handcrafted_features: bool = False,
        cache_feature_tile_size: int = 1024,
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
        self.cache_device = cache_device
        self.cache_handcrafted_features = cache_handcrafted_features
        self.cache_feature_tile_size = cache_feature_tile_size
        if cache_handcrafted_features and cache_device is None:
            raise ValueError(
                "cache_handcrafted_features requires a CUDA image cache"
            )
        all_samples = discover_samples(root)
        self.samples = (
            [all_samples[index] for index in sample_indices]
            if sample_indices is not None
            else all_samples
        )

        self.cached_images = None
        self.cached_masks = None
        self.cached_device_masks = None
        self.cached_context_images = None
        self.cached_handcrafted = None
        self.cached_boundaries = None
        self.cached_boundary_coordinates = None
        self.cached_present_classes = None

        if self.cache_device is not None:
            self.cached_images = []
            self.cached_masks = []
            self.cached_device_masks = []
            self.cached_context_images = []
            self.cached_handcrafted = [] if cache_handcrafted_features else None
            self.cached_boundaries = []
            self.cached_boundary_coordinates = []
            self.cached_present_classes = []
            feature_extractor = (
                HandcraftedFeatures().to(self.cache_device).eval()
                if cache_handcrafted_features
                else None
            )

            for cache_index, (image_path, mask_path) in enumerate(self.samples):
                with Image.open(image_path) as image_file:
                    image = np.array(
                        image_file.convert("RGB"),
                        dtype=np.uint8,
                        copy=True,
                    )

                with Image.open(mask_path) as mask_file:
                    mask = np.array(
                        mask_file,
                        dtype=np.uint8,
                        copy=True,
                    )

                if mask.ndim != 2:
                    raise ValueError(
                        f"{mask_path} must be a single-channel or indexed PNG, "
                        f"received shape {mask.shape}"
                    )
                if image.shape[:2] != mask.shape:
                    raise ValueError(
                        f"Image and mask dimensions differ for "
                        f"{image_path.name}: {image.shape[:2]} versus "
                        f"{mask.shape}"
                    )
                if mask.max() > 5:
                    raise ValueError(
                        f"{mask_path} contains class IDs outside 0..5"
                    )

                image_tensor = (
                    torch.from_numpy(image)
                    .permute(2, 0, 1)
                    .contiguous()
                    .to(self.cache_device)
                )

                self.cached_images.append(image_tensor)

                if self.context_scale == 1:
                    context_tensor = image_tensor
                else:
                    context_height = math.ceil(
                        image_tensor.shape[1] / self.context_scale
                    )
                    context_width = math.ceil(
                        image_tensor.shape[2] / self.context_scale
                    )
                    context_tensor = TF.resize(
                        image_tensor,
                        [context_height, context_width],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ).contiguous()
                self.cached_context_images.append(context_tensor)

                # Retain a CPU copy for class-aware center sampling, while the
                # device copy eliminates one pageable host transfer per patch.
                self.cached_masks.append(mask)
                self.cached_device_masks.append(
                    torch.from_numpy(mask)
                    .unsqueeze(0)
                    .contiguous()
                    .to(self.cache_device)
                )
                full_boundary = bone_fibro_boundary(mask, width=3)
                self.cached_boundaries.append(
                    torch.from_numpy(full_boundary)
                    .to(torch.uint8)
                    .unsqueeze(0)
                    .contiguous()
                    .to(self.cache_device)
                )
                boundary_coordinates = np.column_stack(
                    np.nonzero(bone_fibro_boundary(mask, width=1))
                ).astype(np.int32, copy=False)
                self.cached_boundary_coordinates.append(boundary_coordinates)
                self.cached_present_classes.append(np.unique(mask))

                if feature_extractor is not None:
                    print(
                        f"Caching handcrafted features "
                        f"{cache_index + 1}/{len(self.samples)}: "
                        f"{image_path.name}"
                    )
                    self.cached_handcrafted.append(
                        _precompute_handcrafted_features(
                            image_tensor,
                            feature_extractor,
                            self.cache_feature_tile_size,
                        )
                    )
            del feature_extractor

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def _select_center(
        self, mask: np.ndarray, patch_number: int, sample_index: int
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
            if self.cached_boundary_coordinates is not None:
                coordinates = self.cached_boundary_coordinates[sample_index]
                boundary_size = len(coordinates)
            else:
                boundary_y, boundary_x = np.nonzero(
                    bone_fibro_boundary(mask, width=1)
                )
                coordinates = None
                boundary_size = boundary_y.size
            if boundary_size:
                selected = random.randrange(boundary_size)
                if coordinates is not None:
                    selected_y, selected_x = coordinates[selected]
                else:
                    selected_y = boundary_y[selected]
                    selected_x = boundary_x[selected]
                jitter = self.crop_size // 4
                center_y = int(selected_y) + random.randint(
                    -jitter, jitter
                )
                center_x = int(selected_x) + random.randint(
                    -jitter, jitter
                )
                return (
                    min(height - 1, max(0, center_y)),
                    min(width - 1, max(0, center_x)),
                )

        if choice < (
            self.boundary_focus_probability + self.class_focus_probability
        ):
            present = (
                self.cached_present_classes[sample_index]
                if self.cached_present_classes is not None
                else np.unique(mask)
            )
            foreground = present[present != BACKGROUND_ID]
            selected_class = int(
                random.choice(foreground.tolist() or present.tolist())
            )
            # Rejection sampling avoids retaining coordinate arrays that can
            # consume several times more CPU RAM than the uint8 masks.
            for _ in range(128):
                candidate_y = random.randrange(height)
                candidate_x = random.randrange(width)
                if mask[candidate_y, candidate_x] == selected_class:
                    return candidate_y, candidate_x
            class_y, class_x = np.nonzero(mask == selected_class)
            if class_y.size:
                selected = random.randrange(class_y.size)
                return int(class_y[selected]), int(class_x[selected])

        return random.randrange(height), random.randrange(width)

    def __getitem__(
        self, index: int
    ) -> dict[
        str,
        torch.Tensor | str | int | tuple[float, float, float, float],
    ]:
        sample_index, patch_number = divmod(index, self.patches_per_image)
        image_path, mask_path = self.samples[sample_index]
        if self.cached_images is not None:
            image = self.cached_images[sample_index]
            mask = self.cached_masks[sample_index]
            device_mask = self.cached_device_masks[sample_index]
            context_pyramid = self.cached_context_images[sample_index]
            full_boundary = self.cached_boundaries[sample_index]
            full_handcrafted = (
                self.cached_handcrafted[sample_index]
                if self.cached_handcrafted is not None
                else None
            )
        else:
            with Image.open(image_path) as image_file:
                image_np = np.array(
                    image_file.convert("RGB"),
                    dtype=np.uint8,
                    copy=True,
                )
            image = (
                torch.from_numpy(image_np)
                .permute(2, 0, 1)
                .contiguous()
            )
            with Image.open(mask_path) as mask_file:
                # Preserve indexed-PNG class IDs. Converting a palette mask to
                # luminance would silently replace indices with palette brightness.
                mask = np.array(
                    mask_file,
                    dtype=np.uint8,
                    copy=True,
                )
            context_pyramid = None
            device_mask = None
            full_boundary = None
            full_handcrafted = None
        if self.cached_images is None:
            if mask.ndim != 2:
                raise ValueError(
                    f"{mask_path} must be a single-channel or indexed PNG, "
                    f"received shape {mask.shape}"
                )
            if tuple(image.shape[-2:]) != tuple(mask.shape):
                raise ValueError(
                    f"Image and mask dimensions differ for "
                    f"{image_path.name}: {tuple(image.shape[-2:])} versus "
                    f"{mask.shape}"
                )
            if mask.max() > 5:
                raise ValueError(
                    f"{mask_path} contains class IDs outside 0..5"
                )

        center_y, center_x = self._select_center(
            mask, patch_number, sample_index
        )
        local = _extract_centered_tensor(
            image,
            center_y,
            center_x,
            self.crop_size,
        )

        if device_mask is not None:
            local_mask = _extract_centered_tensor(
                device_mask,
                center_y,
                center_x,
                self.crop_size,
                constant_value=BACKGROUND_ID,
            )
        else:
            local_mask = _extract_centered(
                mask,
                center_y,
                center_x,
                self.crop_size,
                constant_value=BACKGROUND_ID,
            )

        if context_pyramid is not None:
            image_height, image_width = image.shape[-2:]
            pyramid_height, pyramid_width = context_pyramid.shape[-2:]
            context_center_y = round(
                center_y * pyramid_height / image_height
            )
            context_center_x = round(
                center_x * pyramid_width / image_width
            )
            context = _extract_centered_tensor(
                context_pyramid,
                context_center_y,
                context_center_x,
                self.crop_size,
            )
        else:
            context_native = _extract_centered_tensor(
                image,
                center_y,
                center_x,
                self.crop_size * self.context_scale,
            )
            context = TF.resize(
                context_native,
                [self.crop_size, self.crop_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        handcrafted = (
            _extract_centered_tensor(
                full_handcrafted,
                center_y,
                center_x,
                self.crop_size,
            )
            if full_handcrafted is not None
            else None
        )
        boundary = (
            _extract_centered_tensor(
                full_boundary,
                center_y,
                center_x,
                self.crop_size,
                constant_value=0,
            )
            if full_boundary is not None
            else None
        )

        color_jitter = None
        geometry_code = None
        if self.augment:
            flip_horizontal = random.random() < 0.5
            flip_vertical = random.random() < 0.5
            rotation = random.randrange(4)
            if self.cache_device is not None:
                # Defer geometric augmentation so samples sharing the same
                # transform can be processed together after collation.
                geometry_code = (
                    rotation
                    | (int(flip_horizontal) << 2)
                    | (int(flip_vertical) << 3)
                )
            else:
                if flip_horizontal:
                    local = torch.flip(local, dims=[2])
                    context = torch.flip(context, dims=[2])
                    local_mask = local_mask[:, ::-1].copy()
                if flip_vertical:
                    local = torch.flip(local, dims=[1])
                    context = torch.flip(context, dims=[1])
                    local_mask = local_mask[::-1, :].copy()
                local = torch.rot90(local, rotation, dims=(1, 2))
                context = torch.rot90(context, rotation, dims=(1, 2))
                local_mask = np.rot90(local_mask, rotation).copy()

            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            saturation = random.uniform(0.75, 1.25)
            gamma = random.uniform(0.85, 1.15)
            if self.cache_device is not None:
                # Keep cached crops as uint8 and apply all 4 photometric
                # transforms to the assembled batch in one GPU operation.
                color_jitter = (
                    brightness,
                    contrast,
                    saturation,
                    gamma,
                )
            else:
                local = _color_jitter_tensor(
                    local,
                    brightness,
                    contrast,
                    saturation,
                    gamma,
                )
                context = _color_jitter_tensor(
                    context,
                    brightness,
                    contrast,
                    saturation,
                    gamma,
                )
        else:
            local = local.float().div(255.0)
            context = context.float().div(255.0)

        if boundary is None:
            boundary = bone_fibro_boundary(
                local_mask,
                width=3,
            ).astype(np.float32)

        if isinstance(local_mask, torch.Tensor):
            # Defer dtype conversion and Muscle-presence reduction until the
            # full batch has been assembled.
            mask_tensor = local_mask[0]
            boundary_tensor = boundary[0]
            presence_tensor = None
        else:
            muscle_presence = np.float32(np.any(local_mask == 3))
            mask_tensor = torch.from_numpy(local_mask).long()
            boundary_tensor = torch.from_numpy(boundary)
            presence_tensor = torch.tensor(muscle_presence)

        sample = {
            "local": local,
            "context": context,
            "mask": mask_tensor,
            "boundary": boundary_tensor,
            "name": image_path.stem,
        }
        if presence_tensor is not None:
            sample["muscle_presence"] = presence_tensor
        if handcrafted is not None:
            sample["handcrafted"] = handcrafted
        if color_jitter is not None:
            sample["color_jitter"] = color_jitter
        if geometry_code is not None:
            sample["geometry_code"] = geometry_code
        return sample
