from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice, product
from pathlib import Path
import tempfile
from typing import Iterator, Protocol

import numpy as np
import torch
from PIL import Image

from model import AccurateTissueNet, CLASS_NAMES


Image.MAX_IMAGE_PIXELS = None
PALETTE = np.array(
    [
        [0, 59, 115],     # Bone: dark blue
        [121, 199, 255],  # Fibrocartilage: light blue
        [139, 0, 0],      # Cartilage: dark red
        [255, 182, 193],  # Muscle: light pink
        [128, 0, 128],    # Marrow: dark purple
        [255, 255, 255],  # Background: white
    ],
    dtype=np.uint8,
)
RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_SUFFIXES = RASTER_SUFFIXES | {".czi"}


class ImageSource(Protocol):
    width: int
    height: int

    def read_region(
        self, top: int, left: int, height: int, width: int
    ) -> np.ndarray: ...


@dataclass
class ArrayImageSource:
    image: np.ndarray

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    def read_region(
        self, top: int, left: int, height: int, width: int
    ) -> np.ndarray:
        return np.ascontiguousarray(
            self.image[top : top + height, left : left + width]
        )


class CziImageSource:
    def __init__(self, scene: object, include_pyramid: bool = True) -> None:
        self.scene = scene
        self.axes = tuple(scene.dims)
        self.x, self.y, self.width, self.height = scene.bbox
        self.pixeltype = str(scene.pixeltype).lower()
        self.sample_labels = [
            str(value).lower() for value in scene.coords.get("S", [])
        ]
        self.pyramid_sources = []
        if include_pyramid:
            self.pyramid_sources = [
                CziImageSource(level, include_pyramid=False)
                for level in scene.levels[1:]
            ]

    def _to_rgb(self, array: np.ndarray) -> np.ndarray:
        axes = list(self.axes)
        while array.ndim > len(axes):
            array = array[0]
        for axis_index in range(len(axes) - 1, -1, -1):
            axis = axes[axis_index]
            if axis not in {"Y", "X", "S", "C"}:
                array = np.take(array, 0, axis=axis_index)
                axes.pop(axis_index)

        channel_axis = None
        if "S" in axes:
            channel_axis = axes.index("S")
        elif "C" in axes:
            channel_axis = axes.index("C")
        if channel_axis is not None:
            array = np.moveaxis(array, channel_axis, -1)
            axes.pop(channel_axis)
        else:
            array = array[..., None]

        if "Y" in axes and "X" in axes:
            y_axis, x_axis = axes.index("Y"), axes.index("X")
            array = np.moveaxis(array, (y_axis, x_axis), (0, 1))
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] >= 3:
            array = array[..., :3]
        else:
            raise ValueError(
                f"CZI scene has unsupported channel shape {array.shape}"
            )

        if self.sample_labels[:3] == ["blue", "green", "red"]:
            array = array[..., ::-1]
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.integer):
                maximum = np.iinfo(array.dtype).max
                array = np.rint(array.astype(np.float32) * (255 / maximum))
            else:
                finite = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
                maximum = float(finite.max(initial=1.0))
                if maximum <= 1.0:
                    finite = finite * 255.0
                array = np.rint(np.clip(finite, 0, 255))
            array = array.astype(np.uint8)
        return np.ascontiguousarray(array)

    def read_region(
        self, top: int, left: int, height: int, width: int
    ) -> np.ndarray:
        absolute_roi = (self.x + left, self.y + top, width, height)
        try:
            array = self.scene(roi=absolute_roi).asarray(fillvalue=255)
        except ValueError as error:
            if "matches no subblocks" not in str(error):
                raise
            return np.full((height, width, 3), 255, dtype=np.uint8)
        return self._to_rgb(array)

    def read_context(
        self,
        center_y: int,
        center_x: int,
        native_size: int,
        output_size: int,
    ) -> np.ndarray:
        requested_scale = native_size / output_size
        candidates = [self, *self.pyramid_sources]
        level = min(
            candidates,
            key=lambda candidate: abs(
                np.log2(
                    max(self.width / candidate.width, 1e-9)
                    / requested_scale
                )
            ),
        )
        scale_y = level.height / self.height
        scale_x = level.width / self.width
        level_height = max(1, round(native_size * scale_y))
        level_width = max(1, round(native_size * scale_x))
        level_center_y = round(center_y * scale_y)
        level_center_x = round(center_x * scale_x)
        region = read_padded_region(
            level,
            level_center_y - level_height // 2,
            level_center_x - level_width // 2,
            level_height,
            level_width,
        )
        if region.shape[:2] == (output_size, output_size):
            return region
        context_image = Image.fromarray(region)
        try:
            return np.array(
                context_image.resize(
                    (output_size, output_size),
                    Image.Resampling.LANCZOS,
                    reducing_gap=2.0,
                ),
                dtype=np.uint8,
                copy=True,
            )
        finally:
            context_image.close()


def _select_czi_plane(scene: object) -> object:
    selection: dict[str, int | slice] = {}
    for axis_index, axis in enumerate(scene.dims):
        if axis in {"Y", "X", "S"}:
            continue
        start = int(scene.start[axis_index])
        size = int(scene.sizes[axis])
        if axis == "C" and size >= 3:
            selection[axis] = slice(start, start + 3)
        else:
            selection[axis] = start
    return scene(**selection) if selection else scene


@contextmanager
def open_image_sources(
    path: Path,
    czi_scene: int | None,
) -> Iterator[list[tuple[int | None, ImageSource]]]:
    if path.suffix.lower() != ".czi":
        with Image.open(path) as original:
            image = np.array(
                original.convert("RGB"), dtype=np.uint8, copy=True
            )
        yield [(None, ArrayImageSource(image))]
        return

    try:
        import czifile
    except ImportError as error:
        raise RuntimeError(
            "CZI input requires czifile and imagecodecs. Run "
            "'python -m pip install -r requirements.txt'."
        ) from error

    czi = czifile.CziFile(path)
    # Reuse recently decoded mosaic subblocks across overlapping local and
    # context ROIs while keeping the cache bounded.
    czi.maxcache = 64
    try:
        scenes = czi.scenes
        if not scenes:
            raise RuntimeError(f"No readable scenes found in {path}")
        if czi_scene is not None:
            if czi_scene < 0 or czi_scene >= len(scenes):
                raise IndexError(
                    f"{path.name} has {len(scenes)} scene(s); "
                    f"--czi-scene {czi_scene} is out of range"
                )
            scene_indices = [czi_scene]
        else:
            scene_indices = list(range(len(scenes)))
        yield [
            (
                scene_index,
                CziImageSource(_select_czi_plane(scenes[scene_index])),
            )
            for scene_index in scene_indices
        ]
    finally:
        czi.close()


def discover_inputs(
    image: Path | None,
    input_dir: Path | None,
    recursive: bool,
) -> list[Path]:
    if image is not None:
        if not image.is_file():
            raise FileNotFoundError(image)
        if image.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported input suffix: {image.suffix}")
        return [image]
    assert input_dir is not None
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    iterator = input_dir.rglob("*.czi") if recursive else input_dir.glob("*.czi")
    inputs = sorted(path for path in iterator if path.is_file())
    if not inputs:
        scope = "recursively " if recursive else ""
        raise RuntimeError(f"No .czi files found {scope}under {input_dir}")
    return inputs


def prepare_output_layout(
    output_dir: Path,
    save_probabilities: bool,
) -> dict[str, Path]:
    layout: dict[str, Path] = {
        "boundary": output_dir / "Boundary",
        "full": output_dir / "Full Segmentations",
    }
    for name in CLASS_NAMES:
        grayscale = output_dir / name / "Grayscale"
        segmentation = output_dir / name / "Segmentation"
        grayscale.mkdir(parents=True, exist_ok=True)
        segmentation.mkdir(parents=True, exist_ok=True)
        layout[f"{name}:grayscale"] = grayscale
        layout[f"{name}:segmentation"] = segmentation
    layout["boundary"].mkdir(parents=True, exist_ok=True)
    layout["full"].mkdir(parents=True, exist_ok=True)
    if save_probabilities:
        layout["probabilities"] = output_dir / "Probabilities"
        layout["probabilities"].mkdir(parents=True, exist_ok=True)
    return layout


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def blend_window(tile_size: int, overlap: int) -> np.ndarray:
    if overlap == 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    axis = np.hanning(tile_size + 2)[1:-1].astype(np.float32)
    # A small floor retains stable values at the outer image boundary.
    axis = np.maximum(axis, 0.05)
    window = np.outer(axis, axis)
    return window / window.max()


def temporary_path(directory: Path, prefix: str, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=prefix, suffix=suffix, delete=False
    ) as temporary_file:
        return Path(temporary_file.name)


def close_memmap(array: np.memmap) -> None:
    array.flush()
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


def read_padded_region(
    source: ImageSource,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    source_top = max(0, top)
    source_left = max(0, left)
    source_bottom = min(source.height, top + height)
    source_right = min(source.width, left + width)
    if source_bottom <= source_top or source_right <= source_left:
        return np.full((height, width, 3), 255, dtype=np.uint8)
    region = source.read_region(
        source_top,
        source_left,
        source_bottom - source_top,
        source_right - source_left,
    )
    pad_top = source_top - top
    pad_left = source_left - left
    pad_bottom = top + height - source_bottom
    pad_right = left + width - source_right
    if pad_top or pad_bottom or pad_left or pad_right:
        region = np.pad(
            region,
            (
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0),
            ),
            mode="edge",
        )
    return np.ascontiguousarray(region)


def extract_context(
    source: ImageSource,
    center_y: int,
    center_x: int,
    native_size: int,
    output_size: int,
) -> np.ndarray:
    czi_context_reader = getattr(source, "read_context", None)
    if czi_context_reader is not None:
        return czi_context_reader(
            center_y,
            center_x,
            native_size,
            output_size,
        )
    top = center_y - native_size // 2
    left = center_x - native_size // 2
    context = read_padded_region(
        source, top, left, native_size, native_size
    )
    context_image = Image.fromarray(np.ascontiguousarray(context))
    resized = np.array(
        context_image.resize(
            (output_size, output_size),
            Image.Resampling.LANCZOS,
            reducing_gap=2.0,
        ),
        dtype=np.uint8,
        copy=True,
    )
    context_image.close()
    return resized


@torch.inference_mode()
def tiled_prediction(
    model: AccurateTissueNet,
    source: ImageSource,
    device: torch.device,
    tile_size: int,
    overlap: int,
    batch_size: int,
    context_scale: int,
    probability_path: Path,
    use_amp: bool,
) -> tuple[np.memmap, np.memmap, Path]:
    height, width = source.height, source.width
    y_starts = tile_starts(height, tile_size, overlap)
    x_starts = tile_starts(width, tile_size, overlap)
    coordinate_iterator = iter(product(y_starts, x_starts))
    total_tiles = len(y_starts) * len(x_starts)
    window = blend_window(tile_size, overlap)

    probability_sum = np.lib.format.open_memmap(
        probability_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(CLASS_NAMES), height, width),
    )
    probability_sum[:] = 0
    weight_path = temporary_path(
        probability_path.parent, ".blend_weights_", ".dat"
    )
    weight_sum = np.memmap(
        weight_path, mode="w+", dtype=np.float32, shape=(height, width)
    )
    weight_sum[:] = 0
    boundary_path = temporary_path(
        probability_path.parent, ".boundary_", ".dat"
    )
    boundary_sum = np.memmap(
        boundary_path, mode="w+", dtype=np.float32, shape=(height, width)
    )
    boundary_sum[:] = 0

    pin_memory = device.type == "cuda"
    local_host = torch.empty(
        (batch_size, 3, tile_size, tile_size),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    context_host = torch.empty(
        (batch_size, 3, tile_size, tile_size),
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    processed_tiles = 0
    try:
        while True:
            coordinates = list(islice(coordinate_iterator, batch_size))
            if not coordinates:
                break
            valid_shapes = []
            for batch_index, (top, left) in enumerate(coordinates):
                valid_height = min(tile_size, height - top)
                valid_width = min(tile_size, width - left)
                valid_shapes.append((valid_height, valid_width))
                local = read_padded_region(
                    source, top, left, tile_size, tile_size
                )
                center_y = top + tile_size // 2
                center_x = left + tile_size // 2
                context = extract_context(
                    source,
                    center_y,
                    center_x,
                    tile_size * context_scale,
                    tile_size,
                )
                local_tensor = torch.from_numpy(
                    np.ascontiguousarray(local)
                ).permute(2, 0, 1)
                context_tensor = torch.from_numpy(context).permute(2, 0, 1)
                local_host[batch_index].copy_(local_tensor)
                context_host[batch_index].copy_(context_tensor)
                local_host[batch_index].div_(255)
                context_host[batch_index].div_(255)

            local_device = local_host[: len(coordinates)].to(
                device, non_blocking=True
            )
            context_device = context_host[: len(coordinates)].to(
                device, non_blocking=True
            )
            if device.type == "cuda":
                local_device = local_device.contiguous(
                    memory_format=torch.channels_last
                )
                context_device = context_device.contiguous(
                    memory_format=torch.channels_last
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and use_amp,
            ):
                outputs = model(local_device, context_device)
                probabilities = model.probabilities_from_outputs(outputs)
                boundaries = outputs["boundary_logits"].sigmoid()
            probabilities_cpu = probabilities.float().cpu().numpy()
            boundaries_cpu = boundaries[:, 0].float().cpu().numpy()

            for (
                tile_probabilities,
                tile_boundary,
                (top, left),
                (valid_height, valid_width),
            ) in zip(
                probabilities_cpu,
                boundaries_cpu,
                coordinates,
                valid_shapes,
            ):
                bottom, right = top + valid_height, left + valid_width
                tile_weight = window[:valid_height, :valid_width]
                probability_sum[:, top:bottom, left:right] += (
                    tile_probabilities[:, :valid_height, :valid_width]
                    * tile_weight[None]
                )
                boundary_sum[top:bottom, left:right] += (
                    tile_boundary[:valid_height, :valid_width] * tile_weight
                )
                weight_sum[top:bottom, left:right] += tile_weight

            processed_tiles += len(coordinates)
            if processed_tiles == total_tiles or processed_tiles % 50 == 0:
                print(f"Processed {processed_tiles}/{total_tiles} tiles")
            del (
                local_device,
                context_device,
                outputs,
                probabilities,
                boundaries,
                probabilities_cpu,
                boundaries_cpu,
            )

        rows_per_chunk = max(1, min(256, height))
        for top in range(0, height, rows_per_chunk):
            bottom = min(top + rows_per_chunk, height)
            weights = weight_sum[top:bottom]
            for class_id in range(len(CLASS_NAMES)):
                block = probability_sum[class_id, top:bottom]
                np.divide(block, weights, out=block)
            boundary_block = boundary_sum[top:bottom]
            np.divide(boundary_block, weights, out=boundary_block)
        probability_sum.flush()
        boundary_sum.flush()
        return probability_sum, boundary_sum, boundary_path
    except Exception:
        close_memmap(probability_sum)
        close_memmap(boundary_sum)
        boundary_path.unlink(missing_ok=True)
        raise
    finally:
        close_memmap(weight_sum)
        weight_path.unlink(missing_ok=True)


def save_prediction_images(
    probabilities: np.memmap,
    boundary_probability: np.memmap,
    output_dir: Path,
    layout: dict[str, Path],
    output_name: str,
    rows_per_chunk: int = 256,
) -> None:
    _, height, width = probabilities.shape
    label_buffer_path = temporary_path(output_dir, ".label_buffer_", ".dat")
    image_buffer_path = temporary_path(output_dir, ".image_buffer_", ".dat")
    label_buffer = np.memmap(
        label_buffer_path, mode="w+", dtype=np.uint8, shape=(height, width)
    )
    image_buffer = np.memmap(
        image_buffer_path, mode="w+", dtype=np.uint8, shape=(height, width)
    )
    try:
        for top in range(0, height, rows_per_chunk):
            bottom = min(top + rows_per_chunk, height)
            label_buffer[top:bottom] = np.argmax(
                probabilities[:, top:bottom], axis=0
            ).astype(np.uint8)
        label_buffer.flush()
        label_image = Image.fromarray(label_buffer)
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[: len(PALETTE)] = PALETTE
        label_image.putpalette(palette.ravel().tolist())
        label_image.save(layout["full"] / f"{output_name}.png")
        label_image.close()

        for class_id, name in enumerate(CLASS_NAMES):
            for top in range(0, height, rows_per_chunk):
                bottom = min(top + rows_per_chunk, height)
                image_buffer[top:bottom] = np.rint(
                    probabilities[class_id, top:bottom] * 255
                ).astype(np.uint8)
            image_buffer.flush()
            probability_image = Image.fromarray(image_buffer)
            probability_image.save(
                layout[f"{name}:grayscale"] / f"{output_name}.png"
            )
            probability_image.close()

            for top in range(0, height, rows_per_chunk):
                bottom = min(top + rows_per_chunk, height)
                np.equal(
                    label_buffer[top:bottom],
                    class_id,
                    out=image_buffer[top:bottom],
                )
            image_buffer.flush()
            segmentation_image = Image.fromarray(image_buffer)
            class_palette = np.zeros((256, 3), dtype=np.uint8)
            class_palette[1] = PALETTE[class_id]
            segmentation_image.putpalette(class_palette.ravel().tolist())
            # Index 0 is transparent; index 1 contains only this class's
            # pixels in the requested class color.
            segmentation_image.save(
                layout[f"{name}:segmentation"] / f"{output_name}.png",
                transparency=0,
            )
            segmentation_image.close()

        for top in range(0, height, rows_per_chunk):
            bottom = min(top + rows_per_chunk, height)
            image_buffer[top:bottom] = np.rint(
                boundary_probability[top:bottom] * 255
            ).astype(np.uint8)
        image_buffer.flush()
        boundary_image = Image.fromarray(image_buffer)
        boundary_image.save(layout["boundary"] / f"{output_name}.png")
        boundary_image.close()
    finally:
        close_memmap(label_buffer)
        close_memmap(image_buffer)
        label_buffer_path.unlink(missing_ok=True)
        image_buffer_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--image",
        type=Path,
        help="One raster image or CZI file.",
    )
    inputs.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing CZI files for batch prediction.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("prediction"))
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Find CZI files recursively below --input-dir.",
    )
    parser.add_argument(
        "--czi-scene",
        type=int,
        help="Zero-based CZI scene to process. By default all scenes are used.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-save-probabilities", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.tile_size <= 0 or args.tile_size % 32:
        parser.error("--tile-size must be a positive multiple of 32")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        parser.error("--overlap must be at least 0 and smaller than --tile-size")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.recursive and args.input_dir is None:
        parser.error("--recursive requires --input-dir")
    if args.czi_scene is not None and args.czi_scene < 0:
        parser.error("--czi-scene cannot be negative")
    return args


def main() -> None:
    args = parse_args()
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
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if "model_config" not in checkpoint:
        raise RuntimeError(
            "This is a legacy TinyFeatureUNet checkpoint. Retrain with the "
            "updated train.py before using dual-scale prediction."
        )
    model = AccurateTissueNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.to(device).eval()
    context_scale = model.context_scale
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layout = prepare_output_layout(
        args.output_dir,
        save_probabilities=not args.no_save_probabilities,
    )
    input_paths = discover_inputs(
        args.image,
        args.input_dir,
        args.recursive,
    )
    used_output_names: set[str] = set()
    prediction_count = 0
    for input_index, input_path in enumerate(input_paths, start=1):
        print(f"Input {input_index}/{len(input_paths)}: {input_path.name}")
        with open_image_sources(input_path, args.czi_scene) as sources:
            for scene_index, source in sources:
                output_name = input_path.stem
                if scene_index is not None and (
                    len(sources) > 1 or scene_index != 0
                ):
                    output_name += f"__scene_{scene_index:03d}"
                base_output_name = output_name
                collision_index = 2
                while output_name.casefold() in used_output_names:
                    output_name = f"{base_output_name}__{collision_index}"
                    collision_index += 1
                used_output_names.add(output_name.casefold())

                scene_description = (
                    "" if scene_index is None else f", scene {scene_index}"
                )
                print(
                    f"Predicting {output_name}: {source.width}x{source.height}"
                    f"{scene_description}"
                )
                probability_path = (
                    temporary_path(
                        args.output_dir, ".probabilities_", ".npy"
                    )
                    if args.no_save_probabilities
                    else layout["probabilities"] / f"{output_name}.npy"
                )
                probabilities, boundary_probability, boundary_path = (
                    tiled_prediction(
                        model,
                        source,
                        device,
                        tile_size=args.tile_size,
                        overlap=args.overlap,
                        batch_size=args.batch_size,
                        context_scale=context_scale,
                        probability_path=probability_path,
                        use_amp=not args.no_amp,
                    )
                )
                try:
                    save_prediction_images(
                        probabilities,
                        boundary_probability,
                        args.output_dir,
                        layout,
                        output_name,
                    )
                finally:
                    close_memmap(probabilities)
                    close_memmap(boundary_probability)
                    boundary_path.unlink(missing_ok=True)
                    if args.no_save_probabilities:
                        probability_path.unlink(missing_ok=True)
                prediction_count += 1
                print(f"Completed {output_name}")

    classes_text = []
    for class_id, (name, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        hex_color = "#" + "".join(f"{int(value):02X}" for value in color)
        classes_text.append(f"{class_id}: {name}: {hex_color}")
    (args.output_dir / "classes.txt").write_text(
        "\n".join(classes_text), encoding="utf-8"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"Saved {prediction_count} prediction(s) from "
        f"{len(input_paths)} input file(s) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
