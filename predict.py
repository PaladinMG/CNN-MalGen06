from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
from itertools import islice, product
import logging
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Iterator, Protocol

import numpy as np
import torch
from PIL import Image

from model import AccurateTissueNet, CLASS_NAMES


Image.MAX_IMAGE_PIXELS = None
LOGGER = logging.getLogger("tissue_prediction")
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


def configure_logging(output_dir: Path, requested_path: Path | None) -> Path:
    """Configure a flushed console and per-run diagnostic log."""
    if requested_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = output_dir / f"prediction_{timestamp}_pid{os.getpid()}.log"
    elif requested_path.is_absolute():
        log_path = requested_path
    else:
        log_path = output_dir / requested_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)
    return log_path


def _windows_process_metrics() -> dict[str, int]:
    if os.name != "nt":
        return {}
    try:
        import ctypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessHandleCount.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel32.GetProcessHandleCount.restype = ctypes.c_int
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        handle_count = ctypes.c_ulong()
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        metrics: dict[str, int] = {}
        if kernel32.GetProcessHandleCount(
            process, ctypes.byref(handle_count)
        ):
            metrics["handles"] = int(handle_count.value)
        if psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            metrics["working_set_bytes"] = int(counters.WorkingSetSize)
            metrics["private_bytes"] = int(counters.PrivateUsage)
            metrics["pagefile_bytes"] = int(counters.PagefileUsage)
        return metrics
    except Exception:
        LOGGER.debug("Unable to read Windows process metrics", exc_info=True)
        return {}


def _format_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def log_resource_snapshot(
    phase: str,
    output_dir: Path,
    device: torch.device | None = None,
) -> None:
    """Log resources most relevant to long-running Windows batch jobs."""
    fields = [f"phase={phase}", f"pid={os.getpid()}"]
    try:
        disk = shutil.disk_usage(output_dir)
        fields.extend(
            [
                f"disk_free={_format_gib(disk.free)}",
                f"disk_total={_format_gib(disk.total)}",
            ]
        )
    except OSError:
        LOGGER.debug("Unable to read output disk usage", exc_info=True)
    metrics = _windows_process_metrics()
    if "handles" in metrics:
        fields.append(f"handles={metrics['handles']}")
    if "working_set_bytes" in metrics:
        fields.append(
            f"working_set={_format_gib(metrics['working_set_bytes'])}"
        )
    if "private_bytes" in metrics:
        fields.append(f"private={_format_gib(metrics['private_bytes'])}")
    if "pagefile_bytes" in metrics:
        fields.append(f"pagefile={_format_gib(metrics['pagefile_bytes'])}")
    if device is not None and device.type == "cuda":
        fields.extend(
            [
                f"cuda_allocated={_format_gib(torch.cuda.memory_allocated())}",
                f"cuda_reserved={_format_gib(torch.cuda.memory_reserved())}",
                f"cuda_peak={_format_gib(torch.cuda.max_memory_allocated())}",
            ]
        )
    LOGGER.info("Resources | %s", " | ".join(fields))


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


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

    def _read_base_region_at_this_level(
        self,
        base_source: "CziImageSource",
        top: int,
        left: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        """Read a base-coordinate ROI from this pyramid level.

        czifile reports each pyramid level's downsampled dimensions, but its
        ``roi=`` argument remains in level-zero CZI coordinates. Translating
        the ROI through ``read_region`` would therefore divide its position
        and size twice and sample the wrong part of the slide.
        """
        scale_y = self.height / base_source.height
        scale_x = self.width / base_source.width
        output_height = max(1, round(height * scale_y))
        output_width = max(1, round(width * scale_x))

        source_top = max(0, top)
        source_left = max(0, left)
        source_bottom = min(base_source.height, top + height)
        source_right = min(base_source.width, left + width)
        if source_bottom <= source_top or source_right <= source_left:
            return np.full(
                (output_height, output_width, 3), 255, dtype=np.uint8
            )

        absolute_roi = (
            base_source.x + source_left,
            base_source.y + source_top,
            source_right - source_left,
            source_bottom - source_top,
        )
        try:
            array = self.scene(roi=absolute_roi).asarray(fillvalue=255)
        except ValueError as error:
            if "matches no subblocks" not in str(error):
                raise
            return np.full(
                (output_height, output_width, 3), 255, dtype=np.uint8
            )
        region = self._to_rgb(array)

        region_top = round((source_top - top) * scale_y)
        region_left = round((source_left - left) * scale_x)
        region_bottom = round((source_bottom - top) * scale_y)
        region_right = round((source_right - left) * scale_x)
        region_height = max(1, region_bottom - region_top)
        region_width = max(1, region_right - region_left)
        if region.shape[:2] != (region_height, region_width):
            region_image = Image.fromarray(region)
            try:
                region = np.array(
                    region_image.resize(
                        (region_width, region_height),
                        Image.Resampling.LANCZOS,
                        reducing_gap=2.0,
                    ),
                    dtype=np.uint8,
                    copy=True,
                )
            finally:
                region_image.close()

        pad_top = max(0, region_top)
        pad_left = max(0, region_left)
        pad_bottom = max(0, output_height - pad_top - region_height)
        pad_right = max(0, output_width - pad_left - region_width)
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
        return np.ascontiguousarray(region[:output_height, :output_width])

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
        top = center_y - native_size // 2
        left = center_x - native_size // 2
        if level is self:
            region = read_padded_region(
                self, top, left, native_size, native_size
            )
        else:
            region = level._read_base_region_at_this_level(
                self, top, left, native_size, native_size
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

    LOGGER.info("Opening CZI | path=%s", path)
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
        LOGGER.info("Closed CZI | path=%s", path)


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


def allocate_prediction_memmaps(
    probability_path: Path,
    temporary_dir: Path,
    height: int,
    width: int,
    probability_bytes: int,
    single_map_bytes: int,
) -> tuple[np.memmap, np.memmap, Path, np.memmap, Path]:
    """Allocate the three blending maps and remove partial files on failure."""
    probability_sum: np.memmap | None = None
    weight_sum: np.memmap | None = None
    boundary_sum: np.memmap | None = None
    weight_path: Path | None = None
    boundary_path: Path | None = None
    try:
        LOGGER.info(
            "Creating probability memmap | path=%s | size=%s",
            probability_path,
            _format_gib(probability_bytes),
        )
        probability_sum = np.lib.format.open_memmap(
            probability_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(CLASS_NAMES), height, width),
        )
        probability_sum[:] = 0
        LOGGER.info("Initialized probability memmap | path=%s", probability_path)

        weight_path = temporary_path(
            temporary_dir, ".blend_weights_", ".dat"
        )
        LOGGER.info(
            "Creating weight memmap | path=%s | size=%s",
            weight_path,
            _format_gib(single_map_bytes),
        )
        weight_sum = np.memmap(
            weight_path,
            mode="w+",
            dtype=np.float32,
            shape=(height, width),
        )
        weight_sum[:] = 0
        LOGGER.info("Initialized weight memmap | path=%s", weight_path)

        boundary_path = temporary_path(
            temporary_dir, ".boundary_", ".dat"
        )
        LOGGER.info(
            "Creating boundary memmap | path=%s | size=%s",
            boundary_path,
            _format_gib(single_map_bytes),
        )
        boundary_sum = np.memmap(
            boundary_path,
            mode="w+",
            dtype=np.float32,
            shape=(height, width),
        )
        boundary_sum[:] = 0
        LOGGER.info("Initialized boundary memmap | path=%s", boundary_path)
        return (
            probability_sum,
            weight_sum,
            weight_path,
            boundary_sum,
            boundary_path,
        )
    except Exception:
        LOGGER.exception("Failed while allocating prediction memmaps")
        for array in (probability_sum, weight_sum, boundary_sum):
            if array is not None:
                try:
                    close_memmap(array)
                except Exception:
                    LOGGER.debug(
                        "Unable to close a partial memmap", exc_info=True
                    )
        for path in (probability_path, weight_path, boundary_path):
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.debug(
                        "Unable to remove partial memmap | path=%s",
                        path,
                        exc_info=True,
                    )
        raise


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
    temporary_dir: Path,
    use_amp: bool,
    log_every_tiles: int,
) -> tuple[np.memmap, np.memmap, Path]:
    height, width = source.height, source.width
    y_starts = tile_starts(height, tile_size, overlap)
    x_starts = tile_starts(width, tile_size, overlap)
    coordinate_iterator = iter(product(y_starts, x_starts))
    total_tiles = len(y_starts) * len(x_starts)
    window = blend_window(tile_size, overlap)
    pixels = height * width
    probability_bytes = (
        len(CLASS_NAMES) * pixels * np.dtype(np.float32).itemsize
    )
    single_map_bytes = pixels * np.dtype(np.float32).itemsize
    LOGGER.info(
        "Tile plan | image=%dx%d | tiles=%d | tile_size=%d | overlap=%d "
        "| batch_size=%d | probability_map=%s | weight_map=%s | "
        "boundary_map=%s | peak_map_storage=%s",
        width,
        height,
        total_tiles,
        tile_size,
        overlap,
        batch_size,
        _format_gib(probability_bytes),
        _format_gib(single_map_bytes),
        _format_gib(single_map_bytes),
        _format_gib(probability_bytes + 2 * single_map_bytes),
    )
    log_resource_snapshot(
        "before_memmap_allocation", temporary_dir, device
    )
    (
        probability_sum,
        weight_sum,
        weight_path,
        boundary_sum,
        boundary_path,
    ) = allocate_prediction_memmaps(
        probability_path,
        temporary_dir,
        height,
        width,
        probability_bytes,
        single_map_bytes,
    )
    log_resource_snapshot(
        "after_memmap_allocation", temporary_dir, device
    )

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
    prediction_started = time.perf_counter()
    next_progress_log = min(log_every_tiles, total_tiles)
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
            if (
                processed_tiles >= next_progress_log
                or processed_tiles == total_tiles
            ):
                elapsed = max(time.perf_counter() - prediction_started, 1e-9)
                tiles_per_second = processed_tiles / elapsed
                remaining_seconds = (
                    (total_tiles - processed_tiles) / tiles_per_second
                    if tiles_per_second > 0
                    else float("inf")
                )
                LOGGER.info(
                    "Tile progress | processed=%d/%d | percent=%.2f | "
                    "elapsed_seconds=%.1f | tiles_per_second=%.2f | "
                    "eta_seconds=%.1f",
                    processed_tiles,
                    total_tiles,
                    100 * processed_tiles / total_tiles,
                    elapsed,
                    tiles_per_second,
                    remaining_seconds,
                )
                log_resource_snapshot(
                    f"tile_progress_{processed_tiles}",
                    probability_path.parent,
                    device,
                )
                while next_progress_log <= processed_tiles:
                    next_progress_log += log_every_tiles
            del (
                local_device,
                context_device,
                outputs,
                probabilities,
                boundaries,
                probabilities_cpu,
                boundaries_cpu,
            )

        LOGGER.info("Normalizing blended probability and boundary maps")
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
        LOGGER.info(
            "Tiled prediction complete | tiles=%d | elapsed_seconds=%.1f",
            total_tiles,
            time.perf_counter() - prediction_started,
        )
        return probability_sum, boundary_sum, boundary_path
    except Exception:
        LOGGER.exception(
            "Tiled prediction failed | processed_tiles=%d/%d",
            processed_tiles,
            total_tiles,
        )
        close_memmap(probability_sum)
        close_memmap(boundary_sum)
        boundary_path.unlink(missing_ok=True)
        probability_path.unlink(missing_ok=True)
        raise
    finally:
        close_memmap(weight_sum)
        weight_path.unlink(missing_ok=True)
        LOGGER.debug("Removed weight memmap | path=%s", weight_path)


def save_prediction_images(
    probabilities: np.memmap,
    boundary_probability: np.memmap,
    temporary_dir: Path,
    layout: dict[str, Path],
    output_name: str,
    rows_per_chunk: int = 256,
) -> None:
    _, height, width = probabilities.shape
    rendering_started = time.perf_counter()
    LOGGER.info(
        "Rendering output images | name=%s | dimensions=%dx%d",
        output_name,
        width,
        height,
    )
    label_buffer_path = temporary_path(
        temporary_dir, ".label_buffer_", ".dat"
    )
    image_buffer_path = temporary_path(
        temporary_dir, ".image_buffer_", ".dat"
    )
    LOGGER.debug(
        "Rendering buffers | labels=%s | image=%s",
        label_buffer_path,
        image_buffer_path,
    )
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
        LOGGER.debug(
            "Removed rendering buffers | labels=%s | image=%s",
            label_buffer_path,
            image_buffer_path,
        )
    LOGGER.info(
        "Rendered output images | name=%s | elapsed_seconds=%.1f",
        output_name,
        time.perf_counter() - rendering_started,
    )


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
    parser.add_argument(
        "--temp-in-project-dir",
        action="store_true",
        help=(
            "Place temporary memmaps beside predict.py instead of below "
            "--output-dir. Permanent output images still use --output-dir."
        ),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        help=(
            "Diagnostic log path. Relative paths are placed below "
            "--output-dir. By default a timestamped log is created there."
        ),
    )
    parser.add_argument(
        "--log-every-tiles",
        type=int,
        default=250,
        help=(
            "Write progress and resource metrics every N tiles "
            "(default: 250)."
        ),
    )
    args = parser.parse_args()
    if args.tile_size <= 0 or args.tile_size % 32:
        parser.error("--tile-size must be a positive multiple of 32")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        parser.error("--overlap must be at least 0 and smaller than --tile-size")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.log_every_tiles <= 0:
        parser.error("--log-every-tiles must be positive")
    if args.recursive and args.input_dir is None:
        parser.error("--recursive requires --input-dir")
    if args.czi_scene is not None and args.czi_scene < 0:
        parser.error("--czi-scene cannot be negative")
    return args


def run_prediction(args: argparse.Namespace) -> None:
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
    LOGGER.info("Using device | device=%s", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        LOGGER.info(
            "CUDA environment | device_name=%s | cuda_runtime=%s | "
            "cudnn=%s | device_count=%d",
            torch.cuda.get_device_name(device),
            torch.version.cuda,
            torch.backends.cudnn.version(),
            torch.cuda.device_count(),
        )

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
    LOGGER.info(
        "Model loaded | checkpoint=%s | context_scale=%d | "
        "parameters=%d | amp=%s",
        args.checkpoint,
        context_scale,
        sum(parameter.numel() for parameter in model.parameters()),
        device.type == "cuda" and not args.no_amp,
    )
    log_resource_snapshot("model_loaded", args.output_dir, device)
    layout = prepare_output_layout(
        args.output_dir,
        save_probabilities=not args.no_save_probabilities,
    )
    input_paths = discover_inputs(
        args.image,
        args.input_dir,
        args.recursive,
    )
    LOGGER.info(
        "Discovered inputs | count=%d | input_dir=%s | image=%s | recursive=%s",
        len(input_paths),
        args.input_dir,
        args.image,
        args.recursive,
    )
    used_output_names: set[str] = set()
    prediction_count = 0
    for input_index, input_path in enumerate(input_paths, start=1):
        input_started = time.perf_counter()
        try:
            input_size = input_path.stat().st_size
        except OSError:
            input_size = -1
        LOGGER.info(
            "Input start | index=%d/%d | name=%s | path=%s | file_size=%s",
            input_index,
            len(input_paths),
            input_path.name,
            input_path,
            _format_gib(input_size) if input_size >= 0 else "unknown",
        )
        log_resource_snapshot(
            f"input_{input_index}_before_open", args.output_dir, device
        )
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
                LOGGER.info(
                    "Scene start | output_name=%s | dimensions=%dx%d%s",
                    output_name,
                    source.width,
                    source.height,
                    scene_description,
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                probability_path = (
                    temporary_path(
                        args.temporary_dir, ".probabilities_", ".npy"
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
                        temporary_dir=args.temporary_dir,
                        use_amp=not args.no_amp,
                        log_every_tiles=args.log_every_tiles,
                    )
                )
                try:
                    save_prediction_images(
                        probabilities,
                        boundary_probability,
                        args.temporary_dir,
                        layout,
                        output_name,
                    )
                finally:
                    close_memmap(probabilities)
                    close_memmap(boundary_probability)
                    boundary_path.unlink(missing_ok=True)
                    if args.no_save_probabilities:
                        probability_path.unlink(missing_ok=True)
                    LOGGER.debug(
                        "Closed scene memmaps | probability=%s | boundary=%s "
                        "| probability_deleted=%s",
                        probability_path,
                        boundary_path,
                        args.no_save_probabilities,
                    )
                prediction_count += 1
                LOGGER.info("Scene completed | output_name=%s", output_name)
                log_resource_snapshot(
                    f"scene_{prediction_count}_completed",
                    args.output_dir,
                    device,
                )
        LOGGER.info(
            "Input completed | index=%d/%d | name=%s | elapsed_seconds=%.1f",
            input_index,
            len(input_paths),
            input_path.name,
            time.perf_counter() - input_started,
        )
        log_resource_snapshot(
            f"input_{input_index}_closed", args.output_dir, device
        )

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
    LOGGER.info(
        "Batch completed | predictions=%d | inputs=%d | output_dir=%s",
        prediction_count,
        len(input_paths),
        args.output_dir,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temporary_dir = (
        Path(__file__).resolve().parent
        if args.temp_in_project_dir
        else args.output_dir.resolve()
    )
    args.temporary_dir.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(args.output_dir, args.log_file)
    LOGGER.info("Prediction run started | log_file=%s", log_path)
    LOGGER.info("Command | %s", " ".join(sys.argv))
    LOGGER.info(
        "Temporary workspace | path=%s | in_project_dir=%s",
        args.temporary_dir,
        args.temp_in_project_dir,
    )
    LOGGER.info(
        "Environment | platform=%s | python=%s | numpy=%s | torch=%s | "
        "torchvision=%s | pillow=%s | czifile=%s | imagecodecs=%s",
        platform.platform(),
        sys.version.replace("\n", " "),
        np.__version__,
        torch.__version__,
        _package_version("torchvision"),
        _package_version("Pillow"),
        _package_version("czifile"),
        _package_version("imagecodecs"),
    )
    log_resource_snapshot("process_start", args.output_dir)
    try:
        run_prediction(args)
    except BaseException:
        LOGGER.exception("Prediction run terminated with an error")
        log_resource_snapshot("process_failure", args.output_dir)
        raise
    finally:
        LOGGER.info("Prediction run finished | log_file=%s", log_path)
        for handler in LOGGER.handlers:
            handler.flush()


if __name__ == "__main__":
    main()
