from __future__ import annotations

import argparse
from itertools import islice, product
from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image

from model import AccurateTissueNet, CLASS_NAMES


Image.MAX_IMAGE_PIXELS = None
PALETTE = np.array(
    [
        [240, 220, 80],
        [30, 180, 210],
        [60, 210, 110],
        [210, 70, 90],
        [150, 80, 190],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)


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


def extract_context(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    native_size: int,
    output_size: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    top = center_y - native_size // 2
    left = center_x - native_size // 2
    bottom, right = top + native_size, left + native_size
    pad_top, pad_left = max(0, -top), max(0, -left)
    pad_bottom, pad_right = max(0, bottom - height), max(0, right - width)
    top, left = max(0, top), max(0, left)
    bottom, right = min(height, bottom), min(width, right)
    context = image[top:bottom, left:right]
    if pad_top or pad_bottom or pad_left or pad_right:
        context = np.pad(
            context,
            (
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0),
            ),
            mode="edge",
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
    image: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: int,
    batch_size: int,
    context_scale: int,
    probability_path: Path,
    use_amp: bool,
) -> tuple[np.memmap, np.memmap, Path]:
    height, width = image.shape[:2]
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
                local = image[top : top + tile_size, left : left + tile_size]
                valid_height, valid_width = local.shape[:2]
                valid_shapes.append((valid_height, valid_width))
                if valid_height < tile_size or valid_width < tile_size:
                    local = np.pad(
                        local,
                        (
                            (0, tile_size - valid_height),
                            (0, tile_size - valid_width),
                            (0, 0),
                        ),
                        mode="edge",
                    )
                center_y = top + tile_size // 2
                center_x = left + tile_size // 2
                context = extract_context(
                    image,
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
    rows_per_chunk: int = 256,
) -> None:
    _, height, width = probabilities.shape
    image_buffer_path = temporary_path(output_dir, ".image_buffer_", ".dat")
    image_buffer = np.memmap(
        image_buffer_path, mode="w+", dtype=np.uint8, shape=(height, width)
    )
    try:
        for top in range(0, height, rows_per_chunk):
            bottom = min(top + rows_per_chunk, height)
            image_buffer[top:bottom] = np.argmax(
                probabilities[:, top:bottom], axis=0
            ).astype(np.uint8)
        image_buffer.flush()
        label_image = Image.fromarray(image_buffer)
        label_image.save(output_dir / "class_ids.png")
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[: len(PALETTE)] = PALETTE
        label_image.putpalette(palette.ravel().tolist())
        label_image.save(output_dir / "color_mask.png")
        label_image.close()

        for class_id, name in enumerate(CLASS_NAMES):
            for top in range(0, height, rows_per_chunk):
                bottom = min(top + rows_per_chunk, height)
                image_buffer[top:bottom] = np.rint(
                    probabilities[class_id, top:bottom] * 255
                ).astype(np.uint8)
            image_buffer.flush()
            probability_image = Image.fromarray(image_buffer)
            probability_image.save(output_dir / f"{class_id}_{name}.png")
            probability_image.close()

        for top in range(0, height, rows_per_chunk):
            bottom = min(top + rows_per_chunk, height)
            image_buffer[top:bottom] = np.rint(
                boundary_probability[top:bottom] * 255
            ).astype(np.uint8)
        image_buffer.flush()
        boundary_image = Image.fromarray(image_buffer)
        boundary_image.save(output_dir / "bone_fibro_boundary.png")
        boundary_image.close()
    finally:
        close_memmap(image_buffer)
        image_buffer_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("prediction"))
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as original:
        image = np.array(original.convert("RGB"), dtype=np.uint8, copy=True)

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

    probability_path = (
        temporary_path(args.output_dir, ".probabilities_", ".npy")
        if args.no_save_probabilities
        else args.output_dir / "probabilities.npy"
    )
    probabilities, boundary_probability, boundary_path = tiled_prediction(
        model,
        image,
        device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        context_scale=context_scale,
        probability_path=probability_path,
        use_amp=not args.no_amp,
    )
    del image, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        save_prediction_images(
            probabilities,
            boundary_probability,
            args.output_dir,
        )
        (args.output_dir / "classes.txt").write_text(
            "\n".join(
                f"{class_id}: {name}"
                for class_id, name in enumerate(CLASS_NAMES)
            ),
            encoding="utf-8",
        )
    finally:
        close_memmap(probabilities)
        close_memmap(boundary_probability)
        boundary_path.unlink(missing_ok=True)
        if args.no_save_probabilities:
            probability_path.unlink(missing_ok=True)
    print(f"Saved predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
