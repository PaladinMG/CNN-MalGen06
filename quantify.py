from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from model import CLASS_NAMES


Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate hard and probability-weighted tissue areas."
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-name",
        help=(
            "Input/scene stem in the batch layout. Optional when exactly one "
            "full segmentation exists."
        ),
    )
    parser.add_argument("--pixel-width-um", type=float, required=True)
    parser.add_argument("--pixel-height-um", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pixel_width_um <= 0 or args.pixel_height_um <= 0:
        parser.error("Pixel dimensions must be positive")

    legacy_label_path = args.prediction_dir / "class_ids.png"
    if legacy_label_path.exists():
        label_path = legacy_label_path
        prediction_name = "prediction"
        probability_path = args.prediction_dir / "probabilities.npy"
    else:
        full_dir = args.prediction_dir / "Full Segmentations"
        if args.prediction_name:
            prediction_name = args.prediction_name
            label_path = full_dir / f"{prediction_name}.png"
            if not label_path.exists():
                raise FileNotFoundError(label_path)
        else:
            candidates = sorted(full_dir.glob("*.png"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Found {len(candidates)} full segmentations; pass "
                    "--prediction-name to choose one"
                )
            label_path = candidates[0]
            prediction_name = label_path.stem
        probability_path = (
            args.prediction_dir
            / "Probabilities"
            / f"{prediction_name}.npy"
        )
    with Image.open(label_path) as label_image:
        if label_image.mode not in {"P", "L", "I"}:
            raise ValueError(
                f"{label_path} does not preserve indexed class IDs "
                f"(mode={label_image.mode})"
            )
        labels = np.array(label_image, dtype=np.uint8, copy=True)
    height, width = labels.shape
    hard_pixels = np.bincount(
        labels.reshape(-1), minlength=len(CLASS_NAMES)
    ).astype(np.float64)

    expected_pixels = np.full(len(CLASS_NAMES), np.nan, dtype=np.float64)
    if probability_path.exists():
        probabilities = np.load(probability_path, mmap_mode="r")
        if probabilities.shape != (len(CLASS_NAMES), height, width):
            raise ValueError(
                f"Probability shape {probabilities.shape} does not match "
                f"label dimensions {(height, width)}"
            )
        expected_pixels[:] = 0
        for top in range(0, height, 256):
            expected_pixels += probabilities[
                :, top : min(top + 256, height)
            ].sum(axis=(1, 2), dtype=np.float64)

    pixel_area_mm2 = (
        args.pixel_width_um * args.pixel_height_um / 1_000_000.0
    )
    hard_tissue_pixels = hard_pixels[:-1].sum()
    expected_tissue_pixels = np.nansum(expected_pixels[:-1])
    output = args.output or (
        args.prediction_dir
        / "Area Summaries"
        / f"{prediction_name}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "class_id",
                "class_name",
                "hard_pixels",
                "hard_area_mm2",
                "hard_percent_non_background",
                "expected_pixels",
                "expected_area_mm2",
                "expected_percent_non_background",
            ]
        )
        for class_id, name in enumerate(CLASS_NAMES):
            hard_percent = (
                100 * hard_pixels[class_id] / hard_tissue_pixels
                if class_id != len(CLASS_NAMES) - 1 and hard_tissue_pixels
                else np.nan
            )
            expected_percent = (
                100 * expected_pixels[class_id] / expected_tissue_pixels
                if class_id != len(CLASS_NAMES) - 1
                and expected_tissue_pixels
                and not np.isnan(expected_pixels[class_id])
                else np.nan
            )
            writer.writerow(
                [
                    class_id,
                    name,
                    f"{hard_pixels[class_id]:.0f}",
                    f"{hard_pixels[class_id] * pixel_area_mm2:.9f}",
                    f"{hard_percent:.6f}",
                    (
                        f"{expected_pixels[class_id]:.6f}"
                        if not np.isnan(expected_pixels[class_id])
                        else ""
                    ),
                    (
                        f"{expected_pixels[class_id] * pixel_area_mm2:.9f}"
                        if not np.isnan(expected_pixels[class_id])
                        else ""
                    ),
                    (
                        f"{expected_percent:.6f}"
                        if not np.isnan(expected_percent)
                        else ""
                    ),
                ]
            )
    print(
        f"Saved {output}; image={width}x{height}, "
        f"pixel_area={pixel_area_mm2:.9g} mm^2"
    )


if __name__ == "__main__":
    main()
