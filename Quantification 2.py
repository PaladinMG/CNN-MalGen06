import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import argparse
import ctypes
import csv
import gc
import os
import time


Image.MAX_IMAGE_PIXELS = None
SEG_CLASSES = [
    "Bone",
    "Fibrocartilage",
    "Cartilage",
    "Muscle",
    "Marrow",
    "Background",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MEMORY_RESERVE_BYTES = 512 * 1024 ** 2
MEMORY_WAIT_SECONDS = 5
_image_progress_total = 0
_image_progress_completed = 0


def print_progress(message: str) -> None:
    print(f"[Quantification] {message}", flush=True)


def reset_image_progress(total: int) -> None:
    global _image_progress_total, _image_progress_completed
    _image_progress_total = total
    _image_progress_completed = 0
    print_progress(f"Image counter initialized: {total} image(s) remaining")


def print_image_opening(image_path: Path, cls: str, image_type: str) -> None:
    next_index = _image_progress_completed + 1
    if _image_progress_total > 0:
        remaining_after = max(0, _image_progress_total - next_index)
        print_progress(
            f"Opening image {next_index}/{_image_progress_total}: "
            f"{image_path.name} [{cls}/{image_type}] "
            f"({remaining_after} will remain after this image)"
        )
    else:
        print_progress(f"Opening image: {image_path.name} [{cls}/{image_type}]")


def mark_image_released(image_path: Path) -> None:
    global _image_progress_completed
    if _image_progress_total > 0:
        _image_progress_completed += 1
        remaining = max(0, _image_progress_total - _image_progress_completed)
        print_progress(
            f"Released image from memory: {image_path.name}; "
            f"{_image_progress_completed}/{_image_progress_total} processed, "
            f"{remaining} image(s) left"
        )
    else:
        print_progress(f"Released image from memory: {image_path.name}")


def available_memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        ):
            raise ctypes.WinError()
        return int(status.ullAvailPhys)

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except (ValueError, OSError):
            pass
    return None


def estimated_decode_bytes(
    width: int,
    height: int,
    image_type: str,
    image_mode: str,
) -> int:
    # Predictor segmentation PNGs are indexed P images: one byte per source
    # pixel and one byte for the final Boolean mask. Allow two more bytes for
    # decoder/copy slack. Unexpected multichannel inputs use a larger fallback.
    if image_type == "Segmentation" and image_mode not in {"1", "L", "P"}:
        bytes_per_pixel = 8
    else:
        bytes_per_pixel = 4
    return max(64 * 1024 ** 2, width * height * bytes_per_pixel)


def wait_for_image_memory(
    image_path: Path,
    width: int,
    height: int,
    image_type: str,
    image_mode: str,
) -> None:
    required = estimated_decode_bytes(
        width,
        height,
        image_type,
        image_mode,
    )
    required_with_reserve = required + MEMORY_RESERVE_BYTES
    waited = False
    while True:
        available = available_memory_bytes()
        if available is None:
            print_progress(
                "Available system memory could not be determined; "
                f"continuing with {image_path.name}"
            )
            return
        if available >= required_with_reserve:
            if waited:
                print_progress(
                    f"Enough memory is now available; opening {image_path.name}"
                )
            else:
                print_progress(
                    f"Memory check passed for {image_path.name}: "
                    f"approximately {required / 1024 ** 3:.2f} GiB required, "
                    f"{available / 1024 ** 3:.2f} GiB available"
                )
            return
        waited = True
        print_progress(
            f"Not enough memory to open {image_path.name}: approximately "
            f"{required / 1024 ** 3:.2f} GiB is needed plus a "
            f"{MEMORY_RESERVE_BYTES / 1024 ** 3:.2f} GiB reserve, but only "
            f"{available / 1024 ** 3:.2f} GiB is available. The program is "
            "waiting for there to be enough memory."
        )
        time.sleep(MEMORY_WAIT_SECONDS)


class OutputImage(object):

    def __init__(
            self,
            image_name: str,
            image_type: str,
            cls: str,
            image_path: Path,
    ) -> None:
        self.image_name = image_name
        self.image_type = image_type
        self.cls = cls
        self.image_path = image_path
        if not self.image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {self.image_path}"
            )

        try:
            with Image.open(self.image_path) as image:
                wait_for_image_memory(
                    self.image_path,
                    image.width,
                    image.height,
                    self.image_type,
                    image.mode,
                )
                print_image_opening(
                    self.image_path,
                    self.cls,
                    self.image_type,
                )
                if self.image_type == "Grayscale":
                    converted_image = image.convert("L")
                    try:
                        self.image_array = np.array(
                            converted_image,
                            dtype=np.uint8,
                            copy=True,
                        )
                    finally:
                        converted_image.close()
                elif self.image_type == "Segmentation":
                    if image.mode in {"1", "L", "P"}:
                        index_array = np.asarray(image, dtype=np.uint8)
                        self.image_array = np.not_equal(index_array, 0)
                        del index_array
                    else:
                        # Nonstandard RGB/RGBA masks still avoid the previous
                        # three-channel comparison by reducing to luminance.
                        converted_image = image.convert("L")
                        gray_array = None
                        try:
                            gray_array = np.asarray(
                                converted_image,
                                dtype=np.uint8,
                            )
                            self.image_array = np.not_equal(gray_array, 0)
                        finally:
                            if gray_array is not None:
                                del gray_array
                            converted_image.close()
                else:
                    raise ValueError(f"Unknown image type: {self.image_type}")
        except UnidentifiedImageError as error:
            raise ValueError(
                f"Not a valid image file: {self.image_path}"
            ) from error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "image_array"):
            del self.image_array
            gc.collect()
            mark_image_released(self.image_path)

    def confidence(self) -> float:
        if self.image_type != "Grayscale":
            raise TypeError(
                f"Expected Grayscale image, got {self.image_type} image instead"
            )
        probability_array = self.image_array.astype(np.float32) / np.float32(255.0)
        predicted_mask = probability_array >= 0.25
        if np.any(predicted_mask):
            confidence = float(probability_array[predicted_mask].mean())
        else:
            confidence = float(0)
        return confidence

    def area(self) -> int:
        if self.image_type != "Segmentation":
            raise TypeError(
                f"Expected Segmentation image, got {self.image_type} image instead"
            )
        area = np.sum(self.image_array)
        return area

    def percent_area(self, images: list[tuple[str, str, str, Path]]) -> float:
        if self.image_type != "Segmentation":
            raise TypeError(
                f"Expected Segmentation image, got {self.image_type} image instead"
            )
        total_area = self.image_array.shape[0] * self.image_array.shape[1]
        if self.cls == "Background":
            return (self.area() / total_area) * 100
        non_background_area = total_area - get_background_area(
            self.image_name,
            images=images,
        )
        if non_background_area == 0:
            return 0.0
        percent_area = (self.area() / non_background_area) * 100
        return percent_area

    def real_area(self) -> float:
        if self.image_type != "Segmentation":
            raise TypeError(
                f"Expected Segmentation image, got {self.image_type} image instead"
            )
        real_area = self.area() * 0.22 * 0.22
        return real_area



def discover_images(root: str | Path) -> list[tuple[str, str, str, Path]]:
    root = Path(root)
    print_progress(f"Discovering prediction images under: {root.resolve()}")
    images: list[tuple[str, str, str, Path]] = []
    for cls in SEG_CLASSES:
        print_progress(f"Checking class folders: {cls}")
        grayscale_dir, seg_dir = root / cls / "Grayscale", root / cls / "Segmentation"
        if not grayscale_dir.is_dir():
            raise FileNotFoundError(f"Missing Grayscale path for {cls}")
        if not seg_dir.is_dir():
            raise FileNotFoundError(f"Missing Segmentation path for {cls}")
        grayscale_paths = []
        # grayscaleImages = []
        seg_paths = []
        # segImages = []
        for imagePath in grayscale_dir.iterdir():
            if imagePath.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            grayscale_paths.append(imagePath)
            images.append((imagePath.name, "Grayscale", cls, imagePath))
        if not grayscale_paths:
            raise FileNotFoundError(f"No grayscale images found for {cls}")
        for imagePath in seg_dir.iterdir():
            if imagePath.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            seg_paths.append(imagePath)
            images.append((imagePath.name, "Segmentation", cls, imagePath))
        if not seg_paths:
            raise FileNotFoundError(f"No segmentation images found for {cls}")
        print_progress(
            f"Found {len(grayscale_paths)} grayscale and "
            f"{len(seg_paths)} segmentation image(s) for {cls}"
        )
    print_progress("Checking that every scan contains all required class images")
    image_names = sorted({item[0] for item in images})
    available_images = {
        (image_name, image_type, cls)
        for image_name, image_type, cls, _ in images
    }
    missing_images = [
        (image_name, image_type, cls)
        for image_name in image_names
        for cls in SEG_CLASSES
        for image_type in ("Grayscale", "Segmentation")
        if (image_name, image_type, cls) not in available_images
    ]
    if missing_images:
        preview = ", ".join(
            f"{image_name}: {cls}/{image_type}"
            for image_name, image_type, cls in missing_images[:10]
        )
        remaining = len(missing_images) - 10
        if remaining > 0:
            preview += f", and {remaining} more"
        raise FileNotFoundError(f"Missing required class images: {preview}")
    print_progress(
        f"Discovery complete: {len(image_names)} scan(s), "
        f"{len(images)} image file(s)"
    )
    return images

def get_background_area(image_name: str, images: list[tuple[str, str, str, Path]]) -> int:
    background_item = next(
        item
        for item in images
        if item[0] == image_name
        and item[1] == "Segmentation"
        and item[2] == "Background"
    )
    with OutputImage(*background_item) as background_image:
        return background_image.area()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-prediction",
        type=Path,
        default="prediction",
    )
    args = parser.parse_args()
    if not args.path_to_prediction.is_dir():
        raise FileNotFoundError(
            f"{args.path_to_prediction} is not a directory"
        )
    return args

def get_all_areas(images: list[tuple[str, str, str, Path]]) -> list[tuple[str, str, str, Path, float, float]]:
    all_areas: list[tuple[str, str, str, Path, float, float]] = []
    segmentation_images = [i for i in images if i[1] == "Segmentation"]
    background_statistics: dict[str, tuple[int, int]] = {}
    background_images = [i for i in segmentation_images if i[2] == "Background"]
    print_progress(
        f"Reading {len(background_images)} Background mask(s) before class areas"
    )
    for item in background_images:
        print_progress(f"Reading Background mask: {item[0]}")
        with OutputImage(*item) as background_image:
            background_statistics[item[0]] = (
                int(background_image.area()),
                int(background_image.image_array.size),
            )
    print_progress(
        f"Calculating areas for {len(segmentation_images)} segmentation image(s)"
    )
    for index, item in enumerate(segmentation_images, start=1):
        print_progress(
            f"Area {index}/{len(segmentation_images)}: "
            f"{item[0]} [{item[2]}]"
        )
        if item[2] == "Background":
            pixel_area, total_area = background_statistics[item[0]]
        else:
            with OutputImage(*item) as output_image:
                pixel_area = int(output_image.area())
                total_area = int(output_image.image_array.size)
        real_area = pixel_area * 0.22 * 0.22
        if item[2] == "Background":
            percent_area = (pixel_area / total_area) * 100
        else:
            background_area = background_statistics[item[0]][0]
            non_background_area = total_area - background_area
            percent_area = (
                (pixel_area / non_background_area) * 100
                if non_background_area > 0
                else 0.0
            )
        all_areas.append((*item, real_area, percent_area))
    print_progress("Area calculations complete")
    return all_areas

def create_table(images: list[tuple[str, str, str, Path]]) -> list[list[list[str | float]]]:
    print_progress(f"Preparing {len(images)} image record(s) for table creation")
    reset_image_progress(len(images))
    unique_names = sorted({item[0] for item in images})
    output_nested_list = []
    all_areas = get_all_areas(images)
    for cls in SEG_CLASSES:
        print_progress(
            f"Building {cls} table with {len(unique_names)} scan row(s)"
        )
        output_list = []
        for row in range(len(unique_names) + 1):
            if row == 0:
                output_list.append(["Name", f"{cls} Area (um^2)", f"{cls} % Area", f"{cls} Confidence"])
            else:
                name_col = unique_names[row - 1]
                area_tuple = next(item for item in all_areas if item[0] == name_col and item[1] == "Segmentation" and item[2] == cls)[4:]
                confidence_item = next(
                    i
                    for i in images
                    if i[0] == name_col
                    and i[1] == "Grayscale"
                    and i[2] == cls
                )
                print_progress(f"Calculating confidence: {name_col} [{cls}]")
                with OutputImage(*confidence_item) as confidence_image:
                    confidence = confidence_image.confidence()
                output_sub_list = [name_col, *area_tuple, confidence]
                output_list.append(output_sub_list)
        output_nested_list.append(output_list)
    print_progress("All class tables built")
    return output_nested_list

def create_csv_string(nested_list: list[list[tuple[str, str | float, str | float, str | float]]]):
    output_string = "\n".join(", ".join(map(str, inner_list)) for inner_list in nested_list)
    return output_string


def csv_number(value: str | float | int) -> float | int:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned_value = value.strip()
        if cleaned_value.startswith("'"):
            cleaned_value = cleaned_value[1:].strip()
        try:
            return float(cleaned_value)
        except ValueError as error:
            raise ValueError(
                f"Expected a numeric CSV measurement, but received {value!r}"
            ) from error
    raise TypeError(
        f"Expected a numeric CSV measurement, but received {type(value).__name__}"
    )


def create_csv(nested_list: list[list[list[str | float]]]):
    args = parse_args()
    print_progress(f"Writing CSV files under: {args.path_to_prediction.resolve()}")
    for cls, table in zip(SEG_CLASSES, range(len(nested_list))):
        output_path = args.path_to_prediction / f"{cls} area output.csv"
        print_progress(f"Writing {cls} results: {output_path}")
        source_rows = nested_list[table]
        normalized_rows = [[str(value) for value in source_rows[0]]]
        for row in source_rows[1:]:
            normalized_rows.append(
                [str(row[0]), *[csv_number(value) for value in row[1:]]]
            )
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows(normalized_rows)
    print_progress("CSV writing complete")




def main() -> None:
    args = parse_args()
    print_progress("Step 1/4: discovering and validating input images")
    images = discover_images(args.path_to_prediction)
    print_progress("Step 2/4: calculating areas and building result tables")
    table = create_table(images)
    print_progress("Step 3/4: writing class CSV files")
    create_csv(table)
    print_progress("Step 4/4: quantification complete")

if __name__ == "__main__":
    main()
