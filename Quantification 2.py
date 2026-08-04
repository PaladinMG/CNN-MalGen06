import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import argparse
import csv


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
                if self.image_type == "Grayscale":
                    self.image = image.convert("L")
                    self.image_array = np.asarray(
                        self.image,
                        dtype=np.uint8,
                    )
                elif self.image_type == "Segmentation":
                    self.image = image.convert("RGB")
                    rgb_array = np.asarray(
                        self.image,
                        dtype=np.uint8,
                    )
                    self.image_array = np.any(rgb_array != 0, axis=2).astype(np.uint8)
                else:
                    raise ValueError(f"Unknown image type: {self.image_type}")
        except UnidentifiedImageError as error:
            raise ValueError(
                f"Not a valid image file: {self.image_path}"
            ) from error

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
    images: list[tuple[str, str, str, Path]] = []
    for cls in SEG_CLASSES:
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
    return images

def get_background_area(image_name: str, images: list[tuple[str, str, str, Path]]) -> int:
    background_image = OutputImage(*next(item for item in images if item[0] == image_name and item[1] == "Segmentation" and item[2] == "Background"))
    background_area = background_image.area()
    return background_area

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
    for item in [i for i in images if i[1] == "Segmentation"]:
        output_image = OutputImage(*item)
        real_area = output_image.real_area()
        percent_area = output_image.percent_area(images=images)
        all_areas.append((*item, real_area, percent_area))
    return all_areas

def create_table(images: list[tuple[str, str, str, Path]]) -> list[list[list[str | float]]]:
    output_images: list[OutputImage] = []
    for item in images:
        output_image = OutputImage(*item)
        output_images.append(output_image)
    unique_names = sorted({item.image_name for item in output_images})
    output_nested_list = []
    all_areas = get_all_areas(images)
    for cls in SEG_CLASSES:
        output_list = []
        for row in range(len(unique_names) + 1):
            if row == 0:
                output_list.append(["Name", f"{cls} Area (um^2)", f"{cls} % Area", f"{cls} Confidence"])
            else:
                name_col = unique_names[row - 1]
                area_tuple = next(item for item in all_areas if item[0] == name_col and item[1] == "Segmentation" and item[2] == cls)[4:]
                confidence = OutputImage(*next(i for i in images if i[0] == name_col and i[1] == "Grayscale" and i[2] == cls)).confidence()
                output_sub_list = [name_col, *area_tuple, confidence]
                output_list.append(output_sub_list)
        output_nested_list.append(output_list)
    return output_nested_list

def create_csv_string(nested_list: list[list[tuple[str, str | float, str | float, str | float]]]):
    output_string = "\n".join(", ".join(map(str, inner_list)) for inner_list in nested_list)
    return output_string

def create_csv(nested_list: list[list[list[str | float]]]):
    args = parse_args()
    for cls, table in zip(SEG_CLASSES, range(len(nested_list))):
        with Path(f"{args.path_to_prediction}/{cls} area output.csv").open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(nested_list[table])




def main() -> None:
    args = parse_args()
    images = discover_images(args.path_to_prediction)
    table = create_table(images)
    create_csv(table)

if __name__ == "__main__":
    main()
