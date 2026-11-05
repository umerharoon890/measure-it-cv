
from pathlib import Path
import csv

import cv2
import numpy as np


RAW_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/images")
CALIBRATION_FILE = Path("calibration/parameters.npz")

OUTPUT_SIZE = (756, 1008)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration = np.load(CALIBRATION_FILE)

    camera_matrix = calibration["camera_matrix"]
    distortion = calibration["dist_coeffs"]

    calibration_size = (
        int(calibration["image_width"]),
        int(calibration["image_height"])
    )

    image_paths = sorted(RAW_DIR.glob("*.jpg"))

    if len(image_paths) < 70:
        raise RuntimeError("At least 70 original images are required.")

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        camera_matrix,
        calibration_size,
        cv2.CV_32FC1
    )

    records = []
    rotated_count = 0

    for index, path in enumerate(image_paths):
        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(f"Could not read {path.name}")

        height, width = image.shape[:2]
        rotation = "none"

        if width > height:
            image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE
            )
            rotation = "90 clockwise"
            rotated_count += 1

        current_size = (image.shape[1], image.shape[0])

        if current_size != calibration_size:
            raise ValueError(
                f"Unexpected resolution for {path.name}: {current_size}"
            )

        undistorted = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        resized = cv2.resize(
            undistorted,
            OUTPUT_SIZE,
            interpolation=cv2.INTER_AREA
        )

        output_name = f"obj_{index:04d}.png"
        output_path = OUTPUT_DIR / output_name

        saved = cv2.imwrite(str(output_path), resized)

        if not saved:
            raise RuntimeError(f"Could not save {output_name}")

        records.append([output_name, path.name, rotation])

    manifest_path = Path("dataset/image_manifest.csv")

    with manifest_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["processed_image", "original_image", "rotation"]
        )
        writer.writerows(records)

    print(f"Original images: {len(image_paths)}")
    print(f"Processed images: {len(records)}")
    print(f"Landscape images rotated: {rotated_count}")
    print(f"Output resolution: {OUTPUT_SIZE[0]} x {OUTPUT_SIZE[1]}")
    print(f"Images saved to: {OUTPUT_DIR}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
