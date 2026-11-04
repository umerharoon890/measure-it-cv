
from pathlib import Path
import csv

import cv2
import numpy as np


BOARD_SIZE = (9, 7)
SQUARE_SIZE_MM = 18.0

IMAGE_DIR = Path("calibration/images")
OUTPUT_DIR = Path("calibration")


def main():
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError("No JPG images found in calibration/images")

    object_points = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[
        0:BOARD_SIZE[0], 0:BOARD_SIZE[1]
    ].T.reshape(-1, 2)
    object_points *= SQUARE_SIZE_MM

    world_points = []
    image_points = []
    valid_images = []
    image_size = None

    print(f"Found {len(image_paths)} calibration images\n")

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"Could not read: {path.name}")
            continue

        height, width = image.shape[:2]

        if image_size is None:
            image_size = (width, height)

        if (width, height) != image_size:
            print(f"Different image resolution: {path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(
            gray, BOARD_SIZE
        )

        if not found:
            found, corners = cv2.findChessboardCorners(
                gray, BOARD_SIZE
            )

            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS
                    + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                )

                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )

        if found:
            world_points.append(object_points.copy())
            image_points.append(corners)
            valid_images.append(path)
            print(f"Detected: {path.name}")
        else:
            print(f"Not detected: {path.name}")

    print(f"\nSuccessful images: {len(valid_images)}/{len(image_paths)}")

    if len(valid_images) < 20:
        raise RuntimeError(
            "At least 20 successful calibration images are required."
        )

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        world_points,
        image_points,
        image_size,
        None,
        None
    )

    errors = []
    total_squared_error = 0.0
    total_points = 0

    for path, world, observed, rvec, tvec in zip(
        valid_images, world_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            world, rvec, tvec, camera_matrix, distortion
        )

        differences = (
            observed.reshape(-1, 2) - projected.reshape(-1, 2)
        )

        squared_distances = np.sum(differences ** 2, axis=1)
        image_error = np.sqrt(np.mean(squared_distances))

        total_squared_error += np.sum(squared_distances)
        total_points += len(squared_distances)

        errors.append([path.name, float(image_error)])

    reprojection_error = np.sqrt(
        total_squared_error / total_points
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUTPUT_DIR / "parameters.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=distortion,
        image_width=image_size[0],
        image_height=image_size[1],
        square_mm=SQUARE_SIZE_MM,
        valid_images=len(valid_images),
        opencv_rms=rms,
        reprojection_rmse_px=reprojection_error
    )

    with open(
        OUTPUT_DIR / "calibration_views.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image", "reprojection_error_px"])
        writer.writerows(errors)

    example = cv2.imread(str(valid_images[0]))

    corrected = cv2.undistort(
        example, camera_matrix, distortion, None, camera_matrix
    )

    cv2.imwrite(
        str(OUTPUT_DIR / "undistorted_example.jpg"),
        corrected
    )

    print("\nCalibration results")
    print(f"Image resolution: {image_size}")
    print(f"Checkerboard square size: {SQUARE_SIZE_MM} mm")
    print(f"OpenCV RMS error: {rms:.4f} pixels")
    print(f"Reprojection error: {reprojection_error:.4f} pixels")
    print("\nCamera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(distortion.ravel())
    print("\nCalibration files saved successfully.")


if __name__ == "__main__":
    main()
