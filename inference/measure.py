import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.torch_version import TorchVersion


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
from unet import UNet


CALIBRATION_FILE = ROOT / "calibration" / "parameters.npz"
WEIGHTS_FILE = ROOT / "models" / "best_unet.pth"
WORKING_SIZE = (756, 1008)
MODEL_SIZE = (288, 384)
MARKER_SIZE_MM = 40.0
MASK_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.85
MIN_MASK_AREA = 0.005


def prepare_image(path, apply_undistortion=True):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot open image: {path}")

    with np.load(CALIBRATION_FILE) as calibration:
        camera_matrix = calibration["camera_matrix"]
        distortion = calibration["dist_coeffs"]
        expected_size = (
            int(calibration["image_width"]),
            int(calibration["image_height"]),
        )

    rotation = "none"
    if image.shape[1] > image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotation = "90 clockwise"

    size = (image.shape[1], image.shape[0])
    if size != expected_size:
        raise ValueError(
            f"Image size {size} does not match calibration size {expected_size}. "
            "Use a full resolution JPG from the calibrated camera."
        )

    if apply_undistortion:
        image = cv2.undistort(
            image, camera_matrix, distortion, None, camera_matrix
        )

    image = cv2.resize(image, WORKING_SIZE, interpolation=cv2.INTER_AREA)
    return image, rotation


def load_model(device):
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            WEIGHTS_FILE, map_location=device, weights_only=True
        )
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def segment_card(model, image, device):
    resized = cv2.resize(image, MODEL_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(
        np.ascontiguousarray(rgb.transpose(2, 0, 1))
    ).float().unsqueeze(0).to(device) / 255.0

    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    probabilities = cv2.resize(
        probabilities, WORKING_SIZE, interpolation=cv2.INTER_LINEAR
    )
    binary = np.uint8(probabilities >= MASK_THRESHOLD)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    if count <= 1:
        return np.zeros(binary.shape, np.uint8), 0.0

    component = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    mask = np.uint8(labels == component) * 255
    confidence = float(probabilities[mask > 0].mean())
    return mask, confidence


def find_reference(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None:
        return None

    for points, marker_id in zip(corners, ids.flatten()):
        if int(marker_id) == 0:
            return points.reshape(4, 2).astype(np.float32)

    return None


def reference_is_usable(corners, image_shape):
    if corners is None:
        return False

    height, width = image_shape[:2]
    if np.any(corners[:, 0] < 3) or np.any(corners[:, 0] >= width - 3):
        return False
    if np.any(corners[:, 1] < 3) or np.any(corners[:, 1] >= height - 3):
        return False

    area = abs(cv2.contourArea(corners))
    sides = np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)
    return area >= 150 and float(np.min(sides)) >= 10


def measure_in_mm(mask, reference_corners):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    destination = np.array(
        [
            [0.0, 0.0],
            [MARKER_SIZE_MM, 0.0],
            [MARKER_SIZE_MM, MARKER_SIZE_MM],
            [0.0, MARKER_SIZE_MM],
        ],
        dtype=np.float32,
    )
    transformation = cv2.getPerspectiveTransform(reference_corners, destination)
    world_contour = cv2.perspectiveTransform(
        contour.astype(np.float32), transformation
    )
    box = cv2.minAreaRect(world_contour)
    side_a, side_b = box[1]
    if min(side_a, side_b) <= 0 or not np.isfinite([side_a, side_b]).all():
        return None
    return sorted((float(side_a), float(side_b)))


def make_overlay(image, mask, reference, result):
    overlay = image.copy()
    if np.any(mask):
        color = np.zeros_like(image)
        color[mask > 0] = (0, 160, 0)
        overlay = cv2.addWeighted(image, 0.70, color, 0.30, 0)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    if reference is not None:
        cv2.polylines(
            overlay, [np.round(reference).astype(np.int32)],
            isClosed=True, color=(255, 0, 255), thickness=2
        )

    if result["status"] == "measured":
        lines = [
            f"Width: {result['width_mm']:.2f} mm",
            f"Height: {result['height_mm']:.2f} mm",
            f"Mask confidence: {result['confidence']:.3f}",
        ]
    else:
        lines = [
            "Measurement refused",
            result["reason"],
            f"Mask confidence: {result['confidence']:.3f}",
        ]

    cv2.rectangle(overlay, (8, 8), (WORKING_SIZE[0] - 8, 120), (15, 15, 15), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            overlay, line, (18, 37 + 35 * index),
            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2,
            cv2.LINE_AA
        )
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Measure a playing card in millimetres")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-undistortion", action="store_true")
    args = parser.parse_args()

    image_path = args.image.resolve()
    output_path = args.output or (
        ROOT / "inference" / "outputs" / f"{image_path.stem}_overlay.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image, rotation = prepare_image(
        image_path, apply_undistortion=not args.skip_undistortion
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    mask, confidence = segment_card(model, image, device)
    reference = find_reference(image)

    result = {
        "image": str(image_path),
        "status": "refused",
        "reason": "",
        "width_mm": None,
        "height_mm": None,
        "confidence": round(confidence, 4),
        "reference_marker_id": 0,
        "reference_side_mm": MARKER_SIZE_MM,
        "undistorted": not args.skip_undistortion,
        "rotation": rotation,
    }

    area = np.count_nonzero(mask) / mask.size
    border_touch = (
        np.any(mask[0, :]) or np.any(mask[-1, :])
        or np.any(mask[:, 0]) or np.any(mask[:, -1])
    )

    if area < MIN_MASK_AREA:
        result["reason"] = "Object not found"
    elif confidence < MIN_CONFIDENCE:
        result["reason"] = "Low mask confidence"
    elif border_touch:
        result["reason"] = "Object touches image boundary"
    elif reference is None:
        result["reason"] = "Reference marker not found"
    elif not reference_is_usable(reference, image.shape):
        result["reason"] = "Reference marker too small or near image edge"
    else:
        dimensions = measure_in_mm(mask, reference)
        if dimensions is None:
            result["reason"] = "Could not estimate object dimensions"
        else:
            result["status"] = "measured"
            result["width_mm"] = round(dimensions[0], 3)
            result["height_mm"] = round(dimensions[1], 3)
            result["reason"] = None

    overlay = make_overlay(image, mask, reference, result)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"Could not save overlay to {output_path}")

    json_path = output_path.with_name(output_path.stem + ".json")
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Status: {result['status']}")
    if result["status"] == "measured":
        print(f"Width: {result['width_mm']:.2f} mm")
        print(f"Height: {result['height_mm']:.2f} mm")
    else:
        print(f"Reason: {result['reason']}")
    print(f"Mask confidence: {result['confidence']:.4f}")
    print(f"Overlay: {output_path}")
    print(f"Result: {json_path}")


if __name__ == "__main__":
    main()
