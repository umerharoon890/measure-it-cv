import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.measure import (
    MIN_CONFIDENCE,
    MIN_MASK_AREA,
    find_reference,
    load_model,
    make_overlay,
    measure_in_mm,
    prepare_image,
    reference_is_usable,
    segment_card,
)


WARMUP_RUNS = 2
MEASURED_RUNS = 5
TARGET_MS = 1500.0
OUTPUT_DIR = ROOT / "measurement" / "results"


def process_image(image_path, model, device):
    image, _ = prepare_image(image_path)
    mask, confidence = segment_card(model, image, device)
    reference = find_reference(image)

    result = {
        "status": "refused",
        "reason": "",
        "confidence": confidence,
    }

    area = np.count_nonzero(mask) / mask.size
    border_touch = bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
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
            result.update({
                "status": "measured",
                "reason": None,
                "width_mm": round(dimensions[0], 3),
                "height_mm": round(dimensions[1], 3),
            })

    overlay = make_overlay(image, mask, reference, result)
    return result, overlay


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=Path,
        default=ROOT / "inference" / "new_images" / "live_test_01.jpg",
    )
    args = parser.parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    for _ in range(WARMUP_RUNS):
        process_image(image_path, model, device)
        synchronize(device)

    times_ms = []
    result = None
    overlay = None
    for run_number in range(1, MEASURED_RUNS + 1):
        synchronize(device)
        start = time.perf_counter()
        result, overlay = process_image(image_path, model, device)
        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(round(elapsed_ms, 3))
        print(f"Run {run_number}: {elapsed_ms:.2f} ms | {result['status']}")

    summary = {
        "image": str(image_path.relative_to(ROOT)) if image_path.is_relative_to(ROOT) else str(image_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "timings_ms": times_ms,
        "median_ms": round(statistics.median(times_ms), 3),
        "minimum_ms": min(times_ms),
        "maximum_ms": max(times_ms),
        "spread_ms": round(max(times_ms) - min(times_ms), 3),
        "target_ms": TARGET_MS,
        "target_met": statistics.median(times_ms) <= TARGET_MS,
        "last_result": result,
        "timing_scope": "Raw JPG load, undistortion, resize, segmentation, reference detection, measurement, overlay creation. Excludes model loading and saving output files.",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "benchmark.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    cv2.imwrite(str(OUTPUT_DIR / "benchmark_overlay.jpg"), overlay)

    print("\nInference benchmark")
    print(f"Median: {summary['median_ms']:.2f} ms")
    print(f"Range: {summary['minimum_ms']:.2f} to {summary['maximum_ms']:.2f} ms")
    print(f"Spread: {summary['spread_ms']:.2f} ms")
    print(f"MO-2 target met: {summary['target_met']}")
    print(f"Saved: {OUTPUT_DIR / 'benchmark.json'}")


if __name__ == "__main__":
    main()
