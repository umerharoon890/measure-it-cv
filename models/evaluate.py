import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from unet import UNet
from torch.torch_version import TorchVersion


DATASET_DIR = Path("dataset")
WEIGHTS_FILE = Path("models/best_unet.pth")
OUTPUT_DIR = Path("models/evaluation")
IMAGE_SIZE = (288, 384)
MASK_THRESHOLD = 0.5


def largest_component(binary_mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(binary_mask, dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return labels == largest_label


def mask_iou(predicted, actual):
    intersection = np.logical_and(predicted, actual).sum()
    union = np.logical_or(predicted, actual).sum()
    return float(intersection / union) if union else 1.0


def average_precision(predictions, ground_truth_count, iou_threshold):
    ranked = sorted(
        (item for item in predictions if item["has_prediction"]),
        key=lambda item: item["confidence"],
        reverse=True,
    )
    if not ranked or ground_truth_count == 0:
        return 0.0

    true_positives = np.array(
        [float(item["iou"] >= iou_threshold) for item in ranked]
    )
    false_positives = 1.0 - true_positives
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)

    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall = cumulative_tp / ground_truth_count

    # COCO style 101 point interpolation for one class and one object per image.
    return float(np.mean([
        np.max(precision[recall >= level], initial=0.0)
        for level in np.linspace(0.0, 1.0, 101)
    ]))


def main():
    names = [
        name.strip()
        for name in (DATASET_DIR / "splits/test.txt").read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if name.strip()
    ]
    if not names:
        raise RuntimeError("The test split is empty")
    if len(names) != len(set(names)):
        raise RuntimeError("The test split contains repeated filenames")

    for other_split in ("train", "val"):
        other_names = set(
            (DATASET_DIR / f"splits/{other_split}.txt").read_text(
                encoding="utf-8-sig"
            ).splitlines()
        )
        if set(names) & other_names:
            raise RuntimeError(f"The test split overlaps with {other_split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
        WEIGHTS_FILE,
        map_location=device,
        weights_only=True
    )
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = []
    totals = {"tp": 0, "fp": 0, "fn": 0}

    with torch.inference_mode():
        for name in names:
            image = cv2.imread(str(DATASET_DIR / "images" / name))
            mask = cv2.imread(
                str(DATASET_DIR / "masks" / name), cv2.IMREAD_GRAYSCALE
            )
            if image is None or mask is None:
                raise FileNotFoundError(f"Missing image or mask: {name}")
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Image and mask sizes differ: {name}")

            image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
            actual = mask > 127

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(
                np.ascontiguousarray(rgb.transpose(2, 0, 1))
            ).float().unsqueeze(0).to(device) / 255.0
            probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

            predicted = largest_component(probabilities >= MASK_THRESHOLD)
            has_prediction = bool(predicted.any())
            confidence = float(probabilities[predicted].mean()) if has_prediction else 0.0
            iou = mask_iou(predicted, actual)

            tp = int(np.logical_and(predicted, actual).sum())
            fp = int(np.logical_and(predicted, ~actual).sum())
            fn = int(np.logical_and(~predicted, actual).sum())
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn

            predictions.append({
                "image": name,
                "iou": iou,
                "confidence": confidence,
                "has_prediction": has_prediction,
                "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
                "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
            })
            print(f"{name}: IoU={iou:.4f}, confidence={confidence:.4f}")

    thresholds = np.arange(50, 100, 5) / 100.0
    ap_values = [average_precision(predictions, len(names), threshold) for threshold in thresholds]
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)

    summary = {
        "test_images": len(names),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "mask_threshold": MASK_THRESHOLD,
        "mean_mask_iou": float(np.mean([item["iou"] for item in predictions])),
        "map_50": ap_values[0],
        "map_50_95": float(np.mean(ap_values)),
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "ap_by_iou_threshold": {
            f"{threshold:.2f}": ap for threshold, ap in zip(thresholds, ap_values)
        },
        "ap_definition": "101 point interpolated instance mask AP, one proposed object per image",
        "precision_recall_definition": "Foreground pixel precision and recall after largest component filtering",
    }

    with (OUTPUT_DIR / "test_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\nFinal test results")
    print(f"Test images: {summary['test_images']}")
    print(f"Selected checkpoint epoch: {summary['checkpoint_epoch']}")
    print(f"Mean mask IoU: {summary['mean_mask_iou']:.4f}")
    print(f"mAP@0.5: {summary['map_50']:.4f}")
    print(f"mAP@0.5:0.95: {summary['map_50_95']:.4f}")
    print(f"Pixel precision: {summary['pixel_precision']:.4f}")
    print(f"Pixel recall: {summary['pixel_recall']:.4f}")
    print(f"Saved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
