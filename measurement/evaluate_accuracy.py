import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "measurement" / "images"
RESULT_DIR = ROOT / "measurement" / "results"
OVERLAY_DIR = RESULT_DIR / "overlays"
INFERENCE_SCRIPT = ROOT / "inference" / "measure.py"

CARD_WIDTH_MM = 57.0
CARD_HEIGHT_MM = 87.0
TARGET_MAE_MM = 2.0
TARGET_ACCEPTED = 9


def average(values):
    return statistics.mean(values) if values else None


def main():
    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    ) if IMAGE_DIR.is_dir() else []

    if len(images) < 10:
        raise RuntimeError(
            f"Found {len(images)} JPG images in {IMAGE_DIR}. At least 10 are required."
        )

    if not INFERENCE_SCRIPT.is_file():
        raise FileNotFoundError(INFERENCE_SCRIPT)

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    logs = []

    for index, image in enumerate(images, start=1):
        overlay = OVERLAY_DIR / f"{image.stem}_overlay.jpg"
        command = [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--image", str(image),
            "--output", str(overlay),
        ]
        run = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True
        )
        logs.append(f"Image {index}: {image.name}\n{run.stdout}{run.stderr}")
        if run.returncode != 0:
            (RESULT_DIR / "command_log.txt").write_text(
                "\n".join(logs), encoding="utf-8"
            )
            raise RuntimeError(
                f"Measurement failed on {image.name}. "
                "See measurement/results/command_log.txt."
            )

        json_path = overlay.with_suffix(".json")
        if not json_path.is_file():
            raise FileNotFoundError(json_path)
        result = json.loads(json_path.read_text(encoding="utf-8"))
        status = result.get("status")
        if status not in {"measured", "refused"}:
            raise RuntimeError(f"Unexpected result for {image.name}: {status}")

        row = {
            "image": image.name,
            "status": status,
            "reason": result.get("reason") or "",
            "actual_width_mm": CARD_WIDTH_MM,
            "actual_height_mm": CARD_HEIGHT_MM,
            "measured_width_mm": "",
            "measured_height_mm": "",
            "width_abs_error_mm": "",
            "height_abs_error_mm": "",
            "mean_dimension_error_mm": "",
            "width_percentage_error": "",
            "height_percentage_error": "",
            "mean_percentage_error": "",
            "mask_confidence": result.get("confidence", ""),
        }

        if status == "measured":
            width = float(result["width_mm"])
            height = float(result["height_mm"])
            width_error = abs(width - CARD_WIDTH_MM)
            height_error = abs(height - CARD_HEIGHT_MM)
            width_pct = 100 * width_error / CARD_WIDTH_MM
            height_pct = 100 * height_error / CARD_HEIGHT_MM
            row.update({
                "measured_width_mm": width,
                "measured_height_mm": height,
                "width_abs_error_mm": round(width_error, 4),
                "height_abs_error_mm": round(height_error, 4),
                "mean_dimension_error_mm": round((width_error + height_error) / 2, 4),
                "width_percentage_error": round(width_pct, 4),
                "height_percentage_error": round(height_pct, 4),
                "mean_percentage_error": round((width_pct + height_pct) / 2, 4),
            })
            print(
                f"{index:02d}/{len(images)} {image.name}: "
                f"{width:.2f} x {height:.2f} mm, "
                f"mean error {(width_error + height_error) / 2:.2f} mm",
                flush=True,
            )
        else:
            print(
                f"{index:02d}/{len(images)} {image.name}: "
                f"refused ({row['reason']})",
                flush=True,
            )
        rows.append(row)

    csv_path = RESULT_DIR / "per_instance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (RESULT_DIR / "command_log.txt").write_text(
        "\n".join(logs), encoding="utf-8"
    )

    accepted = [row for row in rows if row["status"] == "measured"]
    errors = [float(row["mean_dimension_error_mm"]) for row in accepted]
    pct_errors = [float(row["mean_percentage_error"]) for row in accepted]
    width_errors = [float(row["width_abs_error_mm"]) for row in accepted]
    height_errors = [float(row["height_abs_error_mm"]) for row in accepted]

    summary = {
        "object": "same playing card photographed in separate trials",
        "ground_truth_width_mm": CARD_WIDTH_MM,
        "ground_truth_height_mm": CARD_HEIGHT_MM,
        "total_trials": len(rows),
        "accepted": len(accepted),
        "refused": len(rows) - len(accepted),
        "acceptance_rate_percent": round(100 * len(accepted) / len(rows), 4),
        "mean_absolute_error_mm": round(average(errors), 4) if errors else None,
        "median_absolute_error_mm": round(statistics.median(errors), 4) if errors else None,
        "worst_absolute_error_mm": round(max(errors), 4) if errors else None,
        "mean_percentage_error": round(average(pct_errors), 4) if pct_errors else None,
        "mean_width_absolute_error_mm": round(average(width_errors), 4) if width_errors else None,
        "mean_height_absolute_error_mm": round(average(height_errors), 4) if height_errors else None,
        "definition": (
            "For each accepted trial, mean dimension error is the average of "
            "the absolute width and height errors in millimetres. Mean percentage "
            "error averages the absolute percentage errors of width and height. "
            "Summary errors are calculated only on accepted trials. Refusals "
            "are counted separately, never assigned zero error."
        ),
        "target_mean_absolute_error_mm_at_most": TARGET_MAE_MM,
        "target_accepted_at_least": TARGET_ACCEPTED,
        "target_met": (
            len(accepted) >= TARGET_ACCEPTED
            and bool(errors)
            and average(errors) <= TARGET_MAE_MM
        ),
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\nAccuracy results")
    print(f"Total trials: {summary['total_trials']}")
    print(f"Accepted: {summary['accepted']}")
    print(f"Refused: {summary['refused']}")
    if errors:
        print(f"Mean absolute error: {summary['mean_absolute_error_mm']:.3f} mm")
        print(f"Median absolute error: {summary['median_absolute_error_mm']:.3f} mm")
        print(f"Worst absolute error: {summary['worst_absolute_error_mm']:.3f} mm")
        print(f"Mean percentage error: {summary['mean_percentage_error']:.3f}%")
    else:
        print("No accepted measurements. Accuracy metrics are unavailable.")
    print(f"MO-1 target met: {summary['target_met']}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {RESULT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
