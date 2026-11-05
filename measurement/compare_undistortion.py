import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "measurement" / "images"
RESULT_DIR = ROOT / "measurement" / "results"
OVERLAY_DIR = RESULT_DIR / "distortion_overlays"
MEASURE_SCRIPT = ROOT / "inference" / "measure.py"
GROUND_TRUTH = {"width_mm": 57.0, "height_mm": 87.0}


def run_measurement(image_path, mode):
    output_path = OVERLAY_DIR / f"{image_path.stem}_{mode}.jpg"
    command = [
        sys.executable,
        str(MEASURE_SCRIPT),
        "--image", str(image_path),
        "--output", str(output_path),
    ]
    if mode == "raw":
        command.append("--skip-undistortion")

    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True
    )
    command_text = " ".join(f'"{part}"' if " " in part else part for part in command)
    log = f"Command: {command_text}\n{completed.stdout}{completed.stderr}"
    if completed.returncode != 0:
        raise RuntimeError(
            f"Inference failed for {image_path.name} ({mode}).\n{log}"
        )

    result_file = output_path.with_suffix(".json")
    if not result_file.exists():
        raise FileNotFoundError(result_file)
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if result.get("status") not in {"measured", "refused"}:
        raise ValueError(f"Unexpected inference status in {result_file}")
    expected_undistorted = mode == "undistorted"
    if result.get("undistorted") is not expected_undistorted:
        raise ValueError(f"Unexpected undistortion setting in {result_file}")
    return result, log


def summarize(values):
    if not values:
        return {"mean": None, "median": None, "worst": None}
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "worst": round(max(values), 4),
    }


def main():
    if not MEASURE_SCRIPT.exists():
        raise FileNotFoundError(MEASURE_SCRIPT)
    images = sorted(
        path for path in IMAGE_DIR.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg"} and path.is_file()
    )
    if len(images) < 10:
        raise RuntimeError(f"Expected at least 10 JPG photographs, found {len(images)}")

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    logs = []

    for index, image_path in enumerate(images, start=1):
        row = {"image": image_path.name}
        for mode in ("undistorted", "raw"):
            try:
                result, log = run_measurement(image_path, mode)
            except Exception as error:
                (RESULT_DIR / "distortion_command_log.txt").write_text(
                    "\n\n".join(logs), encoding="utf-8"
                )
                raise RuntimeError(f"Error on {image_path.name} ({mode})") from error

            logs.append(log)
            row[f"{mode}_status"] = result["status"]
            row[f"{mode}_reason"] = result.get("reason") or ""
            row[f"{mode}_confidence"] = result.get("confidence")
            row[f"{mode}_width_mm"] = result.get("width_mm")
            row[f"{mode}_height_mm"] = result.get("height_mm")

            if result["status"] == "measured":
                width_error = abs(float(result["width_mm"]) - GROUND_TRUTH["width_mm"])
                height_error = abs(float(result["height_mm"]) - GROUND_TRUTH["height_mm"])
                row[f"{mode}_width_error_mm"] = round(width_error, 4)
                row[f"{mode}_height_error_mm"] = round(height_error, 4)
                row[f"{mode}_mean_error_mm"] = round((width_error + height_error) / 2, 4)
                row[f"{mode}_mean_percentage_error"] = round(
                    50 * (
                        width_error / GROUND_TRUTH["width_mm"]
                        + height_error / GROUND_TRUTH["height_mm"]
                    ), 4
                )
            else:
                for field in (
                    "width_error_mm", "height_error_mm", "mean_error_mm", "mean_percentage_error"
                ):
                    row[f"{mode}_{field}"] = ""

        paired = (
            row["raw_status"] == "measured"
            and row["undistorted_status"] == "measured"
        )
        row["paired"] = paired
        row["error_reduction_mm"] = (
            round(row["raw_mean_error_mm"] - row["undistorted_mean_error_mm"], 4)
            if paired else ""
        )
        rows.append(row)
        display_raw = (
            f"{row['raw_mean_error_mm']:.2f} mm" if row["raw_status"] == "measured"
            else f"refused ({row['raw_reason']})"
        )
        display_corrected = (
            f"{row['undistorted_mean_error_mm']:.2f} mm"
            if row["undistorted_status"] == "measured"
            else f"refused ({row['undistorted_reason']})"
        )
        print(
            f"{index:02d}/{len(images)} {image_path.name} | "
            f"raw: {display_raw} | corrected: {display_corrected}",
            flush=True,
        )

    csv_path = RESULT_DIR / "undistortion_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (RESULT_DIR / "distortion_command_log.txt").write_text(
        "\n\n".join(logs), encoding="utf-8"
    )

    paired_rows = [row for row in rows if row["paired"]]
    accepted = {
        mode: sum(row[f"{mode}_status"] == "measured" for row in rows)
        for mode in ("raw", "undistorted")
    }
    raw_errors = [row["raw_mean_error_mm"] for row in paired_rows]
    corrected_errors = [row["undistorted_mean_error_mm"] for row in paired_rows]
    reduction = [row["error_reduction_mm"] for row in paired_rows]

    summary = {
        "object": "one playing card, photographed in separate trials",
        "ground_truth_mm": GROUND_TRUTH,
        "total_images": len(rows),
        "raw_accepted": accepted["raw"],
        "raw_refused": len(rows) - accepted["raw"],
        "undistorted_accepted": accepted["undistorted"],
        "undistorted_refused": len(rows) - accepted["undistorted"],
        "paired_accepted_images": len(paired_rows),
        "raw_paired_mean_error_mm": summarize(raw_errors),
        "undistorted_paired_mean_error_mm": summarize(corrected_errors),
        "paired_error_reduction_mm": summarize(reduction),
        "comparison_note": (
            "Positive error reduction means undistortion reduced error. "
            "Errors are compared only on images accepted in both modes. "
            "Refusals are counted separately. The model was trained on undistorted images, "
            "so this comparison reflects the complete pipeline and does not isolate lens geometry alone."
        ),
    }
    summary_path = RESULT_DIR / "undistortion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nRaw versus undistorted comparison")
    print(f"Images: {len(rows)}")
    print(f"Raw accepted: {accepted['raw']}, refused: {len(rows) - accepted['raw']}")
    print(
        f"Undistorted accepted: {accepted['undistorted']}, "
        f"refused: {len(rows) - accepted['undistorted']}"
    )
    print(f"Paired accepted images: {len(paired_rows)}")
    if paired_rows:
        print(f"Raw mean error on paired images: {statistics.mean(raw_errors):.3f} mm")
        print(f"Undistorted mean error on paired images: {statistics.mean(corrected_errors):.3f} mm")
        print(f"Mean error reduction: {statistics.mean(reduction):+.3f} mm")
        print(f"Raw median error: {statistics.median(raw_errors):.3f} mm")
        print(f"Undistorted median error: {statistics.median(corrected_errors):.3f} mm")
        print(f"Raw worst error: {max(raw_errors):.3f} mm")
        print(f"Undistorted worst error: {max(corrected_errors):.3f} mm")
    else:
        print("No common accepted measurements. An error comparison is unavailable.")
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
