from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def label_image(path, destination, position, total):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot open {path}")

    height, width = image.shape[:2]
    scale = min(850 / height, 1000 / width, 1.0)
    display_size = (round(width * scale), round(height * scale))
    points = []
    window = "Playing card annotation"

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            px = min(width - 1, max(0, round(x / scale)))
            py = min(height - 1, max(0, round(y / scale)))
            points.append((px, py))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)

    print(f"\nImage {position}/{total}: {path.name}")
    print("Left click around the CARD ONLY, including its rounded corners.")
    print("S saves, U or right click undoes, R restarts, Q quits.")

    while True:
        preview = image.copy()
        if len(points) >= 3:
            overlay = preview.copy()
            polygon = np.asarray(points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], (0, 180, 0))
            preview = cv2.addWeighted(overlay, 0.25, preview, 0.75, 0)

        if points:
            polygon = np.asarray(points, dtype=np.int32)
            cv2.polylines(preview, [polygon], len(points) >= 3, (0, 255, 0), 2)
            for point in points:
                cv2.circle(preview, point, 4, (0, 0, 255), -1)

        preview = cv2.resize(preview, display_size, interpolation=cv2.INTER_AREA)
        cv2.putText(preview, f"{position}/{total}  Points: {len(points)}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window, preview)

        key = cv2.waitKey(30) & 255
        if key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            return False
        if key in (ord("u"), 8) and points:
            points.pop()
        if key == ord("r"):
            points.clear()
        if key == ord("s"):
            if len(points) < 4:
                print("Select at least four points before saving.")
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
            if not cv2.imwrite(str(destination), mask):
                raise RuntimeError(f"Could not save {destination}")
            print(f"Saved {destination.name}")
            cv2.destroyAllWindows()
            return True


def main():
    parser = ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    image_dir = Path("dataset/images")
    mask_dir = Path("dataset/masks")
    images = sorted(image_dir.glob("obj_*.png"))
    if not images:
        raise RuntimeError("No prepared images found in dataset/images")

    mask_dir.mkdir(parents=True, exist_ok=True)
    remaining = [path for path in images if not (mask_dir / path.name).exists()]
    if args.limit > 0:
        remaining = remaining[:args.limit]

    if not remaining:
        print("All requested images are already labelled.")
        return

    for index, path in enumerate(remaining, start=1):
        if not label_image(path, mask_dir / path.name, index, len(remaining)):
            break

    count = len(list(mask_dir.glob("obj_*.png")))
    print(f"\nSaved masks: {count} of {len(images)}")


if __name__ == "__main__":
    main()
