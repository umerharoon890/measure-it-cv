
from pathlib import Path

import cv2
import numpy as np


image_dir = Path("dataset/images")
mask_dir = Path("dataset/masks")

images = sorted(image_dir.glob("*.png"))
problems = []
areas = []

for image_path in images:
    mask_path = mask_dir / image_path.name

    if not mask_path.exists():
        problems.append(f"{image_path.name}: missing mask")
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        problems.append(f"{image_path.name}: unreadable file")
        continue

    if image.shape[:2] != mask.shape:
        problems.append(f"{image_path.name}: dimensions do not match")
        continue

    if not np.all(np.isin(np.unique(mask), [0, 255])):
        problems.append(f"{image_path.name}: mask is not binary")
        continue

    area = np.count_nonzero(mask) / mask.size
    areas.append(area)

    if area < 0.01 or area > 0.50:
        problems.append(f"{image_path.name}: unusual mask area {area:.2%}")

print(f"Images found: {len(images)}")
print(f"Masks found: {len(list(mask_dir.glob('*.png')))}")
print(f"Masks checked: {len(areas)}")

if areas:
    print(f"Smallest mask area: {min(areas):.2%}")
    print(f"Largest mask area: {max(areas):.2%}")

print(f"Potential problems: {len(problems)}")

for problem in problems:
    print(problem)

if not problems and len(areas) == len(images):
    print("All masks passed the validation checks.")
