
from pathlib import Path

import cv2
import numpy as np


image_dir = Path("dataset/images")
mask_dir = Path("dataset/masks")
train_file = Path("dataset/splits/train.txt")

names = train_file.read_text(encoding="utf-8").splitlines()
indices = np.linspace(0, len(names) - 1, 12, dtype=int)

tiles = []

for index in indices:
    name = names[index]

    image = cv2.imread(str(image_dir / name))
    mask = cv2.imread(
        str(mask_dir / name),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        raise RuntimeError(f"Missing image or mask: {name}")

    image = cv2.resize(image, (200, 267))
    mask = cv2.resize(
        mask, (200, 267),
        interpolation=cv2.INTER_NEAREST
    )

    colored = image.copy()
    colored[mask > 0] = (0, 180, 255)

    overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    tile = np.full((292, 200, 3), 255, dtype=np.uint8)
    tile[25:] = overlay

    cv2.putText(
        tile, name, (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 0, 0), 1
    )

    tiles.append(tile)

rows = [
    np.hstack(tiles[i:i + 4])
    for i in range(0, len(tiles), 4)
]

review = np.vstack(rows)
output = Path("dataset/mask_review.jpg")

cv2.imwrite(str(output), review)

print(f"Reviewed {len(tiles)} training masks")
print(f"Saved review image to {output}")
