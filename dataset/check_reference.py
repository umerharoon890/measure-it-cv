
from pathlib import Path

import cv2


IMAGE_DIR = Path("dataset/images")

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
detector = cv2.aruco.ArucoDetector(dictionary)

image_paths = sorted(IMAGE_DIR.glob("*.png"))

detected = []
missing = []

for path in image_paths:
    image = cv2.imread(str(path))

    if image is None:
        missing.append(path.name)
        continue

    corners, ids, rejected = detector.detectMarkers(image)

    if ids is not None and 0 in ids.flatten():
        detected.append(path.name)
    else:
        missing.append(path.name)

print(f"Total images: {len(image_paths)}")
print(f"Marker detected: {len(detected)}")
print(f"Marker not detected: {len(missing)}")
print(f"Detection rate: {len(detected) / len(image_paths) * 100:.2f}%")

if missing:
    print("\nImages without a detected marker:")
    for name in missing:
        print(name)
