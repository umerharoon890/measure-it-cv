import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from unet import UNet


SEED = 2026
EPOCHS = 35
BATCH_SIZE = 2
LEARNING_RATE = 0.0003
PATIENCE = 8
SIZE = (288, 384)
DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("models")


class CardDataset(Dataset):
    def __init__(self, split, augment=False):
        split_file = DATASET_DIR / "splits" / f"{split}.txt"
        self.names = [name.strip() for name in split_file.read_text(encoding="utf-8-sig").splitlines() if name.strip()]
        self.augment = augment
        if not self.names:
            raise ValueError(f"The {split} split is empty")

        for name in self.names:
            for folder in ("images", "masks"):
                if not (DATASET_DIR / folder / name).is_file():
                    raise FileNotFoundError(DATASET_DIR / folder / name)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = cv2.imread(str(DATASET_DIR / "images" / name))
        mask = cv2.imread(str(DATASET_DIR / "masks" / name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Cannot read {name}")

        image = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, SIZE, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            image = np.clip(image.astype(np.float32) * random.uniform(0.85, 1.15), 0, 255).astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        mask = torch.from_numpy(np.ascontiguousarray((mask > 127)[None])).float()
        return image, mask


def segmentation_loss(logits, target, bce):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = (probabilities + target).sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return bce(logits, target) + dice_loss


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_set = CardDataset("train", augment=True)
    val_set = CardDataset("val")
    if set(train_set.names) & set(val_set.names):
        raise RuntimeError("Training and validation splits overlap")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    bce = nn.BCEWithLogitsLoss()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "architecture": "UNet", "seed": SEED, "max_epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "patience": PATIENCE, "image_width": SIZE[0], "image_height": SIZE[1],
        "training_images": len(train_set), "validation_images": len(val_set),
        "selection_metric": "validation_mask_IoU_at_0.5",
        "device": str(device), "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
    }
    (OUTPUT_DIR / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    best_iou = -1.0
    stale_epochs = 0
    history = []
    print(f"Device: {device} | Train: {len(train_set)} | Validation: {len(val_set)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = segmentation_loss(logits, masks, bce)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        val_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = segmentation_loss(logits, masks, bce)
                val_loss += loss.item() * len(images)
                predicted = torch.sigmoid(logits) >= 0.5
                actual = masks >= 0.5
                intersection = (predicted & actual).sum(dim=(1, 2, 3)).float()
                union = (predicted | actual).sum(dim=(1, 2, 3)).float()
                val_ious.extend(((intersection + 1e-6) / (union + 1e-6)).cpu().tolist())
        val_loss /= len(val_set)
        val_iou = float(np.mean(val_ious))
        history.append((epoch, train_loss, val_loss, val_iou))
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val IoU {val_iou:.4f}", flush=True)

        if val_iou > best_iou + 1e-4:
            best_iou = val_iou
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_iou": val_iou, "config": config}, OUTPUT_DIR / "best_unet.pth")
        else:
            stale_epochs += 1

        with (OUTPUT_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("epoch", "train_loss", "val_loss", "val_iou"))
            writer.writerows(history)

        if stale_epochs >= PATIENCE:
            print(f"Early stopping after {epoch} epochs")
            break

    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Saved weights: {OUTPUT_DIR / 'best_unet.pth'}")


if __name__ == "__main__":
    main()
