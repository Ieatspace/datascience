import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LABELS_CSV = Path("out/labels.csv")
CHARS_DIR = Path("out/chars")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 64
BATCH = 64
EPOCHS = 8
LR = 1e-3


class CharDataset(Dataset):
    def __init__(self, items, class_to_idx):
        self.items = items
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, label = self.items[i]
        img_path = CHARS_DIR / fname
        with Image.open(img_path) as img:
            img = img.convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0

        if arr.shape != (IMG_SIZE, IMG_SIZE):
            raise ValueError(f"Expected {IMG_SIZE}x{IMG_SIZE} image, got {arr.shape} for {img_path}")

        x = torch.from_numpy(arr).unsqueeze(0)
        y = self.class_to_idx[label]
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_labels():
    if not LABELS_CSV.exists():
        raise FileNotFoundError("out/labels.csv not found. Run label_chars.py first.")

    items = []
    with open(LABELS_CSV, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                items.append((row[0], row[1]))

    if not items:
        raise ValueError("out/labels.csv is empty. Label some characters first.")

    random.shuffle(items)
    return items


def preflight_checks():
    if not CHARS_DIR.exists():
        print("Missing character crops: out/chars")
        print("Run this first:")
        print(f"  {Path(sys.executable).name} extract_chars.py")
        return False

    char_files = list(CHARS_DIR.glob("char_*.png"))
    if not char_files:
        print("No character crops found in out/chars.")
        print("Run this first:")
        print(f"  {Path(sys.executable).name} extract_chars.py")
        return False

    if not LABELS_CSV.exists():
        print("Missing labels file: out/labels.csv")
        print("Training needs labeled characters before it can start.")
        print("Run this next:")
        print(f"  {Path(sys.executable).name} label_chars.py")
        print(f"Found {len(char_files)} unlabeled crops in {CHARS_DIR}")
        return False

    return True


def evaluate(model, val_dl):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / max(1, total)


def main():
    if not preflight_checks():
        return

    items = load_labels()
    labels = sorted({lab for _, lab in items})
    class_to_idx = {c: i for i, c in enumerate(labels)}
    n_classes = len(labels)

    print(f"Samples: {len(items)} | Classes: {n_classes} | Device: {DEVICE}")
    if len(items) < 500:
        print("Warning: you should label more (aim 1000+).")
    if n_classes < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    split = int(0.85 * len(items))
    split = max(1, min(split, len(items) - 1))
    train_items = items[:split]
    val_items = items[split:]

    train_ds = CharDataset(train_items, class_to_idx)
    val_ds = CharDataset(val_items, class_to_idx)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    model = SmallCNN(n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_acc = evaluate(model, val_dl)

        print(
            f"Epoch {ep}/{EPOCHS} | loss {train_loss:.4f} | "
            f"train_acc {train_acc:.3f} | val_acc {val_acc:.3f}"
        )

    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "class_to_idx": class_to_idx},
        out_dir / "char_classifier.pt",
    )
    print("Saved model to out/char_classifier.pt")


if __name__ == "__main__":
    main()
