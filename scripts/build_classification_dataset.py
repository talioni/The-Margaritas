"""
Build the YOLOv8 classification dataset:

1. Crop each TACO bounding box into its own image, save by class
2. Add our own webcam photos (already sorted by folder)
3. Split everything 80/20 into train/val

Final layout (what YOLOv8 classify expects):

    dataset_cls/
        train/
            plastic/        *.jpg
            metal/
            drinkkarton/
        val/
            plastic/
            metal/
            drinkkarton/

Run:
    python scripts/build_classification_dataset.py
"""

import random
import shutil
from pathlib import Path
from PIL import Image

random.seed(42)  # reproducible split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Inputs
TACO_IMG_DIR = PROJECT_ROOT / "dataset" / "taco" / "images"
TACO_LBL_DIR = PROJECT_ROOT / "dataset" / "taco" / "labels"
OWN_DIR      = PROJECT_ROOT / "dataset" / "raw_captures"

# Output
OUT_DIR   = PROJECT_ROOT / "dataset_cls"
TRAIN_DIR = OUT_DIR / "train"
VAL_DIR   = OUT_DIR / "val"

# class id (in YOLO label files) -> class name (folder name)
CLASS_NAMES = {0: "plastic", 1: "metal", 2: "drinkkarton"}

# Clean and recreate output dirs
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
for split in (TRAIN_DIR, VAL_DIR):
    for name in CLASS_NAMES.values():
        (split / name).mkdir(parents=True, exist_ok=True)

# Step 1 — collect all (image_path, class_name) samples
samples = []   # list of (PIL crop or path-to-copy, class_name, source_tag)

print("Step 1: cropping TACO images by bounding boxes...")
taco_count = 0
for lbl_path in TACO_LBL_DIR.glob("*.txt"):
    img_path = TACO_IMG_DIR / (lbl_path.stem + ".jpg")
    if not img_path.exists():
        img_path = TACO_IMG_DIR / (lbl_path.stem + ".JPG")
    if not img_path.exists():
        continue
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        continue
    W, H = img.size

    for i, line in enumerate(lbl_path.read_text().strip().splitlines()):
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, nw, nh = map(float, parts[1:])
        # Convert normalized YOLO bbox back to pixel coords
        x1 = int((cx - nw / 2) * W)
        y1 = int((cy - nh / 2) * H)
        x2 = int((cx + nw / 2) * W)
        y2 = int((cy + nh / 2) * H)
        # Clamp + skip degenerate boxes
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        crop = img.crop((x1, y1, x2, y2))
        samples.append((crop, CLASS_NAMES[cls], f"taco_{lbl_path.stem}_{i}"))
        taco_count += 1
print(f"  TACO crops: {taco_count}")

# Step 2 — add our own webcam photos (already class-organized)
print("Step 2: adding our own webcam photos...")
own_count = 0
for cls_name in CLASS_NAMES.values():
    folder = OWN_DIR / cls_name
    if not folder.exists():
        continue
    for img_path in folder.glob("*.jpg"):
        samples.append((img_path, cls_name, f"own_{img_path.stem}"))
        own_count += 1
print(f"  Own photos: {own_count}")
print(f"  TOTAL samples: {len(samples)}")

# Step 3 — shuffle & split 80/20 PER CLASS so each class is balanced
print("Step 3: splitting 80/20 per class...")
by_class = {n: [] for n in CLASS_NAMES.values()}
for s in samples:
    by_class[s[1]].append(s)

for cls_name, items in by_class.items():
    random.shuffle(items)
    split_idx = int(len(items) * 0.8)
    train_items = items[:split_idx]
    val_items   = items[split_idx:]
    print(f"  {cls_name:12s} train={len(train_items)}  val={len(val_items)}")

    for split_dir, batch in ((TRAIN_DIR, train_items), (VAL_DIR, val_items)):
        target = split_dir / cls_name
        for crop_or_path, _, tag in batch:
            out_path = target / f"{tag}.jpg"
            if isinstance(crop_or_path, Path):
                # copy own photo
                shutil.copy(crop_or_path, out_path)
            else:
                # PIL Image (cropped from TACO)
                crop_or_path.save(out_path, "JPEG", quality=90)

print("\nDone. Dataset ready at:", OUT_DIR)
print("Train with:")
print(f'  yolo classify train data="{OUT_DIR}" model=yolov8n-cls.pt epochs=30 imgsz=224')
