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

# ---- What this file does ----
# This is the OLD (v1) dataset builder. It takes the boxed TACO photos that
# download_taco.py saved, cuts each boxed item out into its own little image,
# adds our own webcam photos, and arranges everything into train/ and val/
# folders ready for training. The v2 version is build_dataset_v2.py.

import random   # for shuffling
import shutil   # for copying files
from pathlib import Path
from PIL import Image   # for opening + cropping images

random.seed(42)  # reproducible split  (same shuffle every run)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Inputs
TACO_IMG_DIR = PROJECT_ROOT / "dataset" / "taco" / "images"   # downloaded photos
TACO_LBL_DIR = PROJECT_ROOT / "dataset" / "taco" / "labels"   # their box labels
OWN_DIR      = PROJECT_ROOT / "dataset" / "raw_captures"      # our webcam photos

# Output
OUT_DIR   = PROJECT_ROOT / "dataset_cls"
TRAIN_DIR = OUT_DIR / "train"
VAL_DIR   = OUT_DIR / "val"

# class id (in YOLO label files) -> class name (folder name)
CLASS_NAMES = {0: "plastic", 1: "metal", 2: "drinkkarton"}

# Clean and recreate output dirs
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)               # delete any old dataset first
for split in (TRAIN_DIR, VAL_DIR):       # make train/ and val/ ...
    for name in CLASS_NAMES.values():    # ...with a folder per class
        (split / name).mkdir(parents=True, exist_ok=True)

# Step 1 — collect all (image_path, class_name) samples
samples = []   # list of (PIL crop or path-to-copy, class_name, source_tag)

print("Step 1: cropping TACO images by bounding boxes...")
taco_count = 0
for lbl_path in TACO_LBL_DIR.glob("*.txt"):       # for each label file...
    img_path = TACO_IMG_DIR / (lbl_path.stem + ".jpg")   # find its photo
    if not img_path.exists():
        img_path = TACO_IMG_DIR / (lbl_path.stem + ".JPG")  # maybe upper-case
    if not img_path.exists():
        continue                                   # no photo, skip
    try:
        img = Image.open(img_path).convert("RGB")  # open the photo
    except Exception:
        continue                                   # corrupt, skip
    W, H = img.size

    for i, line in enumerate(lbl_path.read_text().strip().splitlines()):
        parts = line.split()                       # split the label line into 5 parts
        if len(parts) != 5:
            continue                               # malformed line, skip
        cls = int(parts[0])                        # the class id (0/1/2)
        cx, cy, nw, nh = map(float, parts[1:])     # the box, as fractions
        # Convert normalized YOLO bbox back to pixel coords
        # (the reverse of what download_taco.py did)
        x1 = int((cx - nw / 2) * W)
        y1 = int((cy - nh / 2) * H)
        x2 = int((cx + nw / 2) * W)
        y2 = int((cy + nh / 2) * H)
        # Clamp + skip degenerate boxes
        x1, y1 = max(0, x1), max(0, y1)            # keep inside the image
        x2, y2 = min(W, x2), min(H, y2)
        if (x2 - x1) < 20 or (y2 - y1) < 20:       # skip tiny boxes
            continue
        crop = img.crop((x1, y1, x2, y2))          # cut out just that item
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
    for img_path in folder.glob("*.jpg"):           # every photo in this class folder
        samples.append((img_path, cls_name, f"own_{img_path.stem}"))
        own_count += 1
print(f"  Own photos: {own_count}")
print(f"  TOTAL samples: {len(samples)}")

# Step 3 — shuffle & split 80/20 PER CLASS so each class is balanced
print("Step 3: splitting 80/20 per class...")
by_class = {n: [] for n in CLASS_NAMES.values()}   # group samples by class
for s in samples:
    by_class[s[1]].append(s)                        # s[1] is the class name

for cls_name, items in by_class.items():
    random.shuffle(items)                           # mix them up
    split_idx = int(len(items) * 0.8)               # 80% point
    train_items = items[:split_idx]                 # first 80% -> train
    val_items   = items[split_idx:]                 # last 20%  -> val
    print(f"  {cls_name:12s} train={len(train_items)}  val={len(val_items)}")

    for split_dir, batch in ((TRAIN_DIR, train_items), (VAL_DIR, val_items)):
        target = split_dir / cls_name
        for crop_or_path, _, tag in batch:
            out_path = target / f"{tag}.jpg"
            if isinstance(crop_or_path, Path):
                # copy own photo  (it's already a file on disk)
                shutil.copy(crop_or_path, out_path)
            else:
                # PIL Image (cropped from TACO)  (save the in-memory crop)
                crop_or_path.save(out_path, "JPEG", quality=90)

print("\nDone. Dataset ready at:", OUT_DIR)
print("Train with:")
print(f'  yolo classify train data="{OUT_DIR}" model=yolov8n-cls.pt epochs=30 imgsz=224')
