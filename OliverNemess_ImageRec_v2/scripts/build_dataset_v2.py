"""
Build the v2 classification dataset combining everything we have.

Sources:
    organic   : dataset_v2/raw_captures/organic                (296 own webcam)
    pmd       : dataset/taco_v2/pmd                            (945 TACO crops)
              + dataset/raw_captures/{plastic,metal,drinkkarton} (205 own v1)
    restafval : dataset/taco_v2/restafval                      (486 TACO crops)
              + dataset_v2/raw_captures/restafval              (261 own webcam)

To avoid class imbalance bias, we cap each class to MAX_PER_CLASS samples.
Own webcam photos are prioritized (kept first), TACO crops fill the rest.

Output (YOLOv8 classify expects this layout):
    dataset_v2_cls/
        train/{organic,pmd,restafval}/
        val/{organic,pmd,restafval}/

Run:
    python scripts/build_dataset_v2.py
"""

import random
import shutil
from pathlib import Path

random.seed(42)

# Cap each class to this many samples to keep training balanced.
# Set to a higher number (or None) if you want to use everything.
MAX_PER_CLASS = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Class -> list of (source_folder, priority_rank)  lower rank = kept first
SOURCES = {
    "organic": [
        (PROJECT_ROOT / "dataset_v2" / "raw_captures" / "organic", 0),
    ],
    "pmd": [
        # own webcam first (matches deployment environment)
        (PROJECT_ROOT / "dataset" / "raw_captures" / "plastic",     0),
        (PROJECT_ROOT / "dataset" / "raw_captures" / "metal",       0),
        (PROJECT_ROOT / "dataset" / "raw_captures" / "drinkkarton", 0),
        # then TACO crops to fill up
        (PROJECT_ROOT / "dataset" / "taco_v2" / "pmd",              1),
    ],
    "restafval": [
        (PROJECT_ROOT / "dataset_v2" / "raw_captures" / "restafval", 0),
        (PROJECT_ROOT / "dataset" / "taco_v2" / "restafval",         1),
    ],
}

OUT       = PROJECT_ROOT / "dataset_v2_cls"
TRAIN_DIR = OUT / "train"
VAL_DIR   = OUT / "val"

VALID_EXT = {".jpg", ".jpeg", ".png"}

if OUT.exists():
    # Try a clean wipe; if permission-protected files (e.g. YOLO .cache)
    # block it, force-chmod and retry.
    def _onerror(func, path, exc_info):
        import os, stat
        try:
            os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP)
            func(path)
        except Exception:
            pass
    shutil.rmtree(OUT, onerror=_onerror)
for split in (TRAIN_DIR, VAL_DIR):
    for cls in SOURCES:
        (split / cls).mkdir(parents=True, exist_ok=True)

print("Building v2 dataset...\n")
print(f"Cap per class: {MAX_PER_CLASS}\n")

for cls, source_list in SOURCES.items():
    # Collect images grouped by priority
    by_priority = {}
    for folder, rank in source_list:
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.suffix.lower() in VALID_EXT:
                by_priority.setdefault(rank, []).append(p)

    # Keep priority-0 first, then shuffle each priority bucket
    selected = []
    for rank in sorted(by_priority.keys()):
        bucket = by_priority[rank]
        random.shuffle(bucket)
        for p in bucket:
            if MAX_PER_CLASS and len(selected) >= MAX_PER_CLASS:
                break
            selected.append(p)
        if MAX_PER_CLASS and len(selected) >= MAX_PER_CLASS:
            break

    random.shuffle(selected)
    split_idx = int(len(selected) * 0.8)
    train_imgs = selected[:split_idx]
    val_imgs   = selected[split_idx:]

    for p in train_imgs:
        # prefix with source folder name to avoid name collisions
        out_name = f"{p.parent.name}_{p.name}"
        shutil.copy(p, TRAIN_DIR / cls / out_name)
    for p in val_imgs:
        out_name = f"{p.parent.name}_{p.name}"
        shutil.copy(p, VAL_DIR / cls / out_name)

    print(f"  {cls:10s} kept={len(selected):4d}  train={len(train_imgs):4d}  val={len(val_imgs):4d}")

print(f"\nDone. Dataset at: {OUT}")
print("\nTrain with:")
print(f'  yolo classify train data="{OUT}" model=yolov8n-cls.pt epochs=20 imgsz=224 batch=32 workers=0 name=train_v2')
