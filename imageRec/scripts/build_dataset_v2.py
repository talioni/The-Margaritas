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

# ---- What this file does ----
# Before you can TRAIN an AI you must arrange the photos in the exact folder
# layout the trainer expects. This script gathers all our photos (our own +
# the TACO crops), keeps the numbers balanced between classes, and splits them
# into a "train" pile (to learn from) and a "val" pile (to test on).

import random   # to shuffle photos randomly
import shutil   # to copy files
from pathlib import Path

random.seed(42)   # fix the randomness so the same split happens every run

# Cap each class to this many samples to keep training balanced.
# Set to a higher number (or None) if you want to use everything.
MAX_PER_CLASS = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Class -> list of (source_folder, priority_rank)  lower rank = kept first
# So we always keep our OWN webcam photos (rank 0) before TACO crops (rank 1).
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

OUT       = PROJECT_ROOT / "dataset_v2_cls"   # the finished dataset goes here
TRAIN_DIR = OUT / "train"                     # photos to learn from
VAL_DIR   = OUT / "val"                       # photos to test on

VALID_EXT = {".jpg", ".jpeg", ".png"}         # only treat these as images

if OUT.exists():
    # Try a clean wipe; if permission-protected files (e.g. YOLO .cache)
    # block it, force-chmod and retry.
    # In short: delete any old version of the dataset before building a fresh one.
    def _onerror(func, path, exc_info):
        import os, stat
        try:
            os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP)
            func(path)
        except Exception:
            pass
    shutil.rmtree(OUT, onerror=_onerror)
for split in (TRAIN_DIR, VAL_DIR):            # make train/ and val/ ...
    for cls in SOURCES:                        # ...with a folder per class inside
        (split / cls).mkdir(parents=True, exist_ok=True)

print("Building v2 dataset...\n")
print(f"Cap per class: {MAX_PER_CLASS}\n")

for cls, source_list in SOURCES.items():       # handle one class at a time
    # Collect images grouped by priority
    by_priority = {}
    for folder, rank in source_list:
        if not folder.exists():                # skip folders that aren't there
            continue
        for p in folder.iterdir():             # look at every file in the folder
            if p.suffix.lower() in VALID_EXT:  # keep only real images
                by_priority.setdefault(rank, []).append(p)

    # Keep priority-0 first, then shuffle each priority bucket
    selected = []
    for rank in sorted(by_priority.keys()):    # go through rank 0, then 1, ...
        bucket = by_priority[rank]
        random.shuffle(bucket)                 # mix up the photos in this bucket
        for p in bucket:
            if MAX_PER_CLASS and len(selected) >= MAX_PER_CLASS:
                break                          # stop once we hit the cap
            selected.append(p)
        if MAX_PER_CLASS and len(selected) >= MAX_PER_CLASS:
            break

    random.shuffle(selected)                   # final shuffle before splitting
    split_idx = int(len(selected) * 0.8)       # 80% point
    train_imgs = selected[:split_idx]          # first 80% -> training
    val_imgs   = selected[split_idx:]          # last 20%  -> validation

    for p in train_imgs:
        # prefix with source folder name to avoid name collisions
        out_name = f"{p.parent.name}_{p.name}"      # e.g. "metal_metal_123.jpg"
        shutil.copy(p, TRAIN_DIR / cls / out_name)  # copy into train/<class>/
    for p in val_imgs:
        out_name = f"{p.parent.name}_{p.name}"
        shutil.copy(p, VAL_DIR / cls / out_name)    # copy into val/<class>/

    print(f"  {cls:10s} kept={len(selected):4d}  train={len(train_imgs):4d}  val={len(val_imgs):4d}")

print(f"\nDone. Dataset at: {OUT}")
print("\nTrain with:")
# This is the exact command you run next to actually train the model.
print(f'  yolo classify train data="{OUT}" model=yolov8n-cls.pt epochs=20 imgsz=224 batch=32 workers=0 name=train_v2')
