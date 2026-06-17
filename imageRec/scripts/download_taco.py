"""
Download + filter the TACO dataset for our 3 classes:
    plastic (0), metal (1), drinkkarton (2)

What this does:
1. Downloads TACO annotations (COCO format, ~10 MB)
2. Picks only the TACO sub-categories we care about
3. Downloads the matching images from Flickr
4. Converts COCO bounding boxes -> YOLO format .txt files
5. Saves everything to dataset/taco/

Run:
    python scripts/download_taco.py

You can stop and re-run anytime — it skips already-downloaded images.
"""

# ---- What this file does ----
# This is the OLD (v1) TACO downloader. It grabs public trash photos and writes
# a label file (.txt) next to each one in "YOLO format". v1 trained an object
# DETECTOR (boxes), while v2 switched to a CLASSIFIER (whole-image label) —
# that's why download_taco_v2.py crops items into folders instead.

import json                # read the annotations (JSON)
import os                  # (kept for completeness)
import urllib.request      # download files from the internet
from pathlib import Path
from collections import defaultdict

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR     = PROJECT_ROOT / "dataset" / "taco"
IMG_DIR     = OUT_DIR / "images"     # downloaded photos go here
LBL_DIR     = OUT_DIR / "labels"     # matching YOLO label files go here
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

# Where to download TACO's annotation file from.
ANNO_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
ANNO_PATH = OUT_DIR / "annotations.json"

# TACO category names -> our class id (matches data.yaml: 0=plastic 1=metal 2=drinkkarton)
# v1 used NUMBERS for classes (0/1/2) because that's what YOLO detection labels need.
TACO_TO_OURS = {
    # plastic (0)
    "Clear plastic bottle": 0,
    "Other plastic bottle": 0,
    "Plastic bottle cap": 0,
    "Plastic lid": 0,
    "Other plastic": 0,
    "Plastic film": 0,
    "Other plastic wrapper": 0,
    "Plastic container": 0,
    "Disposable plastic cup": 0,
    "Other plastic cup": 0,
    # metal (1)
    "Drink can": 1,
    "Food Can": 1,
    "Aerosol": 1,
    "Metal bottle cap": 1,
    "Metal lid": 1,
    "Aluminium foil": 1,
    "Aluminium blister pack": 1,
    # drinkkarton (2)
    "Drink carton": 2,
    "Other carton": 2,
    "Tupperware": 2,  # remove if you don't want this
}

# --- Step 1: download annotations ---
if not ANNO_PATH.exists():                          # only download once
    print(f"Downloading TACO annotations to {ANNO_PATH} ...")
    urllib.request.urlretrieve(ANNO_URL, ANNO_PATH)
else:
    print("Annotations already downloaded, skipping.")

with open(ANNO_PATH) as f:
    coco = json.load(f)        # load the notes into a dictionary

# Build lookups
cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}  # id -> category name
img_id_to_info = {i["id"]: i for i in coco["images"]}              # id -> image details

# Annotations grouped by image
anns_by_img = defaultdict(list)            # for each image: a list of its items
for ann in coco["annotations"]:
    anns_by_img[ann["image_id"]].append(ann)

# --- Step 2: find images that contain our classes ---
wanted_images = []                 # images that have at least one item we want
counts = defaultdict(int)          # how many of each TACO category we found
for img_id, anns in anns_by_img.items():
    kept = []
    for ann in anns:
        cat_name = cat_id_to_name[ann["category_id"]]
        if cat_name in TACO_TO_OURS:               # is this an item we want?
            our_class = TACO_TO_OURS[cat_name]      # map it to our 0/1/2 id
            kept.append((our_class, ann["bbox"]))   # keep the class + its box
            counts[cat_name] += 1
    if kept:
        wanted_images.append((img_id, kept))

print(f"\nFound {len(wanted_images)} images with our classes.")
print("Annotation counts per TACO category:")
for name, n in sorted(counts.items(), key=lambda x: -x[1]):   # most common first
    print(f"  {name:30s} {n}")

# --- Step 3: download images + write YOLO labels ---
print(f"\nDownloading images to {IMG_DIR} (skipping ones we already have)...")
fail = 0
for idx, (img_id, anns) in enumerate(wanted_images, 1):
    info = img_id_to_info[img_id]
    # TACO image filenames look like "batch_5/000123.JPG"
    safe_name = info["file_name"].replace("/", "_")    # flatten into one name
    img_path = IMG_DIR / safe_name
    lbl_path = LBL_DIR / (Path(safe_name).stem + ".txt")

    # Already done?
    if img_path.exists() and lbl_path.exists():        # skip if we have both
        if idx % 50 == 0:
            print(f"  [{idx}/{len(wanted_images)}] already have, skipping")
        continue

    # Download image (try flickr URL, fallback to taco bucket)
    url = info.get("flickr_640_url") or info.get("flickr_url")
    if not url:
        fail += 1
        continue
    try:
        urllib.request.urlretrieve(url, img_path)      # download the photo
    except Exception as e:
        print(f"  fail: {safe_name} -> {e}")
        fail += 1
        continue

    # Write YOLO label
    # YOLO format: <class> <x_center> <y_center> <width> <height>  (all normalized 0..1)
    # "Normalized" means stored as a fraction of the image size, so it works at
    # any resolution. We convert TACO's pixel boxes into that fraction form.
    W, H = info["width"], info["height"]
    lines = []
    for cls, (x, y, w, h) in anns:
        cx = (x + w / 2) / W       # box centre X as a fraction (0..1)
        cy = (y + h / 2) / H       # box centre Y as a fraction
        nw = w / W                 # box width  as a fraction
        nh = h / H                 # box height as a fraction
        # clip to [0,1]
        cx = max(0, min(1, cx))    # make sure values stay between 0 and 1
        cy = max(0, min(1, cy))
        nw = max(0, min(1, nw))
        nh = max(0, min(1, nh))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    lbl_path.write_text("\n".join(lines))     # save all boxes for this image

    if idx % 25 == 0:
        print(f"  [{idx}/{len(wanted_images)}] downloaded")

print(f"\nDone. Failures: {fail}")
print(f"Images:  {IMG_DIR}")
print(f"Labels:  {LBL_DIR}")
