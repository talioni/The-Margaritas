"""
v2 TACO downloader/filter for the new 3 classes (organic / pmd / restafval).

We REUSE the already-downloaded annotations.json and the existing image cache
at dataset/taco/images/ (set up by v1's download_taco.py). This script
re-classifies each TACO sub-category into the v2 scheme and downloads
the additional images we didn't grab in v1 (mostly restafval items).

Note: TACO has almost no organic data (only 8 'Food waste' annotations),
so the user will still need to capture organic photos with the webcam.

Output: dataset/taco_v2/
    images/<class>/   # one cropped object per file
    Each cropped object goes in its class folder directly (no .txt labels
    needed for classification).

Run:
    python scripts/download_taco_v2.py
"""

import json
import urllib.request
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TACO_DIR = PROJECT_ROOT / "dataset" / "taco"
ANNO_PATH = TACO_DIR / "annotations.json"
IMG_CACHE = TACO_DIR / "images"

OUT_DIR = PROJECT_ROOT / "dataset" / "taco_v2"
for cls in ("organic", "pmd", "restafval"):
    (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

# TACO category -> v2 class
TACO_TO_V2 = {
    # ORGANIC (very limited in TACO)
    "Food waste": "organic",

    # PMD = Plastic + Metal + Drinkkarton
    "Clear plastic bottle": "pmd",
    "Other plastic bottle": "pmd",
    "Plastic bottle cap": "pmd",
    "Plastic lid": "pmd",
    "Other plastic": "pmd",
    "Plastic film": "pmd",
    "Other plastic wrapper": "pmd",
    "Disposable plastic cup": "pmd",
    "Other plastic cup": "pmd",
    "Other plastic container": "pmd",
    "Plastic straw": "pmd",
    "Single-use carrier bag": "pmd",
    "Polypropylene bag": "pmd",
    "Crisp packet": "pmd",
    "Spread tub": "pmd",
    "Tupperware": "pmd",
    "Squeezable tube": "pmd",
    "Plastic utensils": "pmd",
    "Six pack rings": "pmd",
    "Garbage bag": "pmd",
    "Drink can": "pmd",
    "Food Can": "pmd",
    "Aerosol": "pmd",
    "Metal bottle cap": "pmd",
    "Metal lid": "pmd",
    "Aluminium foil": "pmd",
    "Aluminium blister pack": "pmd",
    "Carded blister pack": "pmd",
    "Pop tab": "pmd",
    "Scrap metal": "pmd",
    "Drink carton": "pmd",
    "Other carton": "pmd",

    # RESTAFVAL = residual / general waste
    "Cigarette": "restafval",
    "Unlabeled litter": "restafval",
    "Broken glass": "restafval",
    "Glass bottle": "restafval",
    "Glass cup": "restafval",
    "Glass jar": "restafval",
    "Styrofoam piece": "restafval",
    "Foam cup": "restafval",
    "Foam food container": "restafval",
    "Paper cup": "restafval",
    "Normal paper": "restafval",
    "Magazine paper": "restafval",
    "Wrapping paper": "restafval",
    "Tissues": "restafval",
    "Paper bag": "restafval",
    "Paper straw": "restafval",
    "Toilet tube": "restafval",
    "Corrugated carton": "restafval",
    "Egg carton": "restafval",
    "Meal carton": "restafval",
    "Pizza box": "restafval",
    "Disposable food container": "restafval",
    "Rope & strings": "restafval",
    "Plastic glooves": "restafval",
    "Shoe": "restafval",
    "Battery": "restafval",
}

# Load annotations
with open(ANNO_PATH) as f:
    coco = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
img_id_to_info = {i["id"]: i for i in coco["images"]}

# Group annotations by image and resolve to v2 class
anns_by_img = defaultdict(list)
for ann in coco["annotations"]:
    cat = cat_id_to_name[ann["category_id"]]
    if cat in TACO_TO_V2:
        anns_by_img[ann["image_id"]].append(
            (TACO_TO_V2[cat], ann["bbox"])
        )

print(f"Images with at least one v2-relevant object: {len(anns_by_img)}")

# Process each image: download if needed, then crop each bbox
v2_counts = Counter()
fail = 0
done = 0
for img_id, anns in anns_by_img.items():
    info = img_id_to_info[img_id]
    safe_name = info["file_name"].replace("/", "_")
    img_path = IMG_CACHE / safe_name

    # Download to cache if missing
    if not img_path.exists():
        url = info.get("flickr_640_url") or info.get("flickr_url")
        if not url:
            fail += 1
            continue
        try:
            urllib.request.urlretrieve(url, img_path)
        except Exception:
            fail += 1
            continue

    # Open and crop each bbox
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        fail += 1
        continue
    W, H = img.size

    for i, (cls, (x, y, w, h)) in enumerate(anns):
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        crop = img.crop((x1, y1, x2, y2))
        out = OUT_DIR / cls / f"{Path(safe_name).stem}_{i}.jpg"
        crop.save(out, "JPEG", quality=90)
        v2_counts[cls] += 1

    done += 1
    if done % 100 == 0:
        print(f"  processed {done}/{len(anns_by_img)} images so far ({dict(v2_counts)})")

print("\nDone.")
print("Per-class crop counts:")
for c in ("organic", "pmd", "restafval"):
    print(f"  {c:10s} {v2_counts[c]}")
print(f"Failures: {fail}")
print(f"Crops saved to: {OUT_DIR}")
