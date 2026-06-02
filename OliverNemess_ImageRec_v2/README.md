# Household Trash Sorter v2 — Image Recognition AI

AI that recognizes the three Belgian/Dutch household waste categories from a webcam feed:

- **Organic** (food scraps, peels, plant matter)
- **PMD** (Plastic, Metal, Drinkkarton — bottles, cans, cartons)
- **Restafval** (residual waste — tissues, broken glass, cigarettes, non-recyclables)

Built with YOLOv8 classification trained on a combined dataset of TACO public images and our own webcam captures. Designed to run on a Raspberry Pi 5.

> Built by Oliver Nemess as part of the Margaritas team project.

---

## What's in this folder

```
scripts/
  capture_photos_v2.py        # take training photos with webcam (3 new classes)
  download_taco_v2.py         # pull TACO public trash dataset (PMD + Restafval)
  build_dataset_v2.py         # merge sources, balance classes, split train/val
  live_predict_v2.py          # run trained model on live webcam
runs/classify/train_v2/weights/
  best.pt                     # OUR trained v2 model (3 MB) — ready to use
requirements.txt
README.md
```

---

## Quick start (use the pretrained v2 model)

### 1. Clone the repo and check out this branch

```bash
git clone https://github.com/talioni/The-Margaritas.git
cd The-Margaritas
git checkout OliverNemess_ImageRec_v2
cd OliverNemess_ImageRec_v2
```

### 2. Create a Python virtual environment (Python 3.10+)

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows:**

```bat
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Important on macOS:** Do NOT put the project inside `~/Desktop/` or `~/Documents/` if iCloud Drive sync is enabled — iCloud can offload image files during training and cause OpenCV errors. Use `~/Code/`, `~/dev/`, or anywhere outside the iCloud-synced folders.

### 3. Run the live demo

```bash
python scripts/live_predict_v2.py
```

Hold up an organic item, a PMD item, or a restafval item. The top bar shows the prediction; the bottom bars show class probabilities. Press `q` to quit.

---

## Reproduce the training from scratch

### 1. Capture your own training photos

```bash
python scripts/capture_photos_v2.py
```

- `o` → save as **organic**
- `m` → save as **pmd**
- `r` → save as **restafval**
- `q` → quit

Aim for at least 200 photos per class, with varied items, angles, and the same lighting/background you'll deploy the sorter in.

### 2. Download TACO public dataset

```bash
python scripts/download_taco_v2.py
```

Downloads ~1500 trash photos with annotations. Mapped to our 3 classes (TACO has essentially no organic data, so most organic must come from your webcam).

### 3. Build the classification dataset

```bash
python scripts/build_dataset_v2.py
```

Merges TACO crops + your webcam photos, caps each class at 500 (configurable in the script via `MAX_PER_CLASS`) to balance training, splits 80% train / 20% val.

### 4. Train

```bash
yolo classify train data="$(pwd)/dataset_v2_cls" model=yolov8n-cls.pt epochs=20 imgsz=224 batch=32 workers=0 name=train_v2
```

On an Apple Silicon Mac (CPU) this takes ~10 minutes. The trained model lands in `runs/classify/train_v2/weights/best.pt`.

---

## How the AI was built

- **Base architecture:** YOLOv8 nano (classification head)
- **Pretrained weights:** ImageNet (transfer learning)
- **Dataset (v2):**
  - Organic: 296 own webcam photos
  - PMD: 205 own (v1) + 295 TACO crops (capped at 500)
  - Restafval: 354 own + 146 TACO crops (capped at 500)
- **Classes:** `organic` (0), `pmd` (1), `restafval` (2)
- **Validation accuracy:** 91.2% top-1

---

## What's new vs v1

| | v1 | v2 |
|---|---|---|
| Classes | plastic / metal / drinkkarton (PMD subtypes) | organic / pmd / restafval (household bins) |
| Validation accuracy | 83.4% | **91.2%** |
| Dataset size | ~2000 images (TACO + 205 own) | ~1300 balanced images |
| Use case | Sorting within the PMD bin | Full household 3-bin sort |

---

## Hardware target

- Raspberry Pi 5
- USB webcam
- Pi deployment code coming in a follow-up commit

---

## Troubleshooting

**`Failed to read frame` on macOS** — give camera permission to Terminal/VS Code in *System Settings → Privacy & Security → Camera*, then restart that app.

**`cv2.error ... !_src.empty()` mid-training** — your project is inside an iCloud-synced folder (`~/Desktop/` or `~/Documents/`). Move it to `~/Code/` or disable iCloud sync for those folders.

**`yolo: command not found`** — activate the venv: `source venv/bin/activate`.

---

## License

This project uses the TACO dataset (CC BY 4.0) and the Ultralytics YOLOv8 framework (AGPL-3.0).
