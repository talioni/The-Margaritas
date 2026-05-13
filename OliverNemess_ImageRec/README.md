# PMD Trash Sorter — Image Recognition AI

AI that recognizes the three PMD recycling categories — **Plastic, Metal, Drinkkarton** — from a webcam feed.
Trained with YOLOv8 classification on a combined dataset of public images (TACO) and our own webcam captures.
Designed to run on a Raspberry Pi 4 with a servo motor that physically sorts trash into the correct bin.

> Built by Oliver Nemess as part of the Margaritas team project.

---

## What's in this repo

```
scripts/
  capture_photos.py              # take training photos with your webcam
  download_taco.py               # pull TACO public trash dataset
  build_classification_dataset.py # crop + split into train/val
  live_predict.py                # run the trained model on live webcam
runs/classify/train/weights/
  best.pt                        # OUR trained model (3 MB) — ready to use
dataset/data.yaml                # class config
requirements.txt
README.md
```

---

## Quick start (use the pretrained model right away)

If you just want to see the demo running, skip training and use the included `best.pt`.

### 1. Clone the repo and check out this branch

```bash
git clone https://github.com/talioni/The-Margaritas.git
cd The-Margaritas
git checkout OliverNemess_ImageRec
```

### 2. Set up a Python virtual environment (Python 3.10+)

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

### 3. Run the live demo

```bash
python scripts/live_predict.py
```

A window opens with your webcam feed. Hold up a plastic bottle, a metal can, or a drink carton.
The top bar shows the prediction; the bottom bars show class probabilities. Press `q` to quit.

---

## Reproduce the training from scratch

If you want to retrain the model on your own data:

### 1. Capture your own training photos

```bash
python scripts/capture_photos.py
```

Hold one trash item in frame at a time. Press:
- `p` → save as **plastic**
- `m` → save as **metal**
- `d` → save as **drinkkarton**
- `q` → quit

Aim for 30–50 photos per class. Vary lighting and angle. Photos save to `dataset/raw_captures/<class>/`.

### 2. Download the TACO public dataset

```bash
python scripts/download_taco.py
```

Downloads ~1150 trash photos with bounding boxes for our three classes. Re-runnable (skips files already on disk).

### 3. Build the classification dataset

```bash
python scripts/build_classification_dataset.py
```

Crops each TACO bounding box into its own image, adds your webcam photos, and splits 80% train / 20% validation into `dataset_cls/`.

### 4. Train

```bash
yolo classify train data="$(pwd)/dataset_cls" model=yolov8n-cls.pt epochs=20 imgsz=224 batch=32 workers=0
```

On an Apple Silicon Mac (CPU) this takes ~10 minutes. The trained model lands in `runs/classify/train/weights/best.pt`.

---

## How the AI was built (project notes)

- **Base architecture:** YOLOv8 nano (classification head), a small convolutional network
- **Pretrained weights:** ImageNet (transfer learning)
- **Custom training data:** ~1600 train + ~400 validation images, combined from:
  - TACO public dataset (cropped from bounding boxes)
  - 205 photos taken with our own webcam
- **Classes:** `plastic` (0), `metal` (1), `drinkkarton` (2)
- **Validation accuracy:** ~83% top-1

The `best.pt` file in this repo is the result of training YOLOv8n-cls on our custom PMD dataset. The architecture is YOLOv8's, the weights are ours.

---

## Hardware target

- Raspberry Pi 4 (4 GB or more)
- USB webcam
- SG90 servo motor on a flap to redirect trash into one of three bins
- (Pi deployment code coming in a follow-up commit)

---

## Troubleshooting

**`Failed to read frame` on macOS** — give camera permission to Terminal or VS Code in *System Settings → Privacy & Security → Camera*, then restart that app.

**`yolo: command not found`** — make sure the virtual environment is activated (`source venv/bin/activate`).

**Training is too slow** — reduce `epochs` (e.g. `epochs=10`) or `batch` size.

---

## License

This project uses the TACO dataset (CC BY 4.0) and the Ultralytics YOLOv8 framework (AGPL-3.0).
