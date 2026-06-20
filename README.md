# The Margaritas — AI Trash Sorter

An automatic trash sorter. A webcam looks at a piece of rubbish, an AI model
decides which bin it belongs in, and a motor turns the bin to the right slot
and opens a lid so the item drops in. It runs on a Raspberry Pi.

The waste categories follow the Dutch household scheme:

- **organic** — food scraps, peels, plants
- **pmd** — Plastic, Metal and Drink cartons
- **restafval** — residual / general waste (paper towels, broken things, hygiene items)

> Built by the Margaritas team. Image recognition by Oliver Nemess.

---

## How it works (the big picture)

The project is split into **two programs** that run as separate Docker
containers and talk to each other over a small private network:

1. **`imageRec` (the "camera brain")** — a small web server (FastAPI) that owns
   the USB webcam and runs the AI model (YOLOv8 classification). Other programs
   ask it questions like "look at the camera and tell me what you see".

2. **`hadrwareCtrl` (the "hardware brain")** — the controller that repeatedly
   asks `imageRec` what it sees, and when it is confident, rotates the bin with
   a stepper motor and opens/closes a lid with a servo.

```
   ┌─────────────┐   "what do you see?"    ┌──────────────────┐
   │  webcam ───►│      (HTTP request)     │                  │
   │  imageRec   │◄───────────────────────►│  hadrwareCtrl    │──► stepper motor
   │  (AI brain) │   "pmd, 86% sure"       │  (hardware brain)│──► lid servo
   └─────────────┘                         └──────────────────┘
```

The two containers are wired together by `docker-compose.yaml`.

---

## Repository layout

```
The-Margaritas/
├── docker-compose.yaml          # runs both containers together
├── .env.example                 # copy to .env to enable real GPIO on the Pi
│
├── imageRec/                    # the camera + AI program
│   ├── Dockerfile               # recipe to build the imageRec container
│   ├── api/
│   │   ├── main.py              # the web server + AI endpoints  ★ main file
│   │   └── requirements.txt     # Python packages this program needs
│   ├── scripts/                 # tools used to BUILD and TEST the model
│   │   ├── capture_photos_v2.py # take your own training photos with a webcam
│   │   ├── download_taco_v2.py  # download + sort the public TACO trash dataset
│   │   ├── build_dataset_v2.py  # arrange all photos into train/val folders
│   │   ├── live_predict_v2.py   # live webcam demo of the trained model
│   │   └── *  (the non-v2 files are the older 3-class version, kept for history)
│   └── runs/classify/train_v2-6/weights/best.pt   # OUR trained model
│
├── hadrwareCtrl/                # the motor + lid program
│   ├── Dockerfile               # recipe to build the hadrwareCtrl container
│   ├── main.py                  # the orchestrator (the main loop)  ★ main file
│   ├── tb6600.py                # low-level stepper-motor driver
│   ├── tb6600_test.py           # step-by-step motor troubleshooting tool
│   ├── servo_test.py            # standalone lid-servo test
│   └── requirements.txt
│
├── schematic/schematic.fzz      # wiring diagram (Fritzing)
└── datasets/                    # photo datasets (not committed — regenerate them)
```

Note: the older files without `_v2` (`capture_photos.py`, `download_taco.py`,
`build_classification_dataset.py`, `live_predict.py`) belong to the first
version of the project, which used three classes named *plastic / metal /
drinkkarton*. They are kept for reference; the current project uses the v2
classes *organic / pmd / restafval*.

---

## Quick start — run the whole thing with Docker

This is the normal way to run the finished appliance on the Raspberry Pi.

```bash
git clone https://github.com/talioni/The-Margaritas.git
cd The-Margaritas

# Build and start both containers
docker compose up --build
```

What happens:

- `imageRec` starts, loads the model, and opens the webcam (passed in as
  `/dev/video0`).
- `hadrwareCtrl` waits until `imageRec` is healthy, then begins its loop.

By default GPIO is **off**, so `hadrwareCtrl` runs in **dry-run mode**: it logs
everything it *would* do to the motor and servo without touching any pins. This
lets you test the logic on a laptop with no hardware attached.

To drive the **real** motor and servo on the Pi, enable GPIO:

```bash
cp .env.example .env      # then make sure it contains GPIO_ENABLED=true
docker compose up --build
```

---

## The API endpoints (imageRec)

`imageRec` exposes these over `http://trash-api:8000` (inside Docker) or
`http://127.0.0.1:8000` (on the host):

| Method & path        | What it does                                                                 |
|----------------------|------------------------------------------------------------------------------|
| `GET /health`        | Says whether the model and camera are ready.                                 |
| `GET /predict`       | Grabs one fresh webcam frame and returns the predicted class.                |
| `POST /predict`      | Classifies an **uploaded** image (handy for testing without a camera).       |
| `GET /wait_and_predict` | Watches the tray and only answers once one class stays confidently on top for about a second. This is what the hardware controller uses. |

A prediction looks like this:

```json
{
  "top": "pmd",
  "confidence": 0.86,
  "unsure": false,
  "probabilities": { "organic": 0.05, "pmd": 0.86, "restafval": 0.09 }
}
```

### Why `/wait_and_predict` is clever

A live camera flickers — one frame might read 84%, the next 78%. If we acted on
a single frame we could sort on a fluke. So `/wait_and_predict` waits until the
**same** class stays above a confidence trigger for a short streak, tolerating
small dips (this is called *hysteresis*). Only then does it answer, so the bin
moves only when we are genuinely sure.

---

## Running the live demo (no hardware needed)

To just *see* the AI working on a laptop with a webcam:

```bash
cd imageRec
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r api/requirements.txt
python scripts/live_predict_v2.py
```

A window opens with your webcam feed. Hold up an item; the top bar shows the
prediction and the bottom bars show the score for each class. Press `q` to quit.

To pick a different camera: `CAMERA_INDEX=2 python scripts/live_predict_v2.py`.

---

## Reproducing the model training

You only need this if you want to retrain the AI from scratch.

```bash
# 1. Take your own training photos (o = organic, m = pmd, r = restafval)
python scripts/capture_photos_v2.py

# 2. Download + sort the public TACO trash dataset into our 3 classes
python scripts/download_taco_v2.py

# 3. Combine everything and split into train/val folders (balanced per class)
python scripts/build_dataset_v2.py

# 4. Train. The command is printed at the end of step 3:
yolo classify train data="…/dataset_v2_cls" model=yolov8n-cls.pt \
     epochs=20 imgsz=224 batch=32 workers=0 name=train_v2
```

The trained model lands in `runs/classify/train_v2-…/weights/best.pt`, which is
the file `imageRec/api/main.py` loads.

### How the AI was built (notes for the assessment)

- **Type of task:** image *classification* — given a whole picture, output one
  label. (The very first version tried object *detection* with boxes; the
  project later switched to classification because it is simpler and enough for
  sorting one item at a time.)
- **Base architecture:** YOLOv8 nano classification model (`yolov8n-cls`), a
  small convolutional neural network.
- **Transfer learning:** we start from weights pretrained on ImageNet and
  fine-tune them on our own trash photos, instead of training from zero.
- **Training data:** our own webcam photos + cropped items from the public TACO
  dataset, capped per class to keep the three classes balanced.
- **Classes:** `organic`, `pmd`, `restafval`.

---

## The hardware

- **Raspberry Pi** (Pi 4 / Pi 5) running the two Docker containers.
- **USB webcam** for the camera.
- **Stepper motor** driven by a **TB6600** driver — rotates the bin to one of
  three positions (organic = −60°, pmd = 0°, restafval = +60°).
- **Servo motor** — opens and closes a lid so the item drops into the bin.
- Wiring diagram: `schematic/schematic.fzz` (open with Fritzing).

The TB6600 driver code (`tb6600.py`) talks to the Pi pins directly with `lgpio`.
Before running the full system, test the motor in stages with:

```bash
python hadrwareCtrl/tb6600_test.py      # pins -> tick -> slow -> spin
```

and test just the lid with:

```bash
GPIO_ENABLED=true python hadrwareCtrl/servo_test.py
```

Both files contain detailed wiring notes and a symptom-to-cause guide in their
comments.

---

## Troubleshooting

**Camera not found / `GET /predict` returns 503** — the webcam isn't visible to
the container. Check it is plugged in and passed through as `/dev/video0` in
`docker-compose.yaml`, and set `CAMERA_INDEX` if it isn't index 0.

**Motor locks but never turns** — almost always a wiring/voltage issue. Read the
voltage note at the top of `tb6600.py` (try `PUL+/DIR+` on 3.3 V, not 5 V) and
run `tb6600_test.py` to find the exact stage that fails.

**Everything logs "dry-run"** — that's expected without `GPIO_ENABLED=true`. Set
it in your `.env` file on the Pi to drive the real hardware.

**Camera permission on macOS (live demo)** — allow camera access for your
Terminal / VS Code in *System Settings → Privacy & Security → Camera*.

---

## License / credits

This project uses the **TACO** dataset (CC BY 4.0) and the **Ultralytics
YOLOv8** framework (AGPL-3.0).
