"""
Local HTTP API around the v2 trash classifier (organic / pmd / restafval).

The container owns the USB webcam (passed in as /dev/video0) and runs inference
on demand. The future main program calls these endpoints over the local Docker
network and drives the hardware.

Endpoints:
    GET  /health             -> liveness + camera/model status
    GET  /predict            -> grab a fresh frame from the webcam and classify it
    POST /predict            -> classify an uploaded image (testing aid, no camera)
    GET  /wait_and_predict   -> classify frames until one class stays above the
                                confidence trigger for a second, then return it
                                (else "unsure"). Used by the hadrwareCtrl
                                orchestrator to drive the bin only when sure.

Run (locally, outside Docker):
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# --- Configuration ---------------------------------------------------------

# Default points at where the Dockerfile bakes the weights; overridable for
# running outside the container (e.g. the dev box).
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "runs"
    / "classify"
    / "train_v2-6"
    / "weights"
    / "best.pt"
)
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
IMG_SIZE = 224
MIN_CONF = 0.50  # below this the top prediction is flagged "unsure"
WARMUP_FRAMES = 10  # let the webcam stabilise on open
FLUSH_FRAMES = 4  # drop buffered frames before reading a current one

# --- Confidence-trigger tuning (used by /wait_and_predict) -----------------
# We classify frames continuously and return once one class stays on top with
# confidence above CONF_TRIGGER for HOLD_SECONDS.
#
# Top-1 confidence (0-1) a class must clear to START a streak. "above 80".
CONF_TRIGGER = float(os.getenv("CONF_TRIGGER", "0.80"))
# Hysteresis: once a streak is running, only reset it if confidence drops below
# this (or the top class changes). A live stream flickers 78%<->86% frame to
# frame; without this gap a single dip would wipe the timer and we'd never reach
# a full second. Keep it a bit below CONF_TRIGGER.
CONF_RELEASE = float(os.getenv("CONF_RELEASE", "0.70"))
# How long that confident streak must hold before we return and rotate the bin.
HOLD_SECONDS = float(os.getenv("HOLD_SECONDS", "1.0"))
# Hard timeout for /wait_and_predict so the orchestrator can recover if nothing
# confident ever shows up. We return an "unsure" result instead of hanging.
MAX_WAIT_SECONDS = int(os.getenv("MAX_WAIT_SECONDS", "120"))
# Pause between frame reads inside the wait loop (~20 fps). Keeps CPU low.
FRAME_DELAY_S = 0.05
# Throttle for the live per-frame log so docker logs show what the demo shows
# on screen, without spamming a line every frame.
LOG_EVERY_S = float(os.getenv("LOG_EVERY_S", "0.5"))

# --- Shared state ----------------------------------------------------------

model: YOLO | None = None
camera: cv2.VideoCapture | None = None
_camera_lock = threading.Lock()  # a single VideoCapture is not concurrency-safe


def _open_camera() -> "cv2.VideoCapture | None":
    """Open the USB webcam; return None if it isn't available (e.g. dev box)."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(WARMUP_FRAMES):
        cap.read()
    return cap


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, camera
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Classes: {model.names}")

    camera = _open_camera()
    if camera is None:
        print(
            f"WARNING: webcam (index {CAMERA_INDEX}) not available; "
            f"GET /predict will return 503. POST /predict still works."
        )
    else:
        print("Camera ready.")

    yield

    if camera is not None:
        camera.release()


app = FastAPI(title="Trash Sorter API", version="1.0", lifespan=lifespan)


# --- Inference core (reused from scripts/live_predict_v2.py) ----------------


def _classify(frame) -> dict:
    """Run the model on a frame (cv2 BGR ndarray or PIL RGB image) -> result dict."""
    results = model.predict(source=frame, imgsz=IMG_SIZE, verbose=False)
    r = results[0]
    top_idx = int(r.probs.top1)
    top_conf = float(r.probs.top1conf)
    top_name = model.names[top_idx]
    probs = r.probs.data.tolist()

    return {
        "top": top_name,
        "confidence": round(top_conf, 4),
        "unsure": top_conf < MIN_CONF,
        "probabilities": {
            model.names[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
    }


# --- Endpoints -------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": list(model.names.values()) if model is not None else [],
        "camera": camera is not None and camera.isOpened(),
    }


@app.get("/predict")
def predict_from_camera():
    """Grab a fresh frame from the webcam and classify it."""
    if camera is None or not camera.isOpened():
        raise HTTPException(status_code=503, detail="Camera not available")

    with _camera_lock:
        frame = None
        for _ in range(FLUSH_FRAMES + 1):
            ok, frame = camera.read()
        if not ok or frame is None:
            raise HTTPException(
                status_code=503, detail="Failed to read frame from camera"
            )
        return _classify(frame)


@app.post("/predict")
async def predict_from_upload(file: UploadFile = File(...)):
    """Classify an uploaded image. Testing aid for machines without the webcam."""
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image"
        )
    return _classify(image)


@app.get("/wait_and_predict")
def wait_and_predict():
    """
    Watch the tray and classify frames until one class stays confidently on top
    for a moment, then return it. Used by the hadrwareCtrl orchestrator to drive
    the servo only once we're sure what was deposited.

    Logic is deliberately simple: classify each frame; when a class first clears
    CONF_TRIGGER it starts a streak, and we return once that streak holds for
    HOLD_SECONDS. To survive the normal frame-to-frame flicker of a live stream,
    the streak only resets when the top class changes or confidence falls below
    CONF_RELEASE (hysteresis) — a single dip from 84%% to 78%% does NOT reset it.
    If nothing holds within MAX_WAIT_SECONDS we return an "unsure" result.

    Returns the same shape as /predict, with an extra "waited_seconds" field.
    """
    if camera is None or not camera.isOpened():
        raise HTTPException(status_code=503, detail="Camera not available")

    start = time.time()

    with _camera_lock:
        streak_class = None
        streak_start = None
        last_result = None
        last_log = 0.0

        while time.time() - start < MAX_WAIT_SECONDS:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            result = _classify(frame)
            last_result = result
            top = result["top"]
            conf = result["confidence"]
            now = time.time()

            if (
                streak_class is not None
                and top == streak_class
                and conf >= CONF_RELEASE
            ):
                # Streak continues (tolerating dips down to CONF_RELEASE).
                if now - streak_start >= HOLD_SECONDS:
                    result["waited_seconds"] = round(now - start, 2)
                    print(
                        f"[wait] LOCKED {top} {conf*100:.1f}% "
                        f"(held {now - streak_start:.1f}s) -> rotating bin",
                        flush=True,
                    )
                    return result
            elif conf >= CONF_TRIGGER:
                # New (or first) confident class — (re)start the streak timer.
                streak_class = top
                streak_start = now
            else:
                # Not confident enough to hold anything.
                streak_class = None
                streak_start = None

            # Throttled live log so docker logs mirror the demo's on-screen read.
            if now - last_log >= LOG_EVERY_S:
                held = (now - streak_start) if streak_start is not None else 0.0
                print(f"[wait] {top} {conf*100:.1f}%  streak={held:.1f}s", flush=True)
                last_log = now

            time.sleep(FRAME_DELAY_S)

    # Nothing held above the trigger long enough — tell the caller we're unsure.
    result = last_result or {
        "top": None,
        "confidence": 0.0,
        "probabilities": {},
    }
    result["unsure"] = True
    result["waited_seconds"] = round(time.time() - start, 2)
    return result
