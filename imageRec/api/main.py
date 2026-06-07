"""
Local HTTP API around the v2 trash classifier (organic / pmd / restafval).

The container owns the USB webcam (passed in as /dev/video0) and runs inference
on demand. The future main program calls these endpoints over the local Docker
network and drives the hardware.

Endpoints:
    GET  /health        -> liveness + camera/model status
    GET  /predict       -> grab a fresh frame from the webcam and classify it
    POST /predict       -> classify an uploaded image (testing aid, no camera)

Run (locally, outside Docker):
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# --- Configuration ---------------------------------------------------------

# Default points at where the Dockerfile bakes the weights; overridable for
# running outside the container (e.g. the dev box).
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "runs" / "classify" / "train_v2-6" / "weights" / "best.pt"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
IMG_SIZE = 224
MIN_CONF = 0.50           # below this the top prediction is flagged "unsure"
WARMUP_FRAMES = 10        # let the webcam stabilise on open
FLUSH_FRAMES = 4          # drop buffered frames before reading a current one

# --- Shared state ----------------------------------------------------------

model: YOLO | None = None
camera: cv2.VideoCapture | None = None
_camera_lock = threading.Lock()   # a single VideoCapture is not concurrency-safe


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
        print(f"WARNING: webcam (index {CAMERA_INDEX}) not available; "
              f"GET /predict will return 503. POST /predict still works.")
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
            raise HTTPException(status_code=503, detail="Failed to read frame from camera")
        return _classify(frame)


@app.post("/predict")
async def predict_from_upload(file: UploadFile = File(...)):
    """Classify an uploaded image. Testing aid for machines without the webcam."""
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    return _classify(image)
