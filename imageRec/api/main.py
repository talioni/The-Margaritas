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

# ---- The big picture ----
# This is the "camera brain". It runs a little web server. Other programs can
# ask it questions over the network like "look at the camera and tell me what
# you see". It uses an AI model (YOLOv8) to recognise the trash.
# "API" just means: a set of web addresses (endpoints) other programs can call.

import io          # lets us treat uploaded bytes as if they were a file
import os          # for reading settings from environment variables
import threading   # for a "lock" so two requests don't fight over one camera
import time        # for timing and pauses
from contextlib import asynccontextmanager   # helper for startup/shutdown code
from pathlib import Path                      # nice way to build file paths

import cv2          # OpenCV: grabs frames (pictures) from the webcam
from fastapi import FastAPI, File, HTTPException, UploadFile  # the web framework
from PIL import Image, UnidentifiedImageError   # for opening uploaded images
from ultralytics import YOLO                    # the AI model library

# --- Configuration ---------------------------------------------------------

# Default points at where the Dockerfile bakes the weights; overridable for
# running outside the container (e.g. the dev box).
# This builds the path to our trained model file (best.pt) on disk.
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "runs"
    / "classify"
    / "train_v2-6"
    / "weights"
    / "best.pt"
)
# Use the MODEL_PATH setting if given, otherwise the default path above.
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))  # which camera (0 = first one)
IMG_SIZE = 224     # the AI wants square images 224x224 pixels
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
# These are filled in when the server starts and used by all the endpoints.
model: YOLO | None = None                 # the loaded AI model
camera: cv2.VideoCapture | None = None    # the open webcam
_camera_lock = threading.Lock()  # a single VideoCapture is not concurrency-safe


def _open_camera() -> "cv2.VideoCapture | None":
    """Open the USB webcam; return None if it isn't available (e.g. dev box)."""
    cap = cv2.VideoCapture(CAMERA_INDEX)      # try to open the camera
    if not cap.isOpened():                    # no camera found?
        return None                           # tell the caller "no camera"
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # keep only the newest frame, not a backlog
    for _ in range(WARMUP_FRAMES):            # read & throw away a few frames
        cap.read()                            # so the camera settles (brightness, focus)
    return cap


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This special function runs ONCE when the server starts (before "yield")
    # and ONCE when it shuts down (after "yield").
    global model, camera
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)              # load the trained AI model into memory
    print(f"Classes: {model.names}")     # show which classes it knows

    camera = _open_camera()              # try to open the webcam
    if camera is None:
        print(
            f"WARNING: webcam (index {CAMERA_INDEX}) not available; "
            f"GET /predict will return 503. POST /predict still works."
        )
    else:
        print("Camera ready.")

    yield                                # <-- the server runs while we're paused here

    if camera is not None:               # on shutdown, let go of the camera
        camera.release()


# Create the web application object. "lifespan" tells it to run the setup above.
app = FastAPI(title="Trash Sorter API", version="1.0", lifespan=lifespan)


# --- Inference core (reused from scripts/live_predict_v2.py) ----------------


def _classify(frame) -> dict:
    """Run the model on a frame (cv2 BGR ndarray or PIL RGB image) -> result dict."""
    # "Inference" just means: show the picture to the AI and get its answer.
    results = model.predict(source=frame, imgsz=IMG_SIZE, verbose=False)
    r = results[0]                       # we sent one image, so take the first result
    top_idx = int(r.probs.top1)          # index number of the most likely class
    top_conf = float(r.probs.top1conf)   # how sure it is about that class (0..1)
    top_name = model.names[top_idx]      # turn the index into a name like "pmd"
    probs = r.probs.data.tolist()        # the score for EVERY class, as a list

    # Package the answer up into a neat dictionary to send back.
    return {
        "top": top_name,
        "confidence": round(top_conf, 4),
        "unsure": top_conf < MIN_CONF,   # True if it's below our confidence floor
        "probabilities": {
            # a name->score breakdown for every class
            model.names[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
    }


# --- Endpoints -------------------------------------------------------------
# Each function below is one "endpoint" — a web address other programs can call.


@app.get("/health")
def health():
    # A simple "are you alive?" check. Returns whether the model and camera are ready.
    return {
        "status": "ok",
        "classes": list(model.names.values()) if model is not None else [],
        "camera": camera is not None and camera.isOpened(),
    }


@app.get("/predict")
def predict_from_camera():
    """Grab a fresh frame from the webcam and classify it."""
    if camera is None or not camera.isOpened():     # no camera available
        raise HTTPException(status_code=503, detail="Camera not available")

    with _camera_lock:                  # only one request uses the camera at a time
        frame = None
        for _ in range(FLUSH_FRAMES + 1):   # read a few frames to skip stale ones
            ok, frame = camera.read()        # grab a picture from the camera
        if not ok or frame is None:          # the grab failed
            raise HTTPException(
                status_code=503, detail="Failed to read frame from camera"
            )
        return _classify(frame)              # classify the picture and return the answer


@app.post("/predict")
async def predict_from_upload(file: UploadFile = File(...)):
    """Classify an uploaded image. Testing aid for machines without the webcam."""
    raw = await file.read()              # read the uploaded file's bytes
    try:
        # Turn the raw bytes into an image we can feed to the model.
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):    # not a real image
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image"
        )
    return _classify(image)              # classify and return the answer


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
    # In plain words: keep looking at the camera. Only answer once the SAME class
    # has stayed confidently on top for about a second. That stops us acting on a
    # one-frame fluke. If nothing settles in time, we give up and say "unsure".
    if camera is None or not camera.isOpened():
        raise HTTPException(status_code=503, detail="Camera not available")

    start = time.time()                  # remember when we started waiting

    with _camera_lock:                   # lock the camera for this whole loop
        streak_class = None              # the class currently "on a winning streak"
        streak_start = None              # when that streak began
        last_result = None               # remember the most recent classification
        last_log = 0.0                   # when we last printed a status line

        while time.time() - start < MAX_WAIT_SECONDS:   # keep going until timeout
            ok, frame = camera.read()    # grab a picture
            if not ok or frame is None:  # grab failed, try again shortly
                time.sleep(0.02)
                continue

            result = _classify(frame)    # ask the AI about this frame
            last_result = result
            top = result["top"]          # the winning class this frame
            conf = result["confidence"]  # how sure (0..1)
            now = time.time()

            if (
                streak_class is not None       # a streak is already running
                and top == streak_class        # and it's still the same class
                and conf >= CONF_RELEASE       # and confidence hasn't dropped too far
            ):
                # Streak continues (tolerating dips down to CONF_RELEASE).
                if now - streak_start >= HOLD_SECONDS:   # held long enough?
                    result["waited_seconds"] = round(now - start, 2)
                    print(
                        f"[wait] LOCKED {top} {conf*100:.1f}% "
                        f"(held {now - streak_start:.1f}s) -> rotating bin",
                        flush=True,
                    )
                    return result          # we're sure! send the answer back
            elif conf >= CONF_TRIGGER:
                # New (or first) confident class — (re)start the streak timer.
                streak_class = top
                streak_start = now
            else:
                # Not confident enough to hold anything.
                streak_class = None
                streak_start = None

            # Throttled live log so docker logs mirror the demo's on-screen read.
            if now - last_log >= LOG_EVERY_S:    # only print every half-second
                held = (now - streak_start) if streak_start is not None else 0.0
                print(f"[wait] {top} {conf*100:.1f}%  streak={held:.1f}s", flush=True)
                last_log = now

            time.sleep(FRAME_DELAY_S)        # small pause to keep CPU use low

    # Nothing held above the trigger long enough — tell the caller we're unsure.
    result = last_result or {            # use the last frame's result, or a blank one
        "top": None,
        "confidence": 0.0,
        "probabilities": {},
    }
    result["unsure"] = True              # mark it clearly as "not sure"
    result["waited_seconds"] = round(time.time() - start, 2)
    return result
