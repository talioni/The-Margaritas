"""
Live classification with the v2 model (organic / pmd / restafval).

Controls:
    q  -> quit

Run:
    python scripts/live_predict_v2.py

    # pick a specific camera (e.g. external USB cam on index 2):
    CAMERA_INDEX=2 python scripts/live_predict_v2.py
"""

import os
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "classify" / "train_v2-6" / "weights" / "best.pt"

MIN_CONF = 0.50

# Which camera to use. 0 is usually the built-in webcam; an external USB
# camera typically shows up as 1 or 2. Override without editing the file via
# the CAMERA_INDEX env var, e.g. `CAMERA_INDEX=2 python scripts/live_predict_v2.py`.
# Tip: run `ls /dev/video*` to see available cameras on Linux.
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

# Colors per class (BGR)
COLORS = {
    "organic": (0, 255, 0),  # green
    "pmd": (0, 200, 255),  # orange
    "restafval": (160, 160, 160),  # gray
}

print(f"Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print(f"Classes: {model.names}")

print(f"Opening camera index {CAMERA_INDEX}")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"ERROR: camera {CAMERA_INDEX} not available")
    print("Try a different CAMERA_INDEX (run `ls /dev/video*` to list cameras).")
    exit(1)

for _ in range(10):
    cap.read()
    time.sleep(0.05)

print("Camera ready. Press q to quit.")
fps_t = time.time()
frames = 0
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    results = model.predict(source=frame, imgsz=224, verbose=False)
    r = results[0]
    top_idx = int(r.probs.top1)
    top_conf = float(r.probs.top1conf)
    top_name = model.names[top_idx]
    probs = r.probs.data.tolist()

    if top_conf < MIN_CONF:
        label = f"unsure ({top_name} {top_conf:.2f})"
        color = (0, 0, 255)
    else:
        label = f"{top_name}  {top_conf*100:.1f}%"
        color = COLORS.get(top_name, (255, 255, 255))

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    y = frame.shape[0] - 90
    for i, name in model.names.items():
        bar_w = int(probs[i] * 200)
        cv2.rectangle(frame, (10, y), (10 + 200, y + 18), (40, 40, 40), -1)
        cv2.rectangle(
            frame, (10, y), (10 + bar_w, y + 18), COLORS.get(name, (200, 200, 200)), -1
        )
        cv2.putText(
            frame,
            f"{name}: {probs[i]*100:4.1f}%",
            (220, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 25

    frames += 1
    if frames % 15 == 0:
        fps = frames / (time.time() - fps_t)
    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (frame.shape[1] - 110, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Trash Sorter v2 - Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
