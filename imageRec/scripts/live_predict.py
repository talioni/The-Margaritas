"""
Live classification with the trained YOLOv8 model.

Opens the webcam, sends each frame to the model, draws the predicted
class + confidence on the frame.

Controls:
    q  -> quit

Run:
    python scripts/live_predict.py
"""

# ---- What this file does ----
# The OLD (v1) live demo, for the plastic / metal / drinkkarton model.
# Same idea as live_predict_v2.py: webcam in, AI guess drawn on the video.

import cv2   # webcam + drawing + window
import time  # warm-up + FPS timing
from pathlib import Path
from ultralytics import YOLO   # the AI model library

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "classify" / "train" / "weights" / "best.pt"

# Confidence threshold below which we say "unsure"
MIN_CONF = 0.50

# Colors per class (BGR)   (OpenCV uses Blue-Green-Red order)
COLORS = {
    "plastic":     (0, 200, 255),   # orange
    "metal":       (200, 200, 200), # gray
    "drinkkarton": (0, 255, 0),     # green
}

print(f"Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))        # load the trained model
print(f"Classes: {model.names}")

# Open webcam (AVFoundation on Mac)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)   # Mac camera backend
if not cap.isOpened():
    cap = cv2.VideoCapture(0)                      # fall back to default
if not cap.isOpened():
    print("ERROR: webcam not available")
    exit(1)

# warm up
for _ in range(10):     # discard a few frames so the camera settles
    cap.read()
    time.sleep(0.05)

print("Camera ready. Press q to quit.")
fps_t = time.time()     # start time for FPS
frames = 0
fps = 0.0

while True:
    ok, frame = cap.read()       # grab a frame
    if not ok:
        continue

    # Run inference on the frame (verbose=False to silence per-frame logs)
    results = model.predict(source=frame, imgsz=224, verbose=False)
    r = results[0]
    # Top prediction
    top_idx = int(r.probs.top1)          # index of the most likely class
    top_conf = float(r.probs.top1conf)   # confidence (0..1)
    top_name = model.names[top_idx]      # class name

    # All probs (for showing breakdown)
    probs = r.probs.data.tolist()        # score for every class

    # Decide label text + color
    if top_conf < MIN_CONF:
        label = f"unsure ({top_name} {top_conf:.2f})"
        color = (0, 0, 255)  # red
    else:
        label = f"{top_name}  {top_conf*100:.1f}%"
        color = COLORS.get(top_name, (255, 255, 255))

    # Draw big top label
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)   # black bar
    cv2.putText(frame, label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Draw probability breakdown bottom-left  (a small bar per class)
    y = frame.shape[0] - 90
    for i, name in model.names.items():
        bar_w = int(probs[i] * 200)      # bar length = score * 200 px
        cv2.rectangle(frame, (10, y), (10 + 200, y + 18), (40, 40, 40), -1)   # grey track
        cv2.rectangle(frame, (10, y), (10 + bar_w, y + 18),
                      COLORS.get(name, (200, 200, 200)), -1)                  # coloured fill
        cv2.putText(frame, f"{name}: {probs[i]*100:4.1f}%",
                    (220, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        y += 25

    # FPS counter
    frames += 1
    if frames % 15 == 0:
        fps = frames / (time.time() - fps_t)
    cv2.putText(frame, f"{fps:.1f} FPS",
                (frame.shape[1] - 110, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Trash Sorter - Live", frame)   # show the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):      # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
