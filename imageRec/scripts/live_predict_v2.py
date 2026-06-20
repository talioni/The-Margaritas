"""
Live classification with the v2 model (organic / pmd / restafval).

Controls:
    q  -> quit

Run:
    python scripts/live_predict_v2.py

    # pick a specific camera (e.g. external USB cam on index 2):
    CAMERA_INDEX=2 python scripts/live_predict_v2.py
"""

# ---- What this file does ----
# Opens your webcam in a window and, frame by frame, asks the AI "what is this?"
# It draws the answer and a confidence bar for each class right on the video.
# This is the demo you run on a laptop to SEE the model working live.

import os    # to read the CAMERA_INDEX setting
import cv2   # OpenCV: webcam capture + drawing on images + showing a window
import time  # for warm-up pauses and FPS timing
from pathlib import Path        # nice file paths
from ultralytics import YOLO    # the AI model library

# Work out where the trained model file lives, relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "classify" / "train_v2-6" / "weights" / "best.pt"

MIN_CONF = 0.50   # below this we show "unsure" instead of a class name

# Which camera to use. 0 is usually the built-in webcam; an external USB
# camera typically shows up as 1 or 2. Override without editing the file via
# the CAMERA_INDEX env var, e.g. `CAMERA_INDEX=2 python scripts/live_predict_v2.py`.
# Tip: run `ls /dev/video*` to see available cameras on Linux.
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

# Colors per class (BGR)
# OpenCV uses Blue-Green-Red order (not the usual RGB). One colour per class.
COLORS = {
    "organic": (0, 255, 0),  # green
    "pmd": (0, 200, 255),  # orange
    "restafval": (160, 160, 160),  # gray
}

print(f"Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))        # load the trained AI model
print(f"Classes: {model.names}")    # show the class names it knows

print(f"Opening camera index {CAMERA_INDEX}")
cap = cv2.VideoCapture(CAMERA_INDEX)  # open the webcam
if not cap.isOpened():                # couldn't open it?
    print(f"ERROR: camera {CAMERA_INDEX} not available")
    print("Try a different CAMERA_INDEX (run `ls /dev/video*` to list cameras).")
    exit(1)                           # stop the program with an error code

for _ in range(10):     # read & discard 10 frames so the camera settles
    cap.read()
    time.sleep(0.05)

print("Camera ready. Press q to quit.")
fps_t = time.time()     # start time, used to work out frames-per-second
frames = 0              # how many frames we've processed
fps = 0.0               # the current FPS number we display

while True:                      # main loop — runs until you press q
    ok, frame = cap.read()       # grab one frame (picture) from the camera
    if not ok:                   # grab failed, just try again
        continue

    # Show the frame to the AI and read its answer.
    results = model.predict(source=frame, imgsz=224, verbose=False)
    r = results[0]
    top_idx = int(r.probs.top1)          # index of the most likely class
    top_conf = float(r.probs.top1conf)   # confidence in that class (0..1)
    top_name = model.names[top_idx]      # name of that class
    probs = r.probs.data.tolist()        # scores for ALL classes

    # Choose the label text and colour depending on how sure we are.
    if top_conf < MIN_CONF:
        label = f"unsure ({top_name} {top_conf:.2f})"
        color = (0, 0, 255)              # red = unsure
    else:
        label = f"{top_name}  {top_conf*100:.1f}%"
        color = COLORS.get(top_name, (255, 255, 255))

    # Draw a black bar across the top, then the big label text on it.
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Draw a little bar chart at the bottom showing each class's score.
    y = frame.shape[0] - 90              # starting height near the bottom
    for i, name in model.names.items():
        bar_w = int(probs[i] * 200)      # bar length = score * 200 pixels
        cv2.rectangle(frame, (10, y), (10 + 200, y + 18), (40, 40, 40), -1)   # grey track
        cv2.rectangle(
            frame, (10, y), (10 + bar_w, y + 18), COLORS.get(name, (200, 200, 200)), -1
        )                                # coloured fill = the actual score
        cv2.putText(
            frame,
            f"{name}: {probs[i]*100:4.1f}%",   # text label like "pmd: 83.0%"
            (220, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 25                          # move down for the next bar

    # Work out and show frames-per-second every 15 frames.
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

    cv2.imshow("Trash Sorter v2 - Live", frame)   # show the finished frame in a window
    if cv2.waitKey(1) & 0xFF == ord("q"):         # if the user pressed "q"...
        break                                      # ...leave the loop

cap.release()              # let go of the camera
cv2.destroyAllWindows()    # close the window
