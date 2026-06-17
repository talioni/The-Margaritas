"""
Webcam capture tool for building our trash dataset.

Usage:
    python scripts/capture_photos.py

Controls:
    p  -> save current frame as PLASTIC sample
    m  -> save current frame as METAL sample
    d  -> save current frame as DRINKKARTON sample
    q  -> quit

Goal: ~30-50 photos per class. Vary lighting, angle, distance,
and background. Hold ONE item in frame at a time.

All saved images go to: dataset/raw_captures/<class>/
We'll label them with bounding boxes later using LabelImg.
"""

# ---- What this file does ----
# This is the OLD (v1) photo-capture tool, from when the project used 3 classes
# called plastic / metal / drinkkarton. The newer version is capture_photos_v2.py.
# It opens the webcam and saves a frame to a class folder when you press a key.

import cv2   # webcam + window
import os    # (kept for completeness)
import time  # warm-up + unique filenames
from pathlib import Path

# --- Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw_captures"   # where photos are saved

# Key -> class folder. ord('p') is the number the computer uses for "p".
CLASSES = {
    ord('p'): "plastic",
    ord('m'): "metal",
    ord('d'): "drinkkarton",
}

# Make folders if they don't exist
for cls in CLASSES.values():
    (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)

# Open default webcam — use AVFoundation backend on Mac
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)   # Mac camera backend
if not cap.isOpened():
    print("AVFoundation failed, trying default backend...")
    cap = cv2.VideoCapture(0)                      # fall back to the normal way

if not cap.isOpened():                             # still nothing = stop
    print("ERROR: could not open webcam. On Mac, allow camera access for Terminal/VS Code in System Settings -> Privacy -> Camera.")
    exit(1)

# Warm up — Mac cameras often need a few frames before they return real data
print("Warming up camera...")
for _ in range(10):
    cap.read()
    time.sleep(0.1)

print("Camera ready. Press p / m / d to save, q to quit.")
# Count existing photos so the on-screen counter starts at the right number.
counts = {cls: len(list((RAW_DIR / cls).glob("*.jpg"))) for cls in CLASSES.values()}
print(f"Existing counts: {counts}")

fail_count = 0
while True:
    ok, frame = cap.read()        # grab a frame
    if not ok:                    # grab failed
        fail_count += 1
        if fail_count > 30:       # too many in a row = camera gone
            print("Too many failed frames — camera disconnected. Exiting.")
            break
        time.sleep(0.05)
        continue
    fail_count = 0

    # Show counts on the frame
    y = 30
    for cls, n in counts.items():
        cv2.putText(frame, f"{cls}: {n}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
    cv2.putText(frame, "p=plastic  m=metal  d=drinkkarton  q=quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Capture - Trash Sorter", frame)   # show the live view
    key = cv2.waitKey(1) & 0xFF                    # read a key press

    if key == ord('q'):       # quit
        break
    if key in CLASSES:        # a class key was pressed -> save the photo
        cls = CLASSES[key]
        ts = int(time.time() * 1000)              # unique number for the filename
        path = RAW_DIR / cls / f"{cls}_{ts}.jpg"
        cv2.imwrite(str(path), frame)             # save the frame as a .jpg
        counts[cls] += 1
        print(f"saved {path.name}  ({cls} total: {counts[cls]})")

cap.release()
cv2.destroyAllWindows()
print("Done. Final counts:", counts)
