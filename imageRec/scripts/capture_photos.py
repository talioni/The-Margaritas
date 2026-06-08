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

import cv2
import os
import time
from pathlib import Path

# --- Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw_captures"

CLASSES = {
    ord('p'): "plastic",
    ord('m'): "metal",
    ord('d'): "drinkkarton",
}

# Make folders if they don't exist
for cls in CLASSES.values():
    (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)

# Open default webcam — use AVFoundation backend on Mac
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("AVFoundation failed, trying default backend...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: could not open webcam. On Mac, allow camera access for Terminal/VS Code in System Settings -> Privacy -> Camera.")
    exit(1)

# Warm up — Mac cameras often need a few frames before they return real data
print("Warming up camera...")
for _ in range(10):
    cap.read()
    time.sleep(0.1)

print("Camera ready. Press p / m / d to save, q to quit.")
counts = {cls: len(list((RAW_DIR / cls).glob("*.jpg"))) for cls in CLASSES.values()}
print(f"Existing counts: {counts}")

fail_count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        fail_count += 1
        if fail_count > 30:
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

    cv2.imshow("Capture - Trash Sorter", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key in CLASSES:
        cls = CLASSES[key]
        ts = int(time.time() * 1000)
        path = RAW_DIR / cls / f"{cls}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        counts[cls] += 1
        print(f"saved {path.name}  ({cls} total: {counts[cls]})")

cap.release()
cv2.destroyAllWindows()
print("Done. Final counts:", counts)
