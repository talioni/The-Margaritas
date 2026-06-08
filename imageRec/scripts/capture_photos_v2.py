"""
v2 webcam capture — for the new 3-class household sort:

    o -> ORGANIC      (food scraps, peels, plants)
    m -> PMD          (plastic + metal + drink cartons)
    r -> RESTAFVAL    (residual / general waste: paper towels, broken
                       things, mixed dirty trash, hygiene items)
    q -> quit

Photos go to: dataset_v2/raw_captures/<class>/

Goal: ~40-60 photos per class. Vary lighting, angle, distance, background.
One item in frame at a time. Different items per class help a lot.

Run:
    python scripts/capture_photos_v2.py
"""

import cv2
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset_v2" / "raw_captures"

CLASSES = {
    ord('o'): "organic",
    ord('m'): "pmd",
    ord('r'): "restafval",
}

for cls in CLASSES.values():
    (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: webcam not available. Check System Settings -> Privacy -> Camera.")
    exit(1)

print("Warming up camera...")
for _ in range(10):
    cap.read()
    time.sleep(0.05)

print("Ready. Press o / m / r to save, q to quit.")
counts = {cls: len(list((RAW_DIR / cls).glob("*.jpg"))) for cls in CLASSES.values()}
print(f"Existing counts: {counts}")

fail_count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        fail_count += 1
        if fail_count > 30:
            print("Camera disconnected. Exiting.")
            break
        time.sleep(0.05)
        continue
    fail_count = 0

    y = 30
    for cls, n in counts.items():
        cv2.putText(frame, f"{cls}: {n}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
    cv2.putText(frame,
                "o=organic  m=pmd  r=restafval  q=quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Capture v2 - Trash Sorter", frame)
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
