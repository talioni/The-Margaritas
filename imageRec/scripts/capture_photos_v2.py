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

# ---- What this file does ----
# Opens the webcam so YOU can build training photos. Hold up a piece of trash,
# press a key (o / m / r) to save that frame into the matching class folder.
# More varied photos = a smarter AI later.

import cv2   # webcam capture, drawing, showing the window
import os    # (imported for completeness; paths handled via pathlib below)
import time  # for warm-up and unique filenames
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset_v2" / "raw_captures"   # where photos are saved

# Which keyboard key saves into which class folder.
# ord('o') is just the number the computer uses for the letter "o".
CLASSES = {
    ord('o'): "organic",
    ord('m'): "pmd",
    ord('r'): "restafval",
}

for cls in CLASSES.values():          # make a folder for each class if missing
    (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # open camera (Mac backend)
if not cap.isOpened():                            # if that failed...
    cap = cv2.VideoCapture(0)                      # ...try the normal way
if not cap.isOpened():                            # still no camera = give up
    print("ERROR: webcam not available. Check System Settings -> Privacy -> Camera.")
    exit(1)

print("Warming up camera...")
for _ in range(10):       # discard a few frames so the image settles
    cap.read()
    time.sleep(0.05)

print("Ready. Press o / m / r to save, q to quit.")
# Count how many photos already exist in each folder so the counter is correct.
counts = {cls: len(list((RAW_DIR / cls).glob("*.jpg"))) for cls in CLASSES.values()}
print(f"Existing counts: {counts}")

fail_count = 0            # how many frame-grabs failed in a row
while True:
    ok, frame = cap.read()    # grab a frame
    if not ok:                # grab failed
        fail_count += 1
        if fail_count > 30:   # too many failures = camera probably unplugged
            print("Camera disconnected. Exiting.")
            break
        time.sleep(0.05)
        continue
    fail_count = 0            # success, reset the failure counter

    # Draw the running photo counts in the corner.
    y = 30
    for cls, n in counts.items():
        cv2.putText(frame, f"{cls}: {n}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
    # Draw the key reminder along the bottom.
    cv2.putText(frame,
                "o=organic  m=pmd  r=restafval  q=quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Capture v2 - Trash Sorter", frame)   # show the live view
    key = cv2.waitKey(1) & 0xFF                       # read any key press

    if key == ord('q'):       # q = quit
        break
    if key in CLASSES:        # a class key (o/m/r) was pressed
        cls = CLASSES[key]                    # which class folder
        ts = int(time.time() * 1000)          # a unique number (milliseconds now)
        path = RAW_DIR / cls / f"{cls}_{ts}.jpg"   # build the file name
        cv2.imwrite(str(path), frame)         # save the current frame as a photo
        counts[cls] += 1                      # bump the counter
        print(f"saved {path.name}  ({cls} total: {counts[cls]})")

cap.release()              # release the camera
cv2.destroyAllWindows()    # close the window
print("Done. Final counts:", counts)
