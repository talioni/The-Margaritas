# Builds natively on the Raspberry Pi 5 (arm64). On the Pi:
#   docker compose build && docker compose up
FROM python:3.11-slim

# Runtime libs ultralytics/opencv-headless need. No libgl1 thanks to the
# headless OpenCV wheel; libglib2.0-0 + libgomp1 are still required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer caches across code changes.
# ultralytics depends on the GUI build of opencv-python, which pip installs
# alongside our headless pin. Their native cv2 libs collide and the GUI one
# (needs libxcb/libGL — absent from slim) wins. Strip both and reinstall only
# headless so the container has a clean, X11-free cv2.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless

# Bake in the v2 model weights and the API code.
COPY runs/classify/train_v2-6/weights/best.pt /app/model/best.pt
COPY api/ /app/api/

ENV MODEL_PATH=/app/model/best.pt \
    CAMERA_INDEX=0 \
    YOLO_AUTOINSTALL=false \
    YOLO_OFFLINE=true

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
