# Builds natively on the Raspberry Pi 5 (arm64). On the Pi:
#   docker compose build && docker compose up
FROM python:3.11-slim

# lgpio is needed by gpiozero to actually talk to the Pi 5 GPIO chip.
# It binds against /usr/include/lgpio.h at build time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        liblgpio-dev \
        liblgpio1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer caches across code changes.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code (small — just main.py for now).
COPY main.py /app/main.py

# Where to find the imageRec API on the docker-compose network. Override at
# `docker compose up` time if you rename the service.
ENV IMAGE_REC_URL=http://trash-api:8000

CMD ["python", "-u", "main.py"]
