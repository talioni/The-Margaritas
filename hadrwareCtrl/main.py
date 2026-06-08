"""
Hardware-control orchestrator for the trash sorter.

This service is the brain of the appliance:
    1. Calls the imageRec API (trash-api:8000) /wait_and_predict, which
       blocks until trash has been placed on the tray.
    2. Looks up the bin angle for the predicted class.
    3. Drives the servo to that angle, holds while the item drops, then
       returns to neutral.
    4. Loops forever.

The imageRec service does the camera reading and ML inference; we just
poll it. That's why this container does NOT expose an HTTP API of its own.

GPIO pin numbers are intentionally left as None — fill them in once the
wiring is finalised. See PIN_LAYOUT below.

Run (inside the container):
    python main.py
"""

import logging
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# PIN LAYOUT — fill in once the wiring is finalised.
# ---------------------------------------------------------------------------
# All pin numbers are BCM (the gpiozero default).
# Leave as None to keep the servo in "dry-run" mode (logs only, no GPIO).
SERVO_PIN: int | None = None

# ---------------------------------------------------------------------------
# Servo + sort behaviour
# ---------------------------------------------------------------------------
SERVO_MIN_ANGLE      = -90
SERVO_MAX_ANGLE      =  90
SERVO_MIN_PULSE_S    = 0.0005   # tune for the specific servo if it jitters
SERVO_MAX_PULSE_S    = 0.0025
SERVO_SETTLE_S       = 0.4      # wait after each move before considering it done

NEUTRAL_ANGLE        = 0
HOLD_AFTER_SORT_S    = 3.0      # how long the flap stays tilted to let the item drop
COOLDOWN_AFTER_SORT_S = 1.5     # short pause before asking imageRec for the next item

# Predicted-class -> servo angle. Symmetric 60° spacing as specified.
CLASS_TO_ANGLE: dict[str, int] = {
    "organic":    -60,
    "pmd":          0,
    "restafval":   60,
}

# ---------------------------------------------------------------------------
# imageRec endpoint
# ---------------------------------------------------------------------------
# Inside docker-compose the imageRec container is reachable as "trash-api".
# Override with IMAGE_REC_URL env var for local testing.
IMAGE_REC_URL = os.getenv("IMAGE_REC_URL", "http://trash-api:8000")
# /wait_and_predict can block for up to MAX_WAIT_SECONDS on the API side
# (default 120s); give the HTTP request a bit more headroom.
WAIT_PREDICT_TIMEOUT_S = float(os.getenv("WAIT_PREDICT_TIMEOUT_S", "180"))
# If imageRec is still booting / temporarily down, back off and retry.
API_RETRY_BACKOFF_S = 3.0

# Below this confidence the orchestrator refuses to act (keeps servo at neutral
# so a wrong guess can't dump trash in the wrong bin).
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))

# ---------------------------------------------------------------------------
# Servo abstraction — real on the Pi, dry-run elsewhere or when SERVO_PIN
# hasn't been set yet.
# ---------------------------------------------------------------------------
class _DryRunServo:
    """Stand-in used when SERVO_PIN is None — logs moves without touching GPIO."""

    def __init__(self) -> None:
        self.angle = NEUTRAL_ANGLE

    def move_to(self, angle: int) -> None:
        logging.info("[dry-run servo] moving to %d°", angle)
        self.angle = angle

    def close(self) -> None:
        logging.info("[dry-run servo] closed")


class _RealServo:
    """Real servo driven by gpiozero. Only imported if SERVO_PIN is set."""

    def __init__(self, pin: int) -> None:
        # Imported here so the dry-run path doesn't need gpiozero installed.
        from gpiozero import AngularServo, Device
        from gpiozero.pins.lgpio import LGPIOFactory

        Device.pin_factory = LGPIOFactory()
        self._servo = AngularServo(
            pin,
            min_angle=SERVO_MIN_ANGLE,
            max_angle=SERVO_MAX_ANGLE,
            min_pulse_width=SERVO_MIN_PULSE_S,
            max_pulse_width=SERVO_MAX_PULSE_S,
        )

    def move_to(self, angle: int) -> None:
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, angle))
        logging.info("servo -> %d°", angle)
        self._servo.angle = angle
        time.sleep(SERVO_SETTLE_S)

    def close(self) -> None:
        self._servo.angle = NEUTRAL_ANGLE
        time.sleep(SERVO_SETTLE_S)
        self._servo.close()


def _make_servo():
    if SERVO_PIN is None:
        logging.warning(
            "SERVO_PIN is None — running in dry-run mode (no GPIO output). "
            "Fill it in in hadrwareCtrl/main.py once wiring is finalised."
        )
        return _DryRunServo()
    return _RealServo(SERVO_PIN)


# ---------------------------------------------------------------------------
# imageRec polling
# ---------------------------------------------------------------------------
def _wait_for_prediction() -> dict | None:
    """Block on imageRec's /wait_and_predict; retry on transient errors."""
    url = f"{IMAGE_REC_URL}/wait_and_predict"
    while True:
        try:
            resp = requests.get(url, timeout=WAIT_PREDICT_TIMEOUT_S)
        except requests.exceptions.RequestException as e:
            logging.warning("imageRec not reachable (%s) — retrying in %.1fs",
                            e, API_RETRY_BACKOFF_S)
            time.sleep(API_RETRY_BACKOFF_S)
            continue

        # 408 = imageRec timed out waiting for an item. Loop back immediately.
        if resp.status_code == 408:
            logging.info("imageRec saw no item — looping.")
            continue
        # 503 = camera not available / not initialised yet
        if resp.status_code == 503:
            logging.warning("imageRec reports camera unavailable — retrying in %.1fs",
                            API_RETRY_BACKOFF_S)
            time.sleep(API_RETRY_BACKOFF_S)
            continue
        if resp.status_code != 200:
            logging.warning("imageRec returned HTTP %d: %s — retrying",
                            resp.status_code, resp.text[:200])
            time.sleep(API_RETRY_BACKOFF_S)
            continue

        return resp.json()


def _wait_for_health() -> None:
    """Block on imageRec /health at boot so we don't spam errors before it's up."""
    url = f"{IMAGE_REC_URL}/health"
    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                logging.info("imageRec is up. classes=%s", r.json().get("classes"))
                return
        except requests.exceptions.RequestException:
            pass
        logging.info("waiting for imageRec at %s ...", IMAGE_REC_URL)
        time.sleep(API_RETRY_BACKOFF_S)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("hadrwareCtrl starting. IMAGE_REC_URL=%s", IMAGE_REC_URL)
    logging.info("class -> angle map: %s", CLASS_TO_ANGLE)

    _wait_for_health()
    servo = _make_servo()
    servo.move_to(NEUTRAL_ANGLE)

    try:
        while True:
            result = _wait_for_prediction()
            if result is None:
                continue

            cls   = result.get("top")
            conf  = float(result.get("confidence", 0.0))
            waited = result.get("waited_seconds", "?")
            logging.info("prediction: %s (%.1f%%) after %ss",
                         cls, conf * 100, waited)

            if conf < MIN_CONFIDENCE:
                logging.info("confidence below %.2f — not sorting.", MIN_CONFIDENCE)
            elif cls not in CLASS_TO_ANGLE:
                logging.warning("unknown class %r — not sorting.", cls)
            else:
                target = CLASS_TO_ANGLE[cls]
                servo.move_to(target)
                time.sleep(HOLD_AFTER_SORT_S)
                servo.move_to(NEUTRAL_ANGLE)

            time.sleep(COOLDOWN_AFTER_SORT_S)
    except KeyboardInterrupt:
        logging.info("interrupted by user")
    finally:
        servo.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
