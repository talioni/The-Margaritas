"""
Hardware-control orchestrator for the trash sorter.

This service is the brain of the appliance:
    1. Calls the imageRec API (trash-api:8000) /wait_and_predict, which
       blocks until trash has been placed on the tray.
    2. Looks up the bin position for the predicted class.
    3. Drives the stepper to rotate the bin to that position, then opens
       the lid servo, holds while the item drops, closes the lid and
       returns the bin to home.
    4. Loops forever.

The stepper itself lives in tb6600.py (raw lgpio, no gpiozero). Debug the
motor with tb6600_test.py BEFORE running this — it has a staged diagnostic
and a wiring/symptom guide in its docstring.

With GPIO_ENABLED unset/false, both the stepper and the lid servo run in
"dry-run" mode (logs only, no GPIO) so the logic can be tested without the
real hardware.

Run (inside the container):
    python main.py
"""

import logging
import os
import sys
import time

import requests

# ---------------------------------------------------------------------------
# PIN LAYOUT (BCM numbering)
# ---------------------------------------------------------------------------
STEP_PIN: int | None = 20   # TB6600 PUL-  (PUL+ -> 3.3 V, physical pin 1)
DIR_PIN: int | None = 21    # TB6600 DIR-  (DIR+ -> 3.3 V)
LID_SERVO_PIN: int | None = 25

# Master switch (set by docker-compose / .env). When false, everything stays in
# dry-run mode no matter what the pins above are set to.
GPIO_ENABLED = os.getenv("GPIO_ENABLED", "false").strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Stepper behaviour (bin positioning)
# ---------------------------------------------------------------------------
MICROSTEP       = 8                 # MUST match the TB6600 DIP switches S4-S6
STEPS_PER_REV   = 200 * MICROSTEP   # 1.8°/step motor
STEP_PULSE_S    = 0.0005            # half-period of a STEP pulse (1 kHz rate);
                                    # raise this if the motor stalls/buzzes
HOME_ANGLE      = 0

# Predicted-class -> bin angle the stepper rotates to. Symmetric 60° spacing.
CLASS_TO_ANGLE: dict[str, int] = {
    "organic":    -60,
    "pmd":          0,
    "restafval":   60,
}

# ---------------------------------------------------------------------------
# Lid servo behaviour
# ---------------------------------------------------------------------------
SERVO_MIN_ANGLE   = -90
SERVO_MAX_ANGLE   =  90
SERVO_MIN_PULSE_S = 0.0005  # tune for the specific servo if it jitters
SERVO_MAX_PULSE_S = 0.0025
SERVO_SETTLE_S    = 0.4     # wait after each move before considering it done

LID_OPEN_ANGLE    = 90
LID_CLOSED_ANGLE  = 0

HOLD_LID_OPEN_S      = 3.0  # how long the lid stays open to let the item drop
COOLDOWN_AFTER_SORT_S = 1.5 # short pause before asking imageRec for the next item

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

# Below this confidence the orchestrator refuses to act (keeps the bin at home
# so a wrong guess can't dump trash in the wrong bin).
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))


# ---------------------------------------------------------------------------
# Stepper abstraction — real on the Pi, dry-run otherwise.
# ---------------------------------------------------------------------------
class _DryRunStepper:
    """Stand-in used when GPIO is disabled — logs moves, no GPIO."""

    def __init__(self) -> None:
        self.angle = HOME_ANGLE

    def move_to(self, angle: int) -> None:
        logging.info("[dry-run stepper] rotating to %d°", angle)
        self.angle = angle

    def close(self) -> None:
        logging.info("[dry-run stepper] closed")


class _RealStepper:
    """Thin wrapper around tb6600.TB6600 with the orchestrator's config."""

    def __init__(self, step_pin: int, dir_pin: int) -> None:
        from tb6600 import TB6600

        self._drv = TB6600(step_pin, dir_pin, STEPS_PER_REV,
                           step_pulse_s=STEP_PULSE_S)

    def move_to(self, angle: int) -> None:
        logging.info("stepper -> %d°", angle)
        self._drv.move_to_angle(angle)

    def close(self) -> None:
        self.move_to(HOME_ANGLE)
        self._drv.close()


# ---------------------------------------------------------------------------
# Lid servo abstraction — real on the Pi, dry-run when the pin isn't set.
# ---------------------------------------------------------------------------
class _DryRunServo:
    """Stand-in used when LID_SERVO_PIN is None — logs moves, no GPIO."""

    def open(self) -> None:
        logging.info("[dry-run servo] lid open")

    def close(self) -> None:
        logging.info("[dry-run servo] lid closed")

    def release(self) -> None:
        logging.info("[dry-run servo] released")


class _RealServo:
    """Lid servo via raw lgpio (same approach as TB6600 — no gpiozero)."""

    def __init__(self, pin: int) -> None:
        import lgpio
        from tb6600 import open_gpiochip

        self._lgpio = lgpio
        self._pin = pin
        self._handle, chip, label = open_gpiochip()
        logging.info("servo: gpiochip%d (%s) GPIO%d", chip, label, pin)
        lgpio.gpio_claim_output(self._handle, pin, 0)
        self._move(LID_CLOSED_ANGLE)

    def _move(self, angle: float) -> None:
        frac = (angle - SERVO_MIN_ANGLE) / (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)
        pulse_us = int(
            (SERVO_MIN_PULSE_S + frac * (SERVO_MAX_PULSE_S - SERVO_MIN_PULSE_S))
            * 1_000_000
        )
        self._lgpio.tx_servo(self._handle, self._pin, pulse_us)
        time.sleep(SERVO_SETTLE_S)
        # Stop the PWM signal once the servo has reached position.
        # Continuous software PWM on a non-RT kernel has microsecond-level
        # jitter that makes cheap servos twitch constantly trying to correct.
        self._lgpio.tx_servo(self._handle, self._pin, 0)

    def open(self) -> None:
        logging.info("servo -> lid open (%d°)", LID_OPEN_ANGLE)
        self._move(LID_OPEN_ANGLE)

    def close(self) -> None:
        logging.info("servo -> lid closed (%d°)", LID_CLOSED_ANGLE)
        self._move(LID_CLOSED_ANGLE)

    def release(self) -> None:
        self.close()
        self._lgpio.tx_servo(self._handle, self._pin, 0)
        try:
            self._lgpio.gpio_free(self._handle, self._pin)
        finally:
            self._lgpio.gpiochip_close(self._handle)


def _make_stepper():
    if not GPIO_ENABLED or STEP_PIN is None or DIR_PIN is None:
        logging.warning(
            "GPIO disabled or STEP_PIN/DIR_PIN not set — stepper in dry-run "
            "mode (no GPIO). Set GPIO_ENABLED=true to drive the real motor."
        )
        return _DryRunStepper()
    return _RealStepper(STEP_PIN, DIR_PIN)


def _make_servo():
    if not GPIO_ENABLED or LID_SERVO_PIN is None:
        logging.warning(
            "GPIO disabled or LID_SERVO_PIN not set — lid servo in dry-run "
            "mode (no GPIO). Set GPIO_ENABLED=true to drive the real servo."
        )
        return _DryRunServo()
    return _RealServo(LID_SERVO_PIN)


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
                body = r.json()
                logging.info("imageRec is up. classes=%s", body.get("classes"))
                if not body.get("camera"):
                    logging.warning(
                        "imageRec reports NO camera — /wait_and_predict will "
                        "return 503 until the webcam is attached."
                    )
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
    stepper = _make_stepper()
    servo = _make_servo()
    stepper.move_to(HOME_ANGLE)

    try:
        while True:
            result = _wait_for_prediction()
            if result is None:
                continue

            cls    = result.get("top")
            conf   = float(result.get("confidence", 0.0))
            unsure = bool(result.get("unsure", False))
            waited = result.get("waited_seconds", "?")
            logging.info("prediction: %s (%.1f%%) unsure=%s after %ss",
                         cls, conf * 100, unsure, waited)

            # "unsure" covers /wait_and_predict timing out: the API then returns
            # its LAST frame's result, whose confidence can still read high even
            # though nothing held the trigger — never sort on it.
            if unsure or cls is None:
                logging.info("imageRec is unsure — not sorting.")
            elif conf < MIN_CONFIDENCE:
                logging.info("confidence below %.2f — not sorting.", MIN_CONFIDENCE)
            elif cls not in CLASS_TO_ANGLE:
                logging.warning("unknown class %r — not sorting.", cls)
            else:
                stepper.move_to(CLASS_TO_ANGLE[cls])
                servo.open()
                time.sleep(HOLD_LID_OPEN_S)
                servo.close()
                stepper.move_to(HOME_ANGLE)

            time.sleep(COOLDOWN_AFTER_SORT_S)
    except KeyboardInterrupt:
        logging.info("interrupted by user")
    finally:
        servo.release()
        stepper.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
