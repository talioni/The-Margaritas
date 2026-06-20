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

# ---- The big picture ----
# This program is the "boss". It keeps asking the camera program "what did you
# see?" and, when the camera is sure, it turns the bin to the right slot and
# opens a little lid so the trash drops in. Then it goes back and waits again.

import logging   # for printing tidy status messages with timestamps
import os        # for reading settings from environment variables
import sys        # used to give the operating system a clean exit code
import time      # for pauses (sleep) between actions

import requests  # lets us call the camera program over the network (HTTP)

# ---------------------------------------------------------------------------
# PIN LAYOUT (BCM numbering)
# ---------------------------------------------------------------------------
# These say which Raspberry Pi pins are wired to the motor driver and servo.
STEP_PIN: int | None = 20  # TB6600 PUL-  (PUL+ -> 3.3 V, physical pin 1)
DIR_PIN: int | None = 21  # TB6600 DIR-  (DIR+ -> 3.3 V)
LID_SERVO_PIN: int | None = 25  # the pin that controls the lid servo

# Master switch (set by docker-compose / .env). When false, everything stays in
# dry-run mode no matter what the pins above are set to.
# This reads the GPIO_ENABLED setting and turns it into a simple True/False.
GPIO_ENABLED = os.getenv("GPIO_ENABLED", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)

# ---------------------------------------------------------------------------
# Stepper behaviour (bin positioning)
# ---------------------------------------------------------------------------
MICROSTEP = 4  # MUST match the TB6600 DIP switches S4-S6
STEPS_PER_REV = 200 * MICROSTEP  # 1.8°/step motor -> 200 full steps per turn
STEP_PULSE_S = 0.002  # half-period of a STEP pulse (1 kHz rate);
# raise this if the motor stalls/buzzes
HOME_ANGLE = 0  # the "resting" position the bin returns to

# Predicted-class -> bin angle the stepper rotates to. Symmetric 60° spacing.
# So if the camera says "organic", we rotate the bin to -60 degrees, etc.
CLASS_TO_ANGLE: dict[str, int] = {
    "organic": -60,
    "pmd": 0,
    "restafval": 60,
}

# ---------------------------------------------------------------------------
# Lid servo behaviour
# ---------------------------------------------------------------------------
# A servo is a small motor you tell to go to an exact angle. These settings
# describe its range and timing.
SERVO_MIN_ANGLE = -90
SERVO_MAX_ANGLE = 90
SERVO_MIN_PULSE_S = 0.0005  # tune for the specific servo if it jitters
SERVO_MAX_PULSE_S = 0.0025
SERVO_SETTLE_S = 0.4  # wait after each move before considering it done

LID_OPEN_ANGLE = 90   # angle that counts as "lid open"
LID_CLOSED_ANGLE = 0  # angle that counts as "lid closed"

HOLD_LID_OPEN_S = 3.0  # how long the lid stays open to let the item drop
COOLDOWN_AFTER_SORT_S = 1.5  # short pause before asking imageRec for the next item

# ---------------------------------------------------------------------------
# imageRec endpoint
# ---------------------------------------------------------------------------
# Inside docker-compose the imageRec container is reachable as "trash-api".
# Override with IMAGE_REC_URL env var for local testing.
# This is the web address of the camera program we will talk to.
IMAGE_REC_URL = os.getenv("IMAGE_REC_URL", "http://trash-api:8000")
# /wait_and_predict can block for up to MAX_WAIT_SECONDS on the API side
# (default 120s); give the HTTP request a bit more headroom.
WAIT_PREDICT_TIMEOUT_S = float(os.getenv("WAIT_PREDICT_TIMEOUT_S", "180"))
# If imageRec is still booting / temporarily down, back off and retry.
API_RETRY_BACKOFF_S = 3.0  # how long to wait before trying again after an error

# Below this confidence the orchestrator refuses to act (keeps the bin at home
# so a wrong guess can't dump trash in the wrong bin).
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))


# ---------------------------------------------------------------------------
# Stepper abstraction — real on the Pi, dry-run otherwise.
# ---------------------------------------------------------------------------
# We make TWO versions of the stepper: a fake one (just prints messages) for
# testing on a laptop, and a real one that drives the motor on the Pi. The rest
# of the program uses them the same way, so it doesn't care which it got.
class _DryRunStepper:
    """Stand-in used when GPIO is disabled — logs moves, no GPIO."""

    def __init__(self) -> None:
        self.angle = HOME_ANGLE          # pretend we're at the home position

    def move_to(self, angle: int) -> None:
        logging.info("[dry-run stepper] rotating to %d°", angle)  # just print it
        self.angle = angle

    def close(self) -> None:
        logging.info("[dry-run stepper] closed")


class _RealStepper:
    """Thin wrapper around tb6600.TB6600 with the orchestrator's config."""

    def __init__(self, step_pin: int, dir_pin: int) -> None:
        from tb6600 import TB6600       # import the real driver only when needed
        # Create the real driver with our pins and settings.
        self._drv = TB6600(step_pin, dir_pin, STEPS_PER_REV, step_pulse_s=STEP_PULSE_S)

    def move_to(self, angle: int) -> None:
        logging.info("stepper -> %d°", angle)
        self._drv.move_to_angle(angle)  # actually turn the motor

    def close(self) -> None:
        self.move_to(HOME_ANGLE)        # go back home first
        self._drv.close()               # then release the pins


# ---------------------------------------------------------------------------
# Lid servo abstraction — real on the Pi, dry-run when the pin isn't set.
# ---------------------------------------------------------------------------
# Same idea as the stepper: a fake servo for testing, a real one for the Pi.
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
        import lgpio                          # the Pi pin library
        from tb6600 import open_gpiochip      # reuse the chip-finder from tb6600.py

        self._lgpio = lgpio
        self._pin = pin
        self._handle, chip, label = open_gpiochip()   # open the pin controller
        logging.info("servo: gpiochip%d (%s) GPIO%d", chip, label, pin)
        lgpio.gpio_claim_output(self._handle, pin, 0)  # take control of the pin
        self._move(LID_CLOSED_ANGLE)                   # start with the lid closed

    def _move(self, angle: float) -> None:
        # Turn the angle we want (-90..90) into a fraction from 0.0 to 1.0.
        frac = (angle - SERVO_MIN_ANGLE) / (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)
        # Servos are controlled by a pulse width. Work out the right pulse
        # length for this angle, in microseconds.
        pulse_us = int(
            (SERVO_MIN_PULSE_S + frac * (SERVO_MAX_PULSE_S - SERVO_MIN_PULSE_S))
            * 1_000_000
        )
        self._lgpio.tx_servo(self._handle, self._pin, pulse_us)  # send the pulse
        time.sleep(SERVO_SETTLE_S)                               # wait for it to arrive
        # Stop the PWM signal once the servo has reached position.
        # Continuous software PWM on a non-RT kernel has microsecond-level
        # jitter that makes cheap servos twitch constantly trying to correct.
        self._lgpio.tx_servo(self._handle, self._pin, 0)        # 0 = stop signalling

    def open(self) -> None:
        logging.info("servo -> lid open (%d°)", LID_OPEN_ANGLE)
        self._move(LID_OPEN_ANGLE)      # move to the open angle

    def close(self) -> None:
        logging.info("servo -> lid closed (%d°)", LID_CLOSED_ANGLE)
        self._move(LID_CLOSED_ANGLE)    # move to the closed angle

    def release(self) -> None:
        # Final cleanup: close the lid and let go of the pin and chip.
        self.close()
        self._lgpio.tx_servo(self._handle, self._pin, 0)
        try:
            self._lgpio.gpio_free(self._handle, self._pin)
        finally:
            self._lgpio.gpiochip_close(self._handle)


def _make_stepper():
    # Decide which stepper to build: fake or real.
    if not GPIO_ENABLED or STEP_PIN is None or DIR_PIN is None:
        logging.warning(
            "GPIO disabled or STEP_PIN/DIR_PIN not set — stepper in dry-run "
            "mode (no GPIO). Set GPIO_ENABLED=true to drive the real motor."
        )
        return _DryRunStepper()         # testing mode
    return _RealStepper(STEP_PIN, DIR_PIN)  # real hardware mode


def _make_servo():
    # Decide which servo to build: fake or real.
    if not GPIO_ENABLED or LID_SERVO_PIN is None:
        logging.warning(
            "GPIO disabled or LID_SERVO_PIN not set — lid servo in dry-run "
            "mode (no GPIO). Set GPIO_ENABLED=true to drive the real servo."
        )
        return _DryRunServo()           # testing mode
    return _RealServo(LID_SERVO_PIN)    # real hardware mode


# ---------------------------------------------------------------------------
# imageRec polling
# ---------------------------------------------------------------------------
def _wait_for_prediction() -> dict | None:
    """Block on imageRec's /wait_and_predict; retry on transient errors."""
    # Keep asking the camera program "what did you see?" until we get a good answer.
    url = f"{IMAGE_REC_URL}/wait_and_predict"
    while True:                          # loop forever until we return something
        try:
            resp = requests.get(url, timeout=WAIT_PREDICT_TIMEOUT_S)  # ask the camera
        except requests.exceptions.RequestException as e:
            # The camera program couldn't be reached (maybe still starting up).
            logging.warning(
                "imageRec not reachable (%s) — retrying in %.1fs",
                e,
                API_RETRY_BACKOFF_S,
            )
            time.sleep(API_RETRY_BACKOFF_S)   # wait a moment
            continue                          # then try again

        # 503 = camera not available / not initialised yet
        if resp.status_code == 503:
            logging.warning(
                "imageRec reports camera unavailable — retrying in %.1fs",
                API_RETRY_BACKOFF_S,
            )
            time.sleep(API_RETRY_BACKOFF_S)
            continue
        if resp.status_code != 200:           # any other error code = try again
            logging.warning(
                "imageRec returned HTTP %d: %s — retrying",
                resp.status_code,
                resp.text[:200],
            )
            time.sleep(API_RETRY_BACKOFF_S)
            continue

        return resp.json()                    # success: hand back the answer as a dict


def _wait_for_health() -> None:
    """Block on imageRec /health at boot so we don't spam errors before it's up."""
    # When everything first starts, wait politely until the camera program says
    # "I'm ready" before we start asking it for predictions.
    url = f"{IMAGE_REC_URL}/health"
    while True:
        try:
            r = requests.get(url, timeout=5)  # ask "are you healthy?"
            if r.status_code == 200 and r.json().get("status") == "ok":
                body = r.json()
                logging.info("imageRec is up. classes=%s", body.get("classes"))
                if not body.get("camera"):    # it's up but has no camera attached
                    logging.warning(
                        "imageRec reports NO camera — /wait_and_predict will "
                        "return 503 until the webcam is attached."
                    )
                return                        # ready — stop waiting
        except requests.exceptions.RequestException:
            pass                              # not up yet, ignore and retry
        logging.info("waiting for imageRec at %s ...", IMAGE_REC_URL)
        time.sleep(API_RETRY_BACKOFF_S)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    # Set up logging so every message has a time and a level (INFO/WARNING).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("hadrwareCtrl starting. IMAGE_REC_URL=%s", IMAGE_REC_URL)
    logging.info("class -> angle map: %s", CLASS_TO_ANGLE)

    _wait_for_health()        # don't start until the camera program is ready
    stepper = _make_stepper() # build the (real or fake) motor controller
    servo = _make_servo()     # build the (real or fake) lid servo
    stepper.move_to(HOME_ANGLE)  # park the bin at the home position to start

    try:
        while True:                       # main loop — runs forever
            result = _wait_for_prediction()   # wait for the camera's answer
            if result is None:
                continue                      # nothing useful, go round again

            # Pull the useful fields out of the camera's answer.
            cls = result.get("top")                       # the predicted class name
            conf = float(result.get("confidence", 0.0))   # how sure it is (0..1)
            unsure = bool(result.get("unsure", False))    # a "not sure" flag
            waited = result.get("waited_seconds", "?")     # how long it watched
            logging.info(
                "prediction: %s (%.1f%%) unsure=%s after %ss",
                cls,
                conf * 100,
                unsure,
                waited,
            )

            # "unsure" covers /wait_and_predict timing out: the API then returns
            # its LAST frame's result, whose confidence can still read high even
            # though nothing held the trigger — never sort on it.
            # These checks make sure we ONLY move the bin when we're confident.
            if unsure or cls is None:
                logging.info("imageRec is unsure — not sorting.")
            elif conf < MIN_CONFIDENCE:
                logging.info("confidence below %.2f — not sorting.", MIN_CONFIDENCE)
            elif cls not in CLASS_TO_ANGLE:    # a class we don't have a bin for
                logging.warning("unknown class %r — not sorting.", cls)
            else:
                # All good — actually do the sort:
                stepper.move_to(CLASS_TO_ANGLE[cls])  # turn the bin to the right slot
                servo.open()                          # open the lid
                time.sleep(HOLD_LID_OPEN_S)           # wait so the item can drop
                servo.close()                         # close the lid
                stepper.move_to(HOME_ANGLE)           # return the bin to home

            time.sleep(COOLDOWN_AFTER_SORT_S)         # small rest before next item
    except KeyboardInterrupt:                          # user pressed Ctrl-C
        logging.info("interrupted by user")
    finally:
        # Always clean up the hardware, even if something went wrong.
        servo.release()
        stepper.close()
    return 0                                           # 0 means "finished OK"


# This runs main() only when you launch this file directly (python main.py),
# not when another file imports it.
if __name__ == "__main__":
    sys.exit(main())
