"""
Standalone lid-servo smoke test — no imageRec API needed.

Reuses the pin layout and _RealServo from main.py so you're testing the
exact wiring/config the orchestrator will use. Run it directly on the Pi:

    GPIO_ENABLED=true python servo_test.py

It opens and closes the lid a few times, holding open as long as the real
sort does, so you can confirm travel and that the servo doesn't jitter.

If the lid moves the WRONG way / not far enough: adjust LID_OPEN_ANGLE /
LID_CLOSED_ANGLE in main.py. If it JITTERS or buzzes at rest: tune
SERVO_MIN_PULSE_S / SERVO_MAX_PULSE_S for your specific servo.
"""

# ---- What this file is for ----
# A tiny test program just for the lid servo. It opens and closes the lid a few
# times so you can check the lid moves correctly, WITHOUT needing the camera or
# the rest of the system running.

import logging  # tidy messages
import time     # pauses

# Borrow the real settings and the real servo code from main.py so we test the
# exact same setup the finished product uses.
from main import HOLD_LID_OPEN_S, LID_SERVO_PIN, _RealServo


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if LID_SERVO_PIN is None:                 # no pin set = nothing to test
        logging.error("LID_SERVO_PIN is None in main.py — nothing to test.")
        return 1

    logging.info("Testing lid servo on GPIO%d", LID_SERVO_PIN)
    servo = _RealServo(LID_SERVO_PIN)         # build the real servo (starts closed)

    try:
        for i in range(3):                    # do three open/close cycles
            logging.info("cycle %d: opening lid", i + 1)
            servo.open()                      # open the lid
            time.sleep(HOLD_LID_OPEN_S)       # hold it open like a real sort does
            logging.info("cycle %d: closing lid", i + 1)
            servo.close()                     # close the lid
            time.sleep(1.0)                   # short rest before the next cycle
        logging.info("Done. Lid left closed.")
    except KeyboardInterrupt:                 # Ctrl-C
        logging.info("interrupted")
    finally:
        servo.release()                       # always release the pin at the end
    return 0


# Run main() only when launched directly.
if __name__ == "__main__":
    raise SystemExit(main())
