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

import logging
import time

from main import HOLD_LID_OPEN_S, LID_SERVO_PIN, _RealServo


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if LID_SERVO_PIN is None:
        logging.error("LID_SERVO_PIN is None in main.py — nothing to test.")
        return 1

    logging.info("Testing lid servo on GPIO%d", LID_SERVO_PIN)
    servo = _RealServo(LID_SERVO_PIN)  # constructs at the closed position

    try:
        for i in range(3):
            logging.info("cycle %d: opening lid", i + 1)
            servo.open()
            time.sleep(HOLD_LID_OPEN_S)
            logging.info("cycle %d: closing lid", i + 1)
            servo.close()
            time.sleep(1.0)
        logging.info("Done. Lid left closed.")
    except KeyboardInterrupt:
        logging.info("interrupted")
    finally:
        servo.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
