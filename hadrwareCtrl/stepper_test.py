"""
Standalone TB6600 stepper smoke test — no imageRec API needed.

Reuses the pin layout and _RealStepper from main.py so you're testing the
exact wiring/config the orchestrator will use. Run it directly on the Pi:

    GPIO_ENABLED=true python stepper_test.py

It rotates to each bin angle in turn (organic / pmd / restafval), pausing
between moves so you can watch direction and travel, then returns home.

If the bin turns the WRONG way: swap one motor coil pair (e.g. A+ <-> A-).
If it BUZZES/STALLS instead of turning: raise STEP_PULSE_S in main.py.
"""

import logging
import time

from main import CLASS_TO_ANGLE, DIR_PIN, HOME_ANGLE, STEP_PIN, _RealStepper


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if STEP_PIN is None or DIR_PIN is None:
        logging.error("STEP_PIN/DIR_PIN are None in main.py — nothing to test.")
        return 1

    logging.info("Testing stepper on STEP=GPIO%d DIR=GPIO%d", STEP_PIN, DIR_PIN)
    stepper = _RealStepper(STEP_PIN, DIR_PIN)

    try:
        stepper.move_to(HOME_ANGLE)
        time.sleep(1.0)
        for cls, angle in CLASS_TO_ANGLE.items():
            logging.info("=> %s bin at %d°", cls, angle)
            stepper.move_to(angle)
            time.sleep(1.5)
            stepper.move_to(HOME_ANGLE)
            time.sleep(1.0)
        logging.info("Done. Stepper returned home.")
    except KeyboardInterrupt:
        logging.info("interrupted")
    finally:
        stepper.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
