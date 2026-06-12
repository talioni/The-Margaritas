"""
Standalone DRV8825 stepper smoke test — no imageRec API needed.

NOTE: this does NOT use main.py's _RealStepper. That class is wired for the
TB6600 (active_high=False, for opto-isolated common-anode inputs). The DRV8825
has direct 3.3 V logic inputs that are ACTIVE-HIGH, so this test drives the
pins active-high instead. It reuses only the pin numbers / angles / timing
constants from main.py so there's a single source of truth for those.

Wiring reminder (DRV8825, NOT the same as the TB6600):
    STEP  -> GPIO20            DIR   -> GPIO21
    nRESET + nSLEEP -> tie HIGH to 3.3 V (or the driver stays asleep!)
    VMOT  -> 8.2-45 V motor PSU, with a 100 uF cap across VMOT/GND
    GND   -> motor PSU GND *and* a Pi GND (common ground)
    VDD   -> 3.3 V (logic supply)
    A1/A2/B1/B2 -> the two motor coils
    Set the current limit with the Vref trimmer pot before running.

Run on the Pi:
    python drv8825_test.py

If it does NOTHING: nRESET/nSLEEP probably aren't tied high.
If it turns the WRONG way: swap one motor coil pair (e.g. A1 <-> A2).
If it BUZZES/STALLS: raise STEP_PULSE_S in main.py, or set the Vref higher.
If travel is off by a constant factor: STEPS_PER_REV doesn't match the
M0/M1/M2 microstep jumpers.
"""

import logging
import time

from main import (
    CLASS_TO_ANGLE,
    DIR_PIN,
    HOME_ANGLE,
    STEP_PIN,
    STEP_PULSE_S,
    STEPS_PER_REV,
)


class _DRV8825:
    """Active-high STEP/DIR stepper for the DRV8825, via gpiozero."""

    def __init__(self, step_pin: int, dir_pin: int) -> None:
        from gpiozero import Device, OutputDevice
        from gpiozero.pins.lgpio import LGPIOFactory

        Device.pin_factory = LGPIOFactory()
        # active-high (default): DRV8825 inputs are plain 3.3 V logic.
        self._step = OutputDevice(step_pin)
        self._dir = OutputDevice(dir_pin)
        self._position_steps = 0  # 0 == HOME_ANGLE

    def move_to(self, angle: int) -> None:
        logging.info("stepper -> %d°", angle)
        target_steps = round(angle / 360 * STEPS_PER_REV)
        delta = target_steps - self._position_steps
        self._dir.value = 1 if delta > 0 else 0
        for _ in range(abs(delta)):
            self._step.on()
            time.sleep(STEP_PULSE_S)
            self._step.off()
            time.sleep(STEP_PULSE_S)
        self._position_steps = target_steps

    def close(self) -> None:
        self.move_to(HOME_ANGLE)
        self._step.close()
        self._dir.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if STEP_PIN is None or DIR_PIN is None:
        logging.error("STEP_PIN/DIR_PIN are None in main.py — nothing to test.")
        return 1

    logging.info("Testing DRV8825 on STEP=GPIO%d DIR=GPIO%d (active-high)",
                 STEP_PIN, DIR_PIN)
    stepper = _DRV8825(STEP_PIN, DIR_PIN)

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
