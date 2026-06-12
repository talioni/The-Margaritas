"""
Standalone TB6600 stepper driver test — fully self-contained.

No imports from main.py, no imageRec API, no servo. Just spins the motor so
you can confirm the TB6600 + wiring works. Run directly on the Pi:

    python tb6600_test.py

Wiring assumed (common-anode, active-low — the standard 3.3 V Pi setup):
    PUL+ -> Pi 5 V        PUL- -> GPIO20 (STEP)
    DIR+ -> Pi 5 V        DIR- -> GPIO21 (DIR)
    ENA+/ENA- -> leave unconnected (driver always enabled)
    A+/A- = one motor coil, B+/B- = the other coil
    VCC/GND -> separate motor PSU (common ground with the Pi!)

It spins ~1 full revolution one way, pauses, then ~1 revolution back, a few
times. Watch the shaft.

If it LOCKS / BUZZES but won't rotate: almost always wrong coil pairing.
    Check A+/A- are the SAME coil and B+/B- the other (multimeter: a few ohms
    = same coil; open circuit = different coils).
If it turns the WRONG way: swap one coil pair (A+ <-> A-).
If it STALLS at speed: raise STEP_PULSE_S.
"""

import logging
import time

# --- wiring / behaviour -----------------------------------------------------
STEP_PIN = 20            # BCM. TB6600 PUL- (active-low, common-anode)
DIR_PIN = 21             # BCM. TB6600 DIR-

STEPS_PER_REV = 200     # 200 full steps × 8 (TB6600 DIP set to 1/8 microstep)
STEP_PULSE_S = 0.001     # high/low time per STEP pulse; raise if it stalls
PAUSE_BETWEEN_S = 1.0    # pause between direction changes
CYCLES = 4               # how many forward+back cycles to run
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    from gpiozero import Device, OutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory

    Device.pin_factory = LGPIOFactory()

    # active_high=False: common-anode wiring, so the opto conducts (a pulse is
    # asserted) when the GPIO is driven LOW. .on() therefore means "step".
    step = OutputDevice(STEP_PIN, active_high=False)
    direction = OutputDevice(DIR_PIN, active_high=False)

    def spin(steps: int, forward: bool) -> None:
        direction.value = 1 if forward else 0
        time.sleep(0.001)  # let DIR settle before the first pulse
        for _ in range(steps):
            step.on()
            time.sleep(STEP_PULSE_S)
            step.off()
            time.sleep(STEP_PULSE_S)

    logging.info("TB6600 test: STEP=GPIO%d DIR=GPIO%d, %d steps/rev",
                 STEP_PIN, DIR_PIN, STEPS_PER_REV)
    try:
        for i in range(CYCLES):
            logging.info("cycle %d: forward 1 rev", i + 1)
            spin(STEPS_PER_REV, forward=True)
            time.sleep(PAUSE_BETWEEN_S)
            logging.info("cycle %d: backward 1 rev", i + 1)
            spin(STEPS_PER_REV, forward=False)
            time.sleep(PAUSE_BETWEEN_S)
        logging.info("Done.")
    except KeyboardInterrupt:
        logging.info("interrupted")
    finally:
        step.off()
        step.close()
        direction.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
