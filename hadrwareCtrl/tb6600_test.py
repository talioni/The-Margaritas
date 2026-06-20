"""
Staged TB6600 diagnostic — run this ON THE PI, on the host (not in Docker):

    python tb6600_test.py            # run all stages in order
    python tb6600_test.py pins       # or run a single stage by name
    python tb6600_test.py tick
    python tb6600_test.py slow
    python tb6600_test.py spin

Stages, in order of "how much of the chain they exercise":

  pins  Toggles GPIO20/GPIO21 once per second, motor expectations: NONE.
        Put a multimeter between GPIO20 (physical pin 38) and GND (pin 39):
        it must alternate between ~3.3 V and ~0 V. If it doesn't, the problem
        is software/permissions/wrong pin — stop and fix that first.
  tick  10 single steps, half a second apart. You should HEAR/FEEL one tiny
        tick per step. This proves the TB6600 sees the pulses.
  slow  One full revolution in ~8 s. Proves sustained stepping.
  spin  One revolution forward + one back at normal speed. Proves both
        directions and the speed the orchestrator will actually use.

Expected wiring (common-anode, active-low):
    PUL+ and DIR+ -> Pi 3.3 V (physical pin 1)   [see note in tb6600.py:
                     try 3.3 V first, NOT 5 V — 5 V can leave the opto
                     half-on with 3.3 V logic and then nothing ever steps]
    PUL- -> GPIO20 (physical pin 38)
    DIR- -> GPIO21 (physical pin 40)
    ENA+ / ENA- -> both unconnected
    A+/A- = one motor coil, B+/B- = the other (verify with a multimeter:
            a few ohms = same coil, open = different coils)
    VCC/GND -> separate 12-24 V motor PSU (NOT the Pi's 5 V!)

Symptom guide:
    'pins' fails .............. software/permissions/wrong header pin
    pins OK, no tick, shaft is FREE to turn by hand
                       ........ TB6600 has no motor power, or ENA is wired
                                and disabling it, or DIP current switches
                                are all-off (invalid setting)
    pins OK, no tick, shaft is LOCKED (hard to turn)
                       ........ pulses not registering: PUL+ on 5 V instead
                                of 3.3 V (most likely), or PUL+/PUL- swapped
    ticks but 'slow'/'spin' only buzzes
                       ........ wrong coil pairing (A/B mixed) or current
                                set too low on the DIP switches
    spins the wrong way ....... swap one coil pair (A+ <-> A-), or set
                                invert_direction=True
    travel distance is wrong .. MICROSTEP below doesn't match DIP S4-S6
"""

# ---- What this file is for ----
# This is a TEST / TROUBLESHOOTING tool, not part of the real product. If the
# motor doesn't move, you run this first to find out *where* it's broken
# (the pins? the wiring? the motor power?) by doing easy checks one at a time.

import logging  # for tidy timestamped messages
import sys      # to read the words you typed after the filename
import time     # for pauses between steps

import lgpio    # the Raspberry Pi pin library

from tb6600 import TB6600, open_gpiochip   # reuse our real driver + chip finder

# --- wiring / behaviour ------------------------------------------------------
STEP_PIN = 20          # BCM. TB6600 PUL-  (physical pin 38)
DIR_PIN = 21           # BCM. TB6600 DIR-  (physical pin 40)

MICROSTEP = 8          # MUST match the TB6600 DIP switches S4-S6
STEPS_PER_REV = 200 * MICROSTEP   # how many steps make a full turn here

STEP_PULSE_S = 0.0005  # normal speed: 1 kHz pulse rate
# -----------------------------------------------------------------------------


def stage_pins() -> None:
    """Toggle both pins slowly so a multimeter/LED can verify GPIO output."""
    # Easiest test: just flip the pins HIGH/LOW once a second so you can check
    # with a multimeter that the Pi is really changing the voltage.
    handle, chip, label = open_gpiochip()        # open the pin controller
    logging.info("using gpiochip%d (%s)", chip, label)
    lgpio.gpio_claim_output(handle, STEP_PIN, 1)  # take control of the step pin
    lgpio.gpio_claim_output(handle, DIR_PIN, 1)   # take control of the dir pin
    logging.info("Toggling GPIO%d and GPIO%d once per second for 10 s.",
                 STEP_PIN, DIR_PIN)
    logging.info("Multimeter: GPIO20 = physical pin 38, GND = physical pin 39."
                 " Expect ~3.3 V <-> ~0 V.")
    try:
        for i in range(10):                       # do this 10 times
            level = i % 2                         # alternate 0,1,0,1...
            lgpio.gpio_write(handle, STEP_PIN, level)  # set step pin
            lgpio.gpio_write(handle, DIR_PIN, level)   # set dir pin
            logging.info("pins now %s", "HIGH (3.3 V)" if level else "LOW (0 V)")
            time.sleep(1.0)                       # wait a second so you can read it
    finally:
        # Always put the pins back to HIGH and release them at the end.
        lgpio.gpio_write(handle, STEP_PIN, 1)
        lgpio.gpio_write(handle, DIR_PIN, 1)
        lgpio.gpio_free(handle, STEP_PIN)
        lgpio.gpio_free(handle, DIR_PIN)
        lgpio.gpiochip_close(handle)


def stage_tick(stepper: TB6600) -> None:
    # Take 10 single steps, slowly, so you can hear/feel each tiny "tick".
    logging.info("10 single steps, 0.5 s apart — listen/feel for a tiny tick"
                 " on each one.")
    for i in range(10):
        stepper.step(1, forward=True)   # one step forward
        logging.info("step %d", i + 1)
        time.sleep(0.5)                 # pause so the ticks are separate


def stage_slow(stepper: TB6600) -> None:
    # Turn one full circle slowly to prove it can keep stepping smoothly.
    logging.info("One full revolution, slowly (~8 s)...")
    stepper.step(STEPS_PER_REV, forward=True,
                 step_pulse_s=4.0 / (2 * STEPS_PER_REV))  # spread over ~8 seconds
    logging.info("done.")


def stage_spin(stepper: TB6600) -> None:
    # Turn one way then back, at normal speed, to prove both directions work.
    logging.info("One revolution forward at normal speed...")
    stepper.step(STEPS_PER_REV, forward=True)   # forward one turn
    time.sleep(1.0)
    logging.info("...and one back.")
    stepper.step(STEPS_PER_REV, forward=False)  # backward one turn
    logging.info("done.")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # sys.argv[1:] is whatever words you typed after the filename. If you typed
    # nothing, default to running all four stages in order.
    stages = sys.argv[1:] or ["pins", "tick", "slow", "spin"]
    known = {"pins", "tick", "slow", "spin"}     # the stage names we accept
    bad = set(stages) - known                    # anything you typed that's unknown
    if bad:
        logging.error("unknown stage(s) %s — choose from %s", bad, known)
        return 1                                 # 1 means "error, stopped"

    if "pins" in stages:
        stage_pins()                             # run the pin test first
        stages = [s for s in stages if s != "pins"]  # then drop it from the list
        if stages:
            # Pause and let the user confirm before energising the motor.
            input("If the multimeter showed 3.3V/0V toggling, press Enter to "
                  "continue to the motor stages (Ctrl-C to stop)... ")

    if not stages:
        return 0                                 # only "pins" was asked for — done

    # Build the real driver for the motor stages.
    stepper = TB6600(STEP_PIN, DIR_PIN, STEPS_PER_REV,
                     step_pulse_s=STEP_PULSE_S)
    try:
        for s in stages:
            # Look up the matching function for each stage name and run it.
            {"tick": stage_tick, "slow": stage_slow, "spin": stage_spin}[s](stepper)
            time.sleep(1.0)
    except KeyboardInterrupt:                     # Ctrl-C
        logging.info("interrupted")
    finally:
        stepper.close()                           # always tidy up the pins
    return 0


# Run main() only when launched directly, not when imported.
if __name__ == "__main__":
    raise SystemExit(main())
