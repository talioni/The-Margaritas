"""
Minimal TB6600 step/dir driver for the Raspberry Pi (incl. Pi 5).

Uses lgpio directly — no gpiozero layer in between — so what this code does to
a pin is exactly what you can measure with a multimeter on the header.

Electrical model this code assumes (common-anode wiring):

    PUL+ ──► Pi 3.3 V (physical pin 1)      <- see voltage note below
    DIR+ ──► Pi 3.3 V (physical pin 1)
    PUL- ──► GPIO step_pin
    DIR- ──► GPIO dir_pin
    ENA+ / ENA- ── leave BOTH unconnected (driver is then always enabled)

The TB6600 inputs are opto-coupler LEDs. Current flows through the LED (the
input is ASSERTED) when the GPIO sinks it, i.e. when the pin is LOW:

    GPIO HIGH (3.3 V) = idle, no step
    GPIO LOW  (0 V)   = pulse asserted

VOLTAGE NOTE: the usual TB6600 advice is "PUL+/DIR+ to 5 V", but with 3.3 V
logic that leaves 5 - 3.3 = 1.7 V across the opto LED in the *off* state. On
many TB6600 clones that residual is enough to keep the opto conducting, so the
driver never sees an edge: the motor locks (holding torque) but never moves.
Wiring PUL+/DIR+ to the Pi's 3.3 V pin instead gives a clean full swing and
still drives ~8 mA through the opto, which is plenty. Try 3.3 V first; only
fall back to 5 V if your unit genuinely won't trigger at 3.3 V.
"""

# ---- What this file is ----
# A "driver" is just code that knows how to talk to a piece of hardware.
# This file controls a TB6600, which is the little box that powers a stepper
# motor. We tell the TB6600 "take one step" by flicking a wire (GPIO pin)
# on and off. Do that 200 times and the motor turns once all the way round.

import logging  # lets us print nice timestamped messages instead of print()
import time     # lets us pause the program for tiny amounts of time (sleep)

import lgpio    # the library that actually lets us switch Raspberry Pi pins on/off

# These two names make the code easier to read further down.
# Because of the wiring above, the motor "steps" when the pin is LOW (0),
# and sits still ("idle") when the pin is HIGH (1).
_ASSERTED = 0   # pin LOW  -> opto LED conducting -> this is an active signal
_IDLE = 1       # pin HIGH -> opto LED off        -> nothing happening


def open_gpiochip() -> tuple[int, int, str]:
    """Open the gpiochip that drives the 40-pin header.

    On current Raspberry Pi OS kernels the header is gpiochip0, but on early
    Pi 5 kernels the RP1 chip was gpiochip4 — opening chip 0 there "works"
    and silently drives nothing. So scan for the chip whose label says it is
    the pin controller instead of hardcoding a number.

    Returns (handle, chip_number, label).
    """
    # A "gpiochip" is the controller inside the Pi that owns the physical pins.
    # Different Pi models number it differently, so instead of guessing we try
    # chips 0..7 and pick the one whose name starts with "pinctrl-".
    for n in range(8):                       # try chip 0, 1, 2 ... up to 7
        try:
            handle = lgpio.gpiochip_open(n)  # try to open this chip
        except lgpio.error:
            continue                         # this chip number doesn't exist, skip it
        try:
            info = lgpio.gpio_get_chip_info(handle)  # ask the chip about itself
            # The chip's "label" (its name) can come back in different shapes,
            # so this line just safely pulls the text out of whatever we got.
            label = info[2] if isinstance(info, (list, tuple)) else getattr(info, "label", "")
            if isinstance(label, bytes):     # sometimes the name is raw bytes
                label = label.decode()       # turn bytes into a normal string
        except lgpio.error:
            label = ""                       # couldn't read a name, leave it blank
        # Pi 5: "pinctrl-rp1"; Pi 4: "pinctrl-bcm2711"; Pi 3: "pinctrl-bcm2835"
        if str(label).startswith("pinctrl-"):    # found the real pin controller?
            return handle, n, str(label)          # great — hand it back to the caller
        lgpio.gpiochip_close(handle)              # wrong chip, close it and keep looking
    # If none matched (weird kernel or running in a container), just use chip 0.
    return lgpio.gpiochip_open(0), 0, "unknown (fell back to gpiochip0)"


class TB6600:
    """Step/direction driver for a TB6600 wired common-anode (active-low)."""
    # A "class" is a blueprint. We make one TB6600 object and then ask it to
    # step the motor, turn to an angle, etc. It remembers things like which
    # pins to use and what position the motor is currently in.

    def __init__(
        self,
        step_pin: int,            # which GPIO pin sends the "step" pulses
        dir_pin: int,             # which GPIO pin sets the spin direction
        steps_per_rev: int,       # how many steps make one full turn
        step_pulse_s: float = 0.0005,  # how long each on/off pulse lasts (speed)
        dir_setup_s: float = 0.05,     # tiny wait after changing direction
        invert_direction: bool = False,  # flip this if the motor turns the wrong way
    ) -> None:
        # "self." variables are remembered for as long as this object exists.
        self._step_pin = step_pin
        self._dir_pin = dir_pin
        self._steps_per_rev = steps_per_rev
        self._step_pulse_s = step_pulse_s
        self._dir_setup_s = dir_setup_s
        self._invert = invert_direction
        self._position_steps = 0   # we count steps so we always know where we are

        # Open the pin controller and remember the "handle" (like a file handle).
        self._handle, chip, label = open_gpiochip()
        logging.info("TB6600: using gpiochip%d (%s), STEP=GPIO%d DIR=GPIO%d",
                     chip, label, step_pin, dir_pin)
        # "Claim" the two pins as outputs and start them at IDLE (HIGH) so the
        # motor doesn't accidentally jump the moment the program starts.
        lgpio.gpio_claim_output(self._handle, step_pin, _IDLE)
        lgpio.gpio_claim_output(self._handle, dir_pin, _IDLE)

    # -- low level ----------------------------------------------------------

    def _set_direction(self, forward: bool) -> None:
        # Tell the driver which way to spin by setting the DIR pin.
        if self._invert:
            forward = not forward          # honour the "wrong way round" flag
        lgpio.gpio_write(self._handle, self._dir_pin,
                         _ASSERTED if forward else _IDLE)  # set the pin level
        # The driver needs the direction signal to be steady for a moment before
        # we start stepping, otherwise the first step can go the wrong way.
        time.sleep(self._dir_setup_s)

    def _pulse(self) -> None:
        # One single step = flick the pin LOW then HIGH again, with a short
        # pause each time so the driver actually notices the change.
        lgpio.gpio_write(self._handle, self._step_pin, _ASSERTED)  # pin LOW
        time.sleep(self._step_pulse_s)                             # wait a bit
        lgpio.gpio_write(self._handle, self._step_pin, _IDLE)      # pin HIGH
        time.sleep(self._step_pulse_s)                             # wait a bit

    # -- public API ----------------------------------------------------------

    def step(self, steps: int, forward: bool = True,
             step_pulse_s: float | None = None) -> None:
        """Issue `steps` pulses in one direction. Blocks until done."""
        # Use the speed passed in, or fall back to the default set in __init__.
        pulse = step_pulse_s if step_pulse_s is not None else self._step_pulse_s
        self._set_direction(forward)       # point the motor the right way first
        for _ in range(steps):             # repeat once per step we want
            lgpio.gpio_write(self._handle, self._step_pin, _ASSERTED)  # pin LOW
            time.sleep(pulse)                                          # wait
            lgpio.gpio_write(self._handle, self._step_pin, _IDLE)      # pin HIGH
            time.sleep(pulse)                                          # wait
        # Update our running count of where the motor is now.
        self._position_steps += steps if forward else -steps

    def move_to_angle(self, angle: float) -> None:
        """Rotate to an absolute angle (degrees), 0 = position at startup."""
        # Convert the angle we want into a step number.
        target = round(angle / 360 * self._steps_per_rev)
        delta = target - self._position_steps   # how many steps away we are
        if delta == 0:                          # already there? do nothing
            return
        # Step the difference. If delta is positive go forward, else backward.
        self.step(abs(delta), forward=delta > 0)
        self._position_steps = target           # snap the count to avoid drift

    @property
    def angle(self) -> float:
        # A read-only value: turn our step count back into degrees.
        return self._position_steps / self._steps_per_rev * 360

    def close(self) -> None:
        """Leave both pins idle and release them."""
        # Tidy up when we're finished so we don't leave pins in a weird state.
        try:
            lgpio.gpio_write(self._handle, self._step_pin, _IDLE)  # park step pin HIGH
            lgpio.gpio_write(self._handle, self._dir_pin, _IDLE)   # park dir pin HIGH
            lgpio.gpio_free(self._handle, self._step_pin)          # release the pins
            lgpio.gpio_free(self._handle, self._dir_pin)
        finally:
            lgpio.gpiochip_close(self._handle)                     # close the chip
