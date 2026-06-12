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

import logging
import time

import lgpio

# Physical pin levels for the common-anode (active-low) wiring above.
_ASSERTED = 0   # opto LED conducting
_IDLE = 1       # opto LED off


def open_gpiochip() -> tuple[int, int, str]:
    """Open the gpiochip that drives the 40-pin header.

    On current Raspberry Pi OS kernels the header is gpiochip0, but on early
    Pi 5 kernels the RP1 chip was gpiochip4 — opening chip 0 there "works"
    and silently drives nothing. So scan for the chip whose label says it is
    the pin controller instead of hardcoding a number.

    Returns (handle, chip_number, label).
    """
    for n in range(8):
        try:
            handle = lgpio.gpiochip_open(n)
        except lgpio.error:
            continue
        try:
            info = lgpio.gpio_get_chip_info(handle)
            label = info[2] if isinstance(info, (list, tuple)) else getattr(info, "label", "")
            if isinstance(label, bytes):
                label = label.decode()
        except lgpio.error:
            label = ""
        # Pi 5: "pinctrl-rp1"; Pi 4: "pinctrl-bcm2711"; Pi 3: "pinctrl-bcm2835"
        if str(label).startswith("pinctrl-"):
            return handle, n, str(label)
        lgpio.gpiochip_close(handle)
    # Nothing matched (unusual kernel / container): fall back to chip 0.
    return lgpio.gpiochip_open(0), 0, "unknown (fell back to gpiochip0)"


class TB6600:
    """Step/direction driver for a TB6600 wired common-anode (active-low)."""

    def __init__(
        self,
        step_pin: int,
        dir_pin: int,
        steps_per_rev: int,
        step_pulse_s: float = 0.0005,
        dir_setup_s: float = 0.05,
        invert_direction: bool = False,
    ) -> None:
        self._step_pin = step_pin
        self._dir_pin = dir_pin
        self._steps_per_rev = steps_per_rev
        self._step_pulse_s = step_pulse_s
        self._dir_setup_s = dir_setup_s
        self._invert = invert_direction
        self._position_steps = 0

        self._handle, chip, label = open_gpiochip()
        logging.info("TB6600: using gpiochip%d (%s), STEP=GPIO%d DIR=GPIO%d",
                     chip, label, step_pin, dir_pin)
        # Claim both pins already at the idle (HIGH) level so the driver never
        # sees a spurious pulse during startup.
        lgpio.gpio_claim_output(self._handle, step_pin, _IDLE)
        lgpio.gpio_claim_output(self._handle, dir_pin, _IDLE)

    # -- low level ----------------------------------------------------------

    def _set_direction(self, forward: bool) -> None:
        if self._invert:
            forward = not forward
        lgpio.gpio_write(self._handle, self._dir_pin,
                         _ASSERTED if forward else _IDLE)
        # TB6600 needs DIR stable >5 µs before the first pulse; be generous.
        time.sleep(self._dir_setup_s)

    def _pulse(self) -> None:
        lgpio.gpio_write(self._handle, self._step_pin, _ASSERTED)
        time.sleep(self._step_pulse_s)
        lgpio.gpio_write(self._handle, self._step_pin, _IDLE)
        time.sleep(self._step_pulse_s)

    # -- public API ----------------------------------------------------------

    def step(self, steps: int, forward: bool = True,
             step_pulse_s: float | None = None) -> None:
        """Issue `steps` pulses in one direction. Blocks until done."""
        pulse = step_pulse_s if step_pulse_s is not None else self._step_pulse_s
        self._set_direction(forward)
        for _ in range(steps):
            lgpio.gpio_write(self._handle, self._step_pin, _ASSERTED)
            time.sleep(pulse)
            lgpio.gpio_write(self._handle, self._step_pin, _IDLE)
            time.sleep(pulse)
        self._position_steps += steps if forward else -steps

    def move_to_angle(self, angle: float) -> None:
        """Rotate to an absolute angle (degrees), 0 = position at startup."""
        target = round(angle / 360 * self._steps_per_rev)
        delta = target - self._position_steps
        if delta == 0:
            return
        self.step(abs(delta), forward=delta > 0)
        self._position_steps = target  # avoid rounding drift from step()

    @property
    def angle(self) -> float:
        return self._position_steps / self._steps_per_rev * 360

    def close(self) -> None:
        """Leave both pins idle and release them."""
        try:
            lgpio.gpio_write(self._handle, self._step_pin, _IDLE)
            lgpio.gpio_write(self._handle, self._dir_pin, _IDLE)
            lgpio.gpio_free(self._handle, self._step_pin)
            lgpio.gpio_free(self._handle, self._dir_pin)
        finally:
            lgpio.gpiochip_close(self._handle)
