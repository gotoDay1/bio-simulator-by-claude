"""Inducer dosing events: instantaneous additions to the tank.

An inducer is added as a bolus: the concentration in the tank changes instantly at the
dosing time (the added volume is neglected, as in a concentrated stock). Between doses the
concentration only falls by dilution as the feed raises the volume (no degradation).

Two ways to read "final concentration C":
  mode="set": bring the tank up to C (an addition of max(C - I, 0), so residual inducer from
      earlier doses counts toward the target).
  mode="add": add C on top of whatever is there (I -> I + C).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from biosim.exceptions import InvalidParameterError
from biosim.validation import require_nonnegative

_MODES = ("set", "add")


@dataclass(frozen=True)
class InducerDose:
    """One instantaneous inducer addition at time t (h) with concentration conc (uM)."""

    t: float
    conc: float
    mode: str = "set"

    def __post_init__(self) -> None:
        require_nonnegative("t", self.t)
        require_nonnegative("conc", self.conc)
        if self.mode not in _MODES:
            raise InvalidParameterError(f"mode must be one of {_MODES}, got {self.mode!r}")

    def apply(self, current: float) -> float:
        """Inducer concentration right after this dose, given the concentration just before."""
        if self.mode == "add":
            return current + self.conc
        return max(current, self.conc)


def standard_induction(
    times: Sequence[float] = (32.0, 56.0, 80.0), conc: float = 300.0, mode: str = "set"
) -> list[InducerDose]:
    """The standard recipe: the same final concentration at each time (default 300 uM at 32/56/80 h)."""
    return [InducerDose(t, conc, mode) for t in times]
