"""Culture conditions (temperature, pH) as time-dependent *inputs* to the simulation.

Temperature and pH are not ODE states: like the feed rate F_in(t), they are
prescribed profiles. A ConditionProfile turns t into a Conditions snapshot, which the
orchestrator hands to condition-aware models (those with condition_aware = True).

Temperature dependence uses an Arrhenius factor re-parameterised around a reference
temperature (T_REF_C), so that k_ref and Ea are far less correlated than in the
textbook A*exp(-Ea/RT) form:

    k(T) = k_ref * exp[-Ea/R * (1/T - 1/T_ref)]
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from biosim.exceptions import InvalidParameterError
from biosim.operation_modes import stepwise_feed
from biosim.validation import require_positive

T_REF_C = 30.0
PH_REF = 5.0
_R_KJ_PER_MOL_K = 8.314462618e-3
_KELVIN = 273.15


@dataclass(frozen=True)
class Conditions:
    """Culture conditions at one instant: temperature (deg C), pH and inducer (uM).

    T_C and pH are prescribed by a ConditionProfile; `inducer` is the inducer concentration
    in the tank, which the orchestrator fills in from the state vector (it is 0.0 in a
    snapshot taken straight from a profile).
    """

    T_C: float = T_REF_C
    pH: float = PH_REF
    inducer: float = 0.0


def arrhenius_factor(T_C: float, Ea: float, T_ref_C: float = T_REF_C) -> float:
    """k(T)/k_ref for an activation energy Ea (kJ/mol); equals 1 at T_ref_C."""
    T = T_C + _KELVIN
    T_ref = T_ref_C + _KELVIN
    return math.exp(-Ea / _R_KJ_PER_MOL_K * (1.0 / T - 1.0 / T_ref))


def quadratic_factor(x: float, a: float, b: float, x_ref: float, floor: float = 0.0) -> float:
    """1 + a*(x - x_ref) + b*(x - x_ref)^2, clipped below at `floor`.

    Used for the pH correction (x = pH, x_ref = PH_REF); equals 1 at x_ref. The floor
    keeps a rate multiplier from going negative far from the reference.
    """
    d = x - x_ref
    return max(1.0 + a * d + b * d * d, floor)


class ConditionProfile:
    """Prescribed T(t) [deg C] and pH(t); defaults to the standard 30 C / pH 5.0."""

    def __init__(
        self,
        temperature_fn: Callable[[float], float] | None = None,
        pH_fn: Callable[[float], float] | None = None,
    ):
        self.temperature_fn = temperature_fn or (lambda t: T_REF_C)
        self.pH_fn = pH_fn or (lambda t: PH_REF)

    def at(self, t: float) -> Conditions:
        return Conditions(T_C=float(self.temperature_fn(t)), pH=float(self.pH_fn(t)))


def constant_value(value: float) -> Callable[[float], float]:
    """A profile that holds `value` for all t."""
    require_positive("value", value)

    def _fn(t: float) -> float:
        return value

    return _fn


def shift_value(before: float, after: float, t_shift: float) -> Callable[[float], float]:
    """Hold `before` until t_shift, then `after` (e.g. a 30 -> 24 C shift at 48 h)."""
    require_positive("before", before)
    require_positive("after", after)

    def _fn(t: float) -> float:
        return before if t < t_shift else after

    return _fn


def piecewise_value(times: Sequence[float], values: Sequence[float]) -> Callable[[float], float]:
    """Step-hold profile through (time, value) breakpoints; same semantics as stepwise_feed."""
    return stepwise_feed(times, values)


def ramp_value(
    before: float, after: float, t_start: float, t_end: float
) -> Callable[[float], float]:
    """Hold `before`, ramp linearly to `after` between t_start and t_end, then hold `after`."""
    require_positive("before", before)
    require_positive("after", after)
    if t_end <= t_start:
        raise InvalidParameterError("t_end must be greater than t_start")

    def _fn(t: float) -> float:
        if t <= t_start:
            return before
        if t >= t_end:
            return after
        return before + (after - before) * (t - t_start) / (t_end - t_start)

    return _fn
