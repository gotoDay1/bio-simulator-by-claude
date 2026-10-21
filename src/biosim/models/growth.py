import math
from dataclasses import dataclass

from biosim.conditions import (
    PH_REF,
    T_REF_C,
    Conditions,
    arrhenius_factor,
    quadratic_factor,
)
from biosim.models.base import GrowthModel
from biosim.validation import require_nonnegative, require_positive


@dataclass
class MonodGrowth(GrowthModel):
    """Substrate-limited growth: mu = mu_max * S / (Ks + S).

    mu_max: maximum specific growth rate (1/h), typical range 0.1-1.0
    Ks: half-saturation constant (g/L), typical range 0.01-1.0
    """

    mu_max: float = 0.6
    Ks: float = 0.2
    name = "monod"

    def __post_init__(self) -> None:
        require_positive("mu_max", self.mu_max)
        require_positive("Ks", self.Ks)

    def specific_growth_rate(
        self, X: float, S: float, t: float, C_O2: float | None = None
    ) -> float:
        S = max(S, 0.0)
        return self.mu_max * S / (self.Ks + S)


@dataclass
class LogisticGrowth(GrowthModel):
    """Carrying-capacity limited growth: mu = mu_max * (1 - X / Xmax).

    mu_max: maximum specific growth rate (1/h), typical range 0.1-1.0
    Xmax: carrying capacity / maximum biomass concentration (g/L), typical range 5-50
    """

    mu_max: float = 0.6
    Xmax: float = 10.0
    name = "logistic"

    def __post_init__(self) -> None:
        require_positive("mu_max", self.mu_max)
        require_positive("Xmax", self.Xmax)

    def specific_growth_rate(
        self, X: float, S: float, t: float, C_O2: float | None = None
    ) -> float:
        return self.mu_max * (1.0 - X / self.Xmax)


@dataclass
class GompertzGrowth(GrowthModel):
    """Mechanistic (ODE) Gompertz growth: mu = mu_max * ln(Xmax / X).

    Note: this is the differential/mechanistic form used to plug into an ODE
    mass balance (dX/dt = mu*X = mu_max*X*ln(Xmax/X)), not the closed-form
    regression curve X(t) = Xmax*exp(-exp(...)) sometimes fit to endpoint data.

    mu_max: maximum specific growth rate (1/h), typical range 0.1-1.0
    Xmax: asymptotic maximum biomass concentration (g/L), typical range 5-50
    """

    mu_max: float = 0.6
    Xmax: float = 10.0
    name = "gompertz"

    def __post_init__(self) -> None:
        require_positive("mu_max", self.mu_max)
        require_positive("Xmax", self.Xmax)

    def specific_growth_rate(
        self, X: float, S: float, t: float, C_O2: float | None = None
    ) -> float:
        if X <= 0.0 or X >= self.Xmax:
            return 0.0
        return self.mu_max * math.log(self.Xmax / X)


@dataclass
class MonodGrowthO2(GrowthModel):
    """Dual-substrate Monod growth: mu = mu_max * S/(Ks+S) * C_O2/(Ko2+C_O2).

    Unlike MonodGrowth (substrate-only), this model also gates growth on dissolved
    oxygen availability, so it can represent oxygen-limited growth (e.g. when kLa
    can't keep up with demand). Requires an oxygen model that tracks C_O2 as a state
    (OxygenWithKLa) — BioreactorSimulation rejects pairing this with OxygenDemandOnly.

    mu_max: maximum specific growth rate (1/h), typical range 0.1-1.0
    Ks: half-saturation constant for substrate (g/L), typical range 0.01-1.0
    Ko2: half-saturation constant for dissolved O2 (mg/L), typical range 0.01-0.5
    """

    mu_max: float = 0.6
    Ks: float = 0.2
    Ko2: float = 0.2
    name = "monod_o2"
    requires_oxygen_state = True

    def __post_init__(self) -> None:
        require_positive("mu_max", self.mu_max)
        require_positive("Ks", self.Ks)
        require_positive("Ko2", self.Ko2)

    def specific_growth_rate(
        self, X: float, S: float, t: float, C_O2: float | None = None
    ) -> float:
        S = max(S, 0.0)
        c = max(C_O2, 0.0) if C_O2 is not None else 0.0
        return self.mu_max * S / (self.Ks + S) * c / (self.Ko2 + c)


@dataclass
class PirtMonodGrowth(GrowthModel):
    """Substrate-limited growth from a Pirt mass balance, with temperature/pH dependence.

    Growth is not given a mu_max; it follows from where the substrate goes:

        qS = qSmax(T) * S / (Ks + S)      specific substrate uptake (g/g/h)
        mu = Y(T) * (qS - m(T,pH))        Pirt: uptake -> growth + maintenance

    so under feed limitation (S ~ 0, qS set by the feed) the temperature effect shows up
    through the yield Y and the maintenance coefficient m, as in the theory notes.

        Y(T)     = Yxs_ref * (1 + Yxs_slope * (T - T_REF_C)), floored at a small positive value
        m(T,pH)  = ms_ref * exp[-Ea_ms/R (1/T - 1/T_ref)] * (1 + a*(pH-5) + b*(pH-5)^2)
        qSmax(T) = qSmax_ref * exp[-Ea_qS/R (1/T - 1/T_ref)]

    Reference conditions are T_REF_C (30 C) and PH_REF (5.0). mu can be negative when
    qS < m (net endogenous loss of biomass).

    drives_substrate=True: this model also defines the substrate uptake, so pair it with
    GrowthDrivenSubstrate (BioreactorSimulation enforces this).

    Yxs_ref: biomass yield on substrate at reference conditions (g/g), typical 0.1-0.6
    Yxs_slope: relative change of Y per deg C (1/C), typical -0.02-0.02
    ms_ref: maintenance coefficient at reference conditions (g/g/h), typical 0-0.1
    Ea_ms: activation energy of maintenance (kJ/mol), typical 20-100
    ms_pH_a, ms_pH_b: pH correction of maintenance (linear, quadratic around pH 5.0)
    qSmax_ref: maximum specific substrate uptake at 30 C (g/g/h), typical 0.5-3
    Ea_qS: activation energy of the maximum uptake (kJ/mol), typical 10-80
    Ks: half-saturation constant for substrate (g/L), typical 0.001-1.0
    """

    Yxs_ref: float = 0.5
    Yxs_slope: float = 0.0
    ms_ref: float = 0.02
    Ea_ms: float = 50.0
    ms_pH_a: float = 0.0
    ms_pH_b: float = 0.0
    qSmax_ref: float = 1.0
    Ea_qS: float = 30.0
    Ks: float = 0.1
    name = "pirt_monod"
    condition_aware = True
    drives_substrate = True

    _MIN_YIELD = 1e-3

    def __post_init__(self) -> None:
        require_positive("Yxs_ref", self.Yxs_ref)
        require_nonnegative("ms_ref", self.ms_ref)
        require_nonnegative("Ea_ms", self.Ea_ms)
        require_positive("qSmax_ref", self.qSmax_ref)
        require_nonnegative("Ea_qS", self.Ea_qS)
        require_positive("Ks", self.Ks)

    def yield_at(self, cond: Conditions) -> float:
        return max(self.Yxs_ref * (1.0 + self.Yxs_slope * (cond.T_C - T_REF_C)), self._MIN_YIELD)

    def maintenance_at(self, cond: Conditions) -> float:
        return (
            self.ms_ref
            * arrhenius_factor(cond.T_C, self.Ea_ms)
            * quadratic_factor(cond.pH, self.ms_pH_a, self.ms_pH_b, PH_REF)
        )

    def specific_substrate_uptake(
        self,
        X: float,
        S: float,
        t: float,
        C_O2: float | None = None,
        cond: Conditions | None = None,
    ) -> float:
        cond = cond or Conditions()
        S = max(S, 0.0)
        qS_max = self.qSmax_ref * arrhenius_factor(cond.T_C, self.Ea_qS)
        return qS_max * S / (self.Ks + S)

    def specific_growth_rate(
        self,
        X: float,
        S: float,
        t: float,
        C_O2: float | None = None,
        cond: Conditions | None = None,
    ) -> float:
        cond = cond or Conditions()
        qS = self.specific_substrate_uptake(X, S, t, C_O2, cond)
        return self.yield_at(cond) * (qS - self.maintenance_at(cond))
