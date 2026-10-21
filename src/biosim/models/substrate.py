from dataclasses import dataclass

from biosim.models.base import SubstrateModel
from biosim.validation import require_nonnegative, require_positive


@dataclass
class YieldMaintenanceSubstrate(SubstrateModel):
    """Yield-coefficient + maintenance substrate consumption.

    dS/dt = -(1/Yxs)*dXdt - ms*X [- (1/Yps)*dPdt if Yps is set]

    Yxs: biomass yield on substrate (g biomass / g substrate), typical range 0.1-0.6
    ms: maintenance coefficient (g substrate / g biomass / h), typical range 0-0.1
    Yps: optional product yield on substrate (g product / g substrate); None disables this term
    """

    Yxs: float = 0.5
    ms: float = 0.02
    Yps: float | None = None
    name = "yield_maintenance"

    def __post_init__(self) -> None:
        require_positive("Yxs", self.Yxs)
        require_nonnegative("ms", self.ms)
        if self.Yps is not None:
            require_positive("Yps", self.Yps)

    def consumption_rate(self, X: float, dXdt: float, dPdt: float, t: float) -> float:
        rate = -(1.0 / self.Yxs) * dXdt - self.ms * X
        if self.Yps is not None:
            rate -= (1.0 / self.Yps) * dPdt
        return rate


@dataclass
class GrowthDrivenSubstrate(SubstrateModel):
    """Adds no substrate consumption of its own.

    For growth models with drives_substrate=True (e.g. PirtMonodGrowth), the orchestrator
    takes dS/dt = -qS*X directly from the growth model, so that growth and substrate use
    one consistent set of yield/maintenance parameters. This placeholder has no
    parameters; the substrate side is entirely defined by the growth model.
    """

    name = "growth_driven"

    def consumption_rate(self, X: float, dXdt: float, dPdt: float, t: float) -> float:
        return 0.0
