from dataclasses import dataclass

from biosim.conditions import Conditions, arrhenius_factor
from biosim.models.base import ProductModel
from biosim.validation import require_nonnegative


@dataclass
class LuedekingPiretProduct(ProductModel):
    """Growth- and non-growth-associated product formation: dP/dt = alpha*dXdt + beta*X.

    alpha: growth-associated production coefficient (g product / g biomass), typical range 0-10
    beta: non-growth-associated production coefficient (1/h), typical range 0-0.5
    """

    alpha: float = 2.0
    beta: float = 0.05
    name = "luedeking_piret"

    def __post_init__(self) -> None:
        require_nonnegative("alpha", self.alpha)
        require_nonnegative("beta", self.beta)

    def production_rate(self, X: float, P: float, dXdt: float, t: float) -> float:
        return self.alpha * dXdt + self.beta * X


@dataclass
class NoProduct(ProductModel):
    """No product tracking: dP/dt = 0."""

    name = "none"

    def production_rate(self, X: float, P: float, dXdt: float, t: float) -> float:
        return 0.0


@dataclass
class TemperatureLuedekingPiretProduct(ProductModel):
    """Luedeking-Piret production with temperature-dependent beta and product decay.

        dP/dt = alpha * max(dX/dt, 0) + beta(T) * X - kd(T) * P

        beta(T) = beta_ref * exp[-Ea_beta/R (1/T - 1/T_ref)]
        kd(T)   = kd_ref   * exp[-Ea_kd/R   (1/T - 1/T_ref)]

    The growth-associated term uses max(dX/dt, 0): biomass loss does not un-make product.
    Decay kd(T) is kept separate from beta(T) so that a temperature effect on stability is
    not mistaken for an effect on production capacity (fix kd from a stability test).

    alpha: growth-associated production coefficient (g product / g biomass), typical 0-10
    beta_ref: non-growth-associated coefficient at 30 C (1/h), typical 0-0.5
    Ea_beta: activation energy of beta (kJ/mol), typical 0-100
    kd_ref: product decay rate at 30 C (1/h), typical 0-0.1
    Ea_kd: activation energy of decay (kJ/mol), typical 0-100
    """

    alpha: float = 2.0
    beta_ref: float = 0.05
    Ea_beta: float = 30.0
    kd_ref: float = 0.0
    Ea_kd: float = 50.0
    name = "luedeking_piret_temp"
    condition_aware = True

    def __post_init__(self) -> None:
        require_nonnegative("alpha", self.alpha)
        require_nonnegative("beta_ref", self.beta_ref)
        require_nonnegative("Ea_beta", self.Ea_beta)
        require_nonnegative("kd_ref", self.kd_ref)
        require_nonnegative("Ea_kd", self.Ea_kd)

    def production_rate(
        self, X: float, P: float, dXdt: float, t: float, cond: Conditions | None = None
    ) -> float:
        cond = cond or Conditions()
        beta = self.beta_ref * arrhenius_factor(cond.T_C, self.Ea_beta)
        kd = self.kd_ref * arrhenius_factor(cond.T_C, self.Ea_kd)
        return self.alpha * max(dXdt, 0.0) + beta * X - kd * P
