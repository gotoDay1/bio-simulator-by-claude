from dataclasses import dataclass

from biosim.conditions import Conditions, arrhenius_factor
from biosim.models.base import ProductModel
from biosim.validation import require_nonnegative, require_positive


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


@dataclass
class InducibleLuedekingPiretProduct(ProductModel):
    """Luedeking-Piret production whose non-growth-associated term is switched by an inducer.

        dP/dt = alpha * max(dX/dt, 0) + beta(T, C) * X - kd(T) * P

        beta(T, C) = A(T) * [ beta_basal + beta_max * C^n / (EC50^n + C^n) ]
        A(T)  = exp[-Ea_beta/R (1/T - 1/T_ref)]        (Arrhenius, 1 at 30 C)
        kd(T) = kd_ref * exp[-Ea_kd/R (1/T - 1/T_ref)]

    C is the inducer concentration in the tank (uM), tracked as a state (inducible = True):
    it is raised by dosing events and diluted by the feed. beta_basal is expression without
    inducer, beta_max the extra capacity at saturating inducer, EC50 the concentration giving
    half of it. The induced term is capped by beta_max, so the response saturates (150/300/
    600 uM runs trace the saturation curve). Instantaneous response is assumed.

    alpha: growth-associated production coefficient (g product / g biomass), typical 0-10
    beta_basal: non-growth-associated coefficient without inducer at 30 C (1/h), typical 0-0.05
    beta_max: additional coefficient at saturating inducer at 30 C (1/h), typical 0-0.5
    EC50: inducer concentration for half-maximal induction (uM), typical 10-500
    hill_n: Hill coefficient of the induction curve (-), 1 = hyperbolic, typical 1-4
    Ea_beta: activation energy of beta (kJ/mol), typical 0-100
    kd_ref: product decay rate at 30 C (1/h), typical 0-0.1
    Ea_kd: activation energy of decay (kJ/mol), typical 0-100
    """

    alpha: float = 2.0
    beta_basal: float = 0.005
    beta_max: float = 0.05
    EC50: float = 100.0
    hill_n: float = 1.0
    Ea_beta: float = 30.0
    kd_ref: float = 0.0
    Ea_kd: float = 50.0
    name = "luedeking_piret_induced"
    condition_aware = True
    inducible = True

    def __post_init__(self) -> None:
        require_nonnegative("alpha", self.alpha)
        require_nonnegative("beta_basal", self.beta_basal)
        require_nonnegative("beta_max", self.beta_max)
        require_positive("EC50", self.EC50)
        require_positive("hill_n", self.hill_n)
        require_nonnegative("Ea_beta", self.Ea_beta)
        require_nonnegative("kd_ref", self.kd_ref)
        require_nonnegative("Ea_kd", self.Ea_kd)

    def induction_fraction(self, inducer: float) -> float:
        """C^n / (EC50^n + C^n), in [0, 1]."""
        c = max(inducer, 0.0) ** self.hill_n
        return c / (self.EC50**self.hill_n + c)

    def production_rate(
        self, X: float, P: float, dXdt: float, t: float, cond: Conditions | None = None
    ) -> float:
        cond = cond or Conditions()
        beta = arrhenius_factor(cond.T_C, self.Ea_beta) * (
            self.beta_basal + self.beta_max * self.induction_fraction(cond.inducer)
        )
        kd = self.kd_ref * arrhenius_factor(cond.T_C, self.Ea_kd)
        return self.alpha * max(dXdt, 0.0) + beta * X - kd * P
