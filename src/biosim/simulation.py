import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from biosim.conditions import ConditionProfile, Conditions
from biosim.exceptions import IntegrationError, InvalidParameterError
from biosim.models.base import GrowthModel, OxygenModel, ProductModel, SubstrateModel
from biosim.models.substrate import GrowthDrivenSubstrate
from biosim.operation_modes import OperationMode
from biosim.results import SimulationResults
from biosim.state import InitialConditions, StateLayout


class BioreactorSimulation:
    """Composes the four model categories + an operation mode into one ODE system and runs it.

    The RHS is built once per instance from a single general mass-balance form:
        dC/dt = reaction_rate(C) + (F_in/V) * (C_feed - C)
        dV/dt = F_in - F_out
    which collapses to batch/fed-batch/chemostat depending on what the
    OperationMode supplies for F_in(t), F_out(t), and feed concentrations
    (F_out only affects dV/dt - it cancels out of the concentration balance
    under the standard well-mixed-tank assumption).

    condition_profile supplies temperature/pH as prescribed inputs T(t), pH(t) (default:
    constant 30 C / pH 5.0). Models that declare condition_aware = True receive the
    current Conditions; all others are called exactly as before. A growth model with
    drives_substrate = True (e.g. PirtMonodGrowth) defines substrate uptake itself and
    must be paired with GrowthDrivenSubstrate (and vice versa).
    """

    def __init__(
        self,
        growth_model: GrowthModel,
        product_model: ProductModel,
        substrate_model: SubstrateModel,
        oxygen_model: OxygenModel,
        operation_mode: OperationMode,
        initial_conditions: InitialConditions,
        t_span: tuple[float, float],
        n_points: int = 200,
        solver_kwargs: dict | None = None,
        condition_profile: ConditionProfile | None = None,
    ):
        self.growth_model = growth_model
        self.product_model = product_model
        self.substrate_model = substrate_model
        self.oxygen_model = oxygen_model
        self.operation_mode = operation_mode
        self.initial_conditions = initial_conditions
        self.t_span = t_span
        self.n_points = n_points
        self.solver_kwargs = solver_kwargs or {}
        self.condition_profile = condition_profile or ConditionProfile()
        if growth_model.requires_oxygen_state and not oxygen_model.supports_supply_dynamics:
            raise InvalidParameterError(
                f"{type(growth_model).__name__} requires an oxygen model that tracks "
                f"dissolved-O2 supply dynamics (e.g. OxygenWithKLa), but "
                f"{type(oxygen_model).__name__} does not track C_O2 as a state."
            )
        if growth_model.drives_substrate and not isinstance(
            substrate_model, GrowthDrivenSubstrate
        ):
            raise InvalidParameterError(
                f"{type(growth_model).__name__} defines substrate uptake itself and must be "
                f"paired with GrowthDrivenSubstrate, not {type(substrate_model).__name__}."
            )
        if isinstance(substrate_model, GrowthDrivenSubstrate) and not growth_model.drives_substrate:
            raise InvalidParameterError(
                "GrowthDrivenSubstrate needs a growth model that defines substrate uptake "
                f"(drives_substrate=True); {type(growth_model).__name__} does not."
            )
        self.layout = StateLayout(oxygen_model.supports_supply_dynamics)

    def _growth_rate(self, X: float, S: float, t: float, C_O2, cond: Conditions) -> float:
        if self.growth_model.condition_aware:
            return self.growth_model.specific_growth_rate(X, S, t, C_O2, cond=cond)
        return self.growth_model.specific_growth_rate(X, S, t, C_O2)

    def _product_rate(self, X: float, P: float, dXdt: float, t: float, cond: Conditions) -> float:
        if self.product_model.condition_aware:
            return self.product_model.production_rate(X, P, dXdt, t, cond=cond)
        return self.product_model.production_rate(X, P, dXdt, t)

    def _substrate_rate(
        self, X: float, S: float, t: float, C_O2, dXdt: float, dPdt: float, cond: Conditions
    ) -> float:
        if self.growth_model.drives_substrate:
            qS = self.growth_model.specific_substrate_uptake(X, S, t, C_O2, cond=cond)
            return -qS * X
        return self.substrate_model.consumption_rate(X, dXdt, dPdt, t)

    def _rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        X, S, P, C_O2, V = self.layout.unpack(y)

        cond = self.condition_profile.at(t)
        mu = self._growth_rate(X, S, t, C_O2, cond)
        dXdt_rxn = mu * X
        dPdt_rxn = self._product_rate(X, P, dXdt_rxn, t, cond)
        dSdt_rxn = self._substrate_rate(X, S, t, C_O2, dXdt_rxn, dPdt_rxn, cond)
        our = self.oxygen_model.demand_rate(X, dXdt_rxn, t)

        F_in = self.operation_mode.inflow_rate(t, V)
        F_out = self.operation_mode.outflow_rate(t, V)
        dilution = F_in / V

        dydt = np.zeros_like(y)
        dydt[self.layout.idx_X] = dXdt_rxn + dilution * (self.operation_mode.X_feed - X)

        dS = dSdt_rxn + dilution * (self.operation_mode.S_feed - S)
        dydt[self.layout.idx_S] = self._floor_rate_near_zero(S, dS)

        dydt[self.layout.idx_P] = dPdt_rxn + dilution * (self.operation_mode.P_feed - P)
        if self.layout.has_oxygen_state:
            otr = self.oxygen_model.supply_rate(C_O2, t)
            dC_O2 = our + otr + dilution * (self.operation_mode.C_O2_feed - C_O2)
            dydt[self.layout.idx_C_O2] = self._floor_rate_near_zero(C_O2, dC_O2)
        dydt[self.layout.idx_V] = F_in - F_out
        return dydt

    _RATE_FLOOR_EPS = 1e-6

    @classmethod
    def _floor_rate_near_zero(cls, state: float, rate: float) -> float:
        """Keeps a depleting state (S or C_O2) from going negative without introducing a
        hard discontinuity in the ODE right-hand side.

        A plain `if state <= 0: rate = 0` clamp creates a genuine jump discontinuity at the
        boundary (the unclamped rate does not vanish as state -> 0, since maintenance terms
        like -ms*X are independent of the depleting state itself), which adaptive stiff
        solvers such as LSODA resolve very poorly - integration calls near/after depletion
        can become orders of magnitude slower (observed in fitting, which repeatedly
        integrates many nearby parameter sets). Instead, a negative rate is linearly ramped
        to zero over the tiny window [0, eps], matching the unclamped rate continuously at
        `state=eps` and reaching exactly zero at `state=0`; the state can never cross zero
        and the vector field stays continuous everywhere.
        """
        if rate >= 0.0 or state >= cls._RATE_FLOOR_EPS:
            return rate
        blend = max(state, 0.0) / cls._RATE_FLOOR_EPS
        return rate * blend

    def run(self) -> SimulationResults:
        y0 = self.layout.build_initial_vector(self.initial_conditions)
        t_eval = np.linspace(self.t_span[0], self.t_span[1], self.n_points)

        solver_kwargs = {"method": "LSODA", "rtol": 1e-6, "atol": 1e-9}
        solver_kwargs.update(self.solver_kwargs)

        sol = solve_ivp(self._rhs, self.t_span, y0, t_eval=t_eval, **solver_kwargs)
        if not sol.success:
            raise IntegrationError(f"ODE integration failed: {sol.message}")

        return self._build_results(sol)

    def _build_results(self, sol) -> SimulationResults:
        data: dict[str, np.ndarray] = {"t": sol.t}
        for i, name in enumerate(self.layout.column_names()):
            data[name] = sol.y[i]

        n = len(sol.t)
        mu_vals = np.zeros(n)
        our_vals = np.zeros(n)
        otr_vals = np.zeros(n) if self.layout.has_oxygen_state else None

        for i in range(n):
            X, S, _P, C_O2, _V = self.layout.unpack(sol.y[:, i])
            t = sol.t[i]
            mu = self._growth_rate(X, S, t, C_O2, self.condition_profile.at(t))
            mu_vals[i] = mu
            our_vals[i] = self.oxygen_model.demand_rate(X, mu * X, t)
            if otr_vals is not None:
                otr_vals[i] = self.oxygen_model.supply_rate(C_O2, t)

        data["mu"] = mu_vals
        data["OUR"] = our_vals
        if otr_vals is not None:
            data["OTR"] = otr_vals

        df = pd.DataFrame(data)
        self._warn_if_negative(df)
        return SimulationResults(df)

    @staticmethod
    def _warn_if_negative(df: pd.DataFrame, tol: float = -1e-3) -> None:
        for col in ("X", "S", "P", "C_O2", "V"):
            if col in df.columns and (df[col] < tol).any():
                warnings.warn(
                    f"Simulated '{col}' went negative (min={df[col].min():.4g}). "
                    "'S' and 'C_O2' are floored near zero once depleted (their consumption "
                    "rate is smoothly ramped to zero as they approach zero, see "
                    "_floor_rate_near_zero), so more than a tiny numerical undershoot here "
                    "is an unexpected artifact rather than expected maintenance-term "
                    "behavior; results should be inspected. "
                    "Note that non-growth-associated product formation (the beta*X term "
                    "in LuedekingPiretProduct) is not gated by substrate/oxygen depletion "
                    "and keeps producing 'P' as long as biomass is present.",
                    stacklevel=2,
                )
