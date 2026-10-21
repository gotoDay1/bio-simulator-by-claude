import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from biosim.conditions import ConditionProfile
from biosim.exceptions import FittingError, IntegrationError, InvalidParameterError
from biosim.inducer import InducerDose
from biosim.operation_modes import OperationMode
from biosim.results import SimulationResults
from biosim.simulation import BioreactorSimulation
from biosim.state import InitialConditions

CATEGORY_ORDER = ("growth", "product", "substrate", "oxygen")
_MEASUREMENT_COLUMNS = ("X", "S", "P")
_PENALTY_RESIDUAL_VALUE = 1e6
# Relative step for least_squares' finite-difference Jacobian. The default (~1.5e-8) is far
# below the ODE solver's own error (rtol=1e-6), so the numerical derivative of the
# simulated trajectory is mostly solver noise and the optimizer can stall far from the
# optimum (e.g. a shift-culture fit of the maintenance coefficient stopped at 0.0137 for a
# true 0.030). A step well above solver noise recovers it.
_FD_REL_STEP = 1e-4


@dataclass
class ParameterSpec:
    """One dataclass field of a model, either fixed at a value or free to fit.

    value is the fixed value when fixed=True, or the initial guess when fixed=False.
    is_optional marks Optional[float] fields (e.g. Yps): fixed=True with value=None
    means the field is disabled (passed as None) rather than fixed at a number.
    """

    name: str
    fixed: bool
    value: float | None
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    is_optional: bool = False

    def __post_init__(self) -> None:
        if self.fixed:
            if self.value is None and not self.is_optional:
                raise FittingError(
                    f"Parameter '{self.name}' is fixed but has no value (and is not optional)"
                )
        else:
            if self.value is None:
                raise FittingError(
                    f"Parameter '{self.name}' is marked free-to-fit but has no initial guess"
                )
            if self.lower_bound > self.upper_bound:
                raise FittingError(
                    f"Parameter '{self.name}' has lower_bound > upper_bound "
                    f"({self.lower_bound} > {self.upper_bound})"
                )
            if not (self.lower_bound <= self.value <= self.upper_bound):
                raise FittingError(
                    f"Parameter '{self.name}' initial guess {self.value} is outside "
                    f"its bounds [{self.lower_bound}, {self.upper_bound}]"
                )


@dataclass
class ModelSpec:
    """A chosen model class for one category (growth/product/substrate/oxygen) plus a
    ParameterSpec for each of its dataclass fields."""

    category: str
    model_cls: type
    params: list[ParameterSpec]


@dataclass
class FitResult:
    """Outcome of fitting one batch's experimental data against a fixed model configuration."""

    batch_name: str
    model_names: dict[str, str]
    param_values: dict[str, dict[str, float | None]]
    used_columns: list[str]
    success: bool
    message: str
    cost: float
    simulation_results: SimulationResults | None


def _generate_start_points(
    x0: np.ndarray, lower: np.ndarray, upper: np.ndarray, n_starts: int
) -> list[np.ndarray]:
    """Build n_starts candidate starting points for multi-start optimization.

    Candidate 0 is always x0 itself, so multi-start can only match or improve on a
    single-start fit from the user's own initial guess. The rest are log-space
    multiplicative jitters of x0 (0.1x-10x per free parameter, independently),
    clipped into bounds, using a fixed seed so the same inputs always fit the same way.
    """
    starts = [x0]
    if n_starts > 1:
        rng = np.random.default_rng(0)
        for _ in range(n_starts - 1):
            factors = 10.0 ** rng.uniform(-1.0, 1.0, size=x0.shape)
            candidate = np.clip(x0 * factors, lower, upper)
            starts.append(candidate)
    return starts


def fit_batch(
    batch_name: str,
    model_specs: list[ModelSpec],
    initial_conditions: InitialConditions,
    operation_mode: OperationMode,
    experimental_data: pd.DataFrame,
    n_points: int = 200,
    n_starts: int = 8,
    condition_profile: ConditionProfile | None = None,
    inducer_events: list[InducerDose] | None = None,
) -> FitResult:
    """Fit the free parameters of model_specs against experimental_data for one batch.

    Uses simulate-and-compare (build models -> run BioreactorSimulation -> interpolate at
    the experimental time points -> normalized residual) rather than closed-form regression,
    so it works for any registered model regardless of whether it has a closed-form solution
    (e.g. GompertzGrowth, which is a mechanistic ODE form).

    n_starts controls multi-start optimization: least_squares is a local optimizer, so a
    single run from one initial guess can settle into a local minimum (common with
    correlated parameters like Monod's mu_max/Ks). n_starts >= 1 candidate starting points
    are tried (see _generate_start_points) and the lowest-cost result is kept.

    condition_profile gives this batch's temperature/pH history (default: constant
    30 C / pH 5.0); each batch should be fit with its own profile. inducer_events gives
    the batch's inducer doses (requires an inducible product model).
    """
    specs_by_category = {spec.category: spec for spec in model_specs}
    missing = [c for c in CATEGORY_ORDER if c not in specs_by_category]
    if missing:
        raise FittingError(f"Missing model specs for categories: {missing}")

    for category in CATEGORY_ORDER:
        spec = specs_by_category[category]
        expected_names = {
            f.name for f in dataclasses.fields(spec.model_cls) if f.name != "feed_rate_fn"
        }
        actual_names = {p.name for p in spec.params}
        if expected_names != actual_names:
            raise FittingError(
                f"{category} model {spec.model_cls.__name__} parameter mismatch: "
                f"expected {sorted(expected_names)}, got {sorted(actual_names)}"
            )

    free_index_map: list[tuple[str, int]] = []
    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for category in CATEGORY_ORDER:
        spec = specs_by_category[category]
        for i, p in enumerate(spec.params):
            if not p.fixed:
                free_index_map.append((category, i))
                x0.append(p.value)
                lower.append(p.lower_bound)
                upper.append(p.upper_bound)

    if not free_index_map:
        raise FittingError("At least one parameter must be marked as free to fit.")

    oxygen_spec = specs_by_category["oxygen"]
    if oxygen_spec.model_cls.supports_supply_dynamics and initial_conditions.C_O2_0 is None:
        raise FittingError(
            f"{oxygen_spec.model_cls.__name__} tracks dissolved-O2 supply dynamics; "
            "initial_conditions.C_O2_0 must be set."
        )

    if "t" not in experimental_data.columns:
        raise FittingError("experimental_data must contain a 't' column.")
    t_max = float(experimental_data["t"].max())
    if t_max <= 0:
        raise FittingError(f"experimental_data 't' must reach beyond 0 (got max t={t_max}).")

    simulated_columns = set(_MEASUREMENT_COLUMNS)
    if oxygen_spec.model_cls.supports_supply_dynamics:
        simulated_columns.add("OTR")
    overlap_columns = sorted(simulated_columns & set(experimental_data.columns))
    if not overlap_columns:
        raise FittingError(
            "No overlapping measurement columns between experimental_data "
            f"({sorted(set(experimental_data.columns) - {'t'})}) and the columns the "
            f"selected models can produce ({sorted(simulated_columns)})."
        )

    column_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    residual_length = 0
    for col in overlap_columns:
        sub = experimental_data[["t", col]].dropna(subset=[col])
        if sub.empty:
            continue
        t_meas = sub["t"].to_numpy(dtype=float)
        y_meas = sub[col].to_numpy(dtype=float)
        scale = max(float(np.abs(y_meas).max()), 1e-8)
        column_data[col] = (t_meas, y_meas, scale)
        residual_length += len(t_meas)

    if residual_length == 0:
        raise FittingError("All overlapping measurement columns are empty after dropping NaNs.")

    def _build_kwargs(x: np.ndarray) -> dict[str, dict]:
        kwargs_by_category: dict[str, dict] = {}
        for category in CATEGORY_ORDER:
            spec = specs_by_category[category]
            kwargs_by_category[category] = {p.name: p.value for p in spec.params if p.fixed}
        for (category, i), xi in zip(free_index_map, x, strict=True):
            spec = specs_by_category[category]
            kwargs_by_category[category][spec.params[i].name] = float(xi)
        return kwargs_by_category

    def _build_simulation(x: np.ndarray) -> BioreactorSimulation:
        kwargs_by_category = _build_kwargs(x)
        growth_model = specs_by_category["growth"].model_cls(**kwargs_by_category["growth"])
        product_model = specs_by_category["product"].model_cls(**kwargs_by_category["product"])
        substrate_model = specs_by_category["substrate"].model_cls(
            **kwargs_by_category["substrate"]
        )
        oxygen_model = specs_by_category["oxygen"].model_cls(**kwargs_by_category["oxygen"])
        return BioreactorSimulation(
            growth_model=growth_model,
            product_model=product_model,
            substrate_model=substrate_model,
            oxygen_model=oxygen_model,
            operation_mode=operation_mode,
            initial_conditions=initial_conditions,
            t_span=(0.0, t_max),
            n_points=n_points,
            condition_profile=condition_profile,
            inducer_events=inducer_events,
        )

    x0_arr = np.array(x0, dtype=float)
    lower_arr = np.array(lower, dtype=float)
    upper_arr = np.array(upper, dtype=float)

    try:
        _build_simulation(x0_arr).run()
    except (InvalidParameterError, IntegrationError) as exc:
        raise FittingError(f"Initial parameter configuration is invalid: {exc}") from exc

    def residual(x: np.ndarray) -> np.ndarray:
        try:
            results = _build_simulation(x).run()
        except (InvalidParameterError, IntegrationError):
            return np.full(residual_length, _PENALTY_RESIDUAL_VALUE)
        parts = []
        for col, (t_meas, y_meas, scale) in column_data.items():
            sim_interp = np.interp(t_meas, results.data["t"], results.data[col])
            parts.append((sim_interp - y_meas) / scale)
        return np.concatenate(parts)

    if n_starts < 1:
        raise FittingError(f"n_starts must be >= 1 (got {n_starts}).")

    start_points = _generate_start_points(x0_arr, lower_arr, upper_arr, n_starts)
    opt = None
    for start in start_points:
        candidate = least_squares(
            residual,
            start,
            bounds=(lower_arr, upper_arr),
            max_nfev=2000,
            diff_step=_FD_REL_STEP,
        )
        if opt is None or candidate.cost < opt.cost:
            opt = candidate

    success = bool(opt.success)
    message = str(opt.message)
    if n_starts > 1:
        message = f"{message} (best of {n_starts} starts)"
    try:
        simulation_results: SimulationResults | None = _build_simulation(opt.x).run()
    except (InvalidParameterError, IntegrationError) as exc:
        simulation_results = None
        success = False
        message = f"{message} (final simulation at best-fit parameters failed: {exc})"

    kwargs_by_category = _build_kwargs(opt.x)
    param_values: dict[str, dict[str, float | None]] = {
        category: dict(kwargs_by_category[category]) for category in CATEGORY_ORDER
    }
    model_names = {
        category: specs_by_category[category].model_cls.name for category in CATEGORY_ORDER
    }
    cost = float(np.sqrt(np.mean(opt.fun**2)))

    return FitResult(
        batch_name=batch_name,
        model_names=model_names,
        param_values=param_values,
        used_columns=overlap_columns,
        success=success,
        message=message,
        cost=cost,
        simulation_results=simulation_results,
    )


def fit_results_to_dataframe(results: list[FitResult]) -> pd.DataFrame:
    """Build a wide-format table, one row per batch: batch name, model per category, every
    parameter's final value (fixed or fitted) per category, and fit-quality/status columns."""
    rows = []
    for r in results:
        row: dict[str, object] = {"batch": r.batch_name}
        for category in CATEGORY_ORDER:
            row[f"{category}_model"] = r.model_names[category]
            for name, value in r.param_values[category].items():
                row[f"{category}_{name}"] = value
        row["cost"] = r.cost
        row["success"] = r.success
        row["message"] = r.message
        rows.append(row)
    return pd.DataFrame(rows)
