"""Joint fitting of one set of model parameters to several cultures at once.

fit_batch fits one culture at a time. Many quantities are only identifiable across cultures:
the yield Y and the maintenance coefficient m separate only when the feed level differs, the
activation energy Ea only when temperature differs, EC50 only when the inducer dose differs.
fit_joint therefore simulates every culture (each with its own feed, temperature/pH history and
inducer doses, see biosim.culture.Culture), stacks the residuals and minimises them together,
sharing the free parameters of the model specs across all cultures.

Residuals are normalised per measured column (X, S, P) by a common scale, so cultures and
quantities with different units are weighed on the same footing. Uncertainty comes from the
Jacobian at the optimum (a linearised estimate, s^2 (J^T J)^-1 with s^2 the residual variance):
standard errors, the parameter correlation matrix and a condition number that says how well the
parameters are separated. Parameter pairs with |correlation| near 1, and parameters at a bound,
are flagged in `JointFitResult.flags`, as are parameters whose relative standard error is large.

fit_staged runs a sequence of such fits, freeing a few parameters per stage on a chosen subset
of cultures and fixing what earlier stages found (e.g. growth parameters from the standard and
feed-variation cultures first, then temperature, then production). A stage can also restrict
which measured columns it fits: fitting the biomass (X) alone while the product parameters are
still unknown keeps wrong product guesses from distorting the growth parameters.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from biosim.culture import Culture
from biosim.exceptions import FittingError, IntegrationError, InvalidParameterError
from biosim.fitting import (
    _FD_REL_STEP,
    _PENALTY_RESIDUAL_VALUE,
    CATEGORY_ORDER,
    ModelSpec,
    ParameterSpec,
    _generate_start_points,
)
from biosim.results import SimulationResults
from biosim.simulation import BioreactorSimulation

_FITTED_COLUMNS = ("X", "S", "P")


@dataclass
class JointFitResult:
    """Outcome of one joint fit.

    estimates: one row per free parameter ("category.name"): value, std_error (linearised),
        rel_error (std_error/|value|), lower, upper, at_bound.
    correlation: parameter correlation matrix (None if it could not be computed).
    rmse_by_culture: RMS of the normalised residuals of each culture.
    condition_number: condition number of the relative-sensitivity Jacobian; large values
        (>1e4) mean some combination of parameters is barely constrained by the data.
    flags: human-readable warnings (strongly correlated pairs, parameters at a bound, ...).
    """

    param_values: dict[str, dict[str, float | None]]
    free_names: list[str]
    estimates: pd.DataFrame
    correlation: pd.DataFrame | None
    rmse_by_culture: dict[str, float]
    rmse_total: float
    condition_number: float
    success: bool
    message: str
    n_residuals: int
    simulations: dict[str, SimulationResults | None]
    flags: list[str] = field(default_factory=list)


@dataclass
class FitStage:
    """One stage of fit_staged: free the parameters in `free` ("category.name") on `cultures`
    (names; None = all cultures), fitting only the measured `columns` (subset of X, S, P;
    None = every column), with every other parameter fixed at its current value."""

    name: str
    free: Sequence[str]
    cultures: Sequence[str] | None = None
    columns: Sequence[str] | None = None


def _validate_specs(model_specs: list[ModelSpec]) -> dict[str, ModelSpec]:
    by_category = {spec.category: spec for spec in model_specs}
    missing = [c for c in CATEGORY_ORDER if c not in by_category]
    if missing:
        raise FittingError(f"Missing model specs for categories: {missing}")
    for category in CATEGORY_ORDER:
        spec = by_category[category]
        expected = {f.name for f in dataclasses.fields(spec.model_cls)}
        actual = {p.name for p in spec.params}
        if expected != actual:
            raise FittingError(
                f"{category} model {spec.model_cls.__name__} parameter mismatch: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )
    return by_category


def fit_joint(
    cultures: Sequence[Culture],
    model_specs: list[ModelSpec],
    *,
    n_points: int = 200,
    n_starts: int = 4,
    sigma: dict[str, float] | None = None,
    columns: Sequence[str] | None = None,
    corr_threshold: float = 0.95,
    rel_error_threshold: float = 0.5,
) -> JointFitResult:
    """Fit the free parameters of model_specs to all cultures at once (see module docstring).

    sigma: optional absolute scale per measured column (e.g. {"X": 0.5, "P": 0.05}) used to
        normalise residuals; the default is the largest absolute measurement of that column
        over all cultures.
    columns: restrict the fit to these measured columns (subset of X, S, P); default all.
    n_starts: multi-start count (the first start is the specs' own initial guess).
    corr_threshold: |correlation| at or above which a parameter pair is flagged.
    rel_error_threshold: relative standard error above which a parameter is flagged as
        poorly determined.
    """
    if not cultures:
        raise FittingError("At least one culture is required.")
    names = [c.name for c in cultures]
    if len(set(names)) != len(names):
        raise FittingError(f"Culture names must be unique, got {names}")
    if n_starts < 1:
        raise FittingError(f"n_starts must be >= 1 (got {n_starts}).")
    specs = _validate_specs(model_specs)

    free_map: list[tuple[str, int]] = []
    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for category in CATEGORY_ORDER:
        for i, p in enumerate(specs[category].params):
            if not p.fixed:
                free_map.append((category, i))
                x0.append(p.value)
                lower.append(p.lower_bound)
                upper.append(p.upper_bound)
    if not free_map:
        raise FittingError("At least one parameter must be marked as free to fit.")
    free_names = [f"{cat}.{specs[cat].params[i].name}" for cat, i in free_map]

    if any(c.inducer_events for c in cultures) and not specs["product"].model_cls.inducible:
        raise FittingError(
            "Some cultures have inducer doses but the product model "
            f"{specs['product'].model_cls.__name__} does not respond to an inducer."
        )
    oxygen_cls = specs["oxygen"].model_cls
    if oxygen_cls.supports_supply_dynamics and any(
        c.initial_conditions.C_O2_0 is None for c in cultures
    ):
        raise FittingError(f"{oxygen_cls.__name__} needs C_O2_0 in every culture's initial conditions.")

    fitted = tuple(_FITTED_COLUMNS if columns is None else columns)
    bad = [c for c in fitted if c not in _FITTED_COLUMNS]
    if bad:
        raise FittingError(f"columns must be a subset of {_FITTED_COLUMNS}, got {bad}")

    # measurements: culture -> column -> (t, y)
    measurements: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for c in cultures:
        cols: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for col in fitted:
            if col in c.data.columns:
                sub = c.data[["t", col]].dropna()
                if not sub.empty:
                    cols[col] = (sub["t"].to_numpy(float), sub[col].to_numpy(float))
        if not cols:
            raise FittingError(
                f"Culture '{c.name}' has no measurements in the fitted columns {list(fitted)}."
            )
        measurements[c.name] = cols

    scale: dict[str, float] = {}
    for col in fitted:
        values = [np.abs(m[col][1]).max() for m in measurements.values() if col in m]
        if values:
            scale[col] = max(float(max(values)), 1e-8)
    if sigma:
        for col, s in sigma.items():
            if col not in scale:
                raise FittingError(f"sigma given for column '{col}', which no culture measures.")
            if s <= 0:
                raise FittingError(f"sigma['{col}'] must be positive.")
            scale[col] = float(s)

    def kwargs_for(x: np.ndarray) -> dict[str, dict]:
        out = {cat: {p.name: p.value for p in specs[cat].params if p.fixed} for cat in CATEGORY_ORDER}
        for (cat, i), xi in zip(free_map, x, strict=True):
            out[cat][specs[cat].params[i].name] = float(xi)
        return out

    def simulate(culture: Culture, kwargs: dict[str, dict]) -> SimulationResults:
        models = {cat: specs[cat].model_cls(**kwargs[cat]) for cat in CATEGORY_ORDER}
        return BioreactorSimulation(
            growth_model=models["growth"],
            product_model=models["product"],
            substrate_model=models["substrate"],
            oxygen_model=models["oxygen"],
            operation_mode=culture.operation_mode(),
            initial_conditions=culture.initial_conditions,
            t_span=(0.0, culture.t_end),
            n_points=n_points,
            condition_profile=culture.condition_profile,
            inducer_events=culture.inducer_events or None,
        ).run()

    sizes = {c.name: sum(len(v[1]) for v in measurements[c.name].values()) for c in cultures}

    def residual_blocks(x: np.ndarray) -> dict[str, np.ndarray]:
        kwargs = kwargs_for(x)
        blocks: dict[str, np.ndarray] = {}
        for c in cultures:
            try:
                res = simulate(c, kwargs)
            except (InvalidParameterError, IntegrationError):
                blocks[c.name] = np.full(sizes[c.name], _PENALTY_RESIDUAL_VALUE)
                continue
            parts = []
            for col, (t_meas, y_meas) in measurements[c.name].items():
                sim = np.interp(t_meas, res.data["t"], res.data[col])
                parts.append((sim - y_meas) / scale[col])
            blocks[c.name] = np.concatenate(parts)
        return blocks

    def residual(x: np.ndarray) -> np.ndarray:
        return np.concatenate(list(residual_blocks(x).values()))

    x0_arr = np.array(x0, float)
    lo, hi = np.array(lower, float), np.array(upper, float)
    try:
        for c in cultures:
            simulate(c, kwargs_for(x0_arr))
    except (InvalidParameterError, IntegrationError) as exc:
        raise FittingError(f"Initial parameter configuration is invalid: {exc}") from exc

    best = None
    for start in _generate_start_points(x0_arr, lo, hi, n_starts):
        cand = least_squares(
            residual, start, bounds=(lo, hi), max_nfev=2000, diff_step=_FD_REL_STEP
        )
        if best is None or cand.cost < best.cost:
            best = cand

    x_hat = best.x
    n_res, n_par = best.fun.size, x_hat.size

    # linearised uncertainty from the relative-sensitivity Jacobian
    scale_x = np.maximum(np.abs(x_hat), 1e-12)
    j_rel = best.jac * scale_x  # d residual / d (relative change of parameter)
    sv = np.linalg.svd(j_rel, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    std_error = np.full(n_par, np.nan)
    corr_df: pd.DataFrame | None = None
    if n_res > n_par:
        s2 = 2.0 * best.cost / (n_res - n_par)
        cov_rel = s2 * np.linalg.pinv(j_rel.T @ j_rel)
        std_rel = np.sqrt(np.clip(np.diag(cov_rel), 0.0, None))
        std_error = std_rel * scale_x
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov_rel / np.outer(std_rel, std_rel)
        corr_df = pd.DataFrame(corr, index=free_names, columns=free_names)

    at_bound = [
        bool(np.isclose(v, l_, rtol=1e-6, atol=1e-12) or np.isclose(v, u_, rtol=1e-6, atol=1e-12))
        for v, l_, u_ in zip(x_hat, lo, hi, strict=True)
    ]
    estimates = pd.DataFrame(
        {
            "value": x_hat,
            "std_error": std_error,
            "rel_error": std_error / scale_x,
            "lower": lo,
            "upper": hi,
            "at_bound": at_bound,
        },
        index=pd.Index(free_names, name="parameter"),
    )

    flags: list[str] = []
    if corr_df is not None:
        for i in range(n_par):
            for j in range(i + 1, n_par):
                r = corr_df.iat[i, j]
                if np.isfinite(r) and abs(r) >= corr_threshold:
                    flags.append(
                        f"{free_names[i]} and {free_names[j]} are strongly correlated "
                        f"(r = {r:+.3f}): the data do not separate them"
                    )
    flags += [f"{n} is at a bound" for n, b in zip(free_names, at_bound, strict=True) if b]
    for n, rel in zip(free_names, estimates["rel_error"], strict=True):
        if np.isfinite(rel) and rel > rel_error_threshold:
            flags.append(
                f"{n} is poorly determined (relative standard error {rel:.0%}): "
                "the data barely constrain it"
            )
    if n_res <= n_par:
        flags.append("not enough measurements for the number of free parameters")
    if not np.isfinite(cond) or cond > 1e4:
        flags.append(f"Jacobian condition number is {cond:.3g}: some parameter combination is unconstrained")

    blocks = residual_blocks(x_hat)
    kwargs_final = kwargs_for(x_hat)
    simulations: dict[str, SimulationResults | None] = {}
    for c in cultures:
        try:
            simulations[c.name] = simulate(c, kwargs_final)
        except (InvalidParameterError, IntegrationError):
            simulations[c.name] = None

    return JointFitResult(
        param_values={cat: dict(kwargs_final[cat]) for cat in CATEGORY_ORDER},
        free_names=free_names,
        estimates=estimates,
        correlation=corr_df,
        rmse_by_culture={n: float(np.sqrt(np.mean(b**2))) for n, b in blocks.items()},
        rmse_total=float(np.sqrt(np.mean(best.fun**2))),
        condition_number=cond,
        success=bool(best.success),
        message=str(best.message),
        n_residuals=int(n_res),
        simulations=simulations,
        flags=flags,
    )


def fit_staged(
    cultures: Sequence[Culture],
    model_specs: list[ModelSpec],
    stages: Sequence[FitStage],
    **fit_kwargs,
) -> list[tuple[FitStage, JointFitResult]]:
    """Run fit_joint stage by stage, carrying the fitted values forward.

    model_specs give every parameter's initial value and bounds; their `fixed` flags are ignored
    here: in each stage exactly the parameters named in stage.free are free and all others are
    held at the best values found so far. Extra keyword arguments go to fit_joint (a stage's own
    `columns` overrides one given here).
    """
    specs = _validate_specs(model_specs)
    all_names = {f"{cat}.{p.name}" for cat in CATEGORY_ORDER for p in specs[cat].params}
    by_name = {c.name: c for c in cultures}
    current = {
        cat: {p.name: p.value for p in specs[cat].params} for cat in CATEGORY_ORDER
    }

    results: list[tuple[FitStage, JointFitResult]] = []
    for stage in stages:
        unknown = sorted(set(stage.free) - all_names)
        if unknown:
            raise FittingError(f"Stage '{stage.name}': unknown parameters {unknown}")
        if stage.cultures is None:
            subset = list(cultures)
        else:
            missing = [n for n in stage.cultures if n not in by_name]
            if missing:
                raise FittingError(f"Stage '{stage.name}': unknown cultures {missing}")
            subset = [by_name[n] for n in stage.cultures]

        stage_specs = []
        for cat in CATEGORY_ORDER:
            params = []
            for p in specs[cat].params:
                value = current[cat][p.name]
                if f"{cat}.{p.name}" in stage.free:
                    guess = float(np.clip(value, p.lower_bound, p.upper_bound))
                    params.append(
                        ParameterSpec(p.name, False, guess, p.lower_bound, p.upper_bound)
                    )
                else:
                    params.append(ParameterSpec(p.name, True, value, is_optional=p.is_optional))
            stage_specs.append(ModelSpec(cat, specs[cat].model_cls, params))

        stage_kwargs = dict(fit_kwargs)
        if stage.columns is not None:
            stage_kwargs["columns"] = stage.columns
        result = fit_joint(subset, stage_specs, **stage_kwargs)
        for cat in CATEGORY_ORDER:
            current[cat].update(result.param_values[cat])
        results.append((stage, result))
    return results
