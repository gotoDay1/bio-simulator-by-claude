import numpy as np
import pandas as pd
import pytest

from biosim import (
    BioreactorSimulation,
    ConditionProfile,
    FitStage,
    FittingError,
    GrowthDrivenSubstrate,
    InducerDose,
    InducibleLuedekingPiretProduct,
    InitialConditions,
    LuedekingPiretProduct,
    ModelSpec,
    OxygenDemandOnly,
    ParameterSpec,
    PirtMonodGrowth,
    fit_joint,
    fit_staged,
    shift_value,
    stepwise_feed,
)
from biosim.culture import Culture
from biosim.operation_modes import FedBatch

S_FEED = 100.0
IC = InitialConditions(X0=0.4, S0=0.0, P0=0.0, V0=1.0)
TRUE_GROWTH = {
    "Yxs_ref": 0.5,
    "Yxs_slope": 0.0,
    "ms_ref": 0.02,
    "Ea_ms": 60.0,
    "ms_pH_a": 0.0,
    "ms_pH_b": 0.0,
    "qSmax_ref": 50.0,
    "Ea_qS": 0.0,
    "Ks": 1e-3,
}
TRUE_PRODUCT = {
    "alpha": 0.0,
    "beta_basal": 0.002,
    "beta_max": 0.05,
    "EC50": 200.0,
    "hill_n": 1.0,
    "Ea_beta": 30.0,
    "kd_ref": 0.0,
    "Ea_kd": 50.0,
}
FEED_TIMES = [0.0, 24.0, 48.0, 72.0]
FEED_RATES = [0.02, 0.03, 0.04, 0.05]  # the "standard recipe": stepwise increase (L/h)
SAMPLE_TIMES = np.arange(0.0, 97.0, 8.0)


def _simulate_truth(feed_scale, T_after, conc):
    feed = stepwise_feed(FEED_TIMES, [r * feed_scale for r in FEED_RATES])
    profile = ConditionProfile(shift_value(30.0, T_after, 48.0) if T_after else None)
    doses = [InducerDose(t, conc, "add") for t in (32.0, 56.0, 80.0)] if conc else []
    sim = BioreactorSimulation(
        growth_model=PirtMonodGrowth(**TRUE_GROWTH),
        product_model=InducibleLuedekingPiretProduct(**TRUE_PRODUCT),
        substrate_model=GrowthDrivenSubstrate(),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
        operation_mode=FedBatch(feed_rate_fn=feed, S_feed=S_FEED),
        initial_conditions=IC,
        t_span=(0.0, 96.0),
        n_points=385,
        condition_profile=profile,
        inducer_events=doses,
    )
    return sim.run().data, feed, profile, doses


def make_culture(name, feed_scale=1.0, T_after=None, conc=300.0, noise=0.01, seed=0):
    d, feed, profile, doses = _simulate_truth(feed_scale, T_after, conc)
    rng = np.random.default_rng(seed)
    rows = {"t": SAMPLE_TIMES}
    for col in ("X", "P"):
        y = np.interp(SAMPLE_TIMES, d["t"], d[col])
        rows[col] = y * (1.0 + noise * rng.standard_normal(y.size))
    data = pd.DataFrame(rows)
    data.loc[0, "P"] = np.nan  # no product measured at t = 0
    return Culture(name, data, profile, doses, feed, S_FEED, IC)


@pytest.fixture(scope="module")
def cultures():
    return [
        make_culture("standard", seed=1),
        make_culture("recipe2", feed_scale=0.6, seed=2),
        make_culture("T24", T_after=24.0, seed=3),
        make_culture("T36", T_after=36.0, seed=4),
        make_culture("Cu150", conc=150.0, seed=5),
        make_culture("Cu600", conc=600.0, seed=6),
    ]


def _specs(true_growth=None, true_product=None, free=(), bounds=None, guess=None):
    """Specs with every parameter fixed at its true value except those named in `free`
    ("growth.ms_ref", ...), which start from `guess` (or the true value) within `bounds`."""
    bounds = bounds or {}
    guess = guess or {}
    out = []
    for cat, cls, truth in (
        ("growth", PirtMonodGrowth, true_growth or TRUE_GROWTH),
        ("product", InducibleLuedekingPiretProduct, true_product or TRUE_PRODUCT),
    ):
        params = []
        for name, value in truth.items():
            key = f"{cat}.{name}"
            if key in free:
                lo, hi = bounds.get(key, (value / 20.0, value * 20.0))
                params.append(ParameterSpec(name, False, guess.get(key, value), lo, hi))
            else:
                params.append(ParameterSpec(name, True, value))
        out.append(ModelSpec(cat, cls, params))
    out.append(ModelSpec("substrate", GrowthDrivenSubstrate, []))
    out.append(
        ModelSpec(
            "oxygen",
            OxygenDemandOnly,
            [ParameterSpec("Yxo2", True, 0.9), ParameterSpec("mo2", True, 0.05)],
        )
    )
    return out


def test_joint_fit_recovers_yield_maintenance_and_activation_energy(cultures):
    free = ("growth.Yxs_ref", "growth.ms_ref", "growth.Ea_ms")
    guess = {"growth.Yxs_ref": 0.38, "growth.ms_ref": 0.03, "growth.Ea_ms": 40.0}
    res = fit_joint(cultures, _specs(free=free, guess=guess), n_starts=1)
    assert res.success
    est = res.estimates["value"]
    assert est["growth.Yxs_ref"] == pytest.approx(0.5, rel=0.03)
    assert est["growth.ms_ref"] == pytest.approx(0.02, rel=0.05)
    assert est["growth.Ea_ms"] == pytest.approx(60.0, rel=0.05)
    assert set(res.rmse_by_culture) == {c.name for c in cultures}
    assert res.rmse_total < 0.05  # ~1% measurement noise on normalised residuals


def test_estimates_table_and_uncertainty_are_consistent(cultures):
    free = ("growth.Yxs_ref", "growth.ms_ref")
    res = fit_joint(cultures, _specs(free=free, guess={"growth.ms_ref": 0.03}), n_starts=1)
    assert list(res.estimates.index) == ["growth.Yxs_ref", "growth.ms_ref"]
    assert set(res.estimates.columns) >= {"value", "std_error", "rel_error", "at_bound"}
    assert (res.estimates["std_error"] > 0).all()
    assert (res.estimates["rel_error"] < 0.2).all()  # well determined by this data set
    assert res.correlation.shape == (2, 2)
    assert np.allclose(np.diag(res.correlation.to_numpy()), 1.0)
    assert set(res.simulations) == {c.name for c in cultures}


_YM_FREE = ("growth.Yxs_ref", "growth.ms_ref")
_YM_GUESS = {"growth.Yxs_ref": 0.4, "growth.ms_ref": 0.03}


def _ym_fit(selected):
    return fit_joint(selected, _specs(free=_YM_FREE, guess=_YM_GUESS), n_starts=1)


def test_feed_level_alone_does_not_decorrelate_yield_and_maintenance(cultures):
    # dx/dt = Y (F Sf - m x) is linear in x: scaling the feed only scales the solution, so a
    # second feed *level* adds no new information about Y versus m (it just averages noise).
    one = _ym_fit(cultures[:1])
    two = _ym_fit(cultures[:2])  # standard + recipe2 (feed x0.6)
    assert abs(abs(two.correlation.iat[0, 1]) - abs(one.correlation.iat[0, 1])) < 0.03


def test_more_cultures_tighten_estimates_and_a_hot_shift_lowers_the_y_m_correlation(cultures):
    one = _ym_fit(cultures[:1])
    hot = _ym_fit([cultures[0], cultures[3]])  # standard + 36 C shift
    four = _ym_fit(cultures[:4])  # standard, recipe2, T24, T36
    # a hot shift raises m, which bends the curve more and separates m from Y
    assert abs(hot.correlation.iat[0, 1]) < abs(one.correlation.iat[0, 1]) - 0.02
    for res in (hot, four):
        assert res.estimates.loc["growth.ms_ref", "rel_error"] < one.estimates.loc["growth.ms_ref", "rel_error"]
        assert res.estimates.loc["growth.Yxs_ref", "rel_error"] < one.estimates.loc["growth.Yxs_ref", "rel_error"]
    assert four.estimates.loc["growth.ms_ref", "rel_error"] < 0.5 * one.estimates.loc["growth.ms_ref", "rel_error"]


def test_unidentifiable_parameters_are_flagged(cultures):
    # strictly feed-limited growth (30 C only) is insensitive to qSmax and Ks
    free = ("growth.qSmax_ref", "growth.Ks", "growth.ms_ref")
    limited = [c for c in cultures if c.name in ("standard", "recipe2")]
    res = fit_joint(
        limited,
        _specs(free=free, bounds={"growth.qSmax_ref": (5.0, 500.0), "growth.Ks": (1e-5, 1.0)}),
        n_starts=1,
    )
    assert res.estimates.loc["growth.qSmax_ref", "rel_error"] > 0.5
    assert any("growth.qSmax_ref is poorly determined" in f for f in res.flags)
    # while the parameter that the data do constrain is not flagged
    assert not any("growth.ms_ref is poorly determined" in f for f in res.flags)


def test_staged_fit_recovers_growth_then_temperature_then_production(cultures):
    specs = _specs(
        free=(
            "growth.Yxs_ref",
            "growth.ms_ref",
            "growth.Ea_ms",
            "product.beta_max",
            "product.EC50",
        ),
        guess={
            "growth.Yxs_ref": 0.38,
            "growth.ms_ref": 0.03,
            "growth.Ea_ms": 40.0,
            "product.beta_max": 0.02,
            "product.EC50": 80.0,
        },
    )
    stages = [
        # biomass only: the product parameters are still unknown at this point
        FitStage("growth at 30 C", ["growth.Yxs_ref", "growth.ms_ref"], ["standard", "recipe2"], ["X"]),
        FitStage("temperature", ["growth.Ea_ms"], ["standard", "T24", "T36"], ["X"]),
        FitStage(
            "production", ["product.beta_max", "product.EC50"], ["standard", "Cu150", "Cu600"], ["P"]
        ),
    ]
    out = fit_staged(cultures, specs, stages, n_starts=1)
    assert [s.name for s, _ in out] == [s.name for s in stages]
    final = out[-1][1].param_values
    assert final["growth"]["Yxs_ref"] == pytest.approx(0.5, rel=0.05)
    assert final["growth"]["ms_ref"] == pytest.approx(0.02, rel=0.1)
    assert final["growth"]["Ea_ms"] == pytest.approx(60.0, rel=0.1)
    assert final["product"]["beta_max"] == pytest.approx(0.05, rel=0.15)
    assert final["product"]["EC50"] == pytest.approx(200.0, rel=0.3)
    # earlier stages' values are carried forward, not re-fitted
    assert out[1][1].param_values["growth"]["Yxs_ref"] == out[0][1].param_values["growth"]["Yxs_ref"]


def test_staged_fit_rejects_unknown_names(cultures):
    specs = _specs(free=("growth.ms_ref",))
    with pytest.raises(FittingError, match="unknown parameters"):
        fit_staged(cultures, specs, [FitStage("x", ["growth.nope"])])
    with pytest.raises(FittingError, match="unknown cultures"):
        fit_staged(cultures, specs, [FitStage("x", ["growth.ms_ref"], ["nope"])])


def test_fitting_the_wrong_columns_is_rejected_or_restricted(cultures):
    specs = _specs(free=("growth.ms_ref",))
    with pytest.raises(FittingError, match="subset"):
        fit_joint(cultures, specs, columns=["OD"])
    only_x = fit_joint(cultures[:2], specs, columns=["X"], n_starts=1)
    both = fit_joint(cultures[:2], specs, n_starts=1)
    assert only_x.n_residuals < both.n_residuals


def test_input_validation(cultures):
    specs = _specs(free=("growth.ms_ref",))
    with pytest.raises(FittingError, match="unique"):
        fit_joint([cultures[0], cultures[0]], specs)
    with pytest.raises(FittingError, match="sigma"):
        fit_joint(cultures, specs, sigma={"S": 1.0})
    with pytest.raises(FittingError, match="at least one parameter|At least one"):
        fit_joint(cultures, _specs(free=()))
    plain = _specs(free=("growth.ms_ref",))
    plain[1] = ModelSpec(
        "product",
        LuedekingPiretProduct,
        [ParameterSpec("alpha", True, 0.0), ParameterSpec("beta", True, 0.02)],
    )
    with pytest.raises(FittingError, match="inducer"):
        fit_joint(cultures, plain)
