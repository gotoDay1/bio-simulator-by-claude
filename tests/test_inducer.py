import numpy as np
import pytest

from biosim import (
    BioreactorSimulation,
    FedBatch,
    InducerDose,
    InducibleLuedekingPiretProduct,
    InitialConditions,
    InvalidParameterError,
    ModelSpec,
    OxygenDemandOnly,
    ParameterSpec,
    PirtMonodGrowth,
    TemperatureLuedekingPiretProduct,
    constant_feed,
    fit_batch,
    standard_induction,
)
from biosim.conditions import Conditions, arrhenius_factor
from biosim.models.registry import PRODUCT_MODELS
from biosim.models.substrate import GrowthDrivenSubstrate

F, S_FEED, V0 = 0.05, 100.0, 1.0


def _growth():
    return PirtMonodGrowth(
        Yxs_ref=0.5, ms_ref=0.02, Ea_ms=50.0, qSmax_ref=50.0, Ea_qS=0.0, Ks=1e-3
    )


def _product(**kw):
    params = {"alpha": 0.0, "beta_basal": 0.002, "beta_max": 0.05, "EC50": 100.0}
    params.update(kw)
    return InducibleLuedekingPiretProduct(**params)


def _simulate(events=None, product=None, t_end=100.0, n=201, I0=0.0):
    sim = BioreactorSimulation(
        growth_model=_growth(),
        product_model=product or _product(),
        substrate_model=GrowthDrivenSubstrate(),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
        operation_mode=FedBatch(feed_rate_fn=constant_feed(F), S_feed=S_FEED),
        initial_conditions=InitialConditions(X0=1.0, S0=0.0, P0=0.0, V0=V0, I0=I0),
        t_span=(0.0, t_end),
        n_points=n,
        inducer_events=events,
    )
    return sim.run().data


def _at(d, t, col):
    return float(d.loc[np.isclose(d["t"], t), col].iloc[0])


def test_dose_validation_and_apply():
    assert InducerDose(10.0, 300.0, "set").apply(120.0) == 300.0
    assert InducerDose(10.0, 300.0, "set").apply(400.0) == 400.0  # cannot remove inducer
    assert InducerDose(10.0, 300.0, "add").apply(120.0) == 420.0
    with pytest.raises(InvalidParameterError):
        InducerDose(10.0, 300.0, "reset")
    with pytest.raises(InvalidParameterError):
        InducerDose(-1.0, 300.0)
    with pytest.raises(InvalidParameterError):
        InducerDose(1.0, -5.0)


def test_standard_induction_is_32_56_80_at_300():
    assert [(d.t, d.conc) for d in standard_induction()] == [
        (32.0, 300.0),
        (56.0, 300.0),
        (80.0, 300.0),
    ]


def test_registry_exposes_inducible_product():
    assert PRODUCT_MODELS["luedeking_piret_induced"] is InducibleLuedekingPiretProduct


def test_inducer_is_diluted_by_the_feed_between_doses():
    # I V is conserved between doses: I(t) = I_dose * V(t_dose) / V(t)
    d = _simulate([InducerDose(10.0, 300.0)])
    v10, v40 = V0 + F * 10.0, V0 + F * 40.0
    assert _at(d, 40.0, "I") == pytest.approx(300.0 * v10 / v40, rel=1e-4)
    assert _at(d, 100.0, "I") == pytest.approx(300.0 * v10 / (V0 + F * 100.0), rel=1e-4)


def test_inducer_is_zero_before_the_first_dose_and_jumps_at_the_dose_time():
    d = _simulate([InducerDose(50.0, 300.0)])
    assert (d.loc[d["t"] < 50.0, "I"] == 0.0).all()
    assert _at(d, 50.0, "I") == pytest.approx(300.0)  # reported value is the post-dose one


def test_set_mode_returns_to_target_and_add_mode_accumulates():
    events_set = standard_induction(mode="set")
    d_set = _simulate(events_set)
    for t in (32.0, 56.0, 80.0):
        assert _at(d_set, t, "I") == pytest.approx(300.0)

    d_add = _simulate(standard_induction(mode="add"))
    assert _at(d_add, 56.0, "I") > 300.0
    assert _at(d_add, 80.0, "I") > _at(d_add, 56.0, "I")


def test_output_grid_is_unchanged_by_dosing():
    d = _simulate(standard_induction())
    np.testing.assert_allclose(d["t"].to_numpy(), np.linspace(0.0, 100.0, 201))
    assert d["t"].is_monotonic_increasing and d["t"].is_unique


def test_dose_at_start_acts_on_initial_state_and_late_dose_is_ignored():
    d0 = _simulate([InducerDose(0.0, 300.0)], n=11, t_end=10.0)
    assert _at(d0, 0.0, "I") == pytest.approx(300.0)
    late = _simulate([InducerDose(500.0, 300.0)], n=11, t_end=10.0)
    assert (late["I"] == 0.0).all()


def test_without_doses_only_the_basal_term_produces_product():
    off = _simulate(None)
    assert (off["I"] == 0.0).all()
    basal_only = _simulate(None, product=_product(beta_max=0.0))
    np.testing.assert_allclose(off["P"].to_numpy(), basal_only["P"].to_numpy(), rtol=1e-9)


def test_product_increases_with_inducer_dose():
    def total_product(conc: float) -> float:
        d = _simulate([InducerDose(20.0, conc)])
        return d["P"].iloc[-1] * d["V"].iloc[-1]

    ends = {c: total_product(c) for c in (0.0, 150.0, 300.0, 600.0)}
    assert ends[0.0] < ends[150.0] < ends[300.0] < ends[600.0]


def test_induction_curve_is_hill_saturating():
    m = _product(EC50=100.0, hill_n=1.0)
    assert m.induction_fraction(0.0) == 0.0
    assert m.induction_fraction(100.0) == pytest.approx(0.5)
    assert m.induction_fraction(1e9) == pytest.approx(1.0, abs=1e-6)
    steep = _product(EC50=100.0, hill_n=4.0)
    assert steep.induction_fraction(50.0) < m.induction_fraction(50.0)
    assert steep.induction_fraction(200.0) > m.induction_fraction(200.0)


def test_beta_combines_basal_induced_and_arrhenius():
    m = _product(beta_basal=0.01, beta_max=0.06, EC50=100.0, Ea_beta=40.0)
    cond = Conditions(T_C=36.0, pH=5.0, inducer=100.0)
    expected = arrhenius_factor(36.0, 40.0) * (0.01 + 0.06 * 0.5) * 2.0
    assert m.production_rate(X=2.0, P=0.0, dXdt=0.0, t=0.0, cond=cond) == pytest.approx(expected)


def test_events_require_an_inducible_product_model():
    with pytest.raises(InvalidParameterError, match="inducible"):
        BioreactorSimulation(
            growth_model=_growth(),
            product_model=TemperatureLuedekingPiretProduct(),
            substrate_model=GrowthDrivenSubstrate(),
            oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
            operation_mode=FedBatch(feed_rate_fn=constant_feed(F), S_feed=S_FEED),
            initial_conditions=InitialConditions(X0=1.0, S0=0.0),
            t_span=(0.0, 10.0),
            inducer_events=[InducerDose(5.0, 300.0)],
        )


def _fixed(name, value):
    return ParameterSpec(name=name, fixed=True, value=value)


def test_fit_recovers_induced_capacity_from_a_dosed_batch():
    true_beta_max = 0.08
    events = standard_induction()
    truth = _simulate(events, product=_product(beta_max=true_beta_max))
    obs = truth.iloc[::8][["t", "X", "P"]].reset_index(drop=True)

    growth_specs = [
        _fixed("Yxs_ref", 0.5),
        _fixed("Yxs_slope", 0.0),
        _fixed("ms_ref", 0.02),
        _fixed("Ea_ms", 50.0),
        _fixed("ms_pH_a", 0.0),
        _fixed("ms_pH_b", 0.0),
        _fixed("qSmax_ref", 50.0),
        _fixed("Ea_qS", 0.0),
        _fixed("Ks", 1e-3),
    ]
    product_specs = [
        _fixed("alpha", 0.0),
        _fixed("beta_basal", 0.002),
        ParameterSpec("beta_max", fixed=False, value=0.03, lower_bound=1e-4, upper_bound=0.5),
        _fixed("EC50", 100.0),
        _fixed("hill_n", 1.0),
        _fixed("Ea_beta", 30.0),
        _fixed("kd_ref", 0.0),
        _fixed("Ea_kd", 50.0),
    ]
    specs = [
        ModelSpec("growth", PirtMonodGrowth, growth_specs),
        ModelSpec("product", InducibleLuedekingPiretProduct, product_specs),
        ModelSpec("substrate", GrowthDrivenSubstrate, []),
        ModelSpec("oxygen", OxygenDemandOnly, [_fixed("Yxo2", 0.9), _fixed("mo2", 0.05)]),
    ]
    result = fit_batch(
        "dosed",
        specs,
        InitialConditions(X0=1.0, S0=0.0, P0=0.0, V0=V0),
        FedBatch(feed_rate_fn=constant_feed(F), S_feed=S_FEED),
        obs,
        n_points=201,
        n_starts=2,
        inducer_events=events,
    )
    assert result.success
    assert result.param_values["product"]["beta_max"] == pytest.approx(true_beta_max, rel=0.02)


def test_splitting_the_integration_does_not_change_the_trajectory():
    # a zero-size dose only splits the run into segments; the physics must be unchanged
    whole = _simulate(None)
    split = _simulate([InducerDose(25.0, 0.0, "add"), InducerDose(61.0, 0.0, "add")])
    for col in ("X", "S", "P", "V"):
        np.testing.assert_allclose(
            split[col].to_numpy(), whole[col].to_numpy(), rtol=1e-4, atol=1e-6
        )
