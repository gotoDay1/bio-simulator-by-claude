import numpy as np
import pytest

from biosim import (
    BioreactorSimulation,
    FedBatch,
    InitialConditions,
    InvalidParameterError,
    ModelSpec,
    OxygenDemandOnly,
    ParameterSpec,
    YieldMaintenanceSubstrate,
    constant_feed,
    fit_batch,
)
from biosim.conditions import ConditionProfile, Conditions, constant_value, shift_value
from biosim.models.growth import MonodGrowth, PirtMonodGrowth
from biosim.models.product import TemperatureLuedekingPiretProduct
from biosim.models.registry import GROWTH_MODELS, PRODUCT_MODELS, SUBSTRATE_MODELS
from biosim.models.substrate import GrowthDrivenSubstrate

F, S_FEED = 0.05, 100.0  # 5 g substrate / h
Y, M = 0.5, 0.02


def _growth(**kw):
    params = {"Yxs_ref": Y, "ms_ref": M, "Ea_ms": 50.0, "qSmax_ref": 50.0, "Ea_qS": 0.0, "Ks": 1e-3}
    params.update(kw)
    return PirtMonodGrowth(**params)


def _simulate(growth=None, product=None, temperature_fn=None, pH_fn=None, t_end=100.0, n=201):
    sim = BioreactorSimulation(
        growth_model=growth or _growth(),
        product_model=product or TemperatureLuedekingPiretProduct(alpha=0.0, beta_ref=0.05),
        substrate_model=GrowthDrivenSubstrate(),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
        operation_mode=FedBatch(feed_rate_fn=constant_feed(F), S_feed=S_FEED),
        initial_conditions=InitialConditions(X0=1.0, S0=0.0, P0=0.0, V0=1.0),
        t_span=(0.0, t_end),
        n_points=n,
        condition_profile=ConditionProfile(temperature_fn, pH_fn),
    )
    return sim.run().data


def _total_biomass(d):
    return (d["X"] * d["V"]).to_numpy()


def test_registry_exposes_new_models():
    assert GROWTH_MODELS["pirt_monod"] is PirtMonodGrowth
    assert PRODUCT_MODELS["luedeking_piret_temp"] is TemperatureLuedekingPiretProduct
    assert SUBSTRATE_MODELS["growth_driven"] is GrowthDrivenSubstrate


def test_total_biomass_follows_analytic_feed_limited_solution():
    # dx/dt = Y (F Sf - m x),  x = X V  ->  x(t) = x_ss - (x_ss - x0) exp(-Y m t)
    d = _simulate()
    t = d["t"].to_numpy()
    x_ss = F * S_FEED / M
    expected = x_ss - (x_ss - 1.0) * np.exp(-Y * M * t)
    assert np.max(np.abs(_total_biomass(d) - expected) / expected) < 1e-3
    assert d["S"].max() < 1e-2  # feed-limited: substrate stays near zero


def test_plateau_depends_on_maintenance_not_on_yield():
    x_end = []
    for yxs in (0.3, 0.6):
        d = _simulate(growth=_growth(Yxs_ref=yxs), t_end=1500.0, n=60)
        x_end.append(_total_biomass(d)[-1])
    assert x_end[0] == pytest.approx(F * S_FEED / M, rel=1e-3)
    assert x_end[1] == pytest.approx(F * S_FEED / M, rel=1e-3)


def test_higher_temperature_lowers_biomass_through_maintenance():
    ends = {T: _total_biomass(_simulate(temperature_fn=constant_value(T)))[-1] for T in (22, 30, 36)}
    assert ends[22] > ends[30] > ends[36]


def test_temperature_shift_lands_between_constant_runs():
    shifted = _total_biomass(_simulate(temperature_fn=shift_value(30.0, 36.0, 48.0)))[-1]
    const30 = _total_biomass(_simulate(temperature_fn=constant_value(30.0)))[-1]
    const36 = _total_biomass(_simulate(temperature_fn=constant_value(36.0)))[-1]
    assert const36 < shifted < const30


def test_trajectory_is_unchanged_before_the_shift():
    base = _simulate(t_end=100.0)
    shifted = _simulate(temperature_fn=shift_value(30.0, 24.0, 48.0), t_end=100.0)
    before = base["t"] < 47.0
    np.testing.assert_allclose(
        shifted.loc[before, "X"].to_numpy(), base.loc[before, "X"].to_numpy(), rtol=1e-6
    )


def test_ph_correction_of_maintenance_lowers_biomass_away_from_reference():
    growth = _growth(ms_pH_a=0.0, ms_pH_b=0.3)
    at_ref = _total_biomass(_simulate(growth=growth, pH_fn=constant_value(5.0)))[-1]
    off_ref = _total_biomass(_simulate(growth=growth, pH_fn=constant_value(7.0)))[-1]
    assert off_ref < at_ref


def test_substrate_mass_balance_closes():
    # fed substrate = remaining substrate + substrate taken up (growth + maintenance)
    d = _simulate(t_end=60.0, n=121)
    t = d["t"].to_numpy()
    V = d["V"].to_numpy()
    fed = S_FEED * (V - 1.0)
    remaining = (d["S"] * d["V"]).to_numpy()
    x = _total_biomass(d)
    # integral of x dt (trapezoid) = maintenance sink; growth sink = (x - x0)/Y
    int_x = np.concatenate([[0.0], np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(t))])
    consumed = (x - x[0]) / Y + M * int_x
    np.testing.assert_allclose(fed, remaining + consumed, atol=0.05)


def test_product_decay_reduces_product_and_is_temperature_dependent():
    no_decay = _simulate(product=TemperatureLuedekingPiretProduct(alpha=0.0, beta_ref=0.05))
    decay = _simulate(
        product=TemperatureLuedekingPiretProduct(alpha=0.0, beta_ref=0.05, kd_ref=0.02, Ea_kd=80.0)
    )
    assert (decay["P"] * decay["V"]).iloc[-1] < (no_decay["P"] * no_decay["V"]).iloc[-1]

    hot = _simulate(
        product=TemperatureLuedekingPiretProduct(
            alpha=0.0, beta_ref=0.05, Ea_beta=0.0, kd_ref=0.02, Ea_kd=80.0
        ),
        temperature_fn=constant_value(36.0),
    )
    cold = _simulate(
        product=TemperatureLuedekingPiretProduct(
            alpha=0.0, beta_ref=0.05, Ea_beta=0.0, kd_ref=0.02, Ea_kd=80.0
        ),
        temperature_fn=constant_value(24.0),
    )
    # with beta flat in T, only stability differs; cold keeps more product
    assert (cold["P"] * cold["V"]).iloc[-1] > (hot["P"] * hot["V"]).iloc[-1]


def test_growth_associated_term_ignores_biomass_loss():
    model = TemperatureLuedekingPiretProduct(alpha=3.0, beta_ref=0.0, kd_ref=0.0)
    assert model.production_rate(X=2.0, P=1.0, dXdt=-0.5, t=0.0, cond=Conditions()) == 0.0
    assert model.production_rate(X=2.0, P=1.0, dXdt=0.5, t=0.0, cond=Conditions()) == 1.5


def test_pirt_growth_requires_growth_driven_substrate():
    with pytest.raises(InvalidParameterError, match="GrowthDrivenSubstrate"):
        BioreactorSimulation(
            growth_model=_growth(),
            product_model=TemperatureLuedekingPiretProduct(),
            substrate_model=YieldMaintenanceSubstrate(),
            oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
            operation_mode=FedBatch(feed_rate_fn=constant_feed(F), S_feed=S_FEED),
            initial_conditions=InitialConditions(X0=1.0, S0=0.0),
            t_span=(0.0, 10.0),
        )


def test_growth_driven_substrate_requires_pirt_style_growth():
    with pytest.raises(InvalidParameterError, match="drives_substrate"):
        BioreactorSimulation(
            growth_model=MonodGrowth(),
            product_model=TemperatureLuedekingPiretProduct(),
            substrate_model=GrowthDrivenSubstrate(),
            oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
            operation_mode=FedBatch(feed_rate_fn=constant_feed(F), S_feed=S_FEED),
            initial_conditions=InitialConditions(X0=1.0, S0=0.0),
            t_span=(0.0, 10.0),
        )


def _fixed(name, value):
    return ParameterSpec(name=name, fixed=True, value=value)


def _specs(free_ms_guess):
    growth = [
        _fixed("Yxs_ref", Y),
        _fixed("Yxs_slope", 0.0),
        ParameterSpec("ms_ref", fixed=False, value=free_ms_guess, lower_bound=1e-3, upper_bound=0.1),
        _fixed("Ea_ms", 50.0),
        _fixed("ms_pH_a", 0.0),
        _fixed("ms_pH_b", 0.0),
        _fixed("qSmax_ref", 50.0),
        _fixed("Ea_qS", 0.0),
        _fixed("Ks", 1e-3),
    ]
    product = [
        _fixed("alpha", 0.0),
        _fixed("beta_ref", 0.05),
        _fixed("Ea_beta", 30.0),
        _fixed("kd_ref", 0.0),
        _fixed("Ea_kd", 50.0),
    ]
    return [
        ModelSpec("growth", PirtMonodGrowth, growth),
        ModelSpec("product", TemperatureLuedekingPiretProduct, product),
        ModelSpec("substrate", GrowthDrivenSubstrate, []),
        ModelSpec("oxygen", OxygenDemandOnly, [_fixed("Yxo2", 0.9), _fixed("mo2", 0.05)]),
    ]


def test_fit_recovers_maintenance_from_a_temperature_shift_batch():
    true_ms = 0.03
    profile = ConditionProfile(shift_value(30.0, 36.0, 48.0))
    truth = _simulate(growth=_growth(ms_ref=true_ms), temperature_fn=profile.temperature_fn)
    obs = truth.iloc[::10][["t", "X"]].reset_index(drop=True)

    result = fit_batch(
        "shift36",
        _specs(free_ms_guess=0.012),
        InitialConditions(X0=1.0, S0=0.0, P0=0.0, V0=1.0),
        FedBatch(feed_rate_fn=constant_feed(F), S_feed=S_FEED),
        obs,
        n_points=201,
        n_starts=2,
        condition_profile=profile,
    )
    assert result.success
    assert result.param_values["growth"]["ms_ref"] == pytest.approx(true_ms, rel=0.02)
