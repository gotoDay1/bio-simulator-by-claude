import math

import pytest

from biosim import InvalidParameterError
from biosim.conditions import (
    PH_REF,
    T_REF_C,
    ConditionProfile,
    Conditions,
    arrhenius_factor,
    constant_value,
    piecewise_value,
    quadratic_factor,
    ramp_value,
    shift_value,
)


def test_arrhenius_is_one_at_reference_temperature():
    assert arrhenius_factor(T_REF_C, Ea=50.0) == pytest.approx(1.0)


def test_arrhenius_increases_with_temperature_and_matches_hand_value():
    # Ea = 50 kJ/mol: 30 -> 36 C is ~1.47x, 30 -> 22 C is ~0.58x (values used in the theory doc)
    assert arrhenius_factor(36.0, Ea=50.0) == pytest.approx(1.4696, abs=1e-3)
    assert arrhenius_factor(22.0, Ea=50.0) == pytest.approx(0.5841, abs=1e-3)
    assert arrhenius_factor(36.0, 50.0) > arrhenius_factor(30.0, 50.0) > arrhenius_factor(22.0, 50.0)


def test_arrhenius_zero_activation_energy_is_flat():
    assert arrhenius_factor(40.0, Ea=0.0) == pytest.approx(1.0)


def test_arrhenius_matches_closed_form():
    Ea, T, Tref = 70.0, 36.0 + 273.15, 30.0 + 273.15
    expected = math.exp(-Ea / 8.314462618e-3 * (1 / T - 1 / Tref))
    assert arrhenius_factor(36.0, Ea) == pytest.approx(expected)


def test_quadratic_factor_is_one_at_reference_and_clipped_at_floor():
    assert quadratic_factor(PH_REF, a=0.3, b=0.1, x_ref=PH_REF) == 1.0
    assert quadratic_factor(7.0, a=0.0, b=0.5, x_ref=PH_REF) == pytest.approx(3.0)
    assert quadratic_factor(0.0, a=0.0, b=-1.0, x_ref=PH_REF, floor=0.0) == 0.0


def test_default_profile_is_standard_conditions():
    assert ConditionProfile().at(10.0) == Conditions(T_C=30.0, pH=5.0)


def test_shift_value_switches_at_shift_time():
    fn = shift_value(30.0, 24.0, 48.0)
    assert fn(47.99) == 30.0
    assert fn(48.0) == 24.0


def test_ramp_value_interpolates_linearly():
    fn = ramp_value(30.0, 24.0, 40.0, 50.0)
    assert fn(0.0) == 30.0
    assert fn(45.0) == pytest.approx(27.0)
    assert fn(60.0) == 24.0


def test_ramp_rejects_nonincreasing_interval():
    with pytest.raises(InvalidParameterError):
        ramp_value(30.0, 24.0, 50.0, 50.0)


def test_piecewise_value_holds_each_step():
    fn = piecewise_value([0.0, 48.0, 72.0], [30.0, 24.0, 30.0])
    assert [fn(10.0), fn(48.0), fn(60.0), fn(100.0)] == [30.0, 24.0, 24.0, 30.0]


def test_profile_combines_temperature_and_ph():
    profile = ConditionProfile(shift_value(30.0, 24.0, 48.0), shift_value(5.0, 6.0, 48.0))
    assert profile.at(0.0) == Conditions(30.0, 5.0)
    assert profile.at(50.0) == Conditions(24.0, 6.0)


def test_constant_value_rejects_nonpositive():
    with pytest.raises(InvalidParameterError):
        constant_value(0.0)
