import numpy as np
import pandas as pd
import pytest

from biosim import (
    BioreactorSimulation,
    ExperimentalDataError,
    FedBatch,
    GrowthDrivenSubstrate,
    InducibleLuedekingPiretProduct,
    InitialConditions,
    OxygenDemandOnly,
    PirtMonodGrowth,
)
from biosim.culture import load_operation_log, parse_operation_log

IC = InitialConditions(X0=0.4, S0=0.0, P0=0.0, V0=1.0)

LOG = pd.DataFrame(
    {
        "time": [0, 24, 32, 48, 56, 80],
        "temperature": [None, None, None, 24.0, None, None],
        "pH": [None] * 6,
        "induction": [None, None, 300, None, 300, 300],
        "feed": [0.02, None, 0.04, None, None, 0.06],
        "OD": [0.8, 12.0, None, None, None, None],
        "DCW": [0.35, None, 9.0, None, 20.0, None],
        "P": [None, None, 0.1, None, 0.9, 2.0],
        "note": ["start", "", "dose 1", "shift", "dose 2", "dose 3"],
    }
)


def _parse(df=LOG, **kw):
    params = {"initial_conditions": IC, "S_feed": 100.0, "od_to_dcw": 0.4}
    params.update(kw)
    return parse_operation_log(df, "T24", **params)


def test_temperature_and_ph_are_step_hold_with_standard_before_first_entry():
    c = _parse()
    cond = c.condition_profile
    assert cond.at(0.0).T_C == 30.0 and cond.at(47.9).T_C == 30.0  # standard until the first entry
    assert cond.at(48.0).T_C == 24.0 and cond.at(100.0).T_C == 24.0
    assert cond.at(60.0).pH == 5.0  # pH column entirely blank -> standard


def test_induction_rows_become_add_doses():
    c = _parse()
    assert [(d.t, d.conc, d.mode) for d in c.inducer_events] == [
        (32.0, 300.0, "add"),
        (56.0, 300.0, "add"),
        (80.0, 300.0, "add"),
    ]
    assert all(d.mode == "set" for d in _parse(inducer_mode="set").inducer_events)


def test_feed_is_step_hold_and_zero_before_the_first_entry():
    c = _parse()
    f = c.feed_rate_fn
    assert (f(0.0), f(10.0), f(32.0), f(79.0), f(80.0)) == (0.02, 0.02, 0.04, 0.04, 0.06)
    shifted = LOG.copy()
    shifted["feed"] = [None, None, 0.04, None, None, 0.06]
    assert _parse(shifted).feed_rate_fn(10.0) == 0.0


def test_measurements_use_dcw_first_then_od_and_drop_rows_without_samples():
    c = _parse()
    d = c.data.set_index("t")
    assert d.loc[0.0, "X"] == pytest.approx(0.35)  # DCW wins over OD at the same row
    assert d.loc[24.0, "X"] == pytest.approx(12.0 * 0.4)  # OD only -> converted
    assert d.loc[32.0, "X"] == pytest.approx(9.0)
    assert 48.0 not in d.index  # the temperature-shift row carries no sample
    assert np.isnan(d.loc[24.0, "P"]) and d.loc[56.0, "P"] == pytest.approx(0.9)


def test_ignored_columns_are_reported():
    assert _parse().ignored_columns == ["note"]


def test_od_without_a_factor_or_dcw_is_an_error():
    od_only = LOG.drop(columns=["DCW"])
    with pytest.raises(ExperimentalDataError, match="od_to_dcw"):
        _parse(od_only, od_to_dcw=None)


def test_feed_without_s_feed_is_an_error():
    with pytest.raises(ExperimentalDataError, match="S_feed"):
        _parse(S_feed=None)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns=["time"]), "no time column"),
        (lambda d: d.assign(induction=[None, None, -5, None, None, None]), "negative"),
        (lambda d: d.assign(temperature=["a", None, None, None, None, None]), "non-numeric"),
    ],
)
def test_bad_logs_are_rejected(mutate, message):
    with pytest.raises(ExperimentalDataError, match=message):
        _parse(mutate(LOG))


def test_datetime_time_column_becomes_elapsed_hours():
    df = LOG.copy()
    df["time"] = pd.to_datetime("2026-10-01 09:00") + pd.to_timedelta(LOG["time"], unit="h")
    c = _parse(df)
    assert c.data["t"].tolist() == [0.0, 24.0, 32.0, 56.0, 80.0]
    assert c.condition_profile.at(50.0).T_C == 24.0


def test_japanese_aliases_are_recognised():
    df = LOG.rename(
        columns={"time": "経過時間", "temperature": "温度", "induction": "誘導剤", "feed": "フィード"}
    )
    c = _parse(df)
    assert c.condition_profile.at(60.0).T_C == 24.0 and len(c.inducer_events) == 3


def test_excel_one_sheet_per_culture_and_readme_sheet_is_skipped(tmp_path):
    path = tmp_path / "log.xlsx"
    with pd.ExcelWriter(path) as xw:
        LOG.to_excel(xw, sheet_name="T24", index=False)
        LOG.assign(temperature=[None] * 6).to_excel(xw, sheet_name="standard", index=False)
        pd.DataFrame({"memo": ["how to read this book"]}).to_excel(xw, sheet_name="README", index=False)
    with pytest.warns(UserWarning, match="README"):
        cultures = load_operation_log(path, initial_conditions=IC, S_feed=100.0, od_to_dcw=0.4)
    assert [c.name for c in cultures] == ["T24", "standard"]
    assert cultures[0].condition_profile.at(60.0).T_C == 24.0
    assert cultures[1].condition_profile.at(60.0).T_C == 30.0


def test_excel_batch_column_splits_one_sheet_into_cultures(tmp_path):
    both = pd.concat([LOG.assign(batch="a"), LOG.assign(batch="b", temperature=[None] * 6)])
    path = tmp_path / "log.xlsx"
    both.to_excel(path, index=False, sheet_name="all")
    cultures = load_operation_log(path, initial_conditions=IC, S_feed=100.0, od_to_dcw=0.4)
    assert [c.name for c in cultures] == ["a", "b"]
    assert cultures[0].condition_profile.at(60.0).T_C == 24.0
    assert cultures[1].condition_profile.at(60.0).T_C == 30.0


def test_csv_is_one_culture_named_after_the_file(tmp_path):
    path = tmp_path / "pH6.csv"
    LOG.to_csv(path, index=False)
    (c,) = load_operation_log(path, initial_conditions=IC, S_feed=100.0, od_to_dcw=0.4)
    assert c.name == "pH6"


def test_per_culture_initial_conditions_mapping(tmp_path):
    path = tmp_path / "log.xlsx"
    with pd.ExcelWriter(path) as xw:
        LOG.to_excel(xw, sheet_name="a", index=False)
        LOG.to_excel(xw, sheet_name="b", index=False)
    ics = {"a": InitialConditions(X0=0.4, S0=0.0), "b": InitialConditions(X0=0.9, S0=0.0)}
    a, b = load_operation_log(path, initial_conditions=ics, S_feed=100.0, od_to_dcw=0.4)
    assert (a.initial_conditions.X0, b.initial_conditions.X0) == (0.4, 0.9)
    with pytest.raises(ExperimentalDataError, match="No initial conditions"):
        load_operation_log(path, initial_conditions={"a": ics["a"]}, S_feed=100.0, od_to_dcw=0.4)


def test_culture_inputs_drive_a_simulation():
    c = _parse()
    sim = BioreactorSimulation(
        growth_model=PirtMonodGrowth(Yxs_ref=0.5, ms_ref=0.02, qSmax_ref=50.0, Ea_qS=0.0, Ks=1e-3),
        product_model=InducibleLuedekingPiretProduct(alpha=0.0, beta_max=0.05),
        substrate_model=GrowthDrivenSubstrate(),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
        operation_mode=c.operation_mode(),
        initial_conditions=c.initial_conditions,
        t_span=(0.0, c.t_end),
        n_points=81,
        condition_profile=c.condition_profile,
        inducer_events=c.inducer_events,
    )
    d = sim.run().data
    assert isinstance(c.operation_mode(), FedBatch)
    assert d["I"].iloc[-1] > 300.0  # three additive doses, diluted by the feed
    assert d["P"].iloc[-1] > 0.0
