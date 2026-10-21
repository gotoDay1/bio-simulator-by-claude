"""Operation-log loader: turn a lab operation log (Excel / CSV) into simulation-ready cultures.

A log is a table with one row per time stamp and these columns (case-insensitive; Japanese
aliases accepted, unknown columns are kept in `Culture.ignored_columns` and otherwise ignored):

    time         elapsed time in hours (a datetime column is converted to hours since its first row)
    temperature  setpoint (deg C)
    pH           setpoint
    induction    inducer concentration added at that row (uM, one dose); blank or 0 = no dose
    feed         feed rate (L/h)
    OD, X (DCW)  biomass measurements: X is DCW (g/L); OD is converted with `od_to_dcw` and used
                 only where DCW is blank
    S, P         substrate / product measurements (g/L)
    batch        optional: several cultures in one sheet, one group per value

Blank cells mean "no value". For the setpoint columns (temperature, pH, feed) a blank row
simply records no change: each non-blank value holds (step-hold) until the next non-blank one,
and before the first one the standard condition applies (30 C, pH 5.0, no feed). For the
measurement columns a blank row is a missing sample.

One sheet is one culture (the sheet name is the culture name), unless the sheet has a `batch`
column, in which case each batch value is a culture. A CSV file is one culture (file stem).
"""

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from biosim.conditions import PH_REF, T_REF_C, ConditionProfile, piecewise_value
from biosim.exceptions import ExperimentalDataError
from biosim.inducer import InducerDose
from biosim.operation_modes import Batch, FedBatch, OperationMode, stepwise_feed
from biosim.state import InitialConditions

_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("time", "t", "time_h", "time (h)", "経過時間", "時間"),
    "temperature": ("temperature", "temp", "t_c", "temperature (c)", "温度"),
    "pH": ("ph",),
    "induction": ("induction", "inducer", "誘導", "誘導剤"),
    "feed": ("feed", "feed_rate", "feedrate", "feed (l/h)", "フィード", "フィード速度"),
    "OD": ("od", "od600"),
    "X": ("x", "dcw"),
    "S": ("s", "substrate"),
    "P": ("p", "product"),
    "batch": ("batch", "culture", "培養"),
}
_CANONICAL = {alias: canon for canon, aliases in _ALIASES.items() for alias in aliases}
_MEASUREMENTS = ("X", "S", "P")


@dataclass
class Culture:
    """One culture: its measurements plus everything needed to simulate it.

    data has column `t` plus any of `X` (DCW, g/L), `S`, `P`; blank samples are NaN.
    feed_rate_fn is None for a culture without feed (simulated as a batch).
    """

    name: str
    data: pd.DataFrame
    condition_profile: ConditionProfile
    inducer_events: list[InducerDose]
    feed_rate_fn: Callable[[float], float] | None
    S_feed: float | None
    initial_conditions: InitialConditions
    ignored_columns: list[str] = field(default_factory=list)

    @property
    def t_end(self) -> float:
        return float(self.data["t"].max())

    @property
    def measured_columns(self) -> list[str]:
        return [c for c in _MEASUREMENTS if c in self.data.columns and self.data[c].notna().any()]

    def operation_mode(self) -> OperationMode:
        if self.feed_rate_fn is None:
            return Batch()
        return FedBatch(feed_rate_fn=self.feed_rate_fn, S_feed=self.S_feed or 0.0)


def _canonical_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rename: dict[str, str] = {}
    ignored: list[str] = []
    seen: set[str] = set()
    for col in df.columns:
        canon = _CANONICAL.get(str(col).strip().lower())
        if canon is None or canon in seen:
            ignored.append(str(col))
        else:
            rename[col] = canon
            seen.add(canon)
    out = df[list(rename)].rename(columns=rename)
    return out, ignored


def _elapsed_hours(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series - series.iloc[0]).dt.total_seconds() / 3600.0
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds() / 3600.0
    try:
        return pd.to_numeric(series, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ExperimentalDataError(
            f"Culture '{name}': the time column must be numeric hours or datetimes."
        ) from exc


def _breakpoints(
    t: np.ndarray, v: pd.Series, default: float, name: str, column: str
) -> tuple[list[float], list[float]]:
    """(times, values) of a step-hold setpoint: the non-blank rows, last one winning on a
    repeated time, with the standard value prepended when the first entry is after t=0."""
    numeric = pd.to_numeric(v, errors="coerce")
    if numeric.notna().sum() != v.notna().sum():
        raise ExperimentalDataError(f"Culture '{name}': column '{column}' has non-numeric values.")
    by_time: dict[float, float] = {}
    for ti, vi in zip(t, numeric, strict=True):
        if not np.isnan(vi):
            by_time[float(ti)] = float(vi)
    times = sorted(by_time)
    values = [by_time[ti] for ti in times]
    if times and times[0] < 0:
        raise ExperimentalDataError(f"Culture '{name}': negative time in column '{column}'.")
    if not times or times[0] > 0:
        times.insert(0, 0.0)
        values.insert(0, default)
    return times, values


def parse_operation_log(
    df: pd.DataFrame,
    name: str,
    *,
    initial_conditions: InitialConditions,
    S_feed: float | None = None,
    od_to_dcw: float | None = None,
    inducer_mode: str = "add",
) -> Culture:
    """Build one Culture from a log table (see the module docstring for the layout)."""
    table, ignored = _canonical_columns(df)
    if "time" not in table.columns:
        raise ExperimentalDataError(
            f"Culture '{name}': no time column found (expected one of {_ALIASES['time']}); "
            f"columns are {list(df.columns)}."
        )
    table = table.copy()
    table["time"] = _elapsed_hours(table["time"], name)
    table = table.dropna(subset=["time"]).sort_values("time", kind="stable").reset_index(drop=True)
    if table.empty:
        raise ExperimentalDataError(f"Culture '{name}': the log has no rows with a time stamp.")
    t = table["time"].to_numpy(dtype=float)

    def column(key: str) -> pd.Series:
        return table[key] if key in table.columns else pd.Series(np.nan, index=table.index)

    # conditions
    T_times, T_values = _breakpoints(t, column("temperature"), T_REF_C, name, "temperature")
    pH_times, pH_values = _breakpoints(t, column("pH"), PH_REF, name, "pH")
    profile = ConditionProfile(piecewise_value(T_times, T_values), piecewise_value(pH_times, pH_values))

    # inducer doses
    induction = pd.to_numeric(column("induction"), errors="coerce")
    if induction.notna().sum() != column("induction").notna().sum():
        raise ExperimentalDataError(f"Culture '{name}': column 'induction' has non-numeric values.")
    if (induction < 0).any():
        raise ExperimentalDataError(f"Culture '{name}': column 'induction' has negative values.")
    doses = [
        InducerDose(float(ti), float(c), inducer_mode)
        for ti, c in zip(t, induction, strict=True)
        if not np.isnan(c) and c > 0
    ]

    # feed
    feed_fn: Callable[[float], float] | None = None
    if column("feed").notna().any():
        if S_feed is None:
            raise ExperimentalDataError(
                f"Culture '{name}' has a feed column but S_feed (substrate concentration of "
                "the feed, g/L) was not given."
            )
        f_times, f_values = _breakpoints(t, column("feed"), 0.0, name, "feed")
        feed_fn = stepwise_feed(f_times, f_values)

    # measurements
    data = pd.DataFrame({"t": t})
    dcw = pd.to_numeric(column("X"), errors="coerce")
    od = pd.to_numeric(column("OD"), errors="coerce")
    if od.notna().any() and od_to_dcw is None and not dcw.notna().any():
        raise ExperimentalDataError(
            f"Culture '{name}' has OD but no DCW, and od_to_dcw (the OD -> DCW factor) was not given."
        )
    biomass = dcw.copy()
    if od_to_dcw is not None:
        biomass = dcw.combine_first(od * od_to_dcw)
    if biomass.notna().any():
        data["X"] = biomass.to_numpy()
    for key in ("S", "P"):
        values = pd.to_numeric(column(key), errors="coerce")
        if values.notna().any():
            data[key] = values.to_numpy()
    if not any(c in data.columns for c in _MEASUREMENTS):
        raise ExperimentalDataError(
            f"Culture '{name}' has no measurements (need DCW/OD, S or P values)."
        )
    data = data[data[[c for c in _MEASUREMENTS if c in data.columns]].notna().any(axis=1)]

    return Culture(
        name=name,
        data=data.reset_index(drop=True),
        condition_profile=profile,
        inducer_events=doses,
        feed_rate_fn=feed_fn,
        S_feed=S_feed,
        initial_conditions=initial_conditions,
        ignored_columns=ignored,
    )


def load_operation_log(
    path: str | Path,
    *,
    initial_conditions: InitialConditions | Mapping[str, InitialConditions],
    S_feed: float | None = None,
    od_to_dcw: float | None = None,
    inducer_mode: str = "add",
    sheets: Sequence[str] | None = None,
) -> list[Culture]:
    """Read an operation log (.xlsx/.xls or .csv) into cultures.

    initial_conditions: one InitialConditions for every culture, or a mapping culture name ->
        InitialConditions for per-culture initial states.
    S_feed: substrate concentration of the feed (g/L); required when a feed column is present.
    od_to_dcw: factor DCW = factor * OD, used where DCW is blank.
    inducer_mode: how a value in the `induction` column is read, "add" (added on top of what is
        already in the tank) or "set" (bring the tank up to that concentration).
    sheets: restrict an Excel workbook to these sheets (default: every sheet that has a time
        column; other sheets are skipped with a warning).
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frames = {path.stem: pd.read_csv(path)}
    else:
        frames = pd.read_excel(path, sheet_name=list(sheets) if sheets else None)
        if isinstance(frames, pd.DataFrame):
            frames = {sheets[0] if sheets else path.stem: frames}

    def initial_for(name: str) -> InitialConditions:
        if isinstance(initial_conditions, Mapping):
            if name not in initial_conditions:
                raise ExperimentalDataError(f"No initial conditions given for culture '{name}'.")
            return initial_conditions[name]
        return initial_conditions

    cultures: list[Culture] = []
    for sheet_name, df in frames.items():
        canonical, _ = _canonical_columns(df)
        if "time" not in canonical.columns:
            if sheets:
                raise ExperimentalDataError(f"Sheet '{sheet_name}' has no time column.")
            warnings.warn(f"Skipping sheet '{sheet_name}': no time column.", stacklevel=2)
            continue
        groups: list[tuple[str, pd.DataFrame]]
        batch_col = next((c for c in df.columns if _CANONICAL.get(str(c).strip().lower()) == "batch"), None)
        if batch_col is not None:
            groups = [(str(b), g.drop(columns=[batch_col])) for b, g in df.groupby(batch_col, sort=False)]
        else:
            groups = [(str(sheet_name), df)]
        for name, group in groups:
            cultures.append(
                parse_operation_log(
                    group,
                    name,
                    initial_conditions=initial_for(name),
                    S_feed=S_feed,
                    od_to_dcw=od_to_dcw,
                    inducer_mode=inducer_mode,
                )
            )
    if not cultures:
        raise ExperimentalDataError(f"No cultures found in {path}.")
    names = [c.name for c in cultures]
    if len(set(names)) != len(names):
        raise ExperimentalDataError(f"Duplicate culture names: {sorted({n for n in names if names.count(n) > 1})}")
    return cultures
