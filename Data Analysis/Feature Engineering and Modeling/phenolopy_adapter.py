# phenolopy_adapter.py - run vendored PhenoloPy's calc_phenometrics on one
# (plot, window, index) season and return the 13-key descriptor dict the
# pipeline expects. drop-in replacement for the old custom extract_descriptors.

import os
import sys
import contextlib
import numpy as np
import xarray as xr

# import the vendored, patched phenolopy from <repo>/third_party
_THIRD_PARTY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party",
)
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)
import phenolopy  # noqa: E402

DESCRIPTOR_KEYS = [
    "pos_time", "pos_value", "bse_value", "aos_value",
    "sos_time", "sos_value", "eos_time", "eos_value",
    "los", "roi", "rod", "lios", "sios",
]

# our key -> phenolopy output variable name
_PP_MAP = {
    "pos_time": "pos_times", "pos_value": "pos_values",
    "bse_value": "bse_values", "aos_value": "aos_values",
    "sos_time": "sos_times", "sos_value": "sos_values",
    "eos_time": "eos_times", "eos_value": "eos_values",
    "los": "los_values", "roi": "roi_values", "rod": "rod_values",
    "lios": "lios_values", "sios": "sios_values",
}

MIN_SEASON_POINTS = 5

# task 2 result: phenolopy returns *_times already in day-of-year when fed a
# datetime time-coordinate. observed pos_times=177 for a known peak at doy 180
# (discrete weekly sampling; 177 is the closest sample). _TIME_IS_DOY = True.
_TIME_IS_DOY = True
_BASE = np.datetime64("2001-01-01")  # non-leap base year for doy->datetime


def _nan():
    return {k: np.nan for k in DESCRIPTOR_KEYS}


def _to_doy(value, t):
    """convert a phenolopy *_times output to day-of-year. with _TIME_IS_DOY the
    output is already doy; otherwise treat value as an index into t."""
    if _TIME_IS_DOY or value is None or np.isnan(value):
        return value
    idx = int(round(value))
    return float(t[idx]) if 0 <= idx < len(t) else np.nan


def phenometrics(t, y, factor=0.5):
    """vendored-phenolopy descriptors for one season. t = day-of-year (numeric),
    y = smoothed index values. returns the 13-key descriptor dict."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(t) & ~np.isnan(y)
    t, y = t[m], y[m]
    if len(y) < MIN_SEASON_POINTS:
        return _nan()

    # encode day-of-year as datetime so phenolopy has a real time axis
    times = _BASE + (t.astype(int) - 1).astype("timedelta64[D]")
    da = xr.DataArray(
        y.reshape(-1, 1, 1).astype("float32"),
        dims=("time", "y", "x"),
        coords={"time": times, "y": [0], "x": [0]},
        name="veg_index",
    )
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            out = phenolopy.calc_phenometrics(
                da=da, peak_metric="pos", base_metric="bse",
                method="seasonal_amplitude", factor=factor,
                thresh_sides="two_sided", abs_value=0,
            )
    except Exception:
        return _nan()

    d = {}
    for key, pp in _PP_MAP.items():
        try:
            d[key] = float(out[pp].values.squeeze())
        except Exception:
            d[key] = np.nan
    for tk in ("pos_time", "sos_time", "eos_time"):
        d[tk] = _to_doy(d[tk], t)
    # los in day-of-year units = eos - sos (recompute for consistency with doy)
    if not np.isnan(d["eos_time"]) and not np.isnan(d["sos_time"]):
        d["los"] = d["eos_time"] - d["sos_time"]
    return d
