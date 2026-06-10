# test_eda.py
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eda


def test_summarize_band_groups_and_means():
    df = pd.DataFrame({
        "x": [0, 0, 1, 1],
        "value": [1.0, 3.0, 10.0, 20.0],
        "grp": ["a", "a", "a", "a"],
    })
    out = eda.summarize_band(df, x="x", value="value", group="grp")
    a = out[out["grp"] == "a"].sort_values("x").reset_index(drop=True)
    assert list(a["x"]) == [0, 1]
    assert a.loc[0, "mean"] == 2.0
    assert a.loc[1, "mean"] == 15.0
    assert a.loc[0, "n"] == 2


def test_summarize_band_drops_nan():
    df = pd.DataFrame({
        "x": [0, 0, 0],
        "value": [2.0, 4.0, np.nan],
        "grp": ["a", "a", "a"],
    })
    out = eda.summarize_band(df, x="x", value="value", group="grp")
    assert out.loc[0, "mean"] == 3.0
    assert out.loc[0, "n"] == 2


def test_filter_band_by_support_drops_low_n():
    # sos-aligned overlays smear shoulder points across +/-300 days; those
    # extreme days are backed by only a few windows and must be dropped.
    band = pd.DataFrame({
        "grp": ["a", "a", "a"],
        "x": [-300, 0, 50],
        "mean": [0.1, 0.6, 0.5],
        "std": [0.0, 0.0, 0.0],
        "n": [2, 200, 150],
    })
    out = eda.filter_band_by_support(band, min_n=10)
    assert list(out["x"]) == [0, 50]  # the n=2 tail day is removed


def test_select_control_windows_filters():
    desc = pd.DataFrame({
        "PMT_SITE": ["A", "B", "C", "D", "E"],
        "window":   [0, 0, 0, 0, 0],
        "year":     [2021, 2021, 2021, 2020, 2021],
        "is_organic": [False, True, False, False, False],
        "has_season": [True, True, False, True, True],
        "SAVI_sos_time": [100, 100, 100, 100, 200],
        "SAVI_eos_time": [180, 180, 180, 180, 280],
    })
    # event at doy 130, year 2021, fired on plots {"E"} that year
    out = eda.select_control_windows(desc, event_doy=130, event_year=2021,
                                     event_plots={"E"})
    # A only: B organic, C no-season, D wrong year, E is an event plot + season
    # 200..280 does not contain 130 anyway
    assert list(out["PMT_SITE"]) == ["A"]


def test_calendar_window_slope_over_dates():
    series = pd.DataFrame({
        "date": pd.to_datetime(["2021-05-01", "2021-05-06", "2021-05-11",
                                "2021-06-30"]),
        "value": [1.0, 0.5, 0.0, 5.0],   # -0.1/day over the first 10 days
    })
    s = eda.calendar_window_slope(series, pd.Timestamp("2021-05-01"), days=14)
    assert abs(s - (-0.1)) < 1e-9          # the 2021-06-30 point is outside
