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
