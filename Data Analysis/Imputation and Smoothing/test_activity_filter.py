# test_activity_filter.py
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activity_filter as af


def _windows_csv(tmp_path, name, rows):
    """write a minimal activity-date csv (DD/MM/YYYY) and return its path."""
    p = tmp_path / name
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False)
    return str(p)


def test_parse_aliases_handles_list_and_nan():
    assert af._parse_aliases("['a', 'b']") == ["a", "b"]
    assert af._parse_aliases("['x']") == ["x"]
    assert af._parse_aliases(np.nan) == []
    assert af._parse_aliases("garbage{") == []


def test_load_windows_registers_primary_and_alias(tmp_path):
    path = _windows_csv(tmp_path, "a.csv", [
        {"PMT_SITE": "P1", "PMT_SITE_other": "['P1ALT']",
         "Active_date": "01/03/2021", "Inactive_date": "01/09/2021"},
    ])
    windows, bad = af.load_activity_windows([path])
    assert ("P1" in windows) and ("P1ALT" in windows)
    assert windows["P1"][0][0] == pd.Timestamp("2021-03-01")
    assert windows["P1"][0][1] == pd.Timestamp("2021-09-01")
    assert bad.empty


def test_load_windows_drops_inverted_dates(tmp_path):
    path = _windows_csv(tmp_path, "a.csv", [
        {"PMT_SITE": "BAD", "PMT_SITE_other": "[]",
         "Active_date": "01/01/2020", "Inactive_date": "12/12/2019"},
    ])
    windows, bad = af.load_activity_windows([path])
    assert "BAD" not in windows
    assert len(bad) == 1 and bad.iloc[0]["plot"] == "BAD"


def test_load_windows_unions_across_files(tmp_path):
    p1 = _windows_csv(tmp_path, "y20.csv", [
        {"PMT_SITE": "P", "PMT_SITE_other": "[]",
         "Active_date": "01/01/2020", "Inactive_date": "01/06/2020"},
    ])
    p2 = _windows_csv(tmp_path, "y21.csv", [
        {"PMT_SITE": "P", "PMT_SITE_other": "[]",
         "Active_date": "01/01/2021", "Inactive_date": "01/06/2021"},
    ])
    windows, _ = af.load_activity_windows([p1, p2])
    assert len(windows["P"]) == 2


def test_apply_filter_keeps_only_in_window_rows_and_adds_organic():
    windows = {"P": [(pd.Timestamp("2021-03-01"), pd.Timestamp("2021-09-01"))]}
    df = pd.DataFrame({
        "PMT_SITE": ["P", "P", "P", "Q"],
        "date": pd.to_datetime(["2021-02-01", "2021-05-01", "2021-10-01",
                                "2021-05-01"]),
        "COMM": ["POTATO", "POTATO", "POTATO", "POTATO-ORGANIC"],
    })
    out = af.apply_activity_filter(df, windows, strict=True)
    # only the in-window P row survives in strict mode (Q has no window)
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == ["2021-05-01"]
    assert "is_organic" in out.columns


def test_apply_filter_lenient_keeps_unknown_plots():
    windows = {"P": [(pd.Timestamp("2021-03-01"), pd.Timestamp("2021-09-01"))]}
    df = pd.DataFrame({
        "PMT_SITE": ["P", "Q"],
        "date": pd.to_datetime(["2021-10-01", "2021-05-01"]),
        "COMM": ["POTATO", "POTATO"],
    })
    out = af.apply_activity_filter(df, windows, strict=False)
    # P's row is out of window -> dropped; Q is unknown -> kept (flagged)
    assert set(out["PMT_SITE"]) == {"Q"}


def test_is_organic_flag_values():
    windows = {"P": [(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"))]}
    df = pd.DataFrame({
        "PMT_SITE": ["P", "P"],
        "date": pd.to_datetime(["2021-05-01", "2021-06-01"]),
        "COMM": ["POTATO", "POTATO-ORGANIC"],
    })
    out = af.apply_activity_filter(df, windows, strict=True)
    assert out.sort_values("date")["is_organic"].tolist() == [False, True]


def test_window_known_flag_marks_unmatched_plots():
    # lenient mode keeps unmatched plots unmasked; window_known must mark them
    # False so downstream phases can sensitivity-check with/without them.
    windows = {"P": [(pd.Timestamp("2021-03-01"), pd.Timestamp("2021-09-01"))]}
    df = pd.DataFrame({
        "PMT_SITE": ["P", "Q"],
        "date": pd.to_datetime(["2021-05-01", "2021-05-01"]),
        "COMM": ["POTATO", "POTATO"],
    })
    out = af.apply_activity_filter(df, windows, strict=False)
    flags = out.set_index("PMT_SITE")["window_known"].to_dict()
    assert flags["P"] is True or flags["P"] == True   # matched, in-window
    assert flags["Q"] is False or flags["Q"] == False  # unmatched, kept on faith
