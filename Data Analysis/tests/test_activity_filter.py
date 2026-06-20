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


def test_load_events_parses_dates_and_ingredients(tmp_path):
    p = tmp_path / "events.csv"
    pd.DataFrame([
        {"PMT_SITE": " P1 ", "date": "2020-04-15",
         "Active ingredient": "['Diquat']", "iso_year": 2020},
        {"PMT_SITE": "P2", "date": "2020-05-16",
         "Active ingredient": "['Lambda-cyhalothrin', 'Pyraflufen-ethyl']",
         "iso_year": 2020},
    ]).to_csv(p, index=False)
    ev = af.load_events(str(p))
    assert list(ev.columns) == ["PMT_SITE", "date", "ingredients", "iso_year"]
    assert ev.loc[0, "PMT_SITE"] == "P1"               # stripped
    assert ev.loc[0, "date"] == pd.Timestamp("2020-04-15")
    assert ev.loc[0, "ingredients"] == ["Diquat"]
    assert ev.loc[1, "ingredients"] == ["Lambda-cyhalothrin", "Pyraflufen-ethyl"]
    assert int(ev.loc[1, "iso_year"]) == 2020


def test_build_plot_aliases_groups_equivalent_ids(tmp_path):
    path = _windows_csv(tmp_path, "a.csv", [
        {"PMT_SITE": "P1", "PMT_SITE_other": "['P1ALT', 'P1OLD']",
         "Active_date": "01/03/2021", "Inactive_date": "01/09/2021"},
    ])
    aliases = af.build_plot_aliases([path])
    assert aliases["P1ALT"] == frozenset({"P1", "P1ALT", "P1OLD"})
    assert aliases["P1"] == aliases["P1OLD"]


def test_resolve_event_plot_prefers_direct_then_alias():
    aliases = {"P1": frozenset({"P1", "P1ALT"}),
               "P1ALT": frozenset({"P1", "P1ALT"})}
    indices_plots = {"P1", "X"}
    # direct hit
    assert af.resolve_event_plot("P1", aliases, indices_plots) == "P1"
    # event uses the alias; resolves to the canonical indices id
    assert af.resolve_event_plot("P1ALT", aliases, indices_plots) == "P1"
    # no match anywhere
    assert af.resolve_event_plot("ZZZ", aliases, indices_plots) is None


def test_joinable_events_filters_and_tags_matched_plot():
    events = pd.DataFrame({
        "PMT_SITE": ["P1ALT", "GHOST"],
        "date": pd.to_datetime(["2020-04-15", "2020-04-15"]),
        "ingredients": [["Diquat"], ["Diquat"]],
        "iso_year": pd.array([2020, 2020], dtype="Int64"),
    })
    aliases = {"P1ALT": frozenset({"P1", "P1ALT"})}
    out = af.joinable_events(events, {"P1"}, aliases)
    assert list(out["PMT_SITE"]) == ["P1ALT"]
    assert list(out["matched_plot"]) == ["P1"]


def test_event_coverage_report_counts_and_audits():
    events = pd.DataFrame({
        "PMT_SITE": ["P1", "P2", "GHOST"],
        "date": pd.to_datetime(["2021-05-01", "2021-05-01", "2021-05-01"]),
        "ingredients": [["Diquat"], ["Diquat"], ["Diquat"]],
        "iso_year": pd.array([2021, 2021, 2021], dtype="Int64"),
    })
    joined = pd.DataFrame({
        "PMT_SITE": ["P1", "P2"],
        "date": pd.to_datetime(["2021-05-01", "2021-05-01"]),
        "ingredients": [["Diquat"], ["Diquat"]],
        "iso_year": pd.array([2021, 2021], dtype="Int64"),
        "matched_plot": ["P1", "P2"],
    })
    data = pd.DataFrame({
        "PMT_SITE": ["P1", "P2"],
        "COMM": ["POTATO", "POTATO-ORGANIC"],
        "Treatment status": ["No", "Fungicide"],
    })
    windows = {"P1": [(pd.Timestamp("2021-03-01"), pd.Timestamp("2021-09-01"))]}
    rep = af.event_coverage_report(events, joined, data, windows)
    assert "events total : 3" in rep
    assert "matched      : 2" in rep
    assert "GHOST" in rep                 # unmatched id listed
    assert "P2" in rep                    # organic-with-event audit
    assert "P1" in rep                    # label-noise audit (status "No")
