# test_descriptor_comparison.py
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import descriptor_comparison as dc


def test_cliffs_delta_fully_separated():
    assert dc.cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert dc.cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0


def test_cliffs_delta_identical_is_zero():
    assert dc.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_compare_descriptor_separable():
    rng = np.random.default_rng(0)
    treated = rng.normal(1.0, 0.1, 40)
    untreated = rng.normal(0.0, 0.1, 40)
    values = np.concatenate([treated, untreated])
    labels = np.array([1] * 40 + [0] * 40)
    r = dc.compare_descriptor(values, labels)
    assert r["auc"] > 0.95
    assert abs(r["cliffs_delta"]) > 0.8
    assert r["mw_p"] < 0.01
    assert r["n_treated"] == 40 and r["n_untreated"] == 40


def test_compare_descriptor_drops_nan_and_guards_small_n():
    values = np.array([1.0, np.nan, 2.0, 3.0])
    labels = np.array([1, 1, 0, 0])
    r = dc.compare_descriptor(values, labels)  # only 1 treated after nan drop
    assert np.isnan(r["auc"])
    assert r["n_treated"] == 1


def test_fit_slope_known_line():
    # y = 2x + 1 -> slope 2
    assert abs(dc.fit_slope([0, 1, 2, 3], [1, 3, 5, 7]) - 2.0) < 1e-9


def test_fit_slope_ignores_nan_and_guards():
    assert abs(dc.fit_slope([0, 1, 2], [0.0, np.nan, 2.0]) - 1.0) < 1e-9
    assert np.isnan(dc.fit_slope([5], [1.0]))          # < 2 points
    assert np.isnan(dc.fit_slope([2, 2, 2], [1, 2, 3]))  # zero x-range


def test_event_descriptors_known_drop():
    # value 1.0 at event, falls to 0.0 by day 10, flat before the event.
    # days: -14..21 sampled every 7 days plus day 10.
    days =   [-14, -7,  0,  7, 10, 14, 21]
    values = [1.0, 1.0, 1.0, 0.3, 0.0, 0.0, 0.0]
    d = dc.event_descriptors(days, values)
    assert d["drop_magnitude"] == 1.0                 # 1.0 -> 0.0
    assert d["post_slope_14d"] < 0                    # declining after event
    assert d["time_to_half_drop"] == 7.0              # first day value <= 0.5
    assert d["slope_contrast"] < 0                    # post steeper than flat pre


def test_event_descriptors_guards_empty():
    d = dc.event_descriptors([0, 1], [np.nan, np.nan])
    assert all(np.isnan(v) for v in d.values())
