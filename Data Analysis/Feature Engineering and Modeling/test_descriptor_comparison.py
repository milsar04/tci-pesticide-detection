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
