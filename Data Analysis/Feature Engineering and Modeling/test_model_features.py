# test_model_features.py
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_features as mf


def _toy_df():
    # 6 windows: 1 organic, 1 no-season, 4 usable (3 treated, 1 untreated).
    return pd.DataFrame({
        "PMT_SITE": ["A", "A", "B", "C", "D", "E"],
        "window": [0, 1, 0, 0, 0, 0],
        "year": [2020, 2020, 2020, 2021, 2021, 2020],
        "treated": [True, True, False, True, True, False],
        "treatment_type": ["herbicide", "herbicide", "no", "fungicide", "mixed", "no"],
        "is_organic": [False, False, False, False, True, False],
        "n_obs": [10, 12, 9, 11, 8, 7],
        "window_known": [True, True, True, False, True, True],
        "has_season": [True, True, True, True, True, False],
        # build all 65 whitelist cols with simple values so build_xy works.
        **{c: np.arange(6, dtype=float) for c in mf.FEATURE_COLS},
    })


def test_feature_cols_is_exactly_65():
    assert len(mf.FEATURE_COLS) == 65
    assert len(set(mf.FEATURE_COLS)) == 65


def test_timing_cols_is_20():
    # 5 indices x 4 timing suffixes (pos_time, sos_time, eos_time, los)
    assert len(mf.TIMING_COLS) == 20
    assert set(mf.TIMING_COLS).issubset(set(mf.FEATURE_COLS))


def test_feature_cols_excludes_leakage_and_meta():
    for bad in ["treatment_type", "year", "n_obs", "PMT_SITE", "window",
                "is_organic", "has_season", "window_known", "treated"]:
        assert bad not in mf.FEATURE_COLS


def test_filter_keeps_only_nonorganic_realseason():
    kept = mf.filter_modeling_rows(_toy_df())
    # drops the organic row (D) and the no-season row (E) -> 4 rows
    assert len(kept) == 4
    assert set(kept["PMT_SITE"]) == {"A", "B", "C"}


def test_build_xy_shapes_and_groups():
    # build_xy does NOT filter - pass it the filtered frame.
    kept = mf.filter_modeling_rows(_toy_df())
    X, y, groups, meta = mf.build_xy(kept)
    assert X.shape == (4, 65)
    assert list(X.columns) == mf.FEATURE_COLS
    assert y.tolist() == [1, 1, 0, 1]          # A,A treated; B untreated; C treated
    assert list(groups) == ["A", "A", "B", "C"]
    assert "PMT_SITE" in meta.columns and "treated" in meta.columns


def test_build_xy_drop_timing_removes_20_cols():
    kept = mf.filter_modeling_rows(_toy_df())
    X, _, _, _ = mf.build_xy(kept, drop_timing=True)
    assert X.shape[1] == 45
    assert not any(c in X.columns for c in mf.TIMING_COLS)


def test_real_file_filters_to_405_windows():
    kept = mf.filter_modeling_rows(mf.load_descriptors())
    assert len(kept) == 405
    assert int(kept["treated"].astype(int).sum()) == 309
