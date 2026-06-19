# test_model_diagnostics.py
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_diagnostics as md
import supervised_model as sm


def test_year_masks_are_disjoint_and_complete():
    meta = pd.DataFrame({"year": [2020, 2020, 2021, 2021, 2020]})
    tr, te = md._year_masks(meta, 2021)
    assert not (tr & te).any()
    assert (tr | te).all()
    assert te.sum() == 2 and tr.sum() == 3


def test_permutation_importance_returns_one_row_per_feature():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n)})
    y = pd.Series((X["a"] > 0).astype(int).to_numpy())
    groups = pd.Series([f"P{i % 12}" for i in range(n)])
    imp = md.permutation_importance_grouped(sm.make_booster(class_weight=None),
                                            X, y, groups, n_splits=3, seed=0,
                                            n_repeats=3)
    assert set(imp["feature"]) == {"a", "b", "c"}
    assert len(imp) == 3
    assert {"importance_mean", "importance_std"} <= set(imp.columns)


def test_diagnostics_run_writes_outputs(tmp_path):
    # reduced real-data run: 1 seed, no grid, small permutation repeats -> fast.
    md.run(out_dir=str(tmp_path), quick=True)
    assert os.path.exists(os.path.join(tmp_path, "feature_importance.csv"))
    assert os.path.exists(os.path.join(tmp_path, "robustness.csv"))
    assert os.path.exists(os.path.join(tmp_path, "figures", "feature_importance.png"))
    fi = pd.read_csv(os.path.join(tmp_path, "feature_importance.csv"))
    assert len(fi) == 65  # one row per descriptor
    rob = pd.read_csv(os.path.join(tmp_path, "robustness.csv"))
    assert set(rob["check"]) >= {"seed_stability", "leave_one_year_out",
                                 "imbalance_sensitivity", "timing_ablation"}
