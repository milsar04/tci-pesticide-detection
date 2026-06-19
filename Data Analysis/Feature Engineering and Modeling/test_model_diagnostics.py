# test_model_diagnostics.py
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_diagnostics as md
import supervised_model as sm


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
