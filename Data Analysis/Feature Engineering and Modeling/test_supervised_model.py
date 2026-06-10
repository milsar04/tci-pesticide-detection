# test_supervised_model.py
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supervised_model as sm
import model_features as mf


def test_make_booster_has_predict_proba():
    b = sm.make_booster()
    assert hasattr(b, "fit") and hasattr(b, "predict_proba")


def test_baseline_estimator_uses_single_feature_column():
    Xb = sm.baseline_features(pd.DataFrame(
        {c: np.arange(3, dtype=float) for c in mf.FEATURE_COLS}))
    assert list(Xb.columns) == ["VH_pos_value"]


def test_rollup_by_plot_max_and_mean():
    scores = pd.DataFrame({
        "PMT_SITE": ["A", "A", "B"],
        "treated": [1, 1, 0],
        "p_treated": [0.2, 0.8, 0.5],
    })
    roll = sm.rollup_by_plot(scores)
    a = roll[roll["PMT_SITE"] == "A"].iloc[0]
    assert a["p_treated_max"] == 0.8
    assert abs(a["p_treated_mean"] - 0.5) < 1e-9
    assert a["n_windows"] == 2 and a["any_treated"] == 1


def test_metrics_table_records_baseline_delta():
    baseline = {"roc_auc_mean": 0.70, "pr_auc_mean": 0.80}
    booster = {"roc_auc_mean": 0.78, "pr_auc_mean": 0.85}
    tbl = sm.metrics_table(baseline, booster)
    brow = tbl[tbl["model"] == "booster"].iloc[0]
    assert abs(brow["roc_auc_delta_vs_baseline"] - 0.08) < 1e-9


def test_integration_run_writes_outputs_and_beats_floor(tmp_path):
    # reduced real-data run: 1 repeat, 3 folds, tiny grid -> fast.
    res = sm.run(out_dir=str(tmp_path), n_repeats=1, n_splits=3, quick=True)
    assert os.path.exists(os.path.join(tmp_path, "model_metrics.csv"))
    assert os.path.exists(os.path.join(tmp_path, "model_risk_scores.csv"))
    assert os.path.exists(os.path.join(tmp_path, "model_risk_by_plot.csv"))
    # the booster must clear a sanity floor (single-descriptor baseline ~0.76).
    assert res["booster"]["roc_auc_mean"] > 0.70
    # every exported risk score is a probability.
    scores = pd.read_csv(os.path.join(tmp_path, "model_risk_scores.csv"))
    assert scores["p_treated"].between(0, 1).all()
    assert len(scores) == 405, f"expected 405 rows, got {len(scores)}"
    # the tuned operating-point threshold is recorded in the metrics table.
    metrics = pd.read_csv(os.path.join(tmp_path, "model_metrics.csv"))
    assert "threshold_mean" in metrics.columns
    assert metrics["threshold_mean"].between(0, 1).all()
