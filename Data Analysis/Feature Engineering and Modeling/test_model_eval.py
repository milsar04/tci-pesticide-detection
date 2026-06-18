# test_model_eval.py
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_eval as me


def test_fold_metrics_perfect_separation():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2])
    m = me.fold_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] > 0.99
    assert m["recall_untreated"] == 1.0     # both 0s predicted 0
    assert m["brier"] < 0.1


def test_recall_untreated_counts_zero_class():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])      # one untreated misclassified as treated
    m = me.fold_metrics(y, p)
    assert m["recall_untreated"] == 2 / 3


def test_aggregate_reports_mean_and_std():
    rows = [{"roc_auc": 0.8}, {"roc_auc": 0.6}]
    out = me.aggregate(rows)
    assert abs(out["roc_auc_mean"] - 0.7) < 1e-9
    assert out["roc_auc_std"] > 0


def test_no_group_leakage_across_folds():
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame({"f": rng.normal(size=n)})
    y = np.array([1, 0] * (n // 2))
    groups = np.repeat(np.arange(n // 2), 2)   # 2 windows per plot
    splitter = me.make_splitter(n_splits=4, seed=0)
    for tr, te in splitter.split(X, y, groups):
        assert not (set(groups[tr]) & set(groups[te]))


def test_evaluate_estimator_separable_beats_chance():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    n = 60
    # 30 plots, 2 windows each; treated plots have a higher feature mean.
    plot_treated = rng.integers(0, 2, size=30)
    rows = []
    for pid, t in enumerate(plot_treated):
        for _ in range(2):
            rows.append((pid, t, rng.normal(2.0 if t else 0.0, 0.5)))
    pid, y, f = zip(*rows)
    X = pd.DataFrame({"f": f})
    out = me.evaluate_estimator(LogisticRegression(), X, np.array(y),
                                np.array(pid), n_repeats=1, n_splits=3, seed=0)
    assert out["roc_auc_mean"] > 0.8


def test_fold_metrics_threshold_param_moves_operating_point():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.6, 0.4, 0.45, 0.3])
    m_hi = me.fold_metrics(y, p, threshold=0.5)    # preds [1,0,0,0] -> both 0s caught
    m_lo = me.fold_metrics(y, p, threshold=0.35)   # preds [1,1,1,0] -> one 0 missed
    assert m_hi["recall_untreated"] == 1.0
    assert m_lo["recall_untreated"] == 0.5
    assert m_hi["threshold"] == 0.5                # recorded as the operating point


def test_pick_threshold_meets_untreated_recall_floor():
    from sklearn.metrics import recall_score
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.3, 0.4])   # untreated clearly lower-scored
    t = me.pick_threshold(y, p, min_recall=1.0)
    pred = (p >= t).astype(int)
    assert recall_score(y, pred, pos_label=0) == 1.0   # all untreated caught
    assert t > 0.4                                      # tuned away from 0.5
