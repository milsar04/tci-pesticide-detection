# model_diagnostics.py - explainability + robustness reporting over the trained
# booster. reuses model_features + model_eval + the supervised booster factory;
# does NOT change the production scoring path in supervised_model.run().

import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_features as mf
import model_eval as me
import supervised_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))


# explainability ----

def permutation_importance_grouped(estimator, X, y, groups, n_splits=5, seed=42,
                                   n_repeats=10):
    """grouped OOF permutation importance (n_splits folds x n_repeats column shuffles per fold)."""
    outer = me.make_splitter(n_splits=n_splits, seed=seed)
    accum = []
    for tr, te in outer.split(X, y, groups):
        m = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(m, X.iloc[te], y.iloc[te],
                                   scoring="average_precision",
                                   n_repeats=n_repeats, random_state=seed, n_jobs=1)
        accum.append(r.importances_mean)
    imp = np.vstack(accum)
    return (pd.DataFrame({"feature": list(X.columns),
                          "importance_mean": imp.mean(axis=0),
                          "importance_std": imp.std(axis=0, ddof=1)})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


def gain_importance(estimator, X, y):
    """native feature_importances_ from a fresh fit on all data, or None if unavailable."""
    m = clone(estimator).fit(X, y)
    if not hasattr(m, "feature_importances_"):
        return None
    return (pd.DataFrame({"feature": list(X.columns),
                          "gain": m.feature_importances_})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True))


# robustness ----

def seed_stability(X, y, groups, seeds=(42, 0, 1, 7, 123), n_splits=5):
    """headline metrics across splitter seeds - turns the single number into a range."""
    rows = []
    for s in seeds:
        m = me.evaluate_estimator(sm.make_booster(), X, y, groups,
                                  n_repeats=1, n_splits=n_splits, seed=s,
                                  tune_threshold=True)
        rows.append({"check": "seed_stability", "param": s,
                     "roc_auc": m["roc_auc_mean"], "pr_auc": m["pr_auc_mean"]})
    return pd.DataFrame(rows)


def _year_masks(meta, test_year):
    """(train_mask, test_mask) numpy bool arrays for a leave-one-year-out split."""
    te = (meta["year"] == test_year).to_numpy()
    return ~te, te


def leave_one_year_out(X, y, meta):
    """train on all-but-one year, test on the held-out year. repeats for each year."""
    y = np.asarray(y, int)
    rows = []
    for test_year in sorted(meta["year"].unique()):
        tr, te = _year_masks(meta, test_year)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = sm.make_booster().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        rows.append({"check": "leave_one_year_out", "param": int(test_year),
                     "roc_auc": roc_auc_score(y[te], p),
                     "pr_auc": average_precision_score(y[te], p),
                     "n_test": int(te.sum())})
    return pd.DataFrame(rows)


def imbalance_sensitivity(X, y, groups, weights=(None, "balanced", 5, 10),
                          n_repeats=3, n_splits=5, seed=42):
    """confirm the class_weight knob earns its place."""
    rows = []
    for w in weights:
        cw = {0: w, 1: 1} if isinstance(w, (int, float)) else w
        m = me.evaluate_estimator(sm.make_booster(class_weight=cw), X, y, groups,
                                  n_repeats=n_repeats, n_splits=n_splits, seed=seed,
                                  tune_threshold=True)
        rows.append({"check": "imbalance_sensitivity", "param": str(w),
                     "roc_auc": m["roc_auc_mean"], "pr_auc": m["pr_auc_mean"],
                     "recall_untreated": m["recall_untreated_mean"],
                     "precision_untreated": m["precision_untreated_mean"]})
    return pd.DataFrame(rows)


def timing_ablation(df, n_repeats=3, n_splits=5, seed=42):
    """booster with vs without the 20 timing descriptors."""
    rows = []
    for drop in (False, True):
        X, y, groups, _ = mf.build_xy(df, drop_timing=drop)
        m = me.evaluate_estimator(sm.make_booster(), X, y, groups,
                                  n_repeats=n_repeats, n_splits=n_splits, seed=seed)
        rows.append({"check": "timing_ablation", "param": f"drop_timing={drop}",
                     "n_features": X.shape[1],
                     "roc_auc": m["roc_auc_mean"], "pr_auc": m["pr_auc_mean"]})
    return pd.DataFrame(rows)
