# model_diagnostics.py - explainability + robustness reporting over the trained
# booster. reuses model_features + model_eval + the supervised booster factory;
# does NOT change the production scoring path in supervised_model.run().

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """global permutation importance on held-out grouped folds, averaged across folds."""
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
    """lightgbm native gain importance, or None if the backend has none."""
    m = clone(estimator).fit(X, y)
    if not hasattr(m, "feature_importances_"):
        return None
    return (pd.DataFrame({"feature": list(X.columns),
                          "gain": m.feature_importances_})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True))
