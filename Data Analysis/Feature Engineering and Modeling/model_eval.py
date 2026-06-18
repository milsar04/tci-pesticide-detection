# model_eval.py - shared grouped cross-validation harness and metric helpers.
# group = PMT_SITE so no plot's windows straddle train and test; stratify on treated.

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             recall_score, precision_score, brier_score_loss)


def make_splitter(n_splits=5, seed=42):
    """grouped, stratified k-fold splitter."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def fold_metrics(y_true, p, threshold=0.5):
    """all per-fold metrics for one set of probabilities p = P(treated). the
    threshold-free metrics (pr_auc, roc_auc, brier) ignore `threshold`; the
    operating-point metrics (recall/precision on the untreated class) use it. the
    chosen threshold is echoed so aggregate() can report it as the operating point."""
    y_true = np.asarray(y_true, int)
    p = np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, p),          # treated is positive
        "roc_auc": roc_auc_score(y_true, p),
        "recall_untreated": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "precision_untreated": precision_score(y_true, pred, pos_label=0, zero_division=0),
        "brier": brier_score_loss(y_true, p),
        "threshold": float(threshold),
    }


def pick_threshold(y_true, p, min_recall=0.7):
    """choose a decision threshold for the rare untreated (class 0): the most
    precise threshold whose untreated recall still clears min_recall (catch the
    minority class, then be as precise as possible). candidates are the unique
    probabilities plus 0 and 1; falls back to 0.5 if nothing qualifies. raising
    the threshold predicts more low-scored points as untreated, so untreated
    recall rises with t."""
    y_true = np.asarray(y_true, int)
    p = np.asarray(p, float)
    cands = np.unique(np.concatenate([[0.0], p, [1.0]]))
    best_t, best_score = 0.5, -np.inf
    for t in cands:
        pred = (p >= t).astype(int)
        rec0 = recall_score(y_true, pred, pos_label=0, zero_division=0)
        prec0 = precision_score(y_true, pred, pos_label=0, zero_division=0)
        if rec0 >= min_recall and prec0 >= best_score:
            best_score, best_t = prec0, t
    return float(best_t)


def aggregate(rows):
    """mean +/- std (ddof=1) across folds for every metric key."""
    df = pd.DataFrame(rows)
    out = {}
    for c in df.columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def evaluate_estimator(estimator, X, y, groups, *, param_grid=None,
                       n_repeats=5, n_splits=5, inner_splits=3, seed=42,
                       tune_threshold=False, min_recall=0.7):
    """repeated grouped CV. if param_grid is given, each outer training fold runs a
    grouped GridSearchCV (nested CV, scoring=average_precision); else the estimator
    is fit directly. if tune_threshold, the decision threshold is chosen on the
    training fold only - out-of-fold via an inner grouped split, so the test fold
    never informs the threshold - and applied to the test fold. the threshold-free
    metrics (pr_auc etc.) are unaffected. returns aggregate() of per-fold metrics."""
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    rows = []
    for r in range(n_repeats):
        outer = make_splitter(n_splits=n_splits, seed=seed + r)
        for tr, te in outer.split(X, y, groups):
            Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
            if param_grid:
                inner = make_splitter(n_splits=inner_splits, seed=seed + r)
                model = GridSearchCV(clone(estimator), param_grid,
                                     scoring="average_precision", cv=inner, n_jobs=1)
                model.fit(Xtr, ytr, groups=gtr)
            else:
                model = clone(estimator)
                model.fit(Xtr, ytr)
            thr = 0.5
            if tune_threshold:
                # out-of-fold train probabilities (default hyperparams - the threshold
                # is a coarse operating point and robust to the small grid choice).
                p_tr = cross_val_predict(
                    clone(estimator), Xtr, ytr,
                    cv=make_splitter(n_splits=inner_splits, seed=seed + r),
                    groups=gtr, method="predict_proba")[:, 1]
                thr = pick_threshold(ytr, p_tr, min_recall=min_recall)
            p = model.predict_proba(X.iloc[te])[:, 1]
            rows.append(fold_metrics(y[te], p, threshold=thr))
    return aggregate(rows)
