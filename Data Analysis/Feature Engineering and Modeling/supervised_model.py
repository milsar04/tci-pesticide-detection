# supervised_model.py - supervised track: calibrated P(treated) risk score.
# baseline = logistic regression on VH_pos_value; headline = LightGBM (HistGradientBoosting
# fallback), isotonic-calibrated. evaluated through model_eval's grouped repeated CV.

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import precision_recall_curve

import model_features as mf
import model_eval as me

# booster backend: lightgbm headline, sklearn HistGradientBoosting fallback ----
try:
    from lightgbm import LGBMClassifier
    _HAVE_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAVE_LGBM = False

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_FEATURE = "VH_pos_value"


def make_booster(class_weight="balanced", **params):
    """gradient booster with native NaN handling either way."""
    # class_weight is task-agnostic - the deferred desiccation detector (~80:1)
    # passes its own weight without changing this factory.
    if _HAVE_LGBM:
        defaults = dict(objective="binary", class_weight=class_weight,
                        n_estimators=300, learning_rate=0.05, num_leaves=31,
                        min_child_samples=10, random_state=42, n_jobs=1, verbose=-1)
        defaults.update(params)
        return LGBMClassifier(**defaults)
    defaults = dict(class_weight=class_weight, max_iter=300, learning_rate=0.05,
                    max_leaf_nodes=31, min_samples_leaf=10, random_state=42)
    defaults.update(params)
    return HistGradientBoostingClassifier(**defaults)


def booster_param_grid(quick=False):
    """small grid (lightgbm only). returns None when no tuning applies."""
    if not _HAVE_LGBM:
        # no tuning for histgradientboosting fallback
        return None
    if quick:
        return {"n_estimators": [200], "num_leaves": [31]}
    return {"n_estimators": [200, 400], "num_leaves": [15, 31],
            "learning_rate": [0.03, 0.06], "min_child_samples": [10, 20]}


def baseline_features(X):
    """single-descriptor baseline design matrix."""
    return X[[BASELINE_FEATURE]]


def baseline_estimator(class_weight="balanced"):
    """median-impute -> standardize -> logistic regression on one descriptor.
    class_weight is the same parameterized cost-sensitive knob as make_booster."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight=class_weight, max_iter=1000)),
    ])


def final_calibrated_model(best_params=None):
    """isotonic-calibrated booster for the exported probabilities."""
    base = make_booster(**(best_params or {}))
    # cv=3 uses stratified (not grouped) folds inside CalibratedClassifierCV - intentional.
    # calibration is a post-hoc monotonic transform; within-plot leakage here does not
    # affect the AUC/PR-AUC metrics (which come from the outer grouped CV in model_eval).
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def rollup_by_plot(scores):
    """per-plot max and mean of the window-level risk score."""
    return (scores.groupby("PMT_SITE")
            .agg(p_treated_max=("p_treated", "max"),
                 p_treated_mean=("p_treated", "mean"),
                 n_windows=("p_treated", "count"),
                 any_treated=("treated", "max"))
            .reset_index())


def metrics_table(baseline, booster):
    """tidy table with the booster's delta vs the baseline on the shared metrics."""
    rows = []
    for name, m in [("baseline_lr", baseline), ("booster", booster)]:
        row = {"model": name, **m}
        if name == "booster":
            for k in ("roc_auc_mean", "pr_auc_mean"):
                row[k.replace("_mean", "_delta_vs_baseline")] = m[k] - baseline[k]
        rows.append(row)
    return pd.DataFrame(rows)


def _figures(y, p_oof, out_dir):
    """calibration curve + PR curve from out-of-fold probabilities."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    frac_pos, mean_pred = calibration_curve(y, p_oof, n_bins=8, strategy="quantile")
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(mean_pred, frac_pos, "o-")
    plt.xlabel("mean predicted P(treated)")
    plt.ylabel("observed treated rate")
    plt.title("calibration - isotonic booster (out-of-fold)")
    plt.savefig(os.path.join(fig_dir, "calibration_curve.png"), dpi=120, bbox_inches="tight")
    plt.close()

    prec, rec, _ = precision_recall_curve(y, p_oof)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("recall (treated)")
    plt.ylabel("precision (treated)")
    plt.title("precision-recall - booster (out-of-fold)")
    plt.savefig(os.path.join(fig_dir, "pr_curve.png"), dpi=120, bbox_inches="tight")
    plt.close()


def run(out_dir=HERE, n_repeats=5, n_splits=5, seed=42, quick=False):
    """evaluate baseline + booster, refit a calibrated model on all windows, write outputs."""
    # build_xy does NOT filter - filter first, else you get all 821 windows not 405.
    df = mf.filter_modeling_rows(mf.load_descriptors())
    X, y, groups, meta = mf.build_xy(df)

    # evaluate baseline and booster through the shared grouped-CV harness. the
    # operating-point threshold is tuned per training fold (out-of-fold) so the
    # reported untreated recall/precision are not at the meaningless 0.5 default;
    # the mean tuned threshold lands in model_metrics.csv via fold_metrics.
    baseline = me.evaluate_estimator(baseline_estimator(), baseline_features(X), y, groups,
                                     n_repeats=n_repeats, n_splits=n_splits, seed=seed,
                                     tune_threshold=True)
    booster = me.evaluate_estimator(make_booster(), X, y, groups,
                                    param_grid=booster_param_grid(quick=quick),
                                    n_repeats=n_repeats, n_splits=n_splits, seed=seed,
                                    tune_threshold=True)

    # pick final hyperparameters once on all data (lightgbm only), then calibrate + refit.
    best_params = None
    grid = booster_param_grid(quick=quick)
    if grid:
        gs = GridSearchCV(make_booster(), grid, scoring="average_precision",
                          cv=me.make_splitter(n_splits=3, seed=seed), n_jobs=1)
        gs.fit(X, y, groups=groups)
        best_params = gs.best_params_

    final = final_calibrated_model(best_params)
    # out-of-fold probabilities for honest calibration/PR figures.
    p_oof = cross_val_predict(clone(final), X, y,
                              cv=me.make_splitter(n_splits=n_splits, seed=seed),
                              groups=groups, method="predict_proba")[:, 1]
    _figures(y, p_oof, out_dir)

    # refit on all windows for the exported scores.
    final.fit(X, y)
    p_all = final.predict_proba(X)[:, 1]

    scores = meta[["PMT_SITE", "window", "year", "treated"]].copy()
    scores["p_treated"] = p_all
    scores.to_csv(os.path.join(out_dir, "model_risk_scores.csv"), index=False,
                  float_format="%.4f")
    rollup_by_plot(scores).to_csv(os.path.join(out_dir, "model_risk_by_plot.csv"),
                                  index=False, float_format="%.4f")
    metrics_table(baseline, booster).to_csv(os.path.join(out_dir, "model_metrics.csv"),
                                            index=False, float_format="%.4f")

    backend = "lightgbm" if _HAVE_LGBM else "histgradientboosting (fallback)"
    print(f"booster backend: {backend}")
    print(f"baseline roc_auc {baseline['roc_auc_mean']:.3f} | "
          f"booster roc_auc {booster['roc_auc_mean']:.3f} | "
          f"booster pr_auc {booster['pr_auc_mean']:.3f}")
    return {"baseline": baseline, "booster": booster, "best_params": best_params}


if __name__ == "__main__":
    run()
