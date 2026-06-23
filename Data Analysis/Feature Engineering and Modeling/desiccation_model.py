# desiccation_model.py - desiccation-event detector: calibrated P(desiccation)
# over the event-anchored decline descriptors. mirrors supervised_model.py and
# reuses model_eval + the supervised booster/calibration factories unchanged.
# the positive class (event) is RARE (~80:1), so PR-AUC (floor ~ prevalence) is
# the headline metric, not ROC-AUC.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_features as mf
import model_eval as me
import supervised_model as sm
from descriptor_comparison import event_descriptors, EVENT_DESCRIPTORS

# sibling dirs for pipeline_config (IS) + select_control_windows (EDA).
_IS_DIR = os.path.join(os.path.dirname(HERE), "Imputation and Smoothing")
_EDA_DIR = os.path.join(os.path.dirname(HERE), "Exploratory Data Analysis")
for _d in (_IS_DIR, _EDA_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import pipeline_config as cfg

BASELINE_FEATURE = "RVI_post_slope_14d"          # best single descriptor (AUC 0.86)
EVENT_FEATURE_COLS = [f"{idx}_{k}" for idx in cfg.KEEP_INDICES for k in EVENT_DESCRIPTORS]


def _instance_descriptors(data, plot, edate):
    """the 20 event descriptors (5 indices x 4) for one plot's window around edate."""
    lo = edate - pd.Timedelta(days=cfg.EVENT_PRE_DAYS)
    hi = edate + pd.Timedelta(days=cfg.EVENT_POST_DAYS)
    w = data[(data["PMT_SITE"] == plot) & (data["date"] >= lo) & (data["date"] <= hi)]
    offs = (w["date"] - edate).dt.days.values
    row = {}
    for idx in cfg.KEEP_INDICES:
        d = event_descriptors(offs, w[idx].values) if idx in w.columns else {}
        for k in EVENT_DESCRIPTORS:
            row[f"{idx}_{k}"] = d.get(k, np.nan)
    return row


def _treated_date_index(data):
    """per-plot sorted spray dates (rows whose per-day Treatment status is not
    'No'). Treatment status is per observation row, so this is the day-level
    record of applications. empty if the column is absent (synthetic test
    frames), which makes control screening a no-op."""
    if "Treatment status" not in data.columns:
        return {}
    nonno = data["Treatment status"].fillna("No").astype(str).str.strip().str.lower().ne("no")
    t = data.loc[nonno, ["PMT_SITE", "date"]]
    return {p: sorted(set(pd.to_datetime(g["date"], errors="coerce").dt.normalize().dropna()))
            for p, g in t.groupby("PMT_SITE")}


def _treated_in_window(treated_idx, plot, edate):
    """True if plot has any spray date within the event window around edate."""
    lo = edate - pd.Timedelta(days=cfg.EVENT_PRE_DAYS)
    hi = edate + pd.Timedelta(days=cfg.EVENT_POST_DAYS)
    return any(lo <= d <= hi for d in treated_idx.get(plot, ()))


def build_event_xy(joined, data, desc, clean_controls=True):
    """assemble the detector matrix: one row per event (y=1) and per matched-date
    control (y=0), columns = the 20 event descriptors on the unsmoothed series.
    groups = PMT_SITE (controls repeat plots, so grouped CV is mandatory).
    clean_controls=True drops any control whose per-day Treatment status is non-No
    somewhere in its event window, so the negatives are genuinely untreated rather
    than unlabelled applications the rough event list missed (see _treated_*)."""
    from eda import select_control_windows
    event_plots_by_year = (joined.groupby(joined["date"].dt.year)["matched_plot"]
                           .apply(set).to_dict())
    treated_idx = _treated_date_index(data) if clean_controls else {}
    rows, ys, groups = [], [], []
    for _, ev in joined.iterrows():
        plot, edate = ev["matched_plot"], ev["date"]
        rows.append(_instance_descriptors(data, plot, edate)); ys.append(1); groups.append(plot)
        controls = select_control_windows(desc, edate.dayofyear, edate.year,
                                          event_plots_by_year.get(edate.year, set()))
        for cplot in controls["PMT_SITE"].unique():
            if clean_controls and _treated_in_window(treated_idx, cplot, edate):
                continue
            rows.append(_instance_descriptors(data, cplot, edate)); ys.append(0); groups.append(cplot)
    X = pd.DataFrame(rows, columns=EVENT_FEATURE_COLS).astype(float)
    y = pd.Series(ys, name="event").astype(int)
    g = pd.Series(groups, name="PMT_SITE")
    meta = pd.DataFrame({"PMT_SITE": groups, "event": ys})
    return X, y, g, meta


def _figures(y, p_oof, out_dir):
    """calibration + PR curves for the detector (positive = event)."""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    prevalence = float(np.asarray(y, int).mean())

    frac_pos, mean_pred = calibration_curve(y, p_oof, n_bins=8, strategy="quantile")
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(mean_pred, frac_pos, "o-")
    plt.xlabel("mean predicted P(desiccation)")
    plt.ylabel("observed event rate")
    plt.title("calibration - desiccation detector (grouped out-of-fold)")
    plt.savefig(os.path.join(fig_dir, "desiccation_calibration.png"), dpi=120, bbox_inches="tight")
    plt.close()

    prec, rec, _ = precision_recall_curve(y, p_oof)
    plt.figure()
    plt.plot(rec, prec)
    plt.axhline(prevalence, ls="--", color="gray", label=f"no-skill {prevalence:.3f}")
    plt.xlabel("recall (event)")
    plt.ylabel("precision (event)")
    plt.title("precision-recall - desiccation detector (out-of-fold)")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, "desiccation_pr.png"), dpi=120, bbox_inches="tight")
    plt.close()


def run(out_dir=HERE, n_repeats=3, n_splits=3, seed=42, clean_controls=True):
    """evaluate baseline + booster on the event matrix, calibrate + export. PR-AUC
    (positive=event, no-skill floor ~ prevalence) is the headline number. reuses
    the supervised booster + grouped leak-free calibration unchanged.
    clean_controls=True screens the negatives to genuinely untreated windows
    using the per-day Treatment status (see build_event_xy)."""
    import activity_filter as af
    joined, _ = af.load_desiccation_events(write_report=False)
    data = af.load_imputed_unsmoothed()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    desc = pd.read_csv(mf.IN_FILE)               # phenology_descriptors.csv
    X, y, groups, meta = build_event_xy(joined, data, desc, clean_controls=clean_controls)

    # baseline = LR on the single best descriptor; headline = booster on all 20.
    # tune_threshold left off: the class-0 operating point is not meaningful here,
    # and PR-AUC/ROC-AUC are threshold-free.
    base = me.evaluate_estimator(sm.baseline_estimator(), X[[BASELINE_FEATURE]], y, groups,
                                 n_repeats=n_repeats, n_splits=n_splits, seed=seed)
    boost = me.evaluate_estimator(sm.make_booster(), X, y, groups,
                                  n_repeats=n_repeats, n_splits=n_splits, seed=seed)

    # leak-free grouped calibration, method picked by out-of-fold Brier (reused).
    p_oof, method = sm._choose_calibration(X, y, groups, None, n_splits=n_splits, seed=seed)
    _figures(y, p_oof, out_dir)
    final = sm.fit_calibrated(X, y, groups, method=method, seed=seed)
    p_all = final.predict_proba(X)[:, 1]

    scores = meta.copy()
    scores["p_desiccation"] = p_all
    scores.to_csv(os.path.join(out_dir, "desiccation_risk_scores.csv"),
                  index=False, float_format="%.4f")

    prevalence = float(y.mean())
    rows = []
    for name, m in [("baseline_lr", base), ("booster", boost)]:
        rows.append({"model": name,
                     "pr_auc_mean": m["pr_auc_mean"], "pr_auc_std": m["pr_auc_std"],
                     "roc_auc_mean": m["roc_auc_mean"], "roc_auc_std": m["roc_auc_std"],
                     "brier_mean": m["brier_mean"],
                     "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                     "pr_auc_no_skill": prevalence})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "desiccation_metrics.csv"),
                              index=False, float_format="%.4f")

    print(f"desiccation detector ({int(y.sum())} events / {int((y == 0).sum())} controls):")
    print(f"  baseline PR-AUC {base['pr_auc_mean']:.3f} | booster PR-AUC {boost['pr_auc_mean']:.3f} "
          f"(no-skill {prevalence:.3f}) | booster ROC-AUC {boost['roc_auc_mean']:.3f}")
    return {"baseline": base, "booster": boost}


if __name__ == "__main__":
    run()
