# evaluate_all.py
# runs every imputation + smoothing method from app.py across the whole
# dataset and ranks them to pick the global best ones.
# usage: python evaluate_all.py

import time
import warnings
import numpy as np
import pandas as pd

from app import (
    IMPUTATION_METHODS,
    SMOOTHING_METHODS,
    impute_linear,
    apply_smoother,
)
from activity_filter import load_filtered_data
import pipeline_config as cfg

warnings.filterwarnings("ignore")

# settings -----------------------------------------------------------------
# re-validate on every index (incl. the SAR ones VH/RVI that were never tested
# before). runs on the activity-filtered data so it matches what we model on.
# warning: much slower than the old 5-feature run (expect several hours -
# whittaker + seasonal decomposition dominate). run it in the background.
FEATURES = cfg.ALL_INDICES

# same defaults the app uses for the sliders
SAVGOL_WIN = 11
MA_WIN = 7

# fraction of valid points to hide for the cross-validation
HIDE_FRAC = 0.2

# minimum number of valid points needed to bother evaluating
MIN_VALID = 10

# rng seed so results are reproducible
SEED = 42


def evaluate_one(dates_numeric, raw, valid_idx, rng):
    """run imputation CV + smoothing eval for one (plot, feature) series.
    returns two lists of dicts (imputation rows, smoothing rows)."""

    # pick which known points to hide
    n_hide = max(2, int(HIDE_FRAC * len(valid_idx)))
    hide_idx = rng.choice(valid_idx, size=n_hide, replace=False)
    true_vals = raw[hide_idx].copy()

    masked = raw.copy()
    masked[hide_idx] = np.nan

    imp_rows = []
    for name, func in IMPUTATION_METHODS.items():
        try:
            pred_full = func(dates_numeric, masked.copy())
            pred = pred_full[hide_idx]
            both = ~np.isnan(pred) & ~np.isnan(true_vals)
            if both.sum() < 2:
                continue
            p = pred[both]
            t = true_vals[both]

            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            mae = float(np.mean(np.abs(p - t)))
            ss_res = float(np.sum((p - t) ** 2))
            ss_tot = float(np.sum((t - np.mean(t)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            imp_rows.append({
                "method": name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            })
        except Exception:
            # if something blows up just skip it for this series
            continue

    # smoothing eval - mirror what app.py does: linear-impute first then smooth
    base = impute_linear(dates_numeric, raw.copy())
    if np.any(np.isnan(base)):
        return imp_rows, []

    valid_mask = ~np.isnan(raw)

    sm_rows = []
    for name, func in SMOOTHING_METHODS.items():
        try:
            sm = apply_smoother(name, func, base.copy(), SAVGOL_WIN, MA_WIN)
            sm = np.asarray(sm, dtype=float)

            # roughness = sqrt of mean squared 2nd differences
            if len(sm) > 2:
                d2 = np.diff(sm, n=2)
                roughness = float(np.sqrt(np.mean(d2 ** 2)))
            else:
                roughness = 0.0

            # fidelity = 1 - normalised RMSD from raw points
            kv = raw[valid_mask]
            sv = sm[valid_mask]
            ok = ~np.isnan(kv) & ~np.isnan(sv)
            if ok.sum() > 0:
                deviation = float(np.sqrt(np.mean((kv[ok] - sv[ok]) ** 2)))
                drange = float(np.ptp(kv[ok]))
                fidelity = 1.0 - (deviation / drange) if drange > 0 else 1.0
            else:
                deviation = 0.0
                fidelity = 0.0

            sm_rows.append({
                "method": name,
                "roughness": roughness,
                "rmsd_raw": deviation,
                "fidelity": fidelity,
            })
        except Exception:
            continue

    return imp_rows, sm_rows


def main():
    t_start = time.time()

    print("loading activity-filtered data...")
    df = load_filtered_data(write_report=False)
    print(f"  total rows: {len(df)}")

    # keep only the features that are actually in the file
    features = [f for f in FEATURES if f in df.columns]
    print(f"  features: {features}")

    plots = sorted(df["PMT_SITE"].unique())
    print(f"  plots: {len(plots)}")

    rng = np.random.default_rng(SEED)

    imp_records = []
    sm_records = []

    n_plots = len(plots)
    for i, site in enumerate(plots):
        if i % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  progress {i}/{n_plots} plots  ({elapsed:.0f}s elapsed)")

        sub = df[df["PMT_SITE"] == site].sort_values("date").reset_index(drop=True)
        if len(sub) < MIN_VALID:
            continue

        t0 = sub["date"].min()
        dates_numeric = (sub["date"] - t0).dt.total_seconds().values / 86400.0

        for feature in features:
            raw = sub[feature].values.astype(float)
            valid_idx = np.where(~np.isnan(raw))[0]
            if len(valid_idx) < MIN_VALID:
                continue

            imp_rows, sm_rows = evaluate_one(dates_numeric, raw, valid_idx, rng)

            # tag each row with the source so we can group later
            for r in imp_rows:
                r["site"] = site
                r["feature"] = feature
                imp_records.append(r)
            for r in sm_rows:
                r["site"] = site
                r["feature"] = feature
                sm_records.append(r)

    print(f"\ndone evaluating in {time.time()-t_start:.0f}s")
    print(f"  imputation rows: {len(imp_records)}")
    print(f"  smoothing rows: {len(sm_records)}")

    imp_df = pd.DataFrame(imp_records)
    sm_df = pd.DataFrame(sm_records)

    # save the raw per-series numbers in case we want to look later
    imp_df.to_csv("imputation_eval_raw.csv", index=False)
    sm_df.to_csv("smoothing_eval_raw.csv", index=False)

    # ----------------------------------------------------------------------
    # aggregate imputation - mean across all (plot,feature) tests
    # ----------------------------------------------------------------------
    if not imp_df.empty:
        imp_agg = imp_df.groupby("method").agg(
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            n=("rmse", "count"),
        ).sort_values("mean_rmse")

        # win count = how many times the method had the lowest RMSE
        # for a given (site, feature) pair
        wins = imp_df.loc[
            imp_df.groupby(["site", "feature"])["rmse"].idxmin(), "method"
        ].value_counts()
        imp_agg["wins"] = imp_agg.index.map(wins).fillna(0).astype(int)

        imp_agg.to_csv("imputation_ranking.csv")
        print("\n=== Imputation ranking (sorted by mean RMSE) ===")
        print(imp_agg.to_string())
        best_imp_rmse = imp_agg.index[0]
        best_imp_wins = imp_agg["wins"].idxmax()
    else:
        best_imp_rmse = best_imp_wins = None

    # ----------------------------------------------------------------------
    # aggregate smoothing - same combined score app.py uses on the rec
    # 0.5 * normalised_roughness + 0.5 * (1 - fidelity)
    # ----------------------------------------------------------------------
    if not sm_df.empty:
        sm_agg = sm_df.groupby("method").agg(
            mean_roughness=("roughness", "mean"),
            mean_fidelity=("fidelity", "mean"),
            mean_rmsd=("rmsd_raw", "mean"),
            n=("roughness", "count"),
        )

        r_min = sm_agg["mean_roughness"].min()
        r_max = sm_agg["mean_roughness"].max()
        r_range = (r_max - r_min) if r_max > r_min else 1.0
        sm_agg["score"] = (
            (sm_agg["mean_roughness"] - r_min) / r_range * 0.5
            + (1.0 - sm_agg["mean_fidelity"]) * 0.5
        )
        sm_agg = sm_agg.sort_values("score")

        sm_agg.to_csv("smoothing_ranking.csv")
        print("\n=== Smoothing ranking (sorted by combined score) ===")
        print(sm_agg.to_string())
        best_sm = sm_agg.index[0]
    else:
        best_sm = None

    # ----------------------------------------------------------------------
    # final picks
    # ----------------------------------------------------------------------
    print("\n=== Global best methods ===")
    if best_imp_rmse:
        print(f"  Imputation (lowest mean RMSE): {best_imp_rmse}")
    if best_imp_wins:
        print(f"  Imputation (most wins): {best_imp_wins}")
    if best_sm:
        print(f"  Smoothing (best combined score): {best_sm}")

    print(f"\ntotal runtime: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
