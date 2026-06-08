# process_final.py - impute and smooth all plots, output one clean dataset.
#
# winning methods from evaluate_all.py:
#   imputation : linear interpolation  (mean rmse 0.0490, 537 wins out of 2565)
#   smoothing  : whittaker-eilers      (combined score 0.0339)
#
# usage: python process_final.py

import os
import warnings
import numpy as np
import pandas as pd

from app import impute_linear, smooth_whittaker
from activity_filter import load_filtered_data
import pipeline_config as cfg

warnings.filterwarnings("ignore")

# only the 5 kept indices get imputed + smoothed and written out.
FEATURE_COLS = cfg.KEEP_INDICES

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indices_final.csv")


def process_site(sub, dates_numeric, features):
    """impute then smooth every feature column for one plot's time series."""
    for feat in features:
        raw = sub[feat].values.astype(float)
        imputed = impute_linear(dates_numeric, raw)
        # whittaker requires a complete series; fall back to imputed only if nans remain
        if not np.any(np.isnan(imputed)):
            sub[feat] = smooth_whittaker(imputed)
        else:
            sub[feat] = imputed
    return sub


def main():
    print("loading activity-filtered data...")
    df = load_filtered_data()

    features = [c for c in FEATURE_COLS if c in df.columns]
    plots = sorted(df["PMT_SITE"].unique())

    print(f"  rows     : {len(df):,}")
    print(f"  plots    : {len(plots)}")
    print(f"  features : {len(features)}  ({', '.join(features)})")
    print(f"  imputation : linear interpolation")
    print(f"  smoothing  : whittaker-eilers (lambda=1e4, d=2)")
    print()

    frames = []
    n = len(plots)
    for i, site in enumerate(plots):
        if i % 50 == 0:
            print(f"  processing {i}/{n} plots...")

        sub = (
            df[df["PMT_SITE"] == site]
            .sort_values("date")
            .reset_index(drop=True)
            .copy()
        )
        t0 = sub["date"].min()
        dates_numeric = (sub["date"] - t0).dt.total_seconds().values / 86400.0

        frames.append(process_site(sub, dates_numeric, features))

    print(f"  processing {n}/{n} plots... done")
    print()
    print("combining and saving...")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["PMT_SITE", "date"]).reset_index(drop=True)
    keep = [c for c in cfg.METADATA_COLS if c in out.columns] + cfg.KEEP_INDICES
    out = out[keep]
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"saved {len(out):,} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
