# descriptor_comparison.py - rank phenology descriptors by how well each one
# separates treated from untreated plot-windows. organic windows are excluded.

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

# metadata columns in phenology_descriptors.csv that are NOT descriptors
META_COLS = {"PMT_SITE", "window", "year", "treated",
             "treatment_type", "is_organic", "n_obs",
             "has_season", "window_known"}

MIN_PER_GROUP = 5


def cliffs_delta(a, b):
    """non-parametric effect size in [-1, 1]: mean sign of (a_i - b_j) over all
    pairs. +1 if every a > every b, -1 if reversed, 0 if interleaved."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.sign(np.subtract.outer(a, b)).mean())


def compare_descriptor(values, labels):
    """separation of one descriptor between treated (1) and untreated (0)."""
    values = np.asarray(values, float)
    labels = np.asarray(labels, int)
    keep = ~np.isnan(values)
    values, labels = values[keep], labels[keep]
    treated = values[labels == 1]
    untreated = values[labels == 0]
    res = dict(auc=np.nan, cliffs_delta=np.nan, mw_p=np.nan,
               n_treated=len(treated), n_untreated=len(untreated))
    if len(treated) < MIN_PER_GROUP or len(untreated) < MIN_PER_GROUP:
        return res
    auc_raw = roc_auc_score(labels, values)
    res["auc"] = max(auc_raw, 1 - auc_raw)  # symmetric: direction-agnostic
    res["cliffs_delta"] = cliffs_delta(treated, untreated)
    try:
        res["mw_p"] = float(mannwhitneyu(treated, untreated, alternative="two-sided")[1])
    except ValueError:
        res["mw_p"] = np.nan
    return res


# driver -----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DESC_FILE = os.path.join(HERE, "phenology_descriptors.csv")
OUT_FILE = os.path.join(HERE, "descriptor_comparison.csv")


def _as_bool(s):
    # accept "True"/"False" (csv round-trip of native bool) and "1"/"0"
    lo = s.astype(str).str.strip().str.lower()
    return lo.isin({"true", "1"})


def descriptor_columns(df):
    """all numeric descriptor columns (everything that is not metadata)."""
    return [c for c in df.columns if c not in META_COLS]


def main():
    df = pd.read_csv(DESC_FILE)
    n_all = len(df)
    organic = _as_bool(df["is_organic"])
    n_organic = int(organic.sum())
    main_df = df[~organic].copy()

    n_no_season = 0
    if "has_season" in main_df.columns:
        no_season = ~_as_bool(main_df["has_season"])
        n_no_season = int(no_season.sum())
        main_df = main_df[~no_season].copy()

    print(f"{n_all} windows total; {n_organic} organic excluded; "
          f"{n_no_season} no-season excluded; {len(main_df)} used for comparison")

    labels = main_df["treated"].astype(int).values
    rows = []
    for col in descriptor_columns(main_df):
        r = compare_descriptor(main_df[col].values, labels)
        r["descriptor"] = col
        rows.append(r)

    ranking = pd.DataFrame(rows)
    ranking = ranking[["descriptor", "auc", "cliffs_delta", "mw_p",
                       "n_treated", "n_untreated"]]
    ranking.sort_values("auc", ascending=False, inplace=True)
    ranking.reset_index(drop=True, inplace=True)
    ranking.to_csv(OUT_FILE, index=False, float_format="%.4f")

    print(f"\nranking written to {OUT_FILE}\n")
    print(ranking.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if n_organic > 0:
        print(f"\nnote: {n_organic} organic windows held out - inspect separately")
    if n_no_season > 0:
        print(f"note: {n_no_season} flat/no-season windows excluded (has_season=False)")


if __name__ == "__main__":
    main()
