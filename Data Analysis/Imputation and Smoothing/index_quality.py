# index_quality.py
# quantitative assessment of which vegetation indices are most useful for
# modeling treated vs untreated potato plots. ranks indices on three axes:
#   1. signal quality   - coverage, smoothness, snr, dynamic range
#   2. discriminability - how well the index separates treated from untreated
#   3. redundancy       - pairwise correlation between indices
# usage: python index_quality.py

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# settings --------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# indices we score. raw bands are excluded; add them here if you want them
# in the same ranking.
INDEX_COLS = [
    "NDVI", "GNDVI", "EVI", "SAVI", "CI", "CI_GREEN",
    "RENDVI", "NDREI", "NDRE_(B6)", "MSAVI", "OSAVI",
    "VH", "VV", "RVI", "VH/VV",
]

# peak growing season for potato in northern hemisphere. used to compute the
# per-plot summary value that feeds the discriminability metrics.
PEAK_MONTHS = (6, 7, 8)

# treatment classes scored individually against pure-untreated plots. "No"
# is the negative class and is excluded from this list.
TREATMENT_CLASSES = ["Herbicide", "Fungicide", "Insecticide", "PGR", "Mixed", "Other"]

# minimum positive plots needed to compute a per-class AUC; otherwise NaN
MIN_POSITIVES_PER_CLASS = 5

# rolling window (in valid observations, not days) for snr decomposition
SMOOTH_WIN = 15

# minimum valid observations per plot to score that plot
MIN_VALID = 20

SEED = 42

# data loading ----------------------------------------------------------------

def load_data():
    """load and merge the 2020 and 2021 CSV files."""
    files = [
        os.path.join(DATA_DIR, "indices_2020.csv"),
        os.path.join(DATA_DIR, "indices_2021.csv"),
    ]
    frames = []
    for fp in files:
        if os.path.exists(fp):
            frames.append(pd.read_csv(fp, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"no csv found in {DATA_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values(["PMT_SITE", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# signal quality metrics ------------------------------------------------------

def _per_plot_metrics(values):
    """compute the four signal quality metrics for one plot's series.
    values is a 1d numpy array possibly containing NaNs."""
    valid = values[~np.isnan(values)]
    n_valid = len(valid)
    n_total = len(values)
    coverage = n_valid / n_total if n_total else np.nan

    if n_valid < MIN_VALID:
        return dict(coverage=coverage, autocorr=np.nan, snr_db=np.nan, dyn_range=np.nan)

    # lag-1 autocorrelation of consecutive valid observations. higher = the
    # signal moves smoothly from one observation to the next. caveat: SAR has
    # more frequent valid observations than optical, so a higher autocorr for
    # SAR partly reflects shorter time gaps, not true smoothness.
    autocorr = float(np.corrcoef(valid[:-1], valid[1:])[0, 1]) if n_valid >= 3 else np.nan

    # signal / noise decomposition via rolling mean. signal = smoothed series,
    # noise = residual. snr_db is the usual 10 * log10 of variance ratio.
    s = pd.Series(valid)
    sig = s.rolling(SMOOTH_WIN, center=True, min_periods=SMOOTH_WIN // 2).mean()
    noise = s - sig
    sig_var = float(np.nanvar(sig))
    noise_var = float(np.nanvar(noise))
    snr_db = 10 * np.log10(sig_var / noise_var) if noise_var > 0 and sig_var > 0 else np.nan

    # robust spread, p95 - p5. comparable across indices because we standardize
    # later when forming the composite score.
    dyn_range = float(np.percentile(valid, 95) - np.percentile(valid, 5))

    return dict(coverage=coverage, autocorr=autocorr, snr_db=snr_db, dyn_range=dyn_range)


def signal_quality(df, index_col):
    """mean signal quality across all plots for one index."""
    rows = []
    for _, g in df.groupby("PMT_SITE", sort=False):
        rows.append(_per_plot_metrics(g[index_col].values))
    m = pd.DataFrame(rows)
    return m.mean(numeric_only=True)

# class labels ----------------------------------------------------------------

def plot_labels(df):
    """binary label per plot: 1 if any observation has a non-No treatment."""
    treated_obs = (df["Treatment status"].fillna("No").str.lower() != "no").astype(int)
    return df.assign(_t=treated_obs).groupby("PMT_SITE")["_t"].max()


def plot_class_labels(df):
    """one boolean column per treatment class. a plot is positive for class C
    if any observation has Treatment status == C. all other plots become the
    negative class only when paired with that specific class (i.e. pure-
    untreated plots), the rest are dropped from that class's eval."""
    status = df["Treatment status"].fillna("No")
    out = {}
    for cls in TREATMENT_CLASSES:
        has_cls = (status == cls).astype(int)
        out[cls] = df.assign(_t=has_cls).groupby("PMT_SITE")["_t"].max().astype(bool)
    return pd.DataFrame(out)

# discriminability -----------------------------------------------------------

def peak_summary(df, index_col, agg="median"):
    """one value per plot: aggregate of the index during peak growing months."""
    mask = df["date"].dt.month.isin(PEAK_MONTHS) & df[index_col].notna()
    sub = df[mask]
    g = sub.groupby("PMT_SITE")[index_col]
    return g.median() if agg == "median" else g.mean()


def _fisher(x, y):
    """ratio of between-class to within-class variance."""
    classes = np.unique(y)
    if len(classes) < 2:
        return np.nan
    overall = x.mean()
    num = den = 0.0
    for c in classes:
        m = y == c
        n = m.sum()
        if n < 2:
            continue
        num += n * (x[m].mean() - overall) ** 2
        den += n * x[m].var()
    return float(num / den) if den > 0 else np.nan


def discriminability(df, index_col, labels):
    """auc, fisher, mutual info of peak summary vs treated/untreated label."""
    peak = peak_summary(df, index_col)
    common = peak.index.intersection(labels.index)
    x = peak.loc[common].dropna()
    y = labels.loc[x.index]
    if len(x) < 20 or y.nunique() < 2:
        return dict(auc=np.nan, fisher=np.nan, mi=np.nan)
    auc_raw = roc_auc_score(y, x)
    # symmetric auc: an index that's anticorrelated with treatment is just as
    # useful as one that's positively correlated. flip if needed.
    auc = max(auc_raw, 1 - auc_raw)
    fisher = _fisher(x.values, y.values)
    mi = float(mutual_info_classif(x.values.reshape(-1, 1), y.values, random_state=SEED)[0])
    return dict(auc=auc, fisher=fisher, mi=mi)


def per_class_auc(df, index_col, class_labels, any_treated):
    """one symmetric AUC per treatment class. positives are plots with that
    class, negatives are pure-untreated plots. classes with fewer than
    MIN_POSITIVES_PER_CLASS positive plots return NaN."""
    peak = peak_summary(df, index_col).dropna()
    untreated = any_treated[~any_treated.astype(bool)].index

    out = {}
    for cls in TREATMENT_CLASSES:
        positives = class_labels[cls]
        positives = positives[positives].index
        pos_idx = peak.index.intersection(positives)
        neg_idx = peak.index.intersection(untreated)
        if len(pos_idx) < MIN_POSITIVES_PER_CLASS or len(neg_idx) < MIN_POSITIVES_PER_CLASS:
            out[cls] = np.nan
            continue
        x = pd.concat([peak.loc[pos_idx], peak.loc[neg_idx]])
        y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
        auc_raw = roc_auc_score(y, x.values)
        out[cls] = max(auc_raw, 1 - auc_raw)
    return out

# redundancy ------------------------------------------------------------------

def redundancy_matrix(df, index_cols):
    """pearson correlation of per-plot peak medians, one row/col per index."""
    peaks = pd.DataFrame({ic: peak_summary(df, ic) for ic in index_cols})
    return peaks.corr(method="pearson")

# main ------------------------------------------------------------------------

def main():
    print("loading data ...")
    df = load_data()
    available = [c for c in INDEX_COLS if c in df.columns]
    n_plots = df["PMT_SITE"].nunique()
    print(f"  {len(df):,} rows, {n_plots} plots, {len(available)} indices")

    labels = plot_labels(df)
    print(f"  plots labelled treated: {int(labels.sum())} / {len(labels)}")

    class_labels = plot_class_labels(df)
    n_per_class = class_labels.sum().to_dict()
    print(f"  plots per class: {n_per_class}")

    rows = []
    for ic in available:
        print(f"  scoring {ic} ...", flush=True)
        sq = signal_quality(df, ic)
        disc = discriminability(df, ic, labels)
        per_class = per_class_auc(df, ic, class_labels, labels)
        row = {
            "index": ic,
            "coverage": sq["coverage"],
            "autocorr_lag1": sq["autocorr"],
            "snr_db": sq["snr_db"],
            "dyn_range": sq["dyn_range"],
            "auc_treated": disc["auc"],
            "fisher": disc["fisher"],
            "mutual_info": disc["mi"],
        }
        for cls in TREATMENT_CLASSES:
            row[f"auc_{cls.lower()}"] = per_class[cls]
        rows.append(row)

    ranking = pd.DataFrame(rows)

    # composite score. z-score each ingredient so different scales don't
    # dominate, then sum. quality matters (coverage, autocorr, snr) and so
    # does the actual label signal (auc, mi). dyn_range and fisher are
    # reported but not in the composite (dyn_range is scale-dependent, fisher
    # is essentially redundant with auc here).
    score_cols = ["coverage", "autocorr_lag1", "snr_db", "auc_treated", "mutual_info"]
    z = ranking[score_cols].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    ranking["composite_z"] = z.sum(axis=1)
    ranking.sort_values("composite_z", ascending=False, inplace=True)
    ranking.reset_index(drop=True, inplace=True)

    rank_path = os.path.join(OUT_DIR, "index_quality_ranking.csv")
    ranking.to_csv(rank_path, index=False, float_format="%.4f")
    print(f"\nranking written to {rank_path}\n")
    print(ranking.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    red = redundancy_matrix(df, available)
    red_path = os.path.join(OUT_DIR, "index_redundancy.csv")
    red.to_csv(red_path, float_format="%.3f")
    print(f"\nredundancy matrix written to {red_path}")
    print("\nhighly correlated index pairs (|r| > 0.9):")
    r = red.where(np.triu(np.ones(red.shape), k=1).astype(bool)).stack()
    high = r[r.abs() > 0.9].sort_values(key=lambda s: s.abs(), ascending=False)
    if len(high):
        for (a, b), v in high.items():
            print(f"  {a:12s}  {b:12s}  r = {v:+.3f}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
