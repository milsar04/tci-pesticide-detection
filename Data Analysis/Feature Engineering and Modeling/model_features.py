# model_features.py - reads phenology_descriptors.csv, applies the non-organic
# + real-season filter, and prepares X / y / groups / meta ready for the ml model.

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# shared index list from the pipeline config (single source of truth for the 5 indices).
# cross-folder import via the sibling-dir sys.path pattern (see phenology.py).
_IS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Imputation and Smoothing",
)
if _IS_DIR not in sys.path:
    sys.path.insert(0, _IS_DIR)
import pipeline_config as cfg


# feature definitions ----

# the 13 timesat-style descriptor suffixes phenolopy_adapter emits per index.
SUFFIXES = [
    "pos_time", "pos_value", "bse_value", "aos_value",
    "sos_time", "sos_value", "eos_time", "eos_value",
    "los", "roi", "rod", "lios", "sios",
]

# pure timing descriptors (when-it-happened, not how-much); drop these to check the
# model leans on treatment signal, not the activity-window bounds.
TIMING_SUFFIXES = {"pos_time", "sos_time", "eos_time", "los"}

# explicit 65-col feature whitelist: 5 kept indices x 13 suffixes.
FEATURE_COLS = [f"{idx}_{suf}" for idx in cfg.KEEP_INDICES for suf in SUFFIXES]
TIMING_COLS = [f"{idx}_{suf}" for idx in cfg.KEEP_INDICES for suf in SUFFIXES
               if suf in TIMING_SUFFIXES]


# constants ----

IN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "phenology_descriptors.csv")
TEST_SIZE    = 0.2   # 80% train, 20% test -> n_splits=5
RANDOM_STATE = 42

# columns that are metadata, not features
META_COLS = {"PMT_SITE", "window", "year", "treated",
             "treatment_type", "is_organic", "n_obs",
             "has_season", "window_known"}
META_KEEP = ["PMT_SITE", "window", "year", "treated", "treatment_type", "window_known"]


# helpers ----

def _as_bool(s):
    """accept 'True'/'False' (csv round-trip of native bool) and '1'/'0'."""
    lo = s.astype(str).str.strip().str.lower()
    return lo.isin({"true", "1"})


# load / filter ----

def load_descriptors(path=IN_FILE, verbose=False):
    """load phenology_descriptors.csv and return the raw dataframe (all 821 windows)."""
    df = pd.read_csv(path)
    if verbose:
        print(f"Loaded {len(df)} windows from {os.path.basename(path)}")
    return df


def filter_modeling_rows(df, verbose=False):
    """keep non-organic, real-season windows only (405). mirrors descriptor_comparison.py."""
    is_organic = _as_bool(df["is_organic"])
    has_season = _as_bool(df["has_season"])
    keep = ~is_organic & has_season
    if verbose:
        print(f"  organic (dropped)  : {int(is_organic.sum())}")
        print(f"  no season (dropped): {int((~has_season).sum())}")
        print(f"  kept for model     : {int(keep.sum())}")
    return df[keep].reset_index(drop=True).copy()


# core builder ----

def build_xy(df, drop_timing=False, verbose=False):
    """return (X, y, groups, meta) for the filtered dataframe (no split).

    drop_timing excludes the 20 timing descriptors (pos/sos/eos_time + los per index).
    groups = PMT_SITE - pass to StratifiedGroupKFold so a plot's windows never straddle
    train and test. meta carries treated so predictions can be re-aligned to labels.
    """
    cols = FEATURE_COLS.copy()
    if drop_timing:
        cols = [c for c in cols if c not in TIMING_COLS]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"expected feature columns missing from dataframe: {missing}")

    X = df[cols].astype(float)
    y = df["treated"].astype(int)
    groups = df["PMT_SITE"]
    meta = df[META_KEEP].copy()

    if verbose:
        note = f" (timing dropped, {len(cols)} features)" if drop_timing else ""
        print(f"\nFeatures           : {X.shape[1]}{note}")
        print(f"Samples            : {len(X)} "
              f"({int(y.sum())} treated / {int((y == 0).sum())} untreated)")
        print(f"Unique plots       : {groups.nunique()}")

    return X, y, groups, meta


# imputation + scaling ----

def imputed_scaled_view(X):
    """median-impute then z-score standardize X.

    fits on whatever is passed in. to avoid leakage in a train/test workflow, call on
    X_train only then apply the returned imputer/scaler to X_test via .transform().
    returns (X_scaled ndarray, fitted imputer, fitted scaler).
    """
    arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(arr))
    return X_scaled, imputer, scaler


# split ----

def prepare(path=IN_FILE, drop_timing=False, verbose=False):
    """load, filter, and return a group-aware stratified 80/20 train/test split.

    StratifiedGroupKFold keeps all windows of one plot on the same side (no leakage)
    and preserves the treated/untreated ratio. returns
    X_train, X_test, y_train, y_test, meta_train, meta_test.
    """
    df = filter_modeling_rows(load_descriptors(path, verbose=verbose), verbose=verbose)
    X, y, groups, meta = build_xy(df, drop_timing=drop_timing, verbose=verbose)

    n_splits = int(round(1 / TEST_SIZE))   # 0.2 -> 5 folds -> ~20% test
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    X_train, X_test       = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test       = y.iloc[train_idx], y.iloc[test_idx]
    meta_train, meta_test = meta.iloc[train_idx], meta.iloc[test_idx]

    # no plot may appear in both splits
    overlap = set(meta_train["PMT_SITE"]) & set(meta_test["PMT_SITE"])
    assert not overlap, f"plot leakage in split: {sorted(overlap)}"

    if verbose:
        print(f"\nTrain set          : {len(X_train)} rows  "
              f"({y_train.sum()} treated / {(y_train == 0).sum()} untreated)")
        print(f"Test set           : {len(X_test)} rows  "
              f"({y_test.sum()} treated / {(y_test == 0).sum()} untreated)")
        print(f"Plots in both splits: {len(overlap)}")

    return X_train, X_test, y_train, y_test, meta_train, meta_test


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))

    df = filter_modeling_rows(load_descriptors(verbose=True), verbose=True)

    # full filtered dataset
    X, y, groups, meta = build_xy(df, verbose=True)
    pd.concat([meta, X], axis=1).to_csv(
        os.path.join(HERE, "model_input.csv"), index=False, float_format="%.6f")

    # train / test split
    X_train, X_test, y_train, y_test, meta_train, meta_test = prepare(verbose=False)
    train_out = pd.concat([meta_train, X_train], axis=1)
    test_out  = pd.concat([meta_test, X_test], axis=1)
    train_out.to_csv(os.path.join(HERE, "model_input_train.csv"), index=False, float_format="%.6f")
    test_out.to_csv(os.path.join(HERE, "model_input_test.csv"), index=False, float_format="%.6f")

    print("\nCSVs saved:")
    print(f"  model_input.csv        - full filtered dataset ({len(meta)} rows)")
    print(f"  model_input_train.csv  - train split ({len(train_out)} rows)")
    print(f"  model_input_test.csv   - test split  ({len(test_out)} rows)")
    print("\nReady. Plug in your model:")
    print("  model.fit(X_train, y_train)")
    print("  model.predict(X_test)")
