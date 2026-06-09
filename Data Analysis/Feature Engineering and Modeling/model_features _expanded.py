# model_features.py - reads phenology_descriptors.csv, applies the non-organic
# + real-season filter, and prepares X / y / groups / meta ready for the ML model.


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



# pipeline_config  


class pipeline_config:
    # indices whose phenology descriptors enter the feature matrix
    KEEP_INDICES = ["SAVI", "GNDVI", "RENDVI", "VH", "RVI"]

  
    SUFFIXES = [
        "pos_time", "pos_value", "bse_value", "aos_value",
        "sos_time", "sos_value", "eos_time", "eos_value",
        "los", "roi", "rod", "lios", "sios",
    ]


    TIMING_SUFFIXES = {"pos_time", "sos_time", "eos_time", "los"}



pipeline_config.FEATURE_COLS = [
    f"{idx}_{suf}"
    for idx in pipeline_config.KEEP_INDICES
    for suf in pipeline_config.SUFFIXES
]  

pipeline_config.TIMING_COLS = [
    f"{idx}_{suf}"
    for idx in pipeline_config.KEEP_INDICES
    for suf in pipeline_config.SUFFIXES
    if suf in pipeline_config.TIMING_SUFFIXES
]  


# constants

IN_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "phenology_descriptors.csv",
)

TEST_SIZE    = 0.2   # 80% train, 20% test -> n_splits=5
RANDOM_STATE = 42

# columns that are metadata, not features
META_COLS = {
    "PMT_SITE", "window", "year", "treated",
    "treatment_type", "is_organic", "n_obs",
    "has_season", "window_known",
}


META_KEEP = ["PMT_SITE", "window", "year", "treated", "treatment_type", "window_known"]



# helpers


def _as_bool(s: pd.Series) -> pd.Series:
    """Accept 'True'/'False' (CSV round-trip of native bool) and '1'/'0'."""
    lo = s.astype(str).str.strip().str.lower()
    return lo.isin({"true", "1"})



# load / filter  
def load_descriptors(path: str = IN_FILE, verbose: bool = True) -> pd.DataFrame:
    """Load phenology_descriptors.csv and return the raw DataFrame."""
    df = pd.read_csv(path)
    if verbose:
        print(f"Loaded {len(df)} windows from {os.path.basename(path)}")
    return df


def filter_modeling_rows(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop organic plots and windows with no detectable season.

    Mirrors the filter in descriptor_comparison.py so both scripts always
    operate on identical row sets.
    """
    is_organic = _as_bool(df["is_organic"])
    has_season = _as_bool(df["has_season"])
    keep       = ~is_organic & has_season

    if verbose:
        print(f"  Organic (dropped)  : {int(is_organic.sum())}")
        print(f"  No season (dropped): {int((~has_season).sum())}")
        print(f"  Total dropped      : {int((~keep).sum())}")
        print(f"  Kept for model     : {int(keep.sum())}")

    return df[keep].reset_index(drop=True).copy()



# core builder  

def build_xy(
    df: pd.DataFrame,
    drop_timing: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return (X, y, groups, meta) for the full filtered DataFrame.

    Parameters
    ----------
    df          : filtered DataFrame (output of filter_modeling_rows)
    drop_timing : if True, the 20 timing descriptors (pos/sos/eos_time + los
                  for each index) are excluded from X. Use this to verify the
                  model relies on treatment signal, not activity-window bounds.
    verbose     : print feature / label summary

    Returns
    -------
    X      : DataFrame, shape (n, 65) or (n, 45) when drop_timing=True
    y      : Series of int  (0 = untreated, 1 = treated)
    groups : Series of str  (PMT_SITE - pass to StratifiedGroupKFold to keep
             all windows of one plot on the same side of every fold)
    meta   : DataFrame with [PMT_SITE, window, year, treated,
             treatment_type, window_known] - align predictions to labels
             via meta.index or meta["PMT_SITE"]
    """
    cols = pipeline_config.FEATURE_COLS.copy()
    if drop_timing:
        cols = [c for c in cols if c not in pipeline_config.TIMING_COLS]

    
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Expected feature columns missing from DataFrame: {missing}")

    X      = df[cols].astype(float)
    y      = df["treated"].astype(int)
    groups = df["PMT_SITE"]
    meta   = df[META_KEEP].copy()

    if verbose:
        timing_note = f"  (timing dropped, {len(cols)} features)" if drop_timing else ""
        print(f"\nFeatures           : {X.shape[1]}{timing_note}")
        print(f"Samples            : {len(X)}")
        print(f"  treated          : {int(y.sum())}")
        print(f"  untreated        : {int((y == 0).sum())}")
        print(f"Unique plots       : {groups.nunique()}")

    return X, y, groups, meta


# ---------------------------------------------------------------------------
# imputation + scaling helper
# ---------------------------------------------------------------------------

def imputed_scaled_view(
    X: "pd.DataFrame | np.ndarray",
) -> tuple[np.ndarray, SimpleImputer, StandardScaler]:
    """Median-impute then z-score standardize X.

    Fits on whatever is passed in. To avoid leakage in a train/test workflow,
    call this on X_train only, then apply the returned imputer and scaler to
    X_test via .transform(). Pass the full matrix when fitting unsupervised
    methods that need the complete distribution.

    Returns
    -------
    X_clean  : ndarray, same shape as X, no NaNs, zero-mean unit-variance
    imputer  : fitted SimpleImputer  (call imputer.transform() on held-out data)
    scaler   : fitted StandardScaler (call scaler.transform() on held-out data)
    """
    arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_imputed = imputer.fit_transform(arr)
    X_scaled  = scaler.fit_transform(X_imputed)

    return X_scaled, imputer, scaler




def prepare(
    path: str = IN_FILE,
    drop_timing: bool = False,
    verbose: bool = True,
) -> tuple:
    """Load, filter, and return a group-aware stratified 80/20 train/test split.

    Uses StratifiedGroupKFold so that:
      - all windows of one plot stay on the same side (no plot leakage)
      - treated/untreated ratio is preserved in both splits

    Returns
    -------
    X_train, X_test, y_train, y_test, meta_train, meta_test
    """
    df_raw   = load_descriptors(path, verbose=verbose)
    df_clean = filter_modeling_rows(df_raw, verbose=verbose)
    X, y, groups, meta = build_xy(df_clean, drop_timing=drop_timing, verbose=verbose)

   
    n_splits = int(round(1 / TEST_SIZE))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    X_train,    X_test    = X.iloc[train_idx],    X.iloc[test_idx]
    y_train,    y_test    = y.iloc[train_idx],    y.iloc[test_idx]
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
    

    df_raw   = load_descriptors(verbose=True)
    df_clean = filter_modeling_rows(df_raw, verbose=True)

    # full filtered dataset
    X, y, groups, meta = build_xy(df_clean, drop_timing=False, verbose=True)
    full_out = pd.concat([meta, X], axis=1)
    full_out.to_csv(os.path.join(HERE, "model_expanded_input.csv"), index=False, float_format="%.6f")

    # train / test split
    X_train, X_test, y_train, y_test, meta_train, meta_test = prepare(verbose=False)

    train_out = pd.concat([meta_train, X_train], axis=1)
    train_out.to_csv(os.path.join(HERE, "model_expanded_input_train.csv"), index=False, float_format="%.6f")

    test_out = pd.concat([meta_test, X_test], axis=1)
    test_out.to_csv(os.path.join(HERE, "model_expanded_input_test.csv"), index=False, float_format="%.6f")

    print("\nCSVs saved:")
    print(f"  model_expanded_input.csv        — full filtered dataset ({len(full_out)} rows)")
    print(f"  model_expanded_input_train.csv  — train split ({len(train_out)} rows)")
    print(f"  model_expanded_input_test.csv   — test split  ({len(test_out)} rows)")
    print("\nReady. Plug in your model:")
    print("  model.fit(X_train, y_train)")
    print("  model.predict(X_test)")
