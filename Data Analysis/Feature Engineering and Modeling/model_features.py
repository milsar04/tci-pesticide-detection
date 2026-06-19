# model_features.py - reads phenology_descriptors.csv, applies the non-organic
# + real-season filter, and prepares X / y / groups / meta ready for the ml model.

import os
import sys
import numpy as np
import pandas as pd
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

# the 13 timesat-style descriptor suffixes phenolopy emits per index
# (single source of truth: pipeline_config).
SUFFIXES = cfg.DESCRIPTOR_SUFFIXES

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

# metadata columns carried alongside X so predictions re-align to labels.
META_KEEP = ["PMT_SITE", "window", "year", "treated", "treatment_type", "window_known"]


# load / filter ----

def load_descriptors(path=IN_FILE, verbose=False):
    """load phenology_descriptors.csv and return the raw dataframe (all 821 windows)."""
    df = pd.read_csv(path)
    if verbose:
        print(f"Loaded {len(df)} windows from {os.path.basename(path)}")
    return df


def filter_modeling_rows(df, verbose=False):
    """keep non-organic, real-season windows only (405). mirrors descriptor_comparison.py."""
    is_organic = cfg._as_bool(df["is_organic"])
    has_season = cfg._as_bool(df["has_season"])
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


# treated-share label cleanup ----
# Treated share lives in indices_final.csv (0-1 fraction, present only on treated
# rows). joined per (PMT_SITE, year) - share is a season-level attribute.

def attach_treated_share(df):
    """add a treated_share column (max recorded share per PMT_SITE+year from
    indices_final.csv). untreated / unrecorded -> 0.0. the production label
    (treated) is unchanged; this only feeds the cleanup sensitivity."""
    fin = pd.read_csv(os.path.join(_IS_DIR, "indices_final.csv"),
                      usecols=["PMT_SITE", "date", "Treated share"])
    fin["year"] = pd.to_datetime(fin["date"], errors="coerce").dt.year
    fin["share"] = pd.to_numeric(fin["Treated share"], errors="coerce")
    per = (fin.dropna(subset=["share"]).groupby(["PMT_SITE", "year"])["share"]
           .max().reset_index())
    out = df.merge(per, on=["PMT_SITE", "year"], how="left")
    out["treated_share"] = out["share"].fillna(0.0)
    return out.drop(columns=["share"])


def filter_clean_label(df, min_share=0.5):
    """drop ambiguous partial-treatment windows: keep confident treated
    (treated==1 and treated_share>=min_share) + all confident untreated
    (treated==0). windows with 0 < share < min_share are dropped as label noise.
    needs a treated_share column (see attach_treated_share)."""
    share = pd.to_numeric(df.get("treated_share", 0.0), errors="coerce").fillna(0.0)
    treated = df["treated"].astype(int) == 1
    ambiguous = treated & (share > 0) & (share < min_share)
    return df[~ambiguous].reset_index(drop=True).copy()


def imputed_scaled_view(X):
    """median-impute NaNs then z-score; returns (X_scaled ndarray, fitted imputer, fitted scaler).
    transformers are returned so a train/test workflow can .transform() the test split
    with train-fit objects (no leakage). pass only training data when doing CV."""
    imp = SimpleImputer(strategy="median").fit(X)
    scl = StandardScaler().fit(imp.transform(X))
    X_scaled = scl.transform(imp.transform(X))
    return X_scaled, imp, scl
