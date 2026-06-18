# model_features.py - reads phenology_descriptors.csv, applies the non-organic
# + real-season filter, and prepares X / y / groups / meta ready for the ml model.

import os
import sys
import pandas as pd

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
