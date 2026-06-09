# model_features.py - reads phenology_descriptors.csv, applies the non-organic
# + real-season filter, and prepares X / y + train/test split ready for the ML model

import os
import pandas as pd
from sklearn.model_selection import train_test_split


# config


IN_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "phenology_descriptors.csv")
TEST_SIZE    = 0.2   # 80% train, 20% test
RANDOM_STATE = 42

# metadata columns that are NOT features
META_COLS = {"PMT_SITE", "window", "year", "treated",
             "treatment_type", "is_organic", "n_obs",
             "has_season", "window_known"}



# helpers

def _as_bool(s):
    """accept 'True'/'False' (csv round-trip of native bool) and '1'/'0'."""
    lo = s.astype(str).str.strip().str.lower()
    return lo.isin({"true", "1"})


def descriptor_columns(df):
    """all numeric descriptor columns (everything that is not metadata)."""
    return [c for c in df.columns if c not in META_COLS]



# main


def load_and_filter(path=IN_FILE):
    df = pd.read_csv(path)
    n_all = len(df)

    is_organic = _as_bool(df["is_organic"])
    has_season = _as_bool(df["has_season"])
    keep       = ~is_organic & has_season

    print(f"Total windows      : {n_all}")
    print(f"Organic (dropped)  : {int(is_organic.sum())}")
    print(f"No season (dropped): {int((~has_season).sum())}")
    print(f"Total dropped      : {int((~keep).sum())}")
    print(f"Kept for model     : {int(keep.sum())}")

    return df[keep].copy()


def prepare(path=IN_FILE):
    clean = load_and_filter(path)

    desc_cols = descriptor_columns(clean)

    X = clean[desc_cols].astype(float)
    y = clean["treated"].astype(int)

    meta = clean[["PMT_SITE", "window", "year", "treatment_type", "window_known"]]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # keep treated/untreated ratio the same in both splits
    )

    print(f"\nFeatures           : {X.shape[1]}")
    print(f"Train set          : {len(X_train)} rows  "
          f"({y_train.sum()} treated / {(y_train == 0).sum()} untreated)")
    print(f"Test set           : {len(X_test)} rows  "
          f"({y_test.sum()} treated / {(y_test == 0).sum()} untreated)")

    return X_train, X_test, y_train, y_test, meta_train, meta_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, meta_train, meta_test = prepare()

    #save CSVs
    HERE = os.path.dirname(os.path.abspath(__file__))

    clean = load_and_filter()
    desc_cols = descriptor_columns(clean)
    meta_cols = ["PMT_SITE", "window", "year", "treated", "treatment_type", "window_known"]

    # full filtered dataset
    clean[meta_cols + desc_cols].to_csv(
        os.path.join(HERE, "model_input.csv"), index=False, float_format="%.6f"
    )

    # train split
    train_out = meta_train.copy()
    train_out["treated"] = y_train
    train_out = pd.concat([train_out, X_train], axis=1)
    train_out.to_csv(os.path.join(HERE, "model_input_train.csv"), index=False, float_format="%.6f")

    # test split
    test_out = meta_test.copy()
    test_out["treated"] = y_test
    test_out = pd.concat([test_out, X_test], axis=1)
    test_out.to_csv(os.path.join(HERE, "model_input_test.csv"), index=False, float_format="%.6f")

    print("\nCSVs saved:")
    print("  model_input.csv        — full filtered dataset (405 rows)")
    print("  model_input_train.csv  — train split (324 rows)")
    print("  model_input_test.csv   — test split  (81 rows)")
    print("\nReady. Plug in your model:")
    print("  model.fit(X_train, y_train)")
    print("  model.predict(X_test)")