# timing_confound_check.py - one-off diagnostic: is the treated-vs-untreated
# separation in the phenology TIMING descriptors (sos/pos/eos time) a genuine
# signal or an artifact of the activity-date windowing?
#
# idea: window_known=True plots are row-masked to a TCI activity window, so their
# detected season timing can be shaped by the window bounds. window_known=False
# plots are kept UNMASKED (full record), so their timing reflects detected
# phenology only. if the timing separation is a windowing artifact it should
# weaken on the unmasked subset; if it persists it is not purely windowing.
# amplitude descriptors are carried alongside as a control (vigour should
# separate regardless of windowing). reuses descriptor_comparison's exact AUC.

import os
import pandas as pd
from descriptor_comparison import compare_descriptor, _as_bool

HERE = os.path.dirname(os.path.abspath(__file__))
DESC_FILE = os.path.join(HERE, "phenology_descriptors.csv")

INDICES = ["SAVI", "GNDVI", "RENDVI", "VH", "RVI"]
TIMING = [f"{i}_{k}" for i in INDICES for k in ("sos_time", "pos_time", "eos_time")]
AMPL = [f"{i}_{k}" for i in INDICES for k in ("aos_value", "pos_value", "lios")]


def subset_table(df, cols):
    labels = df["treated"].astype(int).values
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        r = compare_descriptor(df[c].values, labels)
        r["descriptor"] = c
        rows.append(r)
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def report(label, sub):
    n_t = int((sub["treated"].astype(int) == 1).sum())
    n_u = int((sub["treated"].astype(int) == 0).sum())
    print(f"\n=== {label}: {len(sub)} windows ({n_t} treated, {n_u} untreated) ===")
    t = subset_table(sub, TIMING)
    a = subset_table(sub, AMPL)
    print("timing descriptors:")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("amplitude descriptors:")
    print(a.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"mean timing AUC   : {t['auc'].mean():.3f}")
    print(f"mean amplitude AUC: {a['auc'].mean():.3f}")


def main():
    df = pd.read_csv(DESC_FILE)
    df = df[~_as_bool(df["is_organic"]) & _as_bool(df["has_season"])].copy()
    wk = _as_bool(df["window_known"])
    report("ALL compared windows", df)
    report("window_known=True (masked to activity window)", df[wk])
    report("window_known=False (unmasked, full record)", df[~wk])


if __name__ == "__main__":
    main()
