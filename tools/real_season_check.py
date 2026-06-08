"""sensitivity check: does descriptor AUC survive a real-season filter?
filter criterion: SAVI_aos_value >= threshold (a genuine vegetation season)."""

import sys
import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in tools/)
sys.path.insert(0, os.path.join(ROOT, "Data Analysis", "Feature Engineering and Modeling"))
import descriptor_comparison as dc

DESC_FILE = os.path.join(ROOT, "Data Analysis", "Feature Engineering and Modeling",
                         "phenology_descriptors.csv")

# headline descriptors to watch: amplitude/integral, rates, timing
WATCH = [
    "VH_aos_value", "SAVI_aos_value", "RENDVI_sios", "SAVI_sios", "SAVI_los",
    "RVI_aos_value", "VH_rod", "SAVI_roi", "SAVI_sos_time", "VH_sos_time",
]

THRESHOLDS = [0.0, 0.10, 0.20, 0.30]


def run():
    d = pd.read_csv(DESC_FILE)
    organic = d["is_organic"].astype(str).str.lower().isin(["true", "1", "yes"])
    d = d[~organic].copy()

    results = {}
    for thr in THRESHOLDS:
        mask = d["SAVI_aos_value"].fillna(0) >= thr
        sub = d[mask]
        n_t = int((sub["treated"].astype(int) == 1).sum())
        n_u = int((sub["treated"].astype(int) == 0).sum())
        label = f"SAVI_aos >= {thr}"
        print(f"\n--- {label}: {len(sub)} windows ({n_t} treated, {n_u} untreated) ---")
        if n_t < 10 or n_u < 10:
            print("  too few to compare")
            continue
        labels = sub["treated"].astype(int).values
        row = {"label": label, "n_windows": len(sub), "n_treated": n_t, "n_untreated": n_u}
        for col in WATCH:
            if col not in sub.columns:
                continue
            r = dc.compare_descriptor(sub[col].values, labels)
            auc = round(r["auc"], 3) if not np.isnan(r["auc"]) else float("nan")
            dlt = round(r["cliffs_delta"], 3) if not np.isnan(r["cliffs_delta"]) else float("nan")
            nt = r["n_treated"]
            nu = r["n_untreated"]
            auc_s = f"{auc:.3f}" if not np.isnan(auc) else "  nan"
            dlt_s = f"{dlt:+.3f}" if not np.isnan(dlt) else "  nan"
            print(f"  {col:26s}  AUC={auc_s}  delta={dlt_s}  n_t={nt} n_u={nu}")
            row[col + "_auc"] = auc
        results[thr] = row

    # summary: how much did the top AUC move?
    print("\n=== SUMMARY: best AUC at each threshold ===")
    for thr in THRESHOLDS:
        if thr not in results:
            continue
        r = results[thr]
        desc_aucs = {k.replace("_auc", ""): v for k, v in r.items()
                     if k.endswith("_auc") and not np.isnan(v)}
        if not desc_aucs:
            continue
        best = max(desc_aucs, key=desc_aucs.get)
        print(f"  {r['label']:25s}  best={best} AUC={desc_aucs[best]:.3f}"
              f"  n=({r['n_treated']}T + {r['n_untreated']}U)")


if __name__ == "__main__":
    run()
