# align_check.py - diagnostic: is day 0 really the half-amplitude SOS?
# evidence-gathering for the "are seasons aligned correctly?" question.
import numpy as np
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in tools/)
HERE = os.path.join(ROOT, "Data Analysis", "Feature Engineering and Modeling")
al = pd.read_csv(os.path.join(HERE, "aligned_series.csv"))
desc = pd.read_csv(os.path.join(HERE, "phenology_descriptors.csv"))
hs = desc["has_season"].astype(str).str.lower().isin(["true", "1"])
desc = desc[hs].copy()
KEY = ["PMT_SITE", "window", "year"]

print(f"has_season windows: {len(desc)} | aligned rows: {len(al):,}")
print("=" * 78)

for X in ["SAVI", "VH", "GNDVI", "RENDVI", "RVI"]:
    sub = al[al["index"] == X]
    dd = desc[KEY + [f"{X}_bse_value", f"{X}_aos_value", f"{X}_sos_time",
                     f"{X}_pos_time", f"{X}_los"]].copy()
    rows = []
    for k, grp in sub.groupby(KEY):
        grp = grp.sort_values("days_since_sos")
        x = grp["days_since_sos"].values
        y = grp["value"].values
        v0 = np.interp(0.0, x, y) if (x.min() <= 0 <= x.max()) else np.nan
        rows.append((*k, v0, float(x.min()), float(x.max()), len(x)))
    res = pd.DataFrame(rows, columns=KEY + ["v0", "dmin", "dmax", "npts"])
    res = res.merge(dd, on=KEY, how="left")
    aos = res[f"{X}_aos_value"]
    res["frac0"] = (res["v0"] - res[f"{X}_bse_value"]) / aos
    # fallback: SOS pinned at series start -> almost no pre-season (negative) data
    fallback = (res["dmin"] > -1.0).mean() * 100
    f = res["frac0"].dropna()
    print(f"\n[{X}]  windows={len(res)}")
    print(f"  frac0 = (value@day0 - base)/amplitude   (correct alignment -> ~0.50)")
    print(f"    median={f.median():.3f}  mean={f.mean():.3f}  "
          f"q25={f.quantile(.25):.3f}  q75={f.quantile(.75):.3f}")
    print(f"  fallback SOS (no pre-season data, dmin>-1 day): {fallback:.1f}% of windows")
    print(f"  los (season length, days):   median={res[f'{X}_los'].median():.0f}  "
          f"q25={res[f'{X}_los'].quantile(.25):.0f}  q75={res[f'{X}_los'].quantile(.75):.0f}")
    print(f"  sos_time (doy):  median={res[f'{X}_sos_time'].median():.0f}  "
          f"min={res[f'{X}_sos_time'].min():.0f}  max={res[f'{X}_sos_time'].max():.0f}")
    print(f"  pos_time (doy):  median={res[f'{X}_pos_time'].median():.0f}")

# VH vs SAVI anchor: does radar day-0 equal crop-emergence day-0?
print("\n" + "=" * 78)
d2 = desc[KEY + ["VH_sos_time", "SAVI_sos_time"]].dropna()
diff = d2["VH_sos_time"] - d2["SAVI_sos_time"]
print(f"VH_sos_time - SAVI_sos_time (days):  median={diff.median():.0f}  "
      f"mean={diff.mean():.0f}  q25={diff.quantile(.25):.0f}  q75={diff.quantile(.75):.0f}")
print(f"  correlation(VH_sos, SAVI_sos) = {d2['VH_sos_time'].corr(d2['SAVI_sos_time']):.2f}")
print(f"  |diff|>20 days in {(diff.abs()>20).mean()*100:.0f}% of windows")

# reproduce the figure's mean VH curve (treated only, like the orange line)
print("\n" + "=" * 78)
vh = al[(al["index"] == "VH")].copy()
for grp_name, mask in [("treated", vh["treated"] == 1),
                       ("untreated", (vh["treated"] == 0) & (~vh["is_organic"]))]:
    g = vh[mask].copy()
    g["bin"] = g["days_since_sos"].round().astype(int)
    g = g[(g["bin"] >= -45) & (g["bin"] <= 150)]
    m = g.groupby("bin")["value"].agg(["mean", "count"])
    m = m[m["count"] >= 10]
    base = m["mean"].min()
    peak = m["mean"].max()
    v0 = m["mean"].reindex(range(-2, 3)).mean()  # avg near day 0
    print(f"VH figure curve [{grp_name}]: base={base:.4f} peak={peak:.4f} "
          f"value@~0={v0:.4f}  -> frac_fig=(v0-base)/(peak-base)={ (v0-base)/(peak-base):.2f}")
