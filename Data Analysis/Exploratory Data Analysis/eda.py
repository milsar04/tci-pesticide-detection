# eda.py - exploratory plots for treated vs untreated index dynamics.
# two views per index: calendar (day-of-year) and growth-stage (sos-aligned),
# plus per-treatment-type curves and per-descriptor box plots. organic plots are
# kept as their own series, never folded into "untreated".

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: write png files, no display
import matplotlib.pyplot as plt

import sys as _sys
_FE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "Feature Engineering and Modeling")
if _FE_DIR not in _sys.path:
    _sys.path.insert(0, _FE_DIR)
from descriptor_comparison import compare_descriptor, fit_slope

_IS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "Imputation and Smoothing")
if _IS_DIR not in _sys.path:
    _sys.path.insert(0, _IS_DIR)
import pipeline_config as cfg


def summarize_band(df, x, value, group):
    """per (group, x): mean, std, n of value, ignoring nan. long-form output
    used to draw a mean line with a shaded +/-1 std band."""
    d = df[[x, value, group]].copy()
    d = d[d[value].notna()]
    out = (
        d.groupby([group, x])[value]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    out["std"] = out["std"].fillna(0.0)
    return out


_as_bool = cfg._as_bool


# growth-stage display window (days since sos). the masked windows keep long
# post-harvest bare-soil shoulders (treated windows reach +260d), so the raw
# sos-aligned axis smears to +/-300. descriptors are already season-restricted
# (sos..eos, median season ~53d); the figure shows the same growth stage:
# green-up baseline through senescence. shoulders outside this are bare soil.
ALIGNED_DAYS = (-45, 150)

# within the display window, still drop days backed by fewer than this many
# observations - removes residual low-support noise (mainly the untreated tail).
MIN_BAND_N = 10


def filter_band_by_support(band, min_n=MIN_BAND_N):
    """drop band rows (one per group, x) backed by fewer than min_n windows."""
    return band[band["n"] >= min_n].copy()


# desiccation: matched-date control selection ---------------------------------
# a control window for one event is a real-season, non-organic plot-window in
# the same year whose SAVI season span contains the event's day-of-year, on a
# plot that had no desiccation event that year.

def select_control_windows(desc, event_doy, event_year, event_plots, index="SAVI"):
    """control (plot, window) rows from phenology_descriptors for one event."""
    sos = desc[f"{index}_sos_time"]
    eos = desc[f"{index}_eos_time"]
    mask = (
        ~_as_bool(desc["is_organic"])
        & _as_bool(desc["has_season"])
        & (desc["year"] == event_year)
        & (sos <= event_doy) & (event_doy <= eos)
        & (~desc["PMT_SITE"].isin(event_plots))
    )
    return desc[mask].reset_index(drop=True)


def calendar_window_slope(series, start_date, days):
    """slope of value vs day-offset over [start_date, start_date+days] for one
    plot's (date, value) series. NaN if fewer than 2 valid points."""
    end = start_date + pd.Timedelta(days=days)
    w = series[(series["date"] >= start_date) & (series["date"] <= end)]
    off = (w["date"] - start_date).dt.total_seconds().values / 86400.0
    return fit_slope(off, w["value"].values)


def _plot_event_overlay(data, joined, index, out_png):
    """mean event-aligned curve of desiccated plots (day 0 = event date),
    over -EVENT_PRE_DAYS..+EVENT_POST_DAYS, plus a sample of individual series."""
    import pipeline_config as cfg
    rows = []
    for _, ev in joined.iterrows():
        plot, edate = ev["matched_plot"], ev["date"]
        lo = edate - pd.Timedelta(days=cfg.EVENT_PRE_DAYS)
        hi = edate + pd.Timedelta(days=cfg.EVENT_POST_DAYS)
        w = data[(data["PMT_SITE"] == plot) & (data["date"] >= lo) & (data["date"] <= hi)]
        for _, r in w.iterrows():
            d = (r["date"] - edate).days
            rows.append({"d": d, "value": r[index], "key": f"{plot}_{edate.date()}"})
    if not rows:
        return
    al = pd.DataFrame(rows)
    band = al.groupby("d")["value"].agg(mean="mean", std="std", n="count").reset_index()
    band["std"] = band["std"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, sub in al.groupby("key"):
        ax.plot(sub.sort_values("d")["d"], sub.sort_values("d")["value"],
                color="0.8", lw=0.6, zorder=1)
    ax.plot(band["d"], band["mean"], color="C3", lw=2, label="desiccated mean", zorder=3)
    ax.fill_between(band["d"], band["mean"] - band["std"], band["mean"] + band["std"],
                    color="C3", alpha=0.15, zorder=2)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set(xlabel="days since event", ylabel=index, title=f"{index} - event-aligned decline")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def run_desiccation_validation():
    """stage A: test the client's steeper-decline hypothesis. for each kept
    index, compare the post-event decline slope of desiccated plots against
    matched-date controls. writes figures + desiccation_validation.csv and
    prints the gate verdict (any index symmetric AUC >= 0.60)."""
    if IS_DIR not in _sys.path:
        _sys.path.insert(0, IS_DIR)
    import activity_filter as af

    joined, _ = af.load_desiccation_events(write_report=True)
    data = af.load_imputed_unsmoothed()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    desc = pd.read_csv(DESC_FILE)

    # plots that fired an event, per year, to exclude from controls
    event_plots_by_year = (joined.groupby(joined["date"].dt.year)["matched_plot"]
                           .apply(set).to_dict())

    rows = []
    for idx in KEEP_INDICES:
        if idx not in data.columns:
            continue
        treated_slopes, control_slopes = [], []
        for _, ev in joined.iterrows():
            plot, edate = ev["matched_plot"], ev["date"]
            eyear, edoy = edate.year, edate.dayofyear
            series = data.loc[data["PMT_SITE"] == plot, ["date", idx]] \
                         .rename(columns={idx: "value"})
            # post-event slope over day 0..+14 (mirrors Stage B post_slope_14d)
            t = calendar_window_slope(series, edate, 14)
            if not np.isnan(t):
                treated_slopes.append(t)
            controls = select_control_windows(
                desc, edoy, eyear, event_plots_by_year.get(eyear, set()))
            for cplot in controls["PMT_SITE"].unique():
                cs = data.loc[data["PMT_SITE"] == cplot, ["date", idx]] \
                         .rename(columns={idx: "value"})
                cslope = calendar_window_slope(cs, edate, 14)
                if not np.isnan(cslope):
                    control_slopes.append(cslope)
        values = np.array(treated_slopes + control_slopes)
        labels = np.array([1] * len(treated_slopes) + [0] * len(control_slopes))
        r = compare_descriptor(values, labels)
        r["index"] = idx
        rows.append(r)
        os.makedirs(FIG_DIR, exist_ok=True)
        _plot_event_overlay(data, joined, idx, os.path.join(FIG_DIR, f"desiccation_overlay_{idx}.png"))

    out = pd.DataFrame(rows)[["index", "auc", "cliffs_delta", "mw_p",
                              "n_treated", "n_untreated"]]
    out.sort_values("auc", ascending=False, inplace=True)
    out_csv = os.path.join(FE_DIR, "desiccation_validation.csv")
    out.to_csv(out_csv, index=False, float_format="%.4f")
    print("\ndesiccation slope test (post-event slope, desiccated vs controls):")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best = out["auc"].max()
    print(f"\nGATE: best index AUC = {best:.3f} "
          f"({'PASS - proceed to Stage B' if best >= 0.60 else 'FAIL - stop, write up as a finding'})")
    return out


# paths ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))           # .../Exploratory Data Analysis
DATA_ANALYSIS = os.path.dirname(HERE)                       # .../Data Analysis
FE_DIR = os.path.join(DATA_ANALYSIS, "Feature Engineering and Modeling")
IS_DIR = os.path.join(DATA_ANALYSIS, "Imputation and Smoothing")
FIG_DIR = os.path.join(HERE, "figures")

FINAL_FILE = os.path.join(IS_DIR, "indices_final.csv")
ALIGNED_FILE = os.path.join(FE_DIR, "aligned_series.csv")
DESC_FILE = os.path.join(FE_DIR, "phenology_descriptors.csv")

KEEP_INDICES = cfg.KEEP_INDICES

# headline descriptors to box-plot (one optical shape, one optical level, one SAR)
BOX_DESCRIPTORS = [
    "SAVI_aos_value", "SAVI_los", "SAVI_pos_value", "SAVI_roi",
    "VH_aos_value", "RVI_aos_value",
]


def _obs_group(df):
    """per-observation series label for the calendar overlay."""
    organic = _as_bool(df["is_organic"]) if "is_organic" in df else pd.Series(False, index=df.index)
    treated = df["Treatment status"].fillna("No").str.lower().ne("no")
    return np.where(organic, "organic", np.where(treated, "treated", "untreated"))


def _draw_band(ax, band, xcol):
    for grp, sub in band.groupby(band.columns[0]):
        sub = sub.sort_values(xcol)
        ax.plot(sub[xcol], sub["mean"], label=str(grp))
        ax.fill_between(sub[xcol], sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.15)
    ax.legend()


def plot_calendar(final, index, out_png):
    df = final.copy()
    df["doy"] = pd.to_datetime(df["date"], errors="coerce").dt.dayofyear
    df["grp"] = _obs_group(df)
    band = summarize_band(df, x="doy", value=index, group="grp")
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_band(ax, band, "doy")
    ax.set(xlabel="day of year", ylabel=index, title=f"{index} - calendar axis")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def plot_aligned(aligned, index, out_png):
    sub = aligned[aligned["index"] == index].copy()
    if sub.empty:
        return
    sub["grp"] = np.where(_as_bool(sub["is_organic"]), "organic",
                          np.where(sub["treated"].astype(int) == 1, "treated", "untreated"))
    sub["d"] = sub["days_since_sos"].round().astype(int)
    sub = sub[sub["d"].between(*ALIGNED_DAYS)]
    band = filter_band_by_support(summarize_band(sub, x="d", value="value", group="grp"))
    if band.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_band(ax, band, "d")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set(xlabel="days since start-of-season", ylabel=index, title=f"{index} - growth-stage axis")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def plot_treatment_types(aligned, index, out_png):
    sub = aligned[(aligned["index"] == index) & (~_as_bool(aligned["is_organic"]))].copy()
    if sub.empty:
        return
    sub["d"] = sub["days_since_sos"].round().astype(int)
    sub = sub[sub["d"].between(*ALIGNED_DAYS)]
    band = filter_band_by_support(summarize_band(sub, x="d", value="value", group="treatment_type"))
    if band.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw_band(ax, band, "d")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set(xlabel="days since start-of-season", ylabel=index, title=f"{index} - by treatment type")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def plot_descriptor_box(desc, col, out_png):
    # gate to real-season windows (matches descriptor_comparison): exclude
    # organic and flat/no-season windows, so the box is treated-vs-untreated
    # crop, not crop-vs-bare-soil.
    d = desc[~_as_bool(desc["is_organic"])]
    if "has_season" in d.columns:
        d = d[_as_bool(d["has_season"])]
    untreated = d.loc[d["treated"].astype(int) == 0, col].dropna()
    treated = d.loc[d["treated"].astype(int) == 1, col].dropna()
    if len(untreated) < 3 or len(treated) < 3:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot([untreated.values, treated.values], tick_labels=["untreated", "treated"])
    ax.set(ylabel=col, title=col)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    final = pd.read_csv(FINAL_FILE)
    aligned = pd.read_csv(ALIGNED_FILE)
    desc = pd.read_csv(DESC_FILE)

    for idx in KEEP_INDICES:
        if idx in final.columns:
            plot_calendar(final, idx, os.path.join(FIG_DIR, f"calendar_{idx}.png"))
        plot_aligned(aligned, idx, os.path.join(FIG_DIR, f"aligned_{idx}.png"))
        plot_treatment_types(aligned, idx, os.path.join(FIG_DIR, f"types_{idx}.png"))

    for col in BOX_DESCRIPTORS:
        if col in desc.columns:
            plot_descriptor_box(desc, col, os.path.join(FIG_DIR, f"box_{col}.png"))

    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
