# report_extra_figures.py - analytical figures for the research report:
#   1. descriptor_ranking.png - top 15 descriptors by treated-vs-untreated AUC,
#      coloured by index family (visualises the Stage 4 tiers).
#   2. confound_auc.png - mean timing vs amplitude AUC on activity-masked vs
#      unmasked plots (visualises the window_known confound from Stage 4).
# reads descriptor_comparison.csv + phenology_descriptors.csv, writes to figures/.
# reuses descriptor_comparison.compare_descriptor so numbers match the pipeline.

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
FE_DIR = os.path.join(os.path.dirname(HERE), "Feature Engineering and Modeling")
FIG_DIR = os.path.join(HERE, "figures")
sys.path.insert(0, FE_DIR)
from descriptor_comparison import compare_descriptor, _as_bool  # noqa: E402

_IS_DIR = os.path.join(os.path.dirname(HERE), "Imputation and Smoothing")
sys.path.insert(0, _IS_DIR)
import pipeline_config as cfg  # noqa: E402

INDICES = cfg.KEEP_INDICES
COLORS = {"SAVI": "#2ca02c", "GNDVI": "#1f77b4", "RENDVI": "#9467bd",
          "VH": "#d62728", "RVI": "#ff7f0e"}


def ranking_fig(out):
    df = pd.read_csv(os.path.join(FE_DIR, "descriptor_comparison.csv"))
    top = df.nlargest(15, "auc").iloc[::-1]  # ascending so the best is on top
    fam = top["descriptor"].str.split("_").str[0]
    colors = [COLORS.get(i, "gray") for i in fam]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["descriptor"], top["auc"], color=colors)
    ax.set_xlim(0.5, 0.80)
    ax.set(xlabel="symmetric AUC (0.5 = chance)",
           title="Top 15 descriptors by treated-vs-untreated separation")
    handles = [Patch(color=COLORS[i], label=i) for i in INDICES]
    ax.legend(handles=handles, title="index", loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def confound_fig(out):
    d = pd.read_csv(os.path.join(FE_DIR, "phenology_descriptors.csv"))
    d = d[~_as_bool(d["is_organic"]) & _as_bool(d["has_season"])].copy()
    wk = _as_bool(d["window_known"])
    timing = [f"{i}_{k}" for i in INDICES for k in ("sos_time", "pos_time", "eos_time")]
    ampl = [f"{i}_{k}" for i in INDICES for k in ("aos_value", "pos_value", "lios")]

    def mean_auc(sub, cols):
        y = sub["treated"].astype(int).values
        return float(np.mean([compare_descriptor(sub[c].values, y)["auc"]
                              for c in cols if c in sub]))

    masked, unmasked = d[wk], d[~wk]
    timing_vals = [mean_auc(masked, timing), mean_auc(unmasked, timing)]
    ampl_vals = [mean_auc(masked, ampl), mean_auc(unmasked, ampl)]

    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, [timing_vals[0], ampl_vals[0]], w,
                label=f"activity-masked (n={len(masked)})", color="#8c8c8c")
    b2 = ax.bar(x + w / 2, [timing_vals[1], ampl_vals[1]], w,
                label=f"unmasked, full record (n={len(unmasked)})", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(["timing descriptors", "amplitude descriptors"])
    ax.set(ylabel="mean symmetric AUC (0.5 = chance)", ylim=(0.5, 0.80),
           title="Removing the activity windows: timing separation falls, amplitude rises")
    ax.legend()
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.2f", padding=2)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    ranking_fig(os.path.join(FIG_DIR, "descriptor_ranking.png"))
    confound_fig(os.path.join(FIG_DIR, "confound_auc.png"))
    print(f"wrote descriptor_ranking.png and confound_auc.png to {FIG_DIR}")


if __name__ == "__main__":
    main()
