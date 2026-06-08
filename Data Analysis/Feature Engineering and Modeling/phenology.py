# phenology.py - timesat-style growth-stage descriptors for one smoothed season.
# uses phenolopy_adapter for descriptor extraction. operates on per-(plot, window, index) 1d series.

import os
import numpy as np
import pandas as pd
from phenolopy_adapter import phenometrics


def align_to_sos(t, sos_time):
    """shift a time axis so 0 == start of season (days-since-sos)."""
    return np.asarray(t, dtype=float) - float(sos_time)


# driver -----------------------------------------------------------------------
# one season = one contiguous run of dates for a (plot) that stays within the
# same calendar year. the data is already activity-masked, so gaps between
# windows show up as date jumps; we split on jumps larger than SEASON_GAP_DAYS.

SEASON_GAP_DAYS = 60

IN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Imputation and Smoothing", "indices_final.csv",
)
DESC_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "phenology_descriptors.csv")
ALIGNED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "aligned_series.csv")

def _split_windows(sub):
    """yield (window_id, sub_window_df) splitting on date gaps > SEASON_GAP_DAYS
    or year boundaries (so DOY is always monotonic within a window)."""
    sub = sub.sort_values("date").reset_index(drop=True)
    gap = sub["date"].diff().dt.days.fillna(0)
    year_change = (sub["date"].dt.year.diff().fillna(0) != 0)
    win_id = ((gap > SEASON_GAP_DAYS) | year_change).cumsum()
    for wid, g in sub.groupby(win_id):
        yield int(wid), g


def _plot_label(g):
    """1 if any row in the window has a non-No treatment, else 0."""
    return int((g["Treatment status"].fillna("No").str.lower() != "no").any())


def main():
    import sys
    _is_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Imputation and Smoothing",
    )
    if _is_dir not in sys.path:
        sys.path.insert(0, _is_dir)
    import pipeline_config as cfg

    print(f"loading {IN_FILE} ...")
    df = pd.read_csv(IN_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    desc_rows = []
    aligned_rows = []

    for site, sub in df.groupby("PMT_SITE", sort=False):
        for wid, g in _split_windows(sub):
            g = g.sort_values("date")
            doy = g["date"].dt.dayofyear.values.astype(float)
            year = int(g["date"].dt.year.iloc[0])
            treated = _plot_label(g)
            organic = bool(g["is_organic"].iloc[0]) if "is_organic" in g.columns else False
            non_no = g["Treatment status"].fillna("No")
            non_no = non_no[non_no.str.lower() != "no"]
            treat_type = non_no.mode().iloc[0] if len(non_no) else "No"
            window_known = bool(g["window_known"].iloc[0]) if "window_known" in g.columns else True  # constant per plot from activity_filter

            row = {
                "PMT_SITE": site, "window": wid, "year": year,
                "treated": treated, "treatment_type": treat_type,
                "is_organic": organic, "n_obs": len(g),
                "window_known": window_known,
            }
            window_aligned = []
            for idx in cfg.KEEP_INDICES:
                if idx not in g.columns:
                    continue
                d = phenometrics(doy, g[idx].values)
                for k, v in d.items():
                    row[f"{idx}_{k}"] = v
                anchor_sos = d["sos_time"]
                if not np.isnan(anchor_sos):
                    rel = align_to_sos(doy, anchor_sos)
                    for r, val in zip(rel, g[idx].values):
                        window_aligned.append({
                            "PMT_SITE": site, "window": wid, "year": year,
                            "index": idx, "days_since_sos": r, "value": val,
                            "treated": treated, "treatment_type": treat_type,
                            "is_organic": organic,
                        })
            # has_season: detectable SAVI growing season above the amplitude threshold.
            # default False when SAVI is absent or its amplitude is NaN.
            savi_aos = row.get("SAVI_aos_value", float("nan"))
            has_season = not pd.isna(savi_aos) and savi_aos >= cfg.SEASON_AOS_THRESHOLD
            row["has_season"] = has_season
            if not has_season and "SAVI" in cfg.KEEP_INDICES and "SAVI" not in g.columns:
                print(f"warning: SAVI absent for {site} window {wid} - has_season forced False")
            if has_season:
                aligned_rows.extend(window_aligned)
            desc_rows.append(row)

    desc = pd.DataFrame(desc_rows)
    desc.to_csv(DESC_FILE, index=False)
    n_season = int(desc["has_season"].sum())
    print(f"wrote {len(desc):,} (plot, window) descriptor rows -> {DESC_FILE}")
    print(f"  {n_season} real-season windows (has_season=True); "
          f"{len(desc) - n_season} flagged as flat/no-season")

    aligned = pd.DataFrame(aligned_rows)
    aligned.to_csv(ALIGNED_FILE, index=False)
    print(f"wrote {len(aligned):,} aligned points -> {ALIGNED_FILE}")


if __name__ == "__main__":
    main()
