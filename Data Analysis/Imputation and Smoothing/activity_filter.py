# activity_filter.py - filter the indices data to rows where each plot was
# actually an active potato field, using the plot_activity_dates csvs.
# pure module: takes dataframes / paths, never imports app.py.

import ast
import pandas as pd

ORGANIC_VALUE = "POTATO-ORGANIC"


def _parse_aliases(cell):
    """parse the stringified python list in PMT_SITE_other into a list of str."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if not isinstance(cell, str):
        return []
    try:
        val = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val]
    return [str(val).strip()]


def load_activity_windows(paths):
    """build {plot_code: [(active_ts, inactive_ts), ...]} from the activity csvs.
    registers each window under the primary PMT_SITE and every alias. drops and
    collects rows where a date is unparseable or inactive < active.
    returns (windows_dict, bad_rows_dataframe)."""
    windows = {}
    bad = []
    for path in paths:
        adf = pd.read_csv(path)
        for _, row in adf.iterrows():
            primary = str(row["PMT_SITE"]).strip()
            active = pd.to_datetime(row.get("Active_date"), dayfirst=True,
                                    errors="coerce")
            inactive = pd.to_datetime(row.get("Inactive_date"), dayfirst=True,
                                      errors="coerce")
            if pd.isna(active) or pd.isna(inactive) or inactive < active:
                bad.append({
                    "plot": primary,
                    "active": row.get("Active_date"),
                    "inactive": row.get("Inactive_date"),
                    "source": path,
                })
                continue
            codes = [primary] + _parse_aliases(row.get("PMT_SITE_other"))
            for code in codes:
                windows.setdefault(code, []).append((active, inactive))
    return windows, pd.DataFrame(bad)


# desiccation events -----------------------------------------------------------
# the events csv (potential_desiccant_events.csv) has columns PMT_SITE, date
# (ISO), Active ingredient (stringified list), iso_year. parsing reuses
# _parse_aliases for the ingredient list.

def load_events(path):
    """parse the desiccation-events csv. dates are ISO, Active ingredient is a
    stringified python list. keeps all years."""
    ev = pd.read_csv(path)
    ev["PMT_SITE"] = ev["PMT_SITE"].astype(str).str.strip()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev["ingredients"] = ev["Active ingredient"].apply(_parse_aliases)
    if "iso_year" in ev.columns:
        ev["iso_year"] = pd.to_numeric(ev["iso_year"], errors="coerce").astype("Int64")
    else:
        ev["iso_year"] = ev["date"].dt.year.astype("Int64")
    return ev[["PMT_SITE", "date", "ingredients", "iso_year"]]


def build_plot_aliases(paths):
    """map every plot id (primary + aliases) to the frozenset of all ids that
    refer to the same physical plot, using the activity-date files."""
    groups = {}  # id -> mutable set shared by its group

    def union(ids):
        merged = set(ids)
        for i in ids:
            if i in groups:
                merged |= groups[i]
        for i in merged:
            groups[i] = merged

    for path in paths:
        adf = pd.read_csv(path)
        for _, row in adf.iterrows():
            primary = str(row["PMT_SITE"]).strip()
            union({primary, *_parse_aliases(row.get("PMT_SITE_other"))})
    return {k: frozenset(v) for k, v in groups.items()}


def resolve_event_plot(eid, aliases, indices_plots):
    """canonical indices plot id for an event id, alias-aware. None if no match.
    prefers a direct hit on eid, then any equivalent alias."""
    candidates = [eid] + sorted(aliases.get(eid, frozenset()) - {eid})
    for cand in candidates:
        if cand in indices_plots:
            return cand
    return None


def joinable_events(events, indices_plots, aliases):
    """add a matched_plot column (canonical indices id) and keep only the events
    whose plot exists in the indices data."""
    ev = events.copy()
    ev["matched_plot"] = ev["PMT_SITE"].apply(
        lambda e: resolve_event_plot(e, aliases, indices_plots))
    return ev[ev["matched_plot"].notna()].reset_index(drop=True)


def event_coverage_report(events_all, joined, data, windows):
    """human-readable summary of the event join: coverage, organic-with-event,
    out-of-window events, and a label-noise audit (matched plot whose Treatment
    status is No despite a desiccation event). returns a string."""
    comm = data.groupby("PMT_SITE")["COMM"].first()
    status = data.groupby("PMT_SITE")["Treatment status"].first()
    unmatched = sorted(set(events_all["PMT_SITE"]) - set(joined["PMT_SITE"]))
    organic = sorted({p for p in joined["matched_plot"]
                      if str(comm.get(p)) == ORGANIC_VALUE})
    label_noise = sorted({p for p in joined["matched_plot"]
                          if str(status.get(p, "No")).strip().lower() == "no"})
    outside = 0
    for _, r in joined.iterrows():
        if not _in_any_window(r["matched_plot"], r["date"], windows):
            outside += 1
    lines = [
        "desiccation event join report",
        "-----------------------------",
        f"events total : {len(events_all)}",
        f"matched      : {len(joined)}",
        f"unmatched    : {len(unmatched)}",
        f"events outside the plot's activity window: {outside}",
        f"events on organic plots: {len(organic)}",
        f"label-noise (matched plot Treatment status == No): {len(label_noise)}",
    ]
    if unmatched:
        lines.append("  unmatched ids (first 30): " + ", ".join(unmatched[:30]))
    if organic:
        lines.append("  organic-with-event plots: " + ", ".join(organic))
    if label_noise:
        lines.append("  label-noise plots (report to TCI): " + ", ".join(label_noise[:30]))
    return "\n".join(lines)


def _in_any_window(plot, date, windows):
    for active, inactive in windows.get(plot, ()):  # () if plot unknown
        if active <= date <= inactive:
            return True
    return False


def apply_activity_filter(df, windows, strict=True):
    """keep only rows whose date falls inside an active potato window for the
    row's plot. always adds is_organic (COMM == POTATO-ORGANIC) and window_known
    (True if the plot has any window in the date files).
    strict=True : drop rows of plots with no known window.
    strict=False: keep rows of plots that have NO window at all (window_known
                  False, kept unmasked for sensitivity checks), but still
                  drop in-window-known plots whose row is out of window."""
    df = df.copy()
    df["is_organic"] = df["COMM"].eq(ORGANIC_VALUE)
    known = df["PMT_SITE"].isin(windows.keys())
    df["window_known"] = known
    in_window = df.apply(
        lambda r: _in_any_window(r["PMT_SITE"], r["date"], windows), axis=1
    )
    keep = in_window if strict else (in_window | ~known)
    return df[keep].reset_index(drop=True)


def coverage_report(df_before, df_after, windows, bad_df):
    """human-readable summary of what the filter removed. returns a string."""
    plots_before = df_before["PMT_SITE"].nunique()
    plots_after = df_after["PMT_SITE"].nunique()
    unknown = sorted(set(df_before["PMT_SITE"].unique()) - set(windows.keys()))
    lines = [
        "activity filter coverage report",
        "-------------------------------",
        f"rows  : {len(df_before):,} -> {len(df_after):,} "
        f"({len(df_after) / max(len(df_before), 1):.1%} kept)",
        f"plots : {plots_before} -> {plots_after}",
        f"plots with no window in the date files: {len(unknown)}",
        f"bad date rows dropped (inactive < active or unparseable): {len(bad_df)}",
    ]
    if len(bad_df):
        lines.append("  bad rows:")
        for _, r in bad_df.iterrows():
            lines.append(f"    {r['plot']}: {r['active']} -> {r['inactive']}")
    if unknown:
        lines.append("  unmatched plots (first 30): " + ", ".join(unknown[:30]))
    return "\n".join(lines)


# integration helper -----------------------------------------------------------
# imported lazily so unit tests of the pure functions above never load app.py.

def load_filtered_data(strict=False, write_report=True):
    """load the merged indices via app.load_data(), apply the activity filter,
    and (optionally) write activity_filter_report.txt next to this file.
    strict=False by default (team decision): the 80 plots absent from the
    activity files are kept unmasked rather than dropped, since the client
    flagged the date files as incomplete. those rows carry window_known=False
    so later phases can sensitivity-check with/without them."""
    import os
    from app import load_data
    import pipeline_config as cfg

    df = load_data()
    windows, bad = load_activity_windows(cfg.ACTIVITY_FILES)
    filtered = apply_activity_filter(df, windows, strict=strict)

    if write_report:
        report = coverage_report(df, filtered, windows, bad)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "activity_filter_report.txt")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(report)
    return filtered


def load_desiccation_events(write_report=True):
    """load + alias-join the desiccation events to the indices plots. writes
    desiccation_join_report.txt next to this file. returns (joined_events, data)
    where data is the raw merged indices (dates parsed)."""
    import os
    from app import load_data
    import pipeline_config as cfg

    data = load_data()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    events = load_events(cfg.DESICCANT_EVENTS_FILE)
    aliases = build_plot_aliases(cfg.ACTIVITY_FILES)
    joined = joinable_events(events, set(data["PMT_SITE"].unique()), aliases)

    if write_report:
        windows, _ = load_activity_windows(cfg.ACTIVITY_FILES)
        report = event_coverage_report(events, joined, data, windows)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "desiccation_join_report.txt")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(report)
    return joined, data


def load_imputed_unsmoothed():
    """activity-filtered data with the 5 kept indices linear-imputed within each
    plot but NOT smoothed. a desiccant collapse spans ~1-2 weeks, which the
    whittaker smoother attenuates, so the decline analysis uses this series and
    treats indices_final (smoothed) as a sensitivity check only."""
    from app import impute_linear
    import pipeline_config as cfg

    df = load_filtered_data(write_report=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    feats = [c for c in cfg.KEEP_INDICES if c in df.columns]
    parts = []
    for _, sub in df.groupby("PMT_SITE", sort=False):
        sub = sub.sort_values("date").copy()
        t0 = sub["date"].min()
        dn = (sub["date"] - t0).dt.total_seconds().values / 86400.0
        for f in feats:
            sub[f] = impute_linear(dn, sub[f].values.astype(float))
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)
