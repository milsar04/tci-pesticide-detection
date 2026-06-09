# pipeline_config.py - shared constants for the pipeline.
# kept import-light on purpose so any script can import it cheaply.

import os

# the 5 indices kept for modelling - one per physical signal family.
# SAVI broadband optical, GNDVI green, RENDVI red-edge, VH SAR backscatter,
# RVI SAR ratio. see index_quality.py for the selection procedure.
KEEP_INDICES = ["SAVI", "GNDVI", "RENDVI", "VH", "RVI"]

# every vegetation / SAR index (no raw bands) - used for the full method
# re-validation in evaluate_all.py.
ALL_INDICES = [
    "NDVI", "GNDVI", "EVI", "SAVI", "CI", "CI_GREEN",
    "RENDVI", "NDREI", "NDRE_(B6)", "MSAVI", "OSAVI",
    "VH", "VV", "RVI", "VH/VV",
]

# identifier / label columns carried through to the modelling dataset.
METADATA_COLS = [
    "PMT_SITE", "date", "PMT_YEAR", "COMM", "is_organic", "window_known",  # is_organic + window_known added by activity_filter.py
    "Treatment status", "Single/Mixed type", "Single/Mixed ingr",
    "Treated share", "Active ingredient",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared data")

ACTIVITY_FILES = [
    os.path.join(DATA_DIR, "plot_activity_dates_2020.csv"),
    os.path.join(DATA_DIR, "plot_activity_dates_2021.csv"),
]

# minimum SAVI seasonal amplitude (peak - base) for a plot-window to count
# as a real crop season. windows below this are flat/bare-soil and excluded
# from the descriptor comparison. see real_season_check.py for the sensitivity
# analysis that validated this threshold.
SEASON_AOS_THRESHOLD = 0.20

# desiccation events (potential_desiccant_events.csv): the client's dated
# spray events. EVENT_PRE/POST_DAYS bound the decline window measured around
# each event date. see the desiccation-events spec.
DESICCANT_EVENTS_FILE = os.path.join(DATA_DIR, "potential_desiccant_events.csv")
EVENT_PRE_DAYS = 14
EVENT_POST_DAYS = 21
