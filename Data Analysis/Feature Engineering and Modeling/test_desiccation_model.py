# test_desiccation_model.py
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import desiccation_model as dm


def _series(plot, dates, vals):
    return pd.DataFrame({"PMT_SITE": plot, "date": dates,
                         **{i: vals for i in ["SAVI", "GNDVI", "RENDVI", "VH", "RVI"]}})


def test_build_event_xy_shapes_and_labels():
    edate = pd.Timestamp("2021-06-10")
    joined = pd.DataFrame({"matched_plot": ["E"], "date": [edate]})
    dates = pd.to_datetime(["2021-05-30", "2021-06-06", "2021-06-13",
                            "2021-06-20", "2021-06-27"])
    data = pd.concat([_series("E", dates, [0.8, 0.8, 0.4, 0.1, 0.0]),   # declines
                      _series("C", dates, [0.8, 0.8, 0.8, 0.8, 0.8])],  # flat control
                     ignore_index=True)
    edoy = edate.dayofyear
    desc = pd.DataFrame({"PMT_SITE": ["C", "O"], "is_organic": [False, True],
                         "has_season": [True, True], "year": [2021, 2021],
                         "SAVI_sos_time": [edoy - 30, edoy - 30],
                         "SAVI_eos_time": [edoy + 30, edoy + 30]})
    X, y, g, meta = dm.build_event_xy(joined, data, desc)
    assert X.shape == (2, 20)               # 1 event + 1 control, 5 indices x 4 descriptors
    assert list(y) == [1, 0]
    assert list(g) == ["E", "C"]            # O is organic, excluded from controls
    assert X.loc[0, "RVI_post_slope_14d"] < 0   # the event plot declines after day 0


import pytest


def _have_real_data():
    import model_features as mf
    return os.path.exists(mf.IN_FILE) and os.path.exists(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "shared data", "potential_desiccant_events.csv"))


@pytest.mark.skipif(not _have_real_data(), reason="real indices/event data not present")
def test_build_event_xy_real_data_sanity():
    import activity_filter as af
    joined, _ = af.load_desiccation_events(write_report=False)
    data = af.load_imputed_unsmoothed()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    import model_features as mf
    desc = pd.read_csv(mf.IN_FILE)
    X, y, g, meta = dm.build_event_xy(joined, data, desc)
    assert X.shape[1] == 20                       # 5 indices x 4 descriptors
    assert int(y.sum()) > 50                      # ~95 events join in 2020-2021
    assert (y == 0).sum() > int(y.sum())          # controls are the majority
    assert g.nunique() < len(g)                   # plots repeat across events
    # no plot leaks across a grouped fold (inherited from the tested make_splitter)
    import model_eval as me
    tr, te = next(me.make_splitter(n_splits=3).split(X, y, g))
    assert set(g.iloc[tr]).isdisjoint(set(g.iloc[te]))
