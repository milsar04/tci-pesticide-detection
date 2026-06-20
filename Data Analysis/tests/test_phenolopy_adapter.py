# tests for the phenolopy adapter. a known synthetic season pins the mapping
# and the day-of-year time encoding.
import numpy as np
from phenolopy_adapter import phenometrics, DESCRIPTOR_KEYS


def _season(peak_doy=180, base=0.1, amp=0.7, sigma=30.0):
    doy = np.arange(100, 261, 7, dtype=float)
    y = base + amp * np.exp(-((doy - peak_doy) ** 2) / (2 * sigma ** 2))
    return doy, y


def test_returns_all_keys():
    doy, y = _season()
    d = phenometrics(doy, y)
    assert set(d.keys()) == set(DESCRIPTOR_KEYS)


def test_peak_value_and_amplitude_sane():
    doy, y = _season(base=0.1, amp=0.7)
    d = phenometrics(doy, y)
    assert 0.75 <= d["pos_value"] <= 0.81      # base + amp = 0.80
    assert 0.55 <= d["aos_value"] <= 0.75      # amplitude near 0.70


def test_times_are_day_of_year():
    # peak at doy 180; pos_time must be in day-of-year, not an index (~11)
    doy, y = _season(peak_doy=180)
    d = phenometrics(doy, y)
    assert 165 <= d["pos_time"] <= 195
    assert d["sos_time"] < d["pos_time"] < d["eos_time"]
    assert 30 <= d["los"] <= 160               # a real season length in days


def test_short_series_returns_nan():
    d = phenometrics(np.arange(3.0), np.array([0.1, 0.2, 0.1]))
    assert all(np.isnan(v) for v in d.values())


def test_does_not_crash_on_flat_series():
    doy = np.arange(100, 261, 7, dtype=float)
    d = phenometrics(doy, np.full_like(doy, 0.1))
    assert set(d.keys()) == set(DESCRIPTOR_KEYS)   # returns dict, no exception
