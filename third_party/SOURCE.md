# Vendored PhenoloPy

Source: https://github.com/lewistrotter/PhenoloPy (scripts/phenolopy.py)
Vendored: 2026-06-03

## Local patch
Line ~2508 of `calc_phenometrics`:
`xr.merge(da_list)` -> `xr.merge(da_list, compat='override')`
Reason: modern xarray (>=2022) rejects the harmlessly-conflicting `time`
coords the metric arrays carry; older xarray merged leniently. This is the
only change required to run PhenoloPy on this repo's stack.

Only `calc_phenometrics` (and the helpers it calls) is used. The DEA/datacube
image-loading helpers are unused here.

## Time encoding

Probed with a synthetic Gaussian season (peak at day-of-year 180, weekly obs
doy 100..260, step 7).  Observed output from `calc_phenometrics`:

  pos_times = 177.000  (nearest sample to true peak 180 in the weekly grid)
  sos_times = 142.000
  eos_times = 212.000
  los_values = 70.000  (= eos 212 - sos 142, consistent with doy arithmetic)

Conclusion: `*_times` variables are returned as **day-of-year** integers (not
array indices, not ordinal date numbers).  The slight offset from 180 to 177 is
a discrete-sampling artefact - the true argmax is the sample at doy 177, which
is the grid point closest to the Gaussian peak.

`_TIME_IS_DOY = True`  - Task 3 can use `*_times` values directly as doy
without any index-to-doy conversion.