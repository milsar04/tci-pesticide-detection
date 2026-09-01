# TCI - pesticide detection from satellite time series

Saxion HBO-ICT final project for the TCI research group (Saxion / Police Academy
collaboration). Team: Milosz Sarnik, Calle Engelberts, Jim Zuidema.

Goal: help governmental inspectors decide which potato plots are worth a physical
visit, using satellite time series only. This repo holds the analytical workflow -
cleaning, phenology feature engineering, and the models that produce a per-plot
risk score. Methodology is CRISP-DM; the warehouse (PostgreSQL) and the Power BI
dashboard are separate deliverables and are not in this repo.

## Layout

```
Data Analysis/
  Imputation and Smoothing/     data loading, activity masking, imputation + smoothing
    shared data/                the raw client datasets (see below)
  Feature Engineering and Modeling/   phenology descriptors, models, diagnostics
  Exploratory Data Analysis/    figures
  tests/                        pytest suite (63 tests)
third_party/                    vendored PhenoloPy (patched)
tools/                          one-off diagnostics
```

## Setup

Python 3.11.

```bash
pip install -r "Data Analysis/Imputation and Smoothing/requirements.txt"
```

PhenoloPy is not on PyPI. A patched copy is vendored at `third_party/phenolopy.py`;
`third_party/SOURCE.md` records the upstream commit and the one-line patch.

## Data

Everything the pipeline needs is in `Data Analysis/Imputation and Smoothing/shared data/`:

| file | size | contents |
| --- | --- | --- |
| `indices_2020.csv` | 161k rows | per plot-date Sentinel-2 bands, 11 optical indices, 4 SAR indices, crop type and treatment metadata, 2020 season |
| `indices_2021.csv` | 156k rows | same, 2021 season (513 unique plots across both years) |
| `plot_activity_dates_2020.csv`, `_2021.csv` | 512 plots | per-plot potato active/inactive date ranges (dates are DD/MM/YYYY), alias plot ids in `PMT_SITE_other` |
| `potential_desiccant_events.csv` | 115 events | dated desiccant sprays, labels for the desiccation-event detector |

Key columns: `PMT_SITE` (plot id), `date`, `COMM` (POTATO or POTATO-ORGANIC),
`Treatment status`, `Single/Mixed type`, `Treated share`, `Active ingredient`.
Treatment status is per day, not per plot - never collapse it to one value per plot.
Missing index values come from cloud cover and satellite revisit gaps; that is
expected and is what the imputation step handles.

Two derived datasets are deliberately not committed because they are reproducible
in minutes: `indices_final.csv` (from `process_final.py`) and `aligned_series.csv`
(from `phenology.py`).

This is client data and the repository is private. Keep it private.

## Running the pipeline

Modules import each other by bare name, so run each script from its own folder.

```bash
cd "Data Analysis/Imputation and Smoothing"
python index_quality.py       # index selection: scores + redundancy, ~1 min
python process_final.py       # activity mask + linear imputation + whittaker smoothing -> indices_final.csv

cd "../Feature Engineering and Modeling"
python phenology.py           # timesat-style descriptors -> phenology_descriptors.csv, aligned_series.csv
python descriptor_comparison.py   # ranks descriptors by treated-vs-untreated separation
python supervised_model.py    # baseline LR + calibrated LightGBM -> model_metrics.csv, model_risk_scores.csv
python unsupervised_model.py  # PCA + clustering + anomaly scoring (appends to the same outputs)
python model_diagnostics.py   # feature importance + robustness checks
python desiccation_model.py   # desiccation-event detector

cd "../Exploratory Data Analysis"
python eda.py                 # figures/
```

`evaluate_all.py` (in `Imputation and Smoothing`) re-validates all 14 imputation and
9 smoothing methods across all 15 indices. It takes several hours and its winners
(linear interpolation, Whittaker-Eilers with lambda=1e4, d=2) are already baked into
`process_final.py`, so it only needs rerunning if the data changes.

`app.py` is a Dash app for eyeballing imputation and smoothing choices per plot:
`python app.py`.

## Tests

```bash
python -m pytest "Data Analysis" -q
```

63 tests. `Data Analysis/tests/conftest.py` puts the three module folders on
`sys.path`, so the suite runs from the repository root.

## Results

Modelling runs on the 405 plot-windows that are non-organic and show a real crop
season (309 treated, 96 untreated). Evaluation is grouped repeated cross-validation
(StratifiedGroupKFold on `PMT_SITE`, so a plot never appears in both train and test).

| model | ROC-AUC | notes |
| --- | --- | --- |
| baseline LR on `VH_pos_value` | 0.765 | single strongest descriptor |
| LightGBM on all 65 descriptors | 0.892 | headline P(treated) risk score, calibrated |
| desiccation detector (booster) | 0.810 | does not beat its 0.807 single-descriptor baseline |

The strongest single separator is peak-season VH backscatter. Treatment shows up in
vigour and amplitude rather than in season timing: the apparent timing separation is
largely an artifact of the activity-date windows, which
`timing_confound_check.py` demonstrates by splitting on `window_known`.

Main limitation is data, not method: only 96 untreated windows survive the season
gate, and the client's activity-date files are known to be incomplete (80 plots have
no window at all and are kept unmasked, flagged `window_known=False`). More clean
untreated plots and corrected activity dates would move the numbers more than further
modelling would.
