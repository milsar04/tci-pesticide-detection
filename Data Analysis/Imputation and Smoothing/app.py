# app.py - dash app for comparing imputation and smoothing methods on
# satellite time-series (ndvi etc). run with: python app.py

import os
import warnings
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.interpolate import UnivariateSpline, PchipInterpolator, Akima1DInterpolator
from scipy.signal import savgol_filter, butter, filtfilt, medfilt
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import diags, eye as speye
from scipy.sparse.linalg import spsolve

warnings.filterwarnings("ignore")

# kalman is optional, pykalman might not be installed
try:
    from pykalman import KalmanFilter as PyKalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

# data loading ----------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared data")

def load_data():
    """load and merge the 2020 and 2021 CSV files."""
    files = [
        os.path.join(DATA_DIR, "indices_2020.csv"),
        os.path.join(DATA_DIR, "indices_2021.csv"),
    ]
    frames = []
    for fp in files:
        if os.path.exists(fp):
            df = pd.read_csv(fp, low_memory=False)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No data files found in {DATA_DIR}. "
            "Expected indices_2020.csv and/or indices_2021.csv."
        )
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data.sort_values(["PMT_SITE", "date"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


DF = load_data()

PLOT_IDS = sorted(DF["PMT_SITE"].unique())

# columns the user can plot
FEATURE_COLS = [
    "NDVI", "GNDVI", "EVI", "SAVI", "CI", "CI_GREEN",
    "RENDVI", "NDREI", "NDRE_(B6)", "MSAVI", "OSAVI",
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
    "VH", "VV", "RVI", "VH/VV",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in DF.columns]

# treatment metadata columns
TREATMENT_COLS = [
    "Treatment status", "Single/Mixed type",
    "Single/Mixed ingr", "Treated share", "Active ingredient",
]

# imputation methods ----------------------------------------------------------

def impute_linear(dates_numeric, values):
    """linear interpolation for missing values."""
    s = pd.Series(values, dtype=float)
    return s.interpolate(method="linear", limit_direction="both").values


def impute_spline(dates_numeric, values, k=3):
    """cubic spline interpolation (scipy UnivariateSpline)."""
    s = pd.Series(values, dtype=float)
    mask = s.notna()
    if mask.sum() < max(k + 1, 4):
        return impute_linear(dates_numeric, values)
    x_known = dates_numeric[mask]
    y_known = s[mask].values
    try:
        spline = UnivariateSpline(x_known, y_known, k=k, s=0)
        result = s.copy()
        result[~mask] = spline(dates_numeric[~mask])
        return result.values
    except Exception:
        return impute_linear(dates_numeric, values)


def impute_polynomial(dates_numeric, values, degree=3):
    """polynomial interpolation (numpy polyfit)."""
    s = pd.Series(values, dtype=float)
    mask = s.notna()
    if mask.sum() < degree + 1:
        return impute_linear(dates_numeric, values)
    try:
        coeffs = np.polyfit(dates_numeric[mask], s[mask].values, degree)
        poly = np.poly1d(coeffs)
        result = s.copy()
        result[~mask] = poly(dates_numeric[~mask])
        return result.values
    except Exception:
        return impute_linear(dates_numeric, values)


def impute_pchip(dates_numeric, values):
    """pchip - monotone hermite interpolation, no overshoot between known points."""
    s = pd.Series(values, dtype=float)
    mask = s.notna()
    if mask.sum() < 2:
        return impute_linear(dates_numeric, values)
    try:
        interp = PchipInterpolator(dates_numeric[mask], s[mask].values)
        result = s.copy()
        result[~mask] = interp(dates_numeric[~mask])
        return result.values
    except Exception:
        return impute_linear(dates_numeric, values)


def impute_akima(dates_numeric, values):
    """akima interpolation - smooth, less oscillation than cubic spline."""
    s = pd.Series(values, dtype=float)
    mask = s.notna()
    if mask.sum() < 5:
        return impute_linear(dates_numeric, values)
    try:
        interp = Akima1DInterpolator(dates_numeric[mask], s[mask].values)
        result = s.copy()
        result[~mask] = interp(dates_numeric[~mask])
        return result.values
    except Exception:
        return impute_linear(dates_numeric, values)


def impute_nearest(dates_numeric, values):
    """nearest-neighbor interpolation (step function)."""
    s = pd.Series(values, dtype=float)
    return s.interpolate(method="nearest", limit_direction="both").values


def impute_ffill_bfill(dates_numeric, values):
    """forward-fill then backward-fill."""
    s = pd.Series(values, dtype=float)
    s = s.ffill().bfill()
    return s.values


def impute_seasonal(dates_numeric, values, period=30):
    """fills gaps using seasonal decomposition (trend + seasonal component)."""
    s = pd.Series(values, dtype=float)
    mask = s.notna()
    if mask.sum() < 2 * period:
        return impute_linear(dates_numeric, values)
    try:
        # fill linearly first so decomposition can run
        filled = s.interpolate(method="linear", limit_direction="both")
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomp = seasonal_decompose(filled, model="additive", period=period,
                                     extrapolate_trend="freq")
        reconstruction = decomp.trend + decomp.seasonal
        result = s.copy()
        result[~mask] = reconstruction[~mask]
        return result.values
    except Exception:
        return impute_linear(dates_numeric, values)


# whittaker-eilers imputation -------------------------------------------------

def impute_whittaker(dates_numeric, values, lam=1e4, d=2):
    """whittaker-eilers gap-filling: penalised least-squares, weight=0 at gaps."""
    s = pd.Series(values, dtype=float)
    n = len(s)
    y = s.fillna(0).values.astype(float)
    w = (~s.isna()).astype(float).values
    try:
        W = diags(w, 0, shape=(n, n), format="csc")
        # build sparse d-th order difference matrix
        D = speye(n, format="csc")
        for _ in range(d):
            D = D[1:] - D[:-1]
        A = W + lam * D.T.dot(D)
        z = spsolve(A, w * y)
        result = s.copy()
        result[~s.notna()] = z[~s.notna()]
        return result.values
    except Exception:
        return impute_linear(dates_numeric, values)


# hybrid methods - switch strategy based on gap length ------------------------

def _get_gap_segments(dates_numeric, values):
    """returns list of (start, end, gap_days) for each contiguous NaN run."""
    s = pd.Series(values, dtype=float)
    nan_mask = s.isna().values
    segments = []
    i = 0
    n = len(nan_mask)
    while i < n:
        if nan_mask[i]:
            j = i
            while j < n and nan_mask[j]:
                j += 1
            gap_days = dates_numeric[j - 1] - dates_numeric[i] if j > i else 0
            segments.append((i, j - 1, gap_days))
            i = j
        else:
            i += 1
    return segments


def _hybrid_impute(dates_numeric, values, short_fn, long_fn, threshold=14):
    """short_fn for gaps <= threshold days, long_fn for larger gaps."""
    result = values.copy().astype(float)
    segments = _get_gap_segments(dates_numeric, values)
    for start, end, gap_days in segments:
        if gap_days <= threshold:
            filled = short_fn(dates_numeric, values)
        else:
            filled = long_fn(dates_numeric, values)
        result[start:end + 1] = filled[start:end + 1]
    return result


def impute_hybrid_linear_spline(dates_numeric, values):
    """short: linear, long: cubic spline."""
    return _hybrid_impute(dates_numeric, values, impute_linear, impute_spline)


def impute_hybrid_linear_pchip(dates_numeric, values):
    """short: linear, long: pchip."""
    return _hybrid_impute(dates_numeric, values, impute_linear, impute_pchip)


def impute_hybrid_linear_akima(dates_numeric, values):
    """short: linear, long: akima."""
    return _hybrid_impute(dates_numeric, values, impute_linear, impute_akima)


def impute_hybrid_pchip_seasonal(dates_numeric, values):
    """short: pchip, long: seasonal decomposition."""
    return _hybrid_impute(dates_numeric, values, impute_pchip, impute_seasonal,
                          threshold=30)


def impute_hybrid_linear_whittaker(dates_numeric, values):
    """short: linear, long: whittaker-eilers."""
    return _hybrid_impute(dates_numeric, values, impute_linear, impute_whittaker)


IMPUTATION_METHODS = {
    "Linear Interpolation":    impute_linear,
    "Cubic Spline":            impute_spline,
    "Polynomial (deg=3)":      impute_polynomial,
    "PCHIP":                   impute_pchip,
    "Akima":                   impute_akima,
    "Nearest Neighbor":        impute_nearest,
    "Forward/Backward Fill":   impute_ffill_bfill,
    "Seasonal Decomposition":  impute_seasonal,
    "Whittaker-Eilers (impute)": impute_whittaker,
    "Hybrid: Linear+Spline":   impute_hybrid_linear_spline,
    "Hybrid: Linear+PCHIP":    impute_hybrid_linear_pchip,
    "Hybrid: Linear+Akima":    impute_hybrid_linear_akima,
    "Hybrid: PCHIP+Seasonal":  impute_hybrid_pchip_seasonal,
    "Hybrid: Linear+Whittaker": impute_hybrid_linear_whittaker,
}

# smoothing methods -----------------------------------------------------------

def smooth_savgol(values, window=11, polyorder=3):
    """savitzky-golay filter."""
    n = len(values)
    if n < window:
        window = n if n % 2 == 1 else max(n - 1, 3)
    if window < polyorder + 2:
        return values.copy()
    try:
        return savgol_filter(values, window_length=window, polyorder=polyorder)
    except Exception:
        return values.copy()


def smooth_moving_avg(values, window=7):
    """simple rolling mean."""
    s = pd.Series(values)
    smoothed = s.rolling(window=window, center=True, min_periods=1).mean()
    return smoothed.values


def smooth_kalman(values):
    """basic 1d kalman filter (needs pykalman)."""
    if not KALMAN_AVAILABLE:
        return values.copy()
    try:
        masked = np.ma.array(values, mask=np.isnan(values))
        kf = PyKalmanFilter(
            initial_state_mean=np.nanmean(values),
            n_dim_obs=1,
            em_vars=["transition_covariance", "observation_covariance",
                      "initial_state_covariance"],
        )
        kf = kf.em(masked.reshape(-1, 1), n_iter=5)
        smoothed_state, _ = kf.smooth(masked.reshape(-1, 1))
        return smoothed_state.flatten()
    except Exception:
        return values.copy()


def smooth_lowpass(values, cutoff=0.1, fs=1.0, order=3):
    """low-pass butterworth filter."""
    n = len(values)
    if n < 13:
        return values.copy()
    try:
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        normal_cutoff = min(normal_cutoff, 0.99)
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, values)
    except Exception:
        return values.copy()


def smooth_lowess(values, frac=0.1):
    """lowess (locally weighted scatterplot smoothing)."""
    try:
        import statsmodels.api as sm
        x = np.arange(len(values), dtype=float)
        result = sm.nonparametric.lowess(values, x, frac=frac, return_sorted=False)
        return result
    except Exception:
        return values.copy()


def smooth_gaussian(values, sigma=3):
    """gaussian kernel smoothing."""
    try:
        return gaussian_filter1d(values.astype(float), sigma=sigma)
    except Exception:
        return values.copy()


def smooth_ema(values, span=7):
    """exponential moving average."""
    try:
        s = pd.Series(values)
        return s.ewm(span=span, adjust=False).mean().values
    except Exception:
        return values.copy()


def smooth_median(values, kernel_size=7):
    """median filter, good for spike removal."""
    try:
        ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return medfilt(values.astype(float), kernel_size=ks)
    except Exception:
        return values.copy()


def smooth_whittaker(values, lam=1e4, d=2):
    """whittaker-eilers smoother. penalised least-squares on a complete series.
    popular in remote sensing (Atzberger & Eilers 2011)."""
    try:
        n = len(values)
        y = np.array(values, dtype=float)
        E = speye(n, format="csc")
        D = speye(n, format="csc")
        for _ in range(d):
            D = D[1:] - D[:-1]
        A = E + lam * D.T.dot(D)
        smoothed = spsolve(A, y)
        return smoothed
    except Exception:
        return values.copy()


SMOOTHING_METHODS = {
    "Savitzky-Golay": smooth_savgol,
    "Moving Average": smooth_moving_avg,
}
if KALMAN_AVAILABLE:
    SMOOTHING_METHODS["Kalman Filter"] = smooth_kalman
SMOOTHING_METHODS["Low-pass (Butterworth)"] = smooth_lowpass
SMOOTHING_METHODS["LOWESS"] = smooth_lowess
SMOOTHING_METHODS["Gaussian"] = smooth_gaussian
SMOOTHING_METHODS["Exponential MA"] = smooth_ema
SMOOTHING_METHODS["Median Filter"] = smooth_median
SMOOTHING_METHODS["Whittaker-Eilers"] = smooth_whittaker

# dash layout -----------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Crop Index Analysis Tool - Saxion",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.config.suppress_callback_exceptions = True

# colors for plot traces
COLORS = {
    # raw data
    "raw":                    "#636EFA",
    # imputation
    "Linear Interpolation":   "#EF553B",
    "Cubic Spline":           "#00CC96",
    "Polynomial (deg=3)":     "#AB63FA",
    "PCHIP":                  "#FF7F0E",
    "Akima":                  "#D62728",
    "Nearest Neighbor":       "#9467BD",
    "Forward/Backward Fill":  "#8C564B",
    "Seasonal Decomposition":   "#E377C2",
    "Whittaker-Eilers (impute)": "#1B9E77",
    "Hybrid: Linear+Spline":    "#7570B3",
    "Hybrid: Linear+PCHIP":     "#E6AB02",
    "Hybrid: Linear+Akima":     "#A6761D",
    "Hybrid: PCHIP+Seasonal":   "#2CA02C",
    "Hybrid: Linear+Whittaker": "#D95F02",
    # smoothing
    "Savitzky-Golay":           "#FFA15A",
    "Moving Average":           "#19D3F3",
    "Kalman Filter":            "#FF6692",
    "Low-pass (Butterworth)":   "#B6E880",
    "LOWESS":                   "#FECB52",
    "Gaussian":                 "#72B7B2",
    "Exponential MA":           "#F58518",
    "Median Filter":            "#E45756",
    "Whittaker-Eilers":         "#54A24B",
}

# sidebar
sidebar = dbc.Card(
    [
        dbc.CardHeader(
            html.H5("Controls", className="mb-0 text-white"),
            className="bg-primary",
        ),
        dbc.CardBody(
            [
                # dataset year filter
                dbc.Label("Dataset Year", className="fw-bold mt-1"),
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[
                        {"label": "All years", "value": "all"},
                        {"label": "2020", "value": "2020"},
                        {"label": "2021", "value": "2021"},
                    ],
                    value="all",
                    clearable=False,
                    className="mb-3",
                ),

                # plot id
                dbc.Label("Plot ID (PMT_SITE)", className="fw-bold"),
                dcc.Dropdown(
                    id="site-dropdown",
                    options=[{"label": s, "value": s} for s in PLOT_IDS],
                    value=PLOT_IDS[0] if PLOT_IDS else None,
                    clearable=False,
                    searchable=True,
                    className="mb-3",
                ),

                # feature to plot
                dbc.Label("Feature", className="fw-bold"),
                dcc.Dropdown(
                    id="feature-dropdown",
                    options=[{"label": f, "value": f} for f in FEATURE_COLS],
                    value="NDVI",
                    clearable=False,
                    className="mb-3",
                ),

                html.Hr(),

                # imputation methods checklist
                dbc.Label("Imputation Methods", className="fw-bold"),
                dcc.Checklist(
                    id="imputation-checklist",
                    options=[{"label": f"  {m}", "value": m}
                             for m in IMPUTATION_METHODS],
                    value=["Linear Interpolation"],
                    className="mb-3",
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "block", "cursor": "pointer",
                                "marginBottom": "4px"},
                ),

                html.Hr(),

                # smoothing methods checklist
                dbc.Label("Smoothing Methods", className="fw-bold"),
                dcc.Checklist(
                    id="smoothing-checklist",
                    options=[{"label": f"  {m}", "value": m}
                             for m in SMOOTHING_METHODS],
                    value=[],
                    className="mb-3",
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "block", "cursor": "pointer",
                                "marginBottom": "4px"},
                ),

                html.Hr(),

                # which imputed series to apply smoothing on
                dbc.Label("Smooth based on", className="fw-bold"),
                dcc.Dropdown(
                    id="smooth-base-dropdown",
                    options=[{"label": m, "value": m}
                             for m in IMPUTATION_METHODS],
                    value="Linear Interpolation",
                    clearable=False,
                    className="mb-3",
                ),

                html.Hr(),

                # savgol window
                dbc.Label("Savitzky-Golay window size", className="fw-bold"),
                dcc.Slider(
                    id="savgol-window-slider",
                    min=5, max=31, step=2, value=11,
                    marks={i: str(i) for i in range(5, 32, 4)},
                    className="mb-3",
                ),

                # moving average window
                dbc.Label("Moving Average window size", className="fw-bold"),
                dcc.Slider(
                    id="ma-window-slider",
                    min=3, max=21, step=2, value=7,
                    marks={i: str(i) for i in range(3, 22, 4)},
                    className="mb-3",
                ),
            ],
        ),
    ],
    className="shadow-sm",
    style={"position": "sticky", "top": "10px"},
)

# main content area
main_content = dbc.Card(
    [
        dbc.CardHeader(
            html.H5("Time-Series Visualisation", className="mb-0 text-white"),
            className="bg-primary",
        ),
        dbc.CardBody(
            [
                # treatment info badge row
                html.Div(id="treatment-info", className="mb-3"),
                # graph
                dcc.Loading(
                    dcc.Graph(
                        id="main-graph",
                        config={"displayModeBar": True, "scrollZoom": True},
                        style={"height": "62vh"},
                    ),
                    type="circle",
                ),
                # stats table
                html.Div(id="stats-table", className="mt-3"),
            ],
        ),
    ],
    className="shadow-sm",
)

# evaluation panel
eval_panel = dbc.Card(
    [
        dbc.CardHeader(
            html.Div(
                [
                    html.H5(
                        "Method Evaluation (Cross-Validation)",
                        className="mb-0 text-white d-inline",
                    ),
                    dbc.Button(
                        "Run Evaluation",
                        id="run-eval-btn",
                        color="light",
                        size="sm",
                        className="float-end",
                    ),
                ],
            ),
            className="bg-primary",
        ),
        dbc.CardBody(
            [
                dbc.Alert(
                    [
                        html.Strong("How this works: "),
                        "The tool randomly hides 20% of the known (non-missing) data "
                        "points, applies each imputation method without those points, "
                        "then measures how well each method predicted the hidden values. "
                        "Lower RMSE/MAE = better imputation. Higher R² = better fit. "
                        "For smoothing, a Roughness Index measures curve smoothness "
                        "(lower = smoother) while Fidelity (1 - normalised deviation) "
                        "measures how close the smoothed curve stays to the data.",
                    ],
                    color="info",
                    className="small py-2",
                ),
                # recommendation badge
                html.Div(id="eval-recommendation", className="mb-3"),
                # evaluation results
                dcc.Loading(
                    html.Div(id="eval-results"),
                    type="circle",
                ),
            ],
        ),
    ],
    className="shadow-sm mt-3",
)

# put it all together
app.layout = dbc.Container(
    [
        # header
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H3(
                            "Satellite Crop Index Analysis Tool",
                            className="text-white mb-0",
                        ),
                        html.Small(
                            "Saxion University - Detection of Pesticide Treatments",
                            className="text-white-50",
                        ),
                    ],
                    className="bg-primary rounded p-3 mb-3 shadow-sm",
                ),
                width=12,
            ),
        ),
        # body
        dbc.Row(
            [
                dbc.Col(sidebar, lg=3, md=4, sm=12, className="mb-3"),
                dbc.Col(
                    html.Div([main_content, eval_panel]),
                    lg=9, md=8, sm=12,
                ),
            ],
        ),
        # footer
        dbc.Row(
            dbc.Col(
                html.Small(
                    "Built with Plotly Dash | Saxion University 2026",
                    className="text-muted text-center d-block mt-3 mb-2",
                ),
            ),
        ),
    ],
    fluid=True,
    className="px-4 pt-3",
)

# callbacks -------------------------------------------------------------------

@callback(
    Output("main-graph", "figure"),
    Output("treatment-info", "children"),
    Output("stats-table", "children"),
    Input("site-dropdown", "value"),
    Input("feature-dropdown", "value"),
    Input("imputation-checklist", "value"),
    Input("smoothing-checklist", "value"),
    Input("smooth-base-dropdown", "value"),
    Input("savgol-window-slider", "value"),
    Input("ma-window-slider", "value"),
    Input("year-dropdown", "value"),
)
def update_graph(
    site, feature, imp_methods, smooth_methods,
    smooth_base, savgol_window, ma_window, year_filter,
):
    """builds the main time-series figure with all overlays."""

    # empty-state guard
    if not site or not feature:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Select a Plot ID and Feature to begin",
            template="plotly_white",
        )
        return empty_fig, "", ""

    # filter to the selected plot
    sub = DF[DF["PMT_SITE"] == site].copy()

    # optional year filter
    if year_filter and year_filter != "all":
        sub = sub[sub["date"].dt.year == int(year_filter)]

    if sub.empty or feature not in sub.columns:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=f"No data for {site} / {feature}",
            template="plotly_white",
        )
        return empty_fig, dbc.Alert("No data available.", color="warning"), ""

    sub = sub.sort_values("date").reset_index(drop=True)
    dates = sub["date"]
    raw_values = sub[feature].values.astype(float)

    # numeric x-axis in days from first observation
    t0 = dates.min()
    dates_numeric = (dates - t0).dt.total_seconds().values / 86400.0

    # build figure
    fig = go.Figure()

    # raw data trace
    fig.add_trace(
        go.Scatter(
            x=dates, y=raw_values,
            mode="markers+lines",
            name=f"Raw {feature}",
            line=dict(color=COLORS["raw"], width=1, dash="dot"),
            marker=dict(size=4, color=COLORS["raw"]),
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d}<br>" + feature + ": %{y:.4f}<extra>Raw</extra>",
        )
    )

    # imputation overlays
    imputed_cache = {}
    for method_name in (imp_methods or []):
        func = IMPUTATION_METHODS.get(method_name)
        if func is None:
            continue
        imputed = func(dates_numeric, raw_values.copy())
        imputed_cache[method_name] = imputed
        fig.add_trace(
            go.Scatter(
                x=dates, y=imputed,
                mode="lines",
                name=f"Imputed: {method_name}",
                line=dict(color=COLORS.get(method_name, "#888"), width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>" + feature + ": %{y:.4f}<extra>"
                              + method_name + "</extra>",
            )
        )

    # smoothing overlays
    base_imputed = imputed_cache.get(smooth_base)
    if base_imputed is None:
        # fall back to imputing it ourselves
        base_func = IMPUTATION_METHODS.get(smooth_base, impute_linear)
        base_imputed = base_func(dates_numeric, raw_values.copy())

    for method_name in (smooth_methods or []):
        func = SMOOTHING_METHODS.get(method_name)
        if func is None:
            continue
        if method_name == "Savitzky-Golay":
            smoothed = smooth_savgol(base_imputed.copy(), window=savgol_window)
        elif method_name == "Moving Average":
            smoothed = smooth_moving_avg(base_imputed.copy(), window=ma_window)
        elif method_name == "Kalman Filter":
            smoothed = smooth_kalman(base_imputed.copy())
        elif method_name == "Low-pass (Butterworth)":
            smoothed = smooth_lowpass(base_imputed.copy())
        elif method_name == "LOWESS":
            smoothed = smooth_lowess(base_imputed.copy())
        elif method_name == "Gaussian":
            smoothed = smooth_gaussian(base_imputed.copy())
        elif method_name == "Exponential MA":
            smoothed = smooth_ema(base_imputed.copy(), span=ma_window)
        elif method_name == "Median Filter":
            smoothed = smooth_median(base_imputed.copy(), kernel_size=ma_window)
        elif method_name == "Whittaker-Eilers":
            smoothed = smooth_whittaker(base_imputed.copy())
        else:
            smoothed = func(base_imputed.copy())

        fig.add_trace(
            go.Scatter(
                x=dates, y=smoothed,
                mode="lines",
                name=f"Smooth: {method_name}",
                line=dict(
                    color=COLORS.get(method_name, "#888"),
                    width=2.5, dash="dash",
                ),
                hovertemplate="%{x|%Y-%m-%d}<br>" + feature + ": %{y:.4f}<extra>"
                              + method_name + "</extra>",
            )
        )

    # treatment event annotations
    treatment_dates = sub[sub["Treatment status"].notna() &
                          (sub["Treatment status"] != "No")]
    for _, row in treatment_dates.iterrows():
        y_val = row[feature]
        if pd.isna(y_val):
            # place at mean if raw value is NaN
            y_val = np.nanmean(raw_values) if np.any(~np.isnan(raw_values)) else 0
        fig.add_annotation(
            x=row["date"], y=y_val,
            text=str(row["Treatment status"]),
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowcolor="#d62728", font=dict(size=9, color="#d62728"),
            ax=0, ay=-30, opacity=0.8,
        )

    # layout
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"{feature} Time-Series for Plot <b>{site}</b>",
            font=dict(size=16),
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title=feature, showgrid=True, gridcolor="#eee"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11),
        ),
        hovermode="x unified",
        margin=dict(t=80, b=40, l=50, r=20),
    )

    # treatment info badges
    treat_info = sub[TREATMENT_COLS].dropna(how="all").drop_duplicates()
    if treat_info.empty:
        treat_badges = dbc.Alert(
            "No treatment information available for this plot.", color="light",
        )
    else:
        # show top treatment rows (up to 5)
        rows_html = []
        for _, tr in treat_info.head(5).iterrows():
            status = tr.get("Treatment status", "N/A")
            color_map = {
                "Herbicide": "warning", "Fungicide": "info",
                "Insecticide": "danger", "PGR": "success",
                "Mixed": "secondary", "Other": "dark", "No": "light",
            }
            badge_color = color_map.get(status, "primary")
            rows_html.append(
                dbc.Row(
                    [
                        dbc.Col(dbc.Badge(status, color=badge_color,
                                          className="me-2 px-2 py-1"), width="auto"),
                        dbc.Col(html.Small(
                            f"Type: {tr.get('Single/Mixed type', 'N/A')} | "
                            f"Ingredient: {tr.get('Single/Mixed ingr', 'N/A')} | "
                            f"Active: {tr.get('Active ingredient', 'N/A')} | "
                            f"Treated share: {tr.get('Treated share', 'N/A')}",
                            className="text-muted",
                        ), width=True),
                    ],
                    className="mb-1 align-items-center",
                )
            )
        treat_badges = html.Div(
            [html.Strong("Treatment Info: ", className="me-2")] + rows_html,
            className="border rounded p-2 bg-light",
        )

    # summary stats table
    stats_rows = []
    valid = raw_values[~np.isnan(raw_values)]
    stats_rows.append({
        "Series": f"Raw {feature}",
        "Count (valid)": len(valid),
        "Missing": int(np.isnan(raw_values).sum()),
        "Mean": f"{np.nanmean(raw_values):.4f}" if len(valid) else "N/A",
        "Std": f"{np.nanstd(raw_values):.4f}" if len(valid) else "N/A",
        "Min": f"{np.nanmin(raw_values):.4f}" if len(valid) else "N/A",
        "Max": f"{np.nanmax(raw_values):.4f}" if len(valid) else "N/A",
    })
    for mname, imp_vals in imputed_cache.items():
        v = imp_vals[~np.isnan(imp_vals)] if imp_vals is not None else []
        stats_rows.append({
            "Series": f"Imputed: {mname}",
            "Count (valid)": len(v),
            "Missing": int(np.isnan(imp_vals).sum()) if imp_vals is not None else "N/A",
            "Mean": f"{np.nanmean(imp_vals):.4f}" if len(v) else "N/A",
            "Std": f"{np.nanstd(imp_vals):.4f}" if len(v) else "N/A",
            "Min": f"{np.nanmin(imp_vals):.4f}" if len(v) else "N/A",
            "Max": f"{np.nanmax(imp_vals):.4f}" if len(v) else "N/A",
        })

    stats_table = dbc.Table.from_dataframe(
        pd.DataFrame(stats_rows),
        striped=True, bordered=True, hover=True, responsive=True,
        size="sm", className="small",
    )

    return fig, treat_badges, stats_table


@callback(
    Output("eval-results", "children"),
    Output("eval-recommendation", "children"),
    Input("run-eval-btn", "n_clicks"),
    State("site-dropdown", "value"),
    State("feature-dropdown", "value"),
    State("year-dropdown", "value"),
    State("savgol-window-slider", "value"),
    State("ma-window-slider", "value"),
    prevent_initial_call=True,
)
def run_evaluation(n_clicks, site, feature, year_filter, savgol_window, ma_window):
    """20% leave-out CV for imputation; roughness/fidelity eval for smoothing."""

    if not site or not feature:
        return dbc.Alert("Select a Plot ID and Feature first.", color="warning"), ""

    sub = DF[DF["PMT_SITE"] == site].copy()
    if year_filter and year_filter != "all":
        sub = sub[sub["date"].dt.year == int(year_filter)]
    if sub.empty or feature not in sub.columns:
        return dbc.Alert("No data available for evaluation.", color="warning"), ""

    sub = sub.sort_values("date").reset_index(drop=True)
    dates = sub["date"]
    raw_values = sub[feature].values.astype(float)
    t0 = dates.min()
    dates_numeric = (dates - t0).dt.total_seconds().values / 86400.0

    # valid (non-NaN) indices
    valid_mask = ~np.isnan(raw_values)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 10:
        return dbc.Alert(
            f"Only {len(valid_indices)} valid data points - need at least 10 "
            "for meaningful cross-validation.", color="warning",
        ), ""

    # 20% leave-out CV for imputation
    np.random.seed(42)  # reproducible
    n_hide = max(2, int(0.2 * len(valid_indices)))
    hide_idx = np.random.choice(valid_indices, size=n_hide, replace=False)
    true_values = raw_values[hide_idx].copy()

    # mask the hidden points
    masked_values = raw_values.copy()
    masked_values[hide_idx] = np.nan

    imp_results = []
    for method_name, func in IMPUTATION_METHODS.items():
        try:
            imputed = func(dates_numeric, masked_values.copy())
            predicted = imputed[hide_idx]

            both_valid = ~np.isnan(predicted) & ~np.isnan(true_values)
            if both_valid.sum() < 2:
                imp_results.append({
                    "Method": method_name,
                    "RMSE": "N/A", "MAE": "N/A", "R²": "N/A",
                })
                continue

            p = predicted[both_valid]
            t = true_values[both_valid]

            rmse = np.sqrt(np.mean((p - t) ** 2))
            mae = np.mean(np.abs(p - t))
            ss_res = np.sum((p - t) ** 2)
            ss_tot = np.sum((t - np.mean(t)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            imp_results.append({
                "Method": method_name,
                "RMSE": f"{rmse:.6f}",
                "MAE": f"{mae:.6f}",
                "R²": f"{r2:.4f}",
                "_rmse": rmse,
            })
        except Exception as e:
            imp_results.append({
                "Method": method_name,
                "RMSE": "Error", "MAE": "Error", "R²": "Error",
            })

    # smoothing eval - linear-imputed baseline
    base_imputed = impute_linear(dates_numeric, raw_values.copy())

    smooth_results = []
    for method_name, func in SMOOTHING_METHODS.items():
        try:
            if method_name == "Savitzky-Golay":
                smoothed = smooth_savgol(base_imputed.copy(), window=savgol_window)
            elif method_name == "Moving Average":
                smoothed = smooth_moving_avg(base_imputed.copy(), window=ma_window)
            elif method_name == "Kalman Filter":
                smoothed = smooth_kalman(base_imputed.copy())
            elif method_name == "Low-pass (Butterworth)":
                smoothed = smooth_lowpass(base_imputed.copy())
            elif method_name == "LOWESS":
                smoothed = smooth_lowess(base_imputed.copy())
            elif method_name == "Gaussian":
                smoothed = smooth_gaussian(base_imputed.copy())
            elif method_name == "Exponential MA":
                smoothed = smooth_ema(base_imputed.copy(), span=ma_window)
            elif method_name == "Median Filter":
                smoothed = smooth_median(base_imputed.copy(), kernel_size=ma_window)
            elif method_name == "Whittaker-Eilers":
                smoothed = smooth_whittaker(base_imputed.copy())
            else:
                smoothed = func(base_imputed.copy())

            smoothed = np.array(smoothed, dtype=float)

            # roughness = rms of 2nd differences
            if len(smoothed) > 2:
                d2 = np.diff(smoothed, n=2)
                roughness = np.sqrt(np.mean(d2 ** 2))
            else:
                roughness = 0.0

            # fidelity vs raw known points
            known_vals = raw_values[valid_mask]
            smooth_at_known = smoothed[valid_mask]
            both_ok = ~np.isnan(known_vals) & ~np.isnan(smooth_at_known)
            if both_ok.sum() > 0:
                deviation = np.sqrt(np.mean(
                    (known_vals[both_ok] - smooth_at_known[both_ok]) ** 2
                ))
                data_range = np.ptp(known_vals[both_ok])
                fidelity = 1 - (deviation / data_range) if data_range > 0 else 1.0
            else:
                fidelity = 0.0
                deviation = 0.0

            smooth_results.append({
                "Method": method_name,
                "Roughness": f"{roughness:.6f}",
                "RMSD from raw": f"{deviation:.6f}",
                "Fidelity": f"{fidelity:.4f}",
                "_roughness": roughness,
                "_fidelity": fidelity,
            })
        except Exception:
            smooth_results.append({
                "Method": method_name,
                "Roughness": "Error", "RMSD from raw": "Error",
                "Fidelity": "Error",
            })

    # build output tables and chart
    imp_df = pd.DataFrame(imp_results)
    imp_numeric = imp_df[imp_df.get("_rmse", pd.Series(dtype=float)).notna()].copy()
    if "_rmse" in imp_df.columns:
        imp_numeric = imp_df[imp_df["_rmse"].notna()].copy()
    else:
        imp_numeric = pd.DataFrame()

    fig_eval = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Imputation: RMSE (lower = better)",
                        "Smoothing: Roughness vs Fidelity"),
        horizontal_spacing=0.12,
    )

    # imputation bar chart
    if not imp_numeric.empty:
        imp_sorted = imp_numeric.sort_values("_rmse")
        fig_eval.add_trace(
            go.Bar(
                x=imp_sorted["Method"],
                y=imp_sorted["_rmse"],
                marker_color=[
                    COLORS.get(m, "#888") for m in imp_sorted["Method"]
                ],
                text=imp_sorted["RMSE"],
                textposition="auto",
                name="RMSE",
                showlegend=False,
            ),
            row=1, col=1,
        )

    smooth_df = pd.DataFrame(smooth_results)
    # smoothing bar chart
    if "_roughness" in smooth_df.columns:
        sm_numeric = smooth_df[smooth_df["_roughness"].notna()].copy()
        if not sm_numeric.empty:
            fig_eval.add_trace(
                go.Bar(
                    x=sm_numeric["Method"],
                    y=sm_numeric["_roughness"],
                    marker_color=[
                        COLORS.get(m, "#888") for m in sm_numeric["Method"]
                    ],
                    text=sm_numeric["Roughness"],
                    textposition="auto",
                    name="Roughness",
                    showlegend=False,
                ),
                row=1, col=2,
            )

    fig_eval.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(t=40, b=60, l=40, r=20),
    )
    fig_eval.update_xaxes(tickangle=-35, tickfont=dict(size=9))

    imp_display = imp_df.drop(columns=["_rmse"], errors="ignore")
    smooth_display = smooth_df.drop(
        columns=["_roughness", "_fidelity"], errors="ignore"
    )

    result_content = html.Div([
        dcc.Graph(figure=fig_eval, config={"displayModeBar": False}),
        dbc.Row([
            dbc.Col([
                html.H6("Imputation Accuracy",
                        className="text-primary fw-bold mt-2"),
                html.Small(
                    f"Cross-validation: {n_hide} of {len(valid_indices)} "
                    "known points hidden (20%)",
                    className="text-muted d-block mb-2",
                ),
                dbc.Table.from_dataframe(
                    imp_display, striped=True, bordered=True,
                    hover=True, responsive=True, size="sm",
                    className="small",
                ),
            ], lg=6, md=12),
            dbc.Col([
                html.H6("Smoothing Quality",
                        className="text-primary fw-bold mt-2"),
                html.Small(
                    "Based on linear-interpolated data as input",
                    className="text-muted d-block mb-2",
                ),
                dbc.Table.from_dataframe(
                    smooth_display, striped=True, bordered=True,
                    hover=True, responsive=True, size="sm",
                    className="small",
                ),
            ], lg=6, md=12),
        ]),
    ])

    # recommendation boxes
    rec_parts = []

    # best imputation
    if not imp_numeric.empty and "_rmse" in imp_numeric.columns:
        best_imp = imp_numeric.loc[imp_numeric["_rmse"].idxmin()]
        imp_name = best_imp["Method"]
        imp_rmse = best_imp["RMSE"]
        imp_r2 = best_imp.get("R²", "N/A")
        rec_parts.append(
            dbc.Alert(
                [
                    html.I(className="bi bi-trophy-fill me-2"),
                    html.Strong("Best imputation: "),
                    html.Span(
                        f"{imp_name}  (RMSE = {imp_rmse}, R² = {imp_r2})",
                    ),
                    html.Br(),
                    html.Small(
                        "This method had the lowest prediction error on the 20% "
                        "held-out known data points.",
                        className="text-muted",
                    ),
                ],
                color="success",
                className="py-2 mb-2",
            )
        )

    # best smoothing
    if "_roughness" in smooth_df.columns and "_fidelity" in smooth_df.columns:
        sm_valid = smooth_df[
            smooth_df["_roughness"].notna() & smooth_df["_fidelity"].notna()
        ].copy()
        if not sm_valid.empty:
            # combined score = 0.5*normalised_roughness + 0.5*(1-fidelity)
            r_min, r_max = sm_valid["_roughness"].min(), sm_valid["_roughness"].max()
            r_range = r_max - r_min if r_max > r_min else 1.0
            sm_valid["_score"] = (
                (sm_valid["_roughness"] - r_min) / r_range * 0.5
                + (1 - sm_valid["_fidelity"]) * 0.5
            )
            best_sm = sm_valid.loc[sm_valid["_score"].idxmin()]
            sm_name = best_sm["Method"]
            sm_rough = best_sm["Roughness"]
            sm_fidel = best_sm["Fidelity"]
            rec_parts.append(
                dbc.Alert(
                    [
                        html.I(className="bi bi-trophy-fill me-2"),
                        html.Strong("Best smoothing: "),
                        html.Span(
                            f"{sm_name}  "
                            f"(Roughness = {sm_rough}, Fidelity = {sm_fidel})",
                        ),
                        html.Br(),
                        html.Small(
                            "This method achieved the best balance between smoothness "
                            "(low roughness) and staying close to the raw data "
                            "(high fidelity).",
                            className="text-muted",
                        ),
                    ],
                    color="success",
                    className="py-2 mb-2",
                )
            )

    recommendation = html.Div(rec_parts) if rec_parts else ""

    return result_content, recommendation


# run server ------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Satellite Crop Index Analysis Tool")
    print("  Open http://127.0.0.1:8050 in your browser.\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
