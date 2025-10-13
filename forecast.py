import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib.location import Location
import pvlib

# -------------------- CONFIG --------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_IN = os.path.join(SCRIPT_DIR, "data/florida_2023_2025.csv")
CSV_OUT = os.path.join(SCRIPT_DIR, "five_year_data/florida_5y.csv")
CSV_COMBINED = os.path.join(SCRIPT_DIR, "combined/florida.csv")

LAT, LON = 25.76168, -80.191179
LOCAL_TZ = "America/New_York"
SURFACE_TILT, SURFACE_AZIMUTH = 30, 180
FREQ = "1h"  # hourly

# Climatology / bias params
Q = 0.88
SMOOTH_DAYS = 21
RECENT_DAYS = 30
GAIN_CLIP_IRR = (0.7, 1.3)
TAU_DAYS = 20  # decay of recent bias in forecast

# Bias strategy for meteorology-like signals
BIAS_SPEC = {
    "air_temp": {"mode": "add", "clip": (-10, 10)},            # °C offset
    "wind_speed_10m": {"mode": "mul", "clip": (0.5, 1.5)},     # x factor
    "relative_humidity": {"mode": "add", "clip": (-20, 20)},   # %-points
    "precipitation_rate": {"mode": "mul", "clip": (0.5, 2.0)}, # x factor
    "albedo": {"mode": "mul", "clip": (0.7, 1.3)},             # x factor
    "snow_soiling_rooftop": {"mode": "add", "clip": (-2, 5)},  # units in source; nonnegativity enforced later
}

PHYS_CLIPS = {
    "relative_humidity": (0, 100),
    "albedo": (0.0, 1.0),
    "wind_speed_10m": (0.0, None),
    "precipitation_rate": (0.0, None),
    "snow_soiling_rooftop": (0.0, None),
}

# -------------------- HELPERS --------------------
def circular_smooth_rows(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr
    if w % 2 == 0:
        w += 1
    r = w // 2
    pad = np.vstack([arr[-r:], arr, arr[:r]])
    return np.apply_along_axis(lambda x: np.convolve(x, np.ones(w)/w, mode="valid"), 0, pad)

def k_from_doy_hour_nearest(qpivot: pd.DataFrame, ts_index: pd.DatetimeIndex) -> pd.Series:
    qpivot = qpivot.reindex(range(1, 367), fill_value=0.0)
    rows = np.where(ts_index.dayofyear == 366, 365, ts_index.dayofyear) - 1
    h = ts_index.hour.values.astype(int)
    cols = qpivot.columns.values.astype(int)
    diffs = np.abs(h[:, None] - cols[None, :])
    wrap = 24 - diffs
    dist = np.minimum(diffs, wrap)
    j = dist.argmin(axis=1)
    vals = qpivot.values[rows, j]
    return pd.Series(vals, index=ts_index, dtype=float)

def mode_agg(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan

def apply_decay_bias(base_vals: pd.Series, hours: np.ndarray, bias_by_hour: pd.Series, mode: str, tau_days: float):
    steps_per_day = 24
    t_days = np.arange(len(base_vals)) / steps_per_day
    w = np.exp(-t_days / tau_days)
    gvec = np.array([bias_by_hour.get(h, (1.0 if mode=="mul" else 0.0)) for h in hours])
    if mode == "mul":
        return (1.0 + (gvec - 1.0) * w) * base_vals
    else:  # "add"
        return base_vals + gvec * w

def clip_physical(series: pd.Series, key: str) -> pd.Series:
    lo, hi = PHYS_CLIPS.get(key, (None, None))
    if lo is not None:
        series = series.clip(lower=lo)
    if hi is not None:
        series = series.clip(upper=hi)
    return series

# -------------------- LOAD DATA --------------------
print("📥 Load & TZ align…")
df = pd.read_csv(CSV_IN)
df["period_end"] = pd.to_datetime(df["period_end"], utc=True)
df = df.set_index("period_end").sort_index()
df_local = df.tz_convert(LOCAL_TZ)

# -------------------- RESAMPLE HOURLY --------------------
agg = {
    "air_temp": "mean",
    "albedo": "mean",
    "dhi": "mean",
    "dni": "mean",
    "ghi": "mean",
    "gti": "mean",
    "precipitation_rate": "sum",
    "relative_humidity": "mean",
    "snow_soiling_rooftop": "sum",
    # "snow_soiling_ground": "sum",  # removed by request
    "wind_speed_10m": "mean",
    "weather_type": "first",
}

# Keep only keys present in the data to avoid KeyErrors
agg = {k: v for k, v in agg.items() if k in df_local.columns}

df_hour_local = df_local.resample(FREQ, origin="start_day").agg(agg).dropna(how="all")
# Explicitly drop ground soiling if present
if "snow_soiling_ground" in df_hour_local.columns:
    df_hour_local = df_hour_local.drop(columns=["snow_soiling_ground"])

df_hour_utc = df_hour_local.tz_convert("UTC")

# -------------------- CLEAR SKY & SOLAR GEOMETRY --------------------
site = Location(LAT, LON, tz=LOCAL_TZ)
idx = df_hour_local.index
solpos = site.get_solarposition(idx)

# Add historical solar geometry columns (optional but handy)
df_hour_local["solar_azimuth"] = solpos["azimuth"]
df_hour_local["solar_zenith"] = solpos["apparent_zenith"]

cs = site.get_clearsky(idx, model="ineichen")
poa = pvlib.irradiance.get_total_irradiance(
    SURFACE_TILT, SURFACE_AZIMUTH,
    solpos["apparent_zenith"], solpos["azimuth"],
    dni=cs["dni"], ghi=cs["ghi"], dhi=cs["dhi"]
)
cs_hist = pd.DataFrame({
    "ghi_cs": cs["ghi"],
    "dni_cs": cs["dni"],
    "dhi_cs": cs["dhi"],
    "gti_cs": poa["poa_global"],
}, index=idx)
day_hist = cs_hist["ghi_cs"] > 5.0
eps = 1e-6

# -------------------- CLEARNESS INDEX --------------------
K = {}
for col, cs_col in [("gti","gti_cs"), ("ghi","ghi_cs"), ("dni","dni_cs"), ("dhi","dhi_cs")]:
    if col in df_hour_local.columns:
        ki = (df_hour_local[col] / (cs_hist[cs_col] + eps)).clip(0, 1.5)
        mask = ~day_hist.reindex_like(ki).fillna(False)
        ki.loc[mask] = 0.0
        K[f"K_{col}"] = ki
K = pd.DataFrame(K, index=idx)

# -------------------- CLIMATOLOGY (IRRADIANCE) --------------------
doy = idx.dayofyear
hour = idx.hour
qmap = {}
for kcol in [c for c in ["K_gti","K_ghi","K_dni","K_dhi"] if c in K.columns]:
    pivot = K.groupby([doy, hour])[kcol].quantile(Q).unstack(fill_value=0.0)
    pivot = pivot.reindex(range(1, 367), fill_value=0.0)
    if SMOOTH_DAYS >= 3:
        pivot.iloc[:, :] = circular_smooth_rows(pivot.values, SMOOTH_DAYS)
    qmap[kcol] = pivot

# -------------------- RECENT-BIAS (IRRADIANCE) --------------------
recent_start = idx.max() - pd.Timedelta(days=RECENT_DAYS)
recent = K.loc[K.index >= recent_start].copy()
recent_day_mask = day_hist.reindex(recent.index).fillna(False)
recent_filtered = recent.loc[recent_day_mask] if not recent.empty else recent
recent_by_hour = (recent_filtered.groupby(recent_filtered.index.hour).median()
                  .reindex(range(24)).fillna(1.0))

end_doy = int(min(idx[-1].dayofyear, 365))
climo_by_hour = {}
for kcol in qmap.keys():
    piv = qmap[kcol]
    climo_by_hour[kcol] = pd.Series({h: float(piv.loc[end_doy, h]) for h in range(24)})

gain_irr = {}
for kcol in qmap.keys():
    num = recent_by_hour.get(kcol, pd.Series(1.0, index=range(24)))
    den = climo_by_hour[kcol].replace(0, np.nan)
    g = (num / den).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    g = g.clip(GAIN_CLIP_IRR[0], GAIN_CLIP_IRR[1])
    gain_irr[kcol] = g

# -------------------- FUTURE 5-YEAR FORECAST --------------------
print("🔮 Forecasting next 5 years hourly…")
future_steps = 5 * 365 * 24  # 5 years hourly
future_local = pd.date_range(df_hour_local.index[-1] + pd.Timedelta(hours=1),
                             periods=future_steps, freq="1h", tz=LOCAL_TZ)

# Solar / clearsky for future
solpos_f = site.get_solarposition(future_local)
cs_f = site.get_clearsky(future_local, model="ineichen")
poa_f = pvlib.irradiance.get_total_irradiance(
    SURFACE_TILT, SURFACE_AZIMUTH,
    solpos_f["apparent_zenith"], solpos_f["azimuth"],
    dni=cs_f["dni"], ghi=cs_f["ghi"], dhi=cs_f["dhi"]
)
cs_future = pd.DataFrame({
    "ghi_cs": cs_f["ghi"],
    "dni_cs": cs_f["dni"],
    "dhi_cs": cs_f["dhi"],
    "gti_cs": poa_f["poa_global"],
}, index=future_local)
day_future = cs_future["ghi_cs"] > 5.0

# Map climatological K to future timeline + apply decaying recent gain
Kf = pd.DataFrame(index=future_local)
for kcol, qpivot in qmap.items():
    Kf[kcol] = k_from_doy_hour_nearest(qpivot, future_local)

hb = future_local.hour.values
for kcol in qmap.keys():
    gvec = np.array([gain_irr[kcol].get(h, 1.0) for h in hb])
    steps_per_day = 24
    t_days = np.arange(len(future_local)) / steps_per_day
    w = np.exp(-t_days / TAU_DAYS)
    Kf[kcol] = (1.0 + (gvec - 1.0) * w) * Kf[kcol]

# Reconstruct irradiance
fc_future = pd.DataFrame(index=future_local)
for col, cs_col in [("gti","gti_cs"), ("ghi","ghi_cs"), ("dni","dni_cs"), ("dhi","dhi_cs")]:
    if f"K_{col}" in Kf.columns:
        fc_future[col] = (Kf[f"K_{col}"] * cs_future[cs_col]).where(day_future, 0.0)

# -------------------- METEOROLOGY & EXTRA FIELDS (CLIMO + BIAS) --------------------
# Numeric DOY×hour medians (include albedo & snow_soiling_rooftop)
met_numeric_cols = [c for c in [
    "air_temp", "wind_speed_10m", "relative_humidity", "precipitation_rate",
    "albedo", "snow_soiling_rooftop"
] if c in df_hour_local.columns]

met_pivot = df_hour_local[met_numeric_cols].groupby(
    [df_hour_local.index.dayofyear.rename("doy"),
     df_hour_local.index.hour.rename("hour")]
).median()

doy_f = np.where(future_local.dayofyear == 366, 365, future_local.dayofyear)
mh = pd.MultiIndex.from_arrays([doy_f, future_local.hour], names=["doy","hour"])
met_future = met_pivot.reindex(mh)

# If gaps, smooth by hour
if met_future.isna().any().any():
    met_future = met_future.groupby(level="hour").apply(lambda g: g.interpolate(limit_direction="both"))

met_future.index = future_local
for c in met_numeric_cols:
    fc_future[c] = met_future[c].values

# Categorical weather_type via DOY×hour mode
if "weather_type" in df_hour_local.columns:
    wt_pivot = df_hour_local.groupby(
        [df_hour_local.index.dayofyear.rename("doy"),
         df_hour_local.index.hour.rename("hour")]
    )["weather_type"].agg(mode_agg)
    wt_future = wt_pivot.reindex(mh)
    wt_future.index = future_local
    fc_future["weather_type"] = wt_future.values

# Solar geometry into forecast
fc_future["solar_azimuth"] = solpos_f["azimuth"].values
fc_future["solar_zenith"] = solpos_f["apparent_zenith"].values

# -------------------- RECENT-BIAS for MET NUMERIC --------------------
if len(met_numeric_cols) > 0:
    # recent actuals by hour
    recent_met = df_hour_local.loc[df_hour_local.index >= recent_start, met_numeric_cols].copy()
    recent_by_hour_met = recent_met.groupby(recent_met.index.hour).median().reindex(range(24))
    # climatology for "end_doy" by hour
    climo_by_hour_met = {}
    for c in met_numeric_cols:
        # ensure we end up with a 1-level index: hour -> 0..23
        if end_doy in met_pivot.index.get_level_values("doy"):
            # xs(..., drop_level=True) collapses the 'doy' level, leaving an index of 'hour'
            series = met_pivot.xs(end_doy, level="doy")[c]
        else:
            # fallback: climatological hourly median across all doys
            series = met_pivot.groupby(level="hour")[c].median()

        # make sure it's a plain Int64Index [0..23] and numeric
        series = pd.to_numeric(series, errors="coerce")
        series = series.reindex(range(24))  # index is now hours 0..23
        series = series.fillna(series.median())

        climo_by_hour_met[c] = series


    # compute bias per hour & apply with decay
    for c in met_numeric_cols:
        spec = BIAS_SPEC[c]
        if spec["mode"] == "mul":
            num = recent_by_hour_met[c].fillna(climo_by_hour_met[c])
            den = climo_by_hour_met[c].replace(0, np.nan)
            g = (num / den).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            g = g.clip(spec["clip"][0], spec["clip"][1])
        else:  # additive
            num = recent_by_hour_met[c].fillna(climo_by_hour_met[c])
            den = climo_by_hour_met[c]
            g = (num - den).fillna(0.0).clip(spec["clip"][0], spec["clip"][1])

        fc_future[c] = apply_decay_bias(fc_future[c], hb, g, spec["mode"], TAU_DAYS)
        fc_future[c] = clip_physical(fc_future[c], c)

# -------------------- SAVE FORECAST --------------------
# Ensure required columns exist; drop any lingering ground soiling
if "snow_soiling_ground" in fc_future.columns:
    fc_future = fc_future.drop(columns=["snow_soiling_ground"])

fc_future.to_csv(CSV_OUT, index_label="timestamp_local")
print("💾 Forecast saved to:", CSV_OUT)

# -------------------- COMBINED (ACTUALS + FORECAST) --------------------
# Align a consistent set of columns for output
keep_cols = sorted(set(fc_future.columns).union(df_hour_local.columns) - {"snow_soiling_ground"})
hist = df_hour_local.reindex(columns=keep_cols)
fcast = fc_future.reindex(columns=keep_cols)

hist["is_forecast"] = 0
fcast["is_forecast"] = 1

combined = pd.concat([hist, fcast], axis=0)
combined.to_csv(CSV_COMBINED, index_label="timestamp_local")
print("💾 Historical + Forecast saved to:", CSV_COMBINED)

# -------------------- PLOTTING: HISTORICAL vs FORECAST --------------------
# features = [c for c in [
#     "gti","ghi","dni","dhi",
#     "air_temp","wind_speed_10m","relative_humidity","precipitation_rate",
#     "albedo","snow_soiling_rooftop",
#     "solar_zenith","solar_azimuth"
# ] if c in combined.columns]

# for col in features:
#     plt.figure(figsize=(12,4))
#     if col in df_hour_local.columns:
#         plt.plot(df_hour_local.index, df_hour_local[col], label="Historical", alpha=0.7)
#     plt.plot(fc_future.index, fc_future[col], label="Forecast (5y)", alpha=0.7)
#     plt.title(f"{col.upper()}: Historical vs Forecast (Next 5 Years)")
#     plt.xlabel("Time (Local)")
#     plt.ylabel(col)
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()