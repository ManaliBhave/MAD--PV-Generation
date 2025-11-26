#!/usr/bin/env python3
"""
ml_forecast_full.py
 
- Fetch historical (historical-forecast-api.open-meteo.com) and forecast (api.open-meteo.com)
- Merge, feature-engineer (pvlib clearsky + lags + rolling), train ML models, validate, predict
- Saves CSVs, models, and HTML plots.
 
Defaults: Berlin (lat=52.52, lon=13.41), tilt=30, azimuth=180
"""
 
import os
import argparse
from typing import List, Tuple, Optional, Dict
 
import numpy as np
import pandas as pd
import requests
import joblib
import pvlib
 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
 
import plotly.graph_objects as go
import plotly.io as pio
 
# -------------------- CONFIG --------------------
DEFAULT_TARGETS = ["gti", "ghi", "dni", "dhi"]
WEATHER_TARGETS = ["air_temp", "wind_speed_10m", "relative_humidity"]
 
LAGS = [1, 24, 48, 168]
ROLL_WINDOWS = [3, 24]
VAL_DAYS = 30
 
MODEL_DIR = "models_ml_shortterm"
PLOTS_DIR = "plots"
MERGED_CSV = "merged_open_meteo.csv"
FORECAST_CSV = "ml_forecast.csv"
RAW_JSON_DIR = "raw_api_responses"
 
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RAW_JSON_DIR, exist_ok=True)
 
 
# -------------------- HELPERS --------------------
def safe_json_save(prefix: str, obj: dict):
    import json, datetime
    fname = os.path.join(RAW_JSON_DIR, f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%dT%H%M%S')}.json")
    try:
        with open(fname, "w") as f:
            json.dump(obj, f)
        print(f"[INFO] Saved raw JSON to {fname}")
    except Exception as e:
        print("[WARN] Could not save raw JSON:", e)
 
 
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "shortwave_radiation_instant": "ghi",
        "direct_normal_irradiance_instant": "dni",
        "diffuse_radiation_instant": "dhi",
        "global_tilted_irradiance_instant": "gti",
        "shortwave_radiation": "ghi",  # fallback names
        "direct_radiation": "dni",
        "diffuse_radiation": "dhi",
        "global_tilted_irradiance": "gti",
        "temperature_2m": "air_temp",
        "relative_humidity_2m": "relative_humidity",
        "relativehumidity_2m": "relative_humidity",
        "wind_speed_10m": "wind_speed_10m",
        "windspeed_10m": "wind_speed_10m",
        "cloud_cover": "cloud_fraction",
    })
    if "cloud_fraction" in df.columns:
        try:
            if float(np.nanmax(df["cloud_fraction"])) > 1.5:
                df["cloud_fraction"] = df["cloud_fraction"].astype(float) / 100.0
        except Exception:
            pass
    return df
 
 
def try_fetch_historical_forecast(lat: float, lon: float, start_date: str, end_date: str, timezone: str):
    base = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation_instant",
        "direct_normal_irradiance_instant",
        "diffuse_radiation_instant",
        "global_tilted_irradiance_instant",
        "cloud_cover",
    ]
    url = (
        f"{base}?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(hourly_vars)}&tilt=30&azimuth=180&timezone={timezone}"
    )
    print("[INFO] Attempting historical-forecast API:", url)
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        print("[WARN] Historical endpoint failed:", r.status_code, r.text[:400])
        return None
    try:
        data = r.json()
        safe_json_save("historical", data)
        return data
    except Exception as e:
        print("[WARN] JSON decode error from historical endpoint:", e)
        return None
 
 
def fetch_regular_forecast(lat: float, lon: float, horizon_hours: int):
    base = "https://api.open-meteo.com/v1/forecast"
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation_instant",
        "diffuse_radiation_instant",
        "direct_normal_irradiance_instant",
        "global_tilted_irradiance_instant",
    ]
    days = int(np.ceil(horizon_hours / 24.0)) or 1
    url = (
        f"{base}?latitude={lat}&longitude={lon}"
        f"&hourly={','.join(hourly_vars)}"
        f"&forecast_days={days}"
        f"&tilt=30&azimuth=180"
        f"&timezone=UTC"
    )
    print("[INFO] Fetching regular forecast API:", url)
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Regular forecast API failed: {r.status_code} {r.text}")
    data = r.json()
    safe_json_save("forecast", data)
    times = pd.to_datetime(data["hourly"]["time"], utc=True)
 
    def _get(key):
        return data["hourly"].get(key, [np.nan] * len(times))
 
    df = pd.DataFrame({
        "timestamp_utc": times,
        "ghi": _get("shortwave_radiation_instant"),
        "dhi": _get("diffuse_radiation_instant"),
        "dni": _get("direct_normal_irradiance_instant"),
        "gti": _get("global_tilted_irradiance_instant"),
        "air_temp": _get("temperature_2m"),
        "relative_humidity": _get("relative_rh") if "relative_rh" in data["hourly"] else _get("relative_humidity_2m"),
        "wind_speed_10m": _get("wind_speed_10m"),
    }).set_index("timestamp_utc").sort_index()
 
    df = rename_columns(df)
    # ensure continuity and realistic numeric types
    df = df.astype(float).resample("1h").interpolate().ffill()
    return df
 
 
def fetch_open_meteo(lat: float, lon: float, tz: str, hist_days: int, horizon_hours: int,
                     start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("[INFO] Fetching Open-Meteo historical + forecast data...")
    now_utc = pd.Timestamp.now(tz="UTC")
 
    if start_date is not None and end_date is not None:
        sdate = start_date
        edate = end_date
    else:
        sdate = (now_utc - pd.Timedelta(days=hist_days)).strftime("%Y-%m-%d")
        edate = now_utc.strftime("%Y-%m-%d")
 
    hist_json = try_fetch_historical_forecast(lat, lon, sdate, edate, "UTC")
    if hist_json is None:
        # fallback
        max_back_days_for_radiation = 10
        effective_hist_days = min(hist_days, max_back_days_for_radiation)
        start_hist = (now_utc - pd.Timedelta(days=effective_hist_days)).strftime("%Y-%m-%d")
        end_hist = now_utc.strftime("%Y-%m-%d")
        print(f"[INFO] Historical endpoint unavailable; falling back to standard API for past {effective_hist_days} days.")
        base = "https://api.open-meteo.com/v1/forecast"
        hourly_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "shortwave_radiation_instant",
            "direct_normal_irradiance_instant",
            "diffuse_radiation_instant",
            "global_tilted_irradiance_instant",
            "cloud_cover",
        ]
        url = (
            f"{base}?latitude={lat}&longitude={lon}"
            f"&start_date={start_hist}&end_date={end_hist}"
            f"&hourly={','.join(hourly_vars)}&tilt=30&azimuth=180&timezone=UTC"
        )
        print("[INFO] Fallback URL:", url)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        hist_json = r.json()
        safe_json_save("historical_fallback", hist_json)
    else:
        hist_json = hist_json
 
    # parse historical JSON
    times_hist = pd.to_datetime(hist_json["hourly"]["time"], utc=True)
    def _get_hist(key):
        return hist_json["hourly"].get(key, [np.nan] * len(times_hist))
 
    df_hist = pd.DataFrame({
        "timestamp_utc": times_hist,
        "gti": _get_hist("global_tilted_irradiance_instant"),
        "dni": _get_hist("direct_normal_irradiance_instant"),
        "dhi": _get_hist("diffuse_radiation_instant"),
        "ghi": _get_hist("shortwave_radiation_instant"),
        "air_temp": _get_hist("temperature_2m"),
        "relative_humidity": _get_hist("relative_humidity_2m"),
        "wind_speed_10m": _get_hist("wind_speed_10m"),
        "cloud_fraction": _get_hist("cloud_cover"),
    }).set_index("timestamp_utc").sort_index()
 
    try:
        if df_hist["cloud_fraction"].max() > 1.5:
            df_hist["cloud_fraction"] = df_hist["cloud_fraction"] / 100.0
    except Exception:
        pass
 
    df_hist = df_hist.resample("1h").mean(numeric_only=True).ffill()
 
    df_forecast = fetch_regular_forecast(lat, lon, horizon_hours)
    # take forecast after now
    df_forecast = df_forecast.loc[pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1):]
 
    if len(df_forecast) < horizon_hours:
        print(f"[WARN] Regular forecast returned {len(df_forecast)} rows, less than requested horizon {horizon_hours}. Using available forecast.")
    else:
        df_forecast = df_forecast.iloc[:horizon_hours]
 
    # Ensure columns exist
    for col in ["gti", "ghi", "dni", "dhi", "air_temp", "relative_humidity", "wind_speed_10m"]:
        if col not in df_forecast.columns:
            df_forecast[col] = np.nan
 
    # forecast often lacks cloud_fraction: borrow last hist mean or set NaN
    if "cloud_fraction" not in df_forecast.columns:
        if "cloud_fraction" in df_hist.columns:
            df_forecast["cloud_fraction"] = df_hist["cloud_fraction"].iloc[-24:].mean()
        else:
            df_forecast["cloud_fraction"] = np.nan
 
    # ensure types and continuity
    df_forecast = df_forecast.astype(float).resample("1h").interpolate().ffill()
 
    print(f"[INFO] Historical rows: {len(df_hist)}, Forecast rows: {len(df_forecast)}")
    return df_hist, df_forecast
 
 
# -------------------- FEATURE ENGINEERING --------------------
def compute_clearsky(df: pd.DataFrame, lat: float, lon: float, tz: str) -> pd.DataFrame:
    site = pvlib.location.Location(latitude=lat, longitude=lon, tz=tz)
    idx_local = df.index.tz_convert(tz)
    solpos = site.get_solarposition(idx_local)
    cs = site.get_clearsky(idx_local, model="ineichen")
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30,
        surface_azimuth=180,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=cs["dni"], ghi=cs["ghi"], dhi=cs["dhi"]
    )
    out = pd.DataFrame(index=idx_local)
    out["solar_zenith"] = solpos["apparent_zenith"].values
    out["solar_azimuth"] = solpos["azimuth"].values
    out["ghi_cs"] = cs["ghi"].values
    out["dni_cs"] = cs["dni"].values
    out["dhi_cs"] = cs["dhi"].values
    out["gti_cs"] = poa["poa_global"].values
    out.index = out.index.tz_convert("UTC")
    out = out.reindex(df.index)
    return out
 
 
def create_features_for_df(df: pd.DataFrame, lat: float, lon: float, tz: str, candidate_targets: List[str]) -> pd.DataFrame:
    df = df.copy()
    # ensure tz-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
 
    X = pd.DataFrame(index=df.index)
 
    # time features (UTC)
    X["hour"] = df.index.hour
    X["dayofweek"] = df.index.dayofweek
    X["dayofyear"] = df.index.dayofyear
    X["month"] = df.index.month
 
    # clearsky
    cs = compute_clearsky(df, lat, lon, tz)
    for c in cs.columns:
        X[c] = cs[c]
 
    # cloud
    if "cloud_fraction" in df.columns:
        X["cloud_fraction"] = df["cloud_fraction"].astype(float).ffill().bfill()
 
    # numeric candidate targets -> lags + rolling
    numeric_cols = [c for c in candidate_targets if c in df.columns]
    for col in numeric_cols:
        ser = df[col].astype(float).ffill().bfill()
        for l in LAGS:
            X[f"{col}_lag{l}"] = ser.shift(l)
        for w in ROLL_WINDOWS:
            X[f"{col}_r{w}"] = ser.rolling(w, min_periods=1).mean().shift(1)
 
    # also include direct NWP forecast variables if present (they will be used for future rows)
    for v in ["air_temp", "relative_humidity", "wind_speed_10m", "ghi", "dni", "dhi", "gti"]:
        if v in df.columns:
            X[v] = df[v].astype(float).ffill().bfill()
 
    return X.fillna(0.0)
 
 
# -------------------- METRICS / EVAL --------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) + 1e-9
    return np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0
 
 
def masked_mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 10.0) -> float:
    mask = (~np.isnan(y_true)) & (np.abs(y_true) > threshold)
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (np.abs(y_true[mask]) + 1e-9))) * 100.0
 
 
def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")
 
 
# -------------------- MODEL TRAIN / PREDICT --------------------
def train_models(X: pd.DataFrame, y: pd.DataFrame, targets: List[str], model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    models = {}
    scores = {}
 
    val_hours = VAL_DAYS * 24
 
    # time-based split
    if len(X) > val_hours + 200:
        X_train, X_val = X.iloc[:-val_hours], X.iloc[-val_hours:]
        y_train, y_val = y.iloc[:-val_hours], y.iloc[-val_hours:]
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
 
    for tgt in targets:
        if tgt not in y.columns:
            continue
        print(f"[TRAIN] Training {tgt} ...")
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=6)
        mask = y_train[tgt].notna()
        if mask.sum() < 200:
            print(f"[TRAIN] Skipping {tgt}, insufficient data ({mask.sum()} rows)")
            continue
        model.fit(X_train.loc[mask], y_train.loc[mask, tgt])
        # validation predictions
        y_true = y_val[tgt].values
        y_pred = model.predict(X_val)
 
        valid_mask = ~np.isnan(y_true)
        if valid_mask.sum() == 0:
            mae = float("nan")
            rmse = float("nan")
        else:
            mae = mean_absolute_error(y_true[valid_mask], y_pred[valid_mask])
            rmse = np.sqrt(np.mean((y_true[valid_mask] - y_pred[valid_mask]) ** 2))
 
        if tgt in {"gti", "ghi", "dni", "dhi"}:
            mape_val = masked_mape(y_true, y_pred, threshold=10.0)
            mape_label = "masked-MAPE"
        else:
            mape_val = smape(y_true, y_pred)
            mape_label = "SMAPE"
 
        r2 = safe_r2(y_true[valid_mask] if valid_mask.sum() > 0 else y_true, y_pred[valid_mask] if valid_mask.sum() > 0 else y_pred)
        bias = float(np.nanmean(y_pred[valid_mask] - y_true[valid_mask])) if valid_mask.sum() > 0 else float("nan")
 
        print(f"  MAE={mae:.4f} RMSE={rmse:.4f} {mape_label}={mape_val:.2f}% R2={r2:.3f} Bias={bias:.4f}")
 
        models[tgt] = model
        scores[tgt] = mae
        joblib.dump(model, os.path.join(model_dir, f"model_{tgt}.joblib"))
 
    return models, scores, X_val, y_val
 
 
def predict_with_models(models: Dict[str, object], X_future: pd.DataFrame):
    preds = {}
    for tgt, model in models.items():
        try:
            preds[tgt] = model.predict(X_future)
        except Exception:
            preds[tgt] = np.zeros(len(X_future))
    df_pred = pd.DataFrame(preds, index=X_future.index)
    return df_pred
 
 
# -------------------- PLOTTING --------------------
def save_validation_plots(y_val: pd.DataFrame, X_val: pd.DataFrame, models: Dict[str, object], outdir: str = PLOTS_DIR):
    os.makedirs(outdir, exist_ok=True)
    for tgt, model in models.items():
        if tgt not in y_val.columns:
            continue
        y_true = y_val[tgt]
        y_pred = model.predict(X_val)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true.index, y=y_true.values, mode="lines", name="Actual (val)"))
        fig.add_trace(go.Scatter(x=y_true.index, y=y_pred, mode="lines", name="Predicted (val)"))
        fig.update_layout(title=f"Validation – {tgt}", xaxis_title="Time", yaxis_title=tgt)
        pio.write_html(fig, file=os.path.join(outdir, f"{tgt}_validation.html"), auto_open=False)
    print(f"[INFO] Validation plots saved to {outdir}")
 
 
def save_history_forecast_plots(df_hist: pd.DataFrame, df_pred: pd.DataFrame, targets: List[str], outdir: str = PLOTS_DIR):
    os.makedirs(outdir, exist_ok=True)
    split_time = df_hist.index.max()
    for tgt in targets:
        if tgt not in df_hist.columns or tgt not in df_pred.columns:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist[tgt], mode="lines", name="Actual (history)"))
        fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred[tgt], mode="lines", name="Forecast"))
        y_min = min(df_hist[tgt].min(), df_pred[tgt].min())
        y_max = max(df_hist[tgt].max(), df_pred[tgt].max())
        fig.add_shape(type="line", x0=split_time, x1=split_time, y0=y_min, y1=y_max, xref="x", yref="y", line=dict(dash="dash"))
        fig.update_layout(title=f"History + Forecast – {tgt}", xaxis_title="Time", yaxis_title=tgt)
        pio.write_html(fig, file=os.path.join(outdir, f"{tgt}_history_forecast.html"), auto_open=False)
    print(f"[INFO] History+forecast plots saved to {outdir}")
 
 
# -------------------- MAIN --------------------
def main(lat: float, lon: float, tz: str, horizon: int, hist_days: int,
         start_date: Optional[str] = None, end_date: Optional[str] = None):
    # Developer-provided local path (from conversation history) for external usage:
    print("[DEV] Example local file path you can use elsewhere:", "/mnt/data/3eff15b9-d239-4693-90f4-113584382884.png")
 
    # 1) Fetch history and forecast (NWP)
    df_hist, df_fc = fetch_open_meteo(lat, lon, tz, hist_days, horizon, start_date=start_date, end_date=end_date)
 
    # Save merged raw
    df_merged = pd.concat([df_hist, df_fc[df_fc.index > df_hist.index.max()]]).sort_index()
    df_merged.to_csv(MERGED_CSV)
    print(f"[INFO] Saved merged data to {MERGED_CSV} (rows={len(df_merged)})")
 
    # 2) Prepare feature matrices
    candidate_targets = [t for t in DEFAULT_TARGETS + WEATHER_TARGETS if t in df_hist.columns]
    print("[INFO] Candidate targets for ML:", candidate_targets)
 
    # Build features for historical (train) and forecast (future)
    X_hist = create_features_for_df(df_hist, lat, lon, tz, candidate_targets)
    X_fc = create_features_for_df(df_fc, lat, lon, tz, candidate_targets)
 
    # Ensure feature columns match (use hist columns as superset)
    for col in X_hist.columns:
        if col not in X_fc.columns:
            X_fc[col] = 0.0
    X_fc = X_fc[X_hist.columns]
 
    # Build y (targets) from historical
    y_hist = df_hist[[c for c in candidate_targets if c in df_hist.columns]]
 
    # 3) Train models for irradiance targets (and optionally weather)
    models, scores, X_val, y_val = train_models(X_hist, y_hist, candidate_targets, MODEL_DIR)
 
    # 4) Save validation plots
    save_validation_plots(y_val, X_val, models, outdir=PLOTS_DIR)
 
    # 5) Predict on future where features come from NWP forecast (X_fc)
    df_pred = predict_with_models(models, X_fc)
    df_pred.to_csv(FORECAST_CSV, index_label="timestamp")
    print(f"[INFO] Saved ML forecast CSV to {FORECAST_CSV} (rows={len(df_pred)})")
 
    # 6) Save history+forecast plots
    save_history_forecast_plots(df_hist, df_pred, candidate_targets, outdir=PLOTS_DIR)
 
    # 7) Print summary metrics vs persistence baseline on validation window
    print("[INFO] Validation metrics vs persistence baseline:")
    for tgt, model in models.items():
        if tgt not in y_val.columns:
            continue
        y_true = y_val[tgt].values
        y_pred = model.predict(X_val)
        # persistence = last observed value (lag1)
        if f"{tgt}_lag1" in X_val.columns:
            persistence = X_val[f"{tgt}_lag1"].values
        else:
            persistence = np.zeros_like(y_true)
 
        mask = ~np.isnan(y_true)
        if mask.sum() == 0:
            continue
        mae_model = mean_absolute_error(y_true[mask], y_pred[mask])
        mae_pers = mean_absolute_error(y_true[mask], persistence[mask])
        print(f" {tgt}: MAE_model={mae_model:.3f}, MAE_persistence={mae_pers:.3f}, improvement={(mae_pers-mae_model)/mae_pers*100:.1f}%")
 
    print("[DONE] ML integration finished.")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=168, help="forecast horizon in hours")
    parser.add_argument("--hist_days", type=int, default=365, help="how many past days to pull (if start/end not provided)")
    parser.add_argument("--start_date", type=str, default=None, help="YYYY-MM-DD explicit start date for historical fetch (optional)")
    parser.add_argument("--end_date", type=str, default=None, help="YYYY-MM-DD explicit end date for historical fetch (optional)")
    parser.add_argument("--lat", type=float, default=37.390026, help="latitude (default Berlin)")
    parser.add_argument("--lon", type=float, default=-122.08123, help="longitude (default Berlin)")
    parser.add_argument("--tz", type=str, default="America/Los_Angeles", help="local timezone for pvlib (default Europe/Berlin)")
    args = parser.parse_args()
 
    main(lat=args.lat, lon=args.lon, tz=args.tz, horizon=args.horizon, hist_days=args.hist_days,
         start_date=args.start_date, end_date=args.end_date)