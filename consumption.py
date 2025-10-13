# pv_net_metering.py
import os
import glob
import math
import pandas as pd
import numpy as np

# ----------------------------
# Input / Output folders
# ----------------------------
IN_DIR = "shading"            # where *_pv_result.csv live
LOAD_DIR = "loads"            # optional: put per-state load profiles here (columns: timestamp, load_kW)
OUT_DIR = "net_metering"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Sites (from your message)
# ----------------------------
SITES = {
    "california":      {"lat": 37.390026, "lon": -122.08123,  "tz": "America/Los_Angeles"},
    "north_carolinas": {"lat": 35.759573, "lon": -79.019300,  "tz": "America/New_York"},
    "texas":           {"lat": 29.760077, "lon": -95.370111,  "tz": "America/Chicago"},
    "north_dakota":    {"lat": 46.539175, "lon": -102.868223, "tz": "America/Chicago"},
    "colorado":        {"lat": 39.306108, "lon": -102.269356, "tz": "America/Denver"},
    "michigan":        {"lat": 45.421402, "lon": -83.81833,   "tz": "America/Detroit"},
    "maine":           {"lat": 44.952297, "lon": -67.660831,  "tz": "America/New_York"},
    "washington":      {"lat": 47.606139, "lon": -122.332848, "tz": "America/Los_Angeles"},  # Seattle
    "missouri":        {"lat": 36.083959, "lon": -89.829251,  "tz": "America/Chicago"},
    "nevada":          {"lat": 41.947679, "lon": -116.098709, "tz": "America/Los_Angeles"},
    "florida":         {"lat": 25.76168,  "lon": -80.191179,  "tz": "America/New_York"},
}

# ----------------------------
# Tariffs – edit as needed (USD/kWh)
# ----------------------------
DEFAULT_BUY  = 0.20  # what the home pays to import from grid
DEFAULT_SELL = 0.08  # what the utility credits for exports (net metering / feed-in)
TARIFFS = {
    # override any state; omitted states use defaults
    # "california": {"buy": 0.28, "sell": 0.08},
}

# ----------------------------
# Household load model (synthetic)
# ----------------------------
DEFAULT_DAILY_KWH = 25.0   # typical daily household consumption
# You can override per state if you like:
DAILY_KWH = {
    # "texas": 30.0,
}

def normalized_residential_shape(hours):
    """
    Simple 24h shape (array of length=hours) with morning & evening peaks.
    Returns values that sum to 1 over a 24h window.
    """
    h = np.arange(hours, dtype=float) % 24
    # two cosine bumps: morning 7–10, evening 18–22
    morning = np.clip(np.cos((h-8)/3 * np.pi), 0, None)
    evening = np.clip(np.cos((h-20)/4 * np.pi), 0, None)
    base = 0.3  # small baseload
    shape = base + 0.7*(0.5*morning + evening)  # weight evening higher
    # ensure no negatives and normalize per 24h
    shape = np.maximum(shape, 0)
    # normalize by average over each full day
    return shape

def scale_load_to_daily(df_ts, daily_kwh):
    """
    Build load_kW series for df_ts (DatetimeIndex) so each local day sums to daily_kwh.
    """
    # Build a repeating 24h shape
    shape = normalized_residential_shape(len(df_ts))
    # Compute per-day scaler so each local day hits daily_kwh
    s = pd.Series(shape, index=df_ts, dtype=float)
    # initial unscaled daily energy assuming 1h steps
    daily_unscaled = s.resample("D").sum(min_count=1)
    # avoid divide by zero
    daily_unscaled = daily_unscaled.replace(0, np.nan)
    # scaler per day
    scaler = (daily_kwh / daily_unscaled).reindex(daily_unscaled.index).fillna(0.0)
    # broadcast to hourly
    scaler_hr = scaler.reindex(s.index, method="ffill")
    load_kW = s * scaler_hr
    return load_kW

def try_load_profile(state_key, tz):
    """
    If you have a measured load file, place it in LOAD_DIR as:
    loads/<state_key>_load.csv with columns: timestamp, load_kW
    It will be parsed (tz-aware). Otherwise we return None.
    """
    path = os.path.join(LOAD_DIR, f"{state_key}_load.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_convert(tz)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    if "load_kW" not in df.columns:
        raise ValueError(f"{path} must contain a 'load_kW' column")
    df = df.set_index(ts_col)
    return df["load_kW"]

def load_pv_result(state_key, tz):
    """
    Load one *_pv_result.csv; return DataFrame indexed by local tz:
    columns: PV_P_kW, PV_E_kWh, dt_hr (computed if needed)
    """
    # find file like shading/<state>_pv_result.csv
    pattern = os.path.join(IN_DIR, f"{state_key}_pv_result.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No PV file found: {pattern}")
    path = matches[0]

    df = pd.read_csv(path)

    # timestamp column (your files use timestamp_local)
    ts = "timestamp_local" if "timestamp_local" in df.columns else "period_end"
    if ts not in df.columns:
        raise ValueError(f"{path}: missing timestamp column 'timestamp_local' or 'period_end'.")

    # parse timestamps; bring to local tz
    # If ts looks tz-aware already, parse with utc then convert; otherwise localize.
    ser = pd.to_datetime(df[ts], errors="coerce", utc=True)
    # If your CSV timestamps are naive local time, uncomment next line instead:
    # ser = pd.to_datetime(df[ts], errors="coerce").dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    ser = ser.dt.tz_convert(tz)

    df = df.set_index(ser).drop(columns=[ts])
    df.index.name = "ts"

    # pick post-shading power & energy
    p_col = "P_actual_new_kW" if "P_actual_new_kW" in df.columns else "P_ac"
    PV_P = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0)

    if "E_kWh_new" in df.columns:
        PV_E = pd.to_numeric(df["E_kWh_new"], errors="coerce").fillna(0.0)
        dt_hr = None
    else:
        # compute dt hours and energy from power
        dt = df.index.to_series().diff().dt.total_seconds().div(3600.0)
        dt.iloc[0] = dt.median() if not math.isnan(dt.median()) else 1.0
        dt_hr = dt.clip(lower=0.0, upper=2.0)
        PV_E = PV_P * dt_hr

    out = pd.DataFrame({"PV_P_kW": PV_P, "PV_E_kWh": PV_E}, index=df.index)
    if dt_hr is None:
        # infer interval from index differences (assume hourly if missing)
        dt_hr = out.index.to_series().diff().dt.total_seconds().div(3600.0)
        dt_hr.iloc[0] = dt_hr.median() if not math.isnan(dt_hr.median()) else 1.0
        dt_hr = dt_hr.clip(lower=0.0, upper=2.0)
    out["dt_hr"] = dt_hr.values
    return out

def get_rates(state_key):
    spec = TARIFFS.get(state_key.lower(), {})
    return float(spec.get("buy", DEFAULT_BUY)), float(spec.get("sell", DEFAULT_SELL))

# ----------------------------
# Core computation
# ----------------------------
per_state_summaries = []

for state_key, meta in SITES.items():
    tz = meta["tz"]

    try:
        pv = load_pv_result(state_key, tz)
    except Exception as e:
        print(f"[{state_key}] skipped: {e}")
        continue

    # Load profile: measured if available, else synthetic scaled per day
    measured = try_load_profile(state_key, tz)
    if measured is not None:
        # align to PV index
        load_kW = measured.reindex(pv.index).interpolate(limit_direction="both").fillna(0.0)
    else:
        daily_kwh = DAILY_KWH.get(state_key, DEFAULT_DAILY_KWH)
        load_kW = scale_load_to_daily(pv.index, daily_kwh)

    # Energy per step from load
    load_E = load_kW * pv["dt_hr"]

    # Net power/energy (positive = export; negative = import)
    net_P = pv["PV_P_kW"] - load_kW
    net_E = pv["PV_E_kWh"] - load_E

    # Separate import/export kWh
    export_kWh = net_E.clip(lower=0.0)
    import_kWh = (-net_E).clip(lower=0.0)

    # Costs & savings
    buy_rate, sell_rate = get_rates(state_key)
    cost_without_pv = (load_E.sum()) * buy_rate
    grid_cost = (import_kWh.sum()) * buy_rate
    export_credit = (export_kWh.sum()) * sell_rate
    net_bill = grid_cost - export_credit
    savings = cost_without_pv - net_bill

    # Assemble hourly result
    out = pd.DataFrame({
        "PV_P_kW": pv["PV_P_kW"],
        "PV_E_kWh": pv["PV_E_kWh"],
        "Load_kW": load_kW,
        "Load_E_kWh": load_E,
        "Net_P_kW": net_P,
        "Net_E_kWh": net_E,
        "Import_kWh": import_kWh,
        "Export_kWh": export_kWh,
    }, index=pv.index)
    out.index.name = "timestamp_local"

    # Daily & monthly rollups
    daily = out.resample("D").sum(min_count=1)
    monthly = out.resample("MS").sum(min_count=1)  # month starts

    daily["Buy_$"]  = daily["Import_kWh"] * buy_rate
    daily["Sell_$"] = daily["Export_kWh"] * sell_rate
    daily["NetBill_$"] = daily["Buy_$"] - daily["Sell_$"]

    monthly["Buy_$"]  = monthly["Import_kWh"] * buy_rate
    monthly["Sell_$"] = monthly["Export_kWh"] * sell_rate
    monthly["NetBill_$"] = monthly["Buy_$"] - monthly["Sell_$"]

    # Save per state
    out_path = os.path.join(OUT_DIR, f"{state_key}_hourly_net.csv")
    day_path = os.path.join(OUT_DIR, f"{state_key}_daily_net.csv")
    mon_path = os.path.join(OUT_DIR, f"{state_key}_monthly_net.csv")
    out.to_csv(out_path)
    daily.to_csv(day_path)
    monthly.to_csv(mon_path)

    # Summary row
    per_state_summaries.append({
        "state": state_key,
        "buy_rate_$per_kWh": buy_rate,
        "sell_rate_$per_kWh": sell_rate,
        "total_PV_kWh": out["PV_E_kWh"].sum(),
        "total_Load_kWh": out["Load_E_kWh"].sum(),
        "total_Export_kWh": export_kWh.sum(),
        "total_Import_kWh": import_kWh.sum(),
        "baseline_cost_$": cost_without_pv,
        "grid_cost_$": grid_cost,
        "export_credit_$": export_credit,
        "net_bill_$": net_bill,
        "savings_$": savings,
    })

# Combined summary
summary_df = pd.DataFrame(per_state_summaries).sort_values("state")
summary_df.to_csv(os.path.join(OUT_DIR, "summary_all_states.csv"), index=False)

print("✅ Finished.")
print(f"Saved hourly/daily/monthly results per state in: {OUT_DIR}/")
print("Top-level summary: summary_all_states.csv")
