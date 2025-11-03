# pv_net_metering.py / consumption.py
import os
import sys
import glob
import math
import argparse
import pandas as pd
import numpy as np

# ----------------------------
# Defaults (folders can be overridden by CLI)
# ----------------------------
DEFAULT_IN_DIR   = "shading"            # where *_pv_result.csv (or *_pv_result_session.csv) live
DEFAULT_LOAD_DIR = "loads"              # optional measured loads: <state>_load.csv with timestamp, load_kW
DEFAULT_OUT_DIR  = "net_metering"

# ----------------------------
# Sites (from your project)
# ----------------------------
SITES = {
    "california":      {"lat": 37.390026, "lon": -122.08123,  "tz": "America/Los_Angeles"},
    "north_carolinas": {"lat": 35.759573, "lon": -79.019300,  "tz": "America/New_York"},
    "texas":           {"lat": 29.760077, "lon": -95.370111,  "tz": "America/Chicago"},
    "north_dakota":    {"lat": 46.539175, "lon": -102.868223, "tz": "America/Chicago"},
    "colorado":        {"lat": 39.306108, "lon": -102.269356, "tz": "America/Denver"},
    "michigan":        {"lat": 45.421402, "lon": -83.81833,   "tz": "America/Detroit"},
    "maine":           {"lat": 44.952297, "lon": -67.660831,  "tz": "America/New_York"},
    "washington":      {"lat": 47.606139, "lon": -122.332848, "tz": "America/Los_Angeles"},
    "missouri":        {"lat": 36.083959, "lon": -89.829251,  "tz": "America/Chicago"},
    "nevada":          {"lat": 41.947679, "lon": -116.098709, "tz": "America/Los_Angeles"},
    "florida":         {"lat": 25.76168,  "lon": -80.191179,  "tz": "America/New_York"},
}

# ----------------------------
# Tariffs – edit as needed (USD/kWh)
# ----------------------------
DEFAULT_BUY  = 0.20  # import from grid
DEFAULT_SELL = 0.08  # export credit
TARIFFS = {
    "california":      {"buy": 0.30, "sell": 0.15},
    "north_carolinas": {"buy": 0.13, "sell": 0.065},
    "texas":           {"buy": 0.14, "sell": 0.070},
    "north_dakota":    {"buy": 0.11, "sell": 0.055},
    "colorado":        {"buy": 0.14, "sell": 0.070},
    "michigan":        {"buy": 0.19, "sell": 0.095},
    "maine":           {"buy": 0.28, "sell": 0.140},
    "washington":      {"buy": 0.11, "sell": 0.055},
    "missouri":        {"buy": 0.13, "sell": 0.065},
    "nevada":          {"buy": 0.16, "sell": 0.080},
    "florida":         {"buy": 0.15, "sell": 0.075},
}

# ----------------------------
# Household load model (synthetic)
# ----------------------------
DEFAULT_DAILY_KWH = 25.0
DAILY_KWH = {
    # "texas": 30.0,
}

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute net-metering results from shading PV outputs.")
    p.add_argument("--state", help="State key (e.g., texas). If omitted, run all states found in SITES.", default=None)
    p.add_argument("--session", action="store_true", help="Use *_session inputs if present and write *_session outputs.")
    p.add_argument("--in", dest="in_dir", default=DEFAULT_IN_DIR, help="Input shading dir (default: shading)")
    p.add_argument("--out", dest="out_dir", default=DEFAULT_OUT_DIR, help="Output net-metering dir (default: net_metering)")
    p.add_argument("--loads", dest="load_dir", default=DEFAULT_LOAD_DIR, help="Measured loads dir (default: loads)")
    return p.parse_args()

# ----------------------------
# Load helpers
# ----------------------------
def normalized_residential_shape(hours):
    h = np.arange(hours, dtype=float) % 24
    morning = np.clip(np.cos((h-8)/3 * np.pi), 0, None)
    evening = np.clip(np.cos((h-20)/4 * np.pi), 0, None)
    base = 0.3
    shape = base + 0.7*(0.5*morning + evening)
    return np.maximum(shape, 0)

def scale_load_to_daily(df_ts, daily_kwh):
    shape = normalized_residential_shape(len(df_ts))
    s = pd.Series(shape, index=df_ts, dtype=float)
    daily_unscaled = s.resample("D").sum(min_count=1).replace(0, np.nan)
    scaler = (daily_kwh / daily_unscaled).reindex(daily_unscaled.index).fillna(0.0)
    scaler_hr = scaler.reindex(s.index, method="ffill")
    load_kW = s * scaler_hr
    return load_kW

def try_load_profile(state_key, tz, load_dir):
    path = os.path.join(load_dir, f"{state_key}_load.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    ser = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_convert(tz)
    df = df.set_index(ser).drop(columns=[ts_col])
    if "load_kW" not in df.columns:
        raise ValueError(f"{path} must contain 'load_kW' column")
    return df["load_kW"]

def load_pv_result_for_state(state_key, tz, in_dir, use_session=False):
    """
    Load one PV result CSV:
      - If use_session=True, prefer shading/<state>_pv_result_session.csv (fallback to non-session if missing)
      - Else use shading/<state>_pv_result.csv
    Returns DataFrame indexed by local tz with columns: PV_P_kW, PV_E_kWh, dt_hr
    """
    patterns = []
    if use_session:
        patterns.append(os.path.join(in_dir, f"{state_key}_pv_result_session.csv"))
    patterns.append(os.path.join(in_dir, f"{state_key}_pv_result.csv"))

    path = None
    for patt in patterns:
        matches = glob.glob(patt)
        if matches:
            path = matches[0]
            break
    if path is None:
        raise FileNotFoundError(f"No PV file found for '{state_key}'. Looked for: {', '.join(patterns)}")

    df = pd.read_csv(path)

    # timestamp column (commonly 'timestamp_local' from your pipeline)
    ts_col = "timestamp_local" if "timestamp_local" in df.columns else ("period_end" if "period_end" in df.columns else None)
    if ts_col is None or ts_col not in df.columns:
        raise ValueError(f"{path}: missing timestamp column 'timestamp_local' or 'period_end'.")

    # parse timestamps → local tz index
    ser = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_convert(tz)
    df = df.set_index(ser).drop(columns=[ts_col])
    df.index.name = "ts"

    # post-shading power & energy
    p_col = "P_actual_new_kW" if "P_actual_new_kW" in df.columns else "P_ac"
    PV_P = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0)

    if "E_kWh_new" in df.columns:
        PV_E = pd.to_numeric(df["E_kWh_new"], errors="coerce").fillna(0.0)
        dt_hr = None
    else:
        dt = df.index.to_series().diff().dt.total_seconds().div(3600.0)
        dt.iloc[0] = dt.median() if not math.isnan(dt.median()) else 1.0
        dt_hr = dt.clip(lower=0.0, upper=2.0)
        PV_E = PV_P * dt_hr

    out = pd.DataFrame({"PV_P_kW": PV_P, "PV_E_kWh": PV_E}, index=df.index)
    if dt_hr is None:
        dt_hr = out.index.to_series().diff().dt.total_seconds().div(3600.0)
        dt_hr.iloc[0] = dt_hr.median() if not math.isnan(dt_hr.median()) else 1.0
        dt_hr = dt_hr.clip(lower=0.0, upper=2.0)
    out["dt_hr"] = dt_hr.values
    return out

def get_rates(state_key):
    spec = TARIFFS.get(state_key.lower(), {})
    return float(spec.get("buy", DEFAULT_BUY)), float(spec.get("sell", DEFAULT_SELL))

# ----------------------------
# Per-state computation
# ----------------------------
def compute_and_write(state_key, tz, in_dir, out_dir, load_dir, use_session=False):
    pv = load_pv_result_for_state(state_key, tz, in_dir, use_session=use_session)

    measured = try_load_profile(state_key, tz, load_dir)
    if measured is not None:
        load_kW = measured.reindex(pv.index).interpolate(limit_direction="both").fillna(0.0)
    else:
        daily_kwh = DAILY_KWH.get(state_key, DEFAULT_DAILY_KWH)
        load_kW = scale_load_to_daily(pv.index, daily_kwh)

    load_E = load_kW * pv["dt_hr"]
    net_P = pv["PV_P_kW"] - load_kW
    net_E = pv["PV_E_kWh"] - load_E

    export_kWh = net_E.clip(lower=0.0)
    import_kWh = (-net_E).clip(lower=0.0)

    buy_rate, sell_rate = get_rates(state_key)
    cost_without_pv = (load_E.sum()) * buy_rate
    grid_cost = (import_kWh.sum()) * buy_rate
    export_credit = (export_kWh.sum()) * sell_rate
    net_bill = grid_cost - export_credit
    savings = cost_without_pv - net_bill

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

    # rollups
    daily = out.resample("D").sum(min_count=1)
    monthly = out.resample("MS").sum(min_count=1)

    daily["Buy_$"]  = daily["Import_kWh"] * buy_rate
    daily["Sell_$"] = daily["Export_kWh"] * sell_rate
    daily["NetBill_$"] = daily["Buy_$"] - daily["Sell_$"]

    monthly["Buy_$"]  = monthly["Import_kWh"] * buy_rate
    monthly["Sell_$"] = monthly["Export_kWh"] * sell_rate
    monthly["NetBill_$"] = monthly["Buy_$"] - monthly["Sell_$"]

    # output paths (session-safe)
    suf = "_session" if use_session else ""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{state_key}_hourly_net{suf}.csv")
    day_path = os.path.join(out_dir, f"{state_key}_daily_net{suf}.csv")
    mon_path = os.path.join(out_dir, f"{state_key}_monthly_net{suf}.csv")

    out.to_csv(out_path)
    daily.to_csv(day_path)
    monthly.to_csv(mon_path)

    summary = {
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
        "_hourly_path": out_path,
        "_monthly_path": mon_path,
        "_daily_path": day_path,
    }
    return summary

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    in_dir   = args.in_dir
    out_dir  = args.out_dir
    load_dir = args.load_dir
    use_session = args.session

    summaries = []

    states_to_run = [args.state.lower()] if args.state else list(SITES.keys())
    for state_key in states_to_run:
        if state_key not in SITES:
            print(f"[{state_key}] skipped: not in SITES dict.")
            continue
        tz = SITES[state_key]["tz"]
        try:
            s = compute_and_write(state_key, tz, in_dir, out_dir, load_dir, use_session=use_session)
            summaries.append(s)
            tag = "SESSION" if use_session else "BASE"
            print(f"✅ {state_key} ({tag}) → hourly/daily/monthly written to {out_dir}/")
        except Exception as e:
            print(f"[{state_key}] skipped: {e}")

    # Top-level CSV (just for convenience; contains only states processed in this run)
    if summaries:
        suf = "_session" if use_session else ""
        summary_df = pd.DataFrame(summaries).sort_values("state")
        os.makedirs(out_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(out_dir, f"summary_all_states{suf}.csv"), index=False)
        print(f"Summary saved: {os.path.join(out_dir, f'summary_all_states{suf}.csv')}")
    else:
        print("No states processed.")

if __name__ == "__main__":
    main()
