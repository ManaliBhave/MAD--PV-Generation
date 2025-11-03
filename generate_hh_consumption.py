import os
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
 
# ----------------------------
# Config
# ----------------------------
OUT_DIR = "loads"
os.makedirs(OUT_DIR, exist_ok=True)
 
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
# Enhanced mock load generator
# ----------------------------
def mock_load_profile(lat, tz_name, start_year=2023, end_year=2030, avg_daily_kwh=25.0):
    tz = pytz.timezone(tz_name)

    # ---- CHANGED: fixed multi-year window 2023-01-01 00:00 to 2031-01-01 00:00 (left-inclusive) ----
    start = tz.localize(datetime(start_year, 1, 1, 0, 0, 0))
    end   = tz.localize(datetime(end_year + 1, 1, 1, 0, 0, 0))
    idx = pd.date_range(start=start, end=end, freq="h", tz=tz, inclusive="left")

    # derive number of whole days in range for daily randoms
    total_days = (end - start).days
 
    hours = idx.hour
    weekdays = idx.weekday  # 0=Mon, 6=Sun

    # keep the original behavior (single month scalar) for minimal change
    month = datetime.now().month
 
    # --- Base daily shape ---
    morning_peak = np.clip(np.cos((hours - 7) / 3 * np.pi), 0, None)
    evening_peak = np.clip(np.cos((hours - 20) / 4 * np.pi), 0, None)
    base = 0.25
    shape = base + 0.75 * (0.4 * morning_peak + evening_peak)
 
    # --- Weekend boost ---
    weekend_boost = np.where(weekdays >= 5, 1.1, 1.0)
 
    # --- Seasonal variation (summer vs winter) ---
    # Peak summer (July): +15%, peak winter (Jan): +10%
    seasonal_factor = 1.0 + 0.15 * math.sin((month - 7) / 12 * 2 * math.pi)
 
    # --- Temperature proxy using latitude ---
    # Higher latitude → colder → more heating use
    temp_factor = 1.0 + (abs(lat - 30) / 60) * 0.15
 
    # --- Random per-day variation ---
    # (repeat per day, assume 24 samples/day; we’ll trim to a 24-multiple below)
    daily_factors = np.repeat(np.random.uniform(0.9, 1.1, size=total_days), 24)
    noise = np.random.normal(1.0, 0.05, size=len(idx))
 
    shape = np.array(shape) * daily_factors[:len(shape)] * noise * weekend_boost * seasonal_factor * temp_factor
 
    # Normalize to desired daily energy
    trim_len = len(shape) - (len(shape) % 24)
    shape = shape[:trim_len]
    daily_energy = shape.reshape(-1, 24).sum(axis=1)
    scale = avg_daily_kwh / np.mean(daily_energy)
    load_kw = shape * scale
 
    df = pd.DataFrame({
        "timestamp": idx[:len(shape)],
        "load_kW": load_kw
    })
    return df
 
# ----------------------------
# Generate CSVs
# ----------------------------
def main():
    for state, info in SITES.items():
        avg_kwh = 25.0
        if state in ["texas", "florida", "nevada"]:
            avg_kwh = 30.0  # hotter = more cooling
        elif state in ["maine", "michigan", "north_dakota"]:
            avg_kwh = 28.0  # colder = more heating
 
        # ---- CHANGED: call with the fixed 2023–2030 window ----
        df = mock_load_profile(info["lat"], info["tz"], start_year=2023, end_year=2030, avg_daily_kwh=avg_kwh)
        path = os.path.join(OUT_DIR, f"{state}_load.csv")
        df.to_csv(path, index=False)
        print(f"✅ Created realistic mock load for {state}: {path}")
 
if __name__ == "__main__":
    main()
