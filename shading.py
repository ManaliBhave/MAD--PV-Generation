import os
import glob
import pandas as pd
import numpy as np
from math import atan, tan, radians, degrees

# ----------------------------
# Paths
# ----------------------------
PV_DIR = "pv_generation"
OBJ_DIR = "object"
OUT_DIR = "shading"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Discover state keys from filenames
# pv files: pv_generation_<state>.csv
# obj files: <state>_object.csv
# ----------------------------
def state_key_from_pv(path):
    # pv_generation_<state>.csv -> <state>
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("pv_generation_", "", 1)

def state_key_from_obj(path):
    # <state>_object.csv -> <state>
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("_object", "", 1)

pv_files = {state_key_from_pv(p): p for p in glob.glob(os.path.join(PV_DIR, "pv_generation_*.csv"))}
obj_files = {state_key_from_obj(p): p for p in glob.glob(os.path.join(OBJ_DIR, "*_object.csv"))}

common_states = sorted(set(pv_files) & set(obj_files))
if not common_states:
    raise SystemExit("No matching PV/Object file pairs found. Check your folder names.")

print(f"Found {len(common_states)} state(s): {', '.join(common_states)}")

# ----------------------------
# Shading calculator
# ----------------------------
def compute_shading_for_row(row, obj_df_cached):
    """
    Returns (total_shading, contrib_list)
    total_shading in [0..0.95], contrib_list = list of object_ids
    """
    sun_az = row["solar_azimuth"]
    solar_elev = row["solar_elevation"]
    gti = row.get("gti", np.nan)

    # No sun / night / no irradiance
    if (solar_elev is None) or (pd.isna(solar_elev)) or (solar_elev <= 0) or (pd.isna(gti)) or (gti == 0):
        return 0.0, []

    shaded_ids = []
    total = 0.0

    for obj in obj_df_cached:
        # solar_azimuth difference (0..180)
        diff = abs(sun_az - obj["azimuth_deg"])
        if diff > 180:
            diff = 360 - diff

        # object roughly in front of sun?
        if diff <= obj["angular_half_width_plus_margin"]:
            # shadow length at ground
            # protect tan() around 0 elevation (already filtered <=0 above)
            sl = obj["height_m"] / tan(radians(solar_elev))
            if sl >= obj["distance_m"]:
                total += obj["shading_intensity"]
                shaded_ids.append(obj["object_id"])

    return min(total, 0.95), shaded_ids

# ----------------------------
# Per-state processing
# ----------------------------
for state in common_states:
    pv_path = pv_files[state]
    obj_path = obj_files[state]
    out_path = os.path.join(OUT_DIR, f"{state}_pv_result.csv")

    print(f"\n=== Processing {state} ===")
    print(f"PV:  {pv_path}")
    print(f"OBJ: {obj_path}")

    # --- load
    pv_df = pd.read_csv(pv_path)
    obj_df = pd.read_csv(obj_path)

    # --- ensure datetime
    if "timestamp_local" in pv_df.columns:
        pv_df["timestamp_local"] = pd.to_datetime(pv_df["timestamp_local"], errors="coerce")
    else:
        raise ValueError(f"'timestamp_local' column not found in {pv_path}")

    # --- sanity checks
    needed_pv_cols = ["solar_azimuth", "solar_zenith", "P_ac"]
    missing = [c for c in needed_pv_cols if c not in pv_df.columns]
    if missing:
        raise ValueError(f"Missing PV columns in {pv_path}: {missing}")

    needed_obj_cols = ["object_id", "width_m", "distance_m", "azimuth_deg", "height_m", "shading_intensity"]
    missing = [c for c in needed_obj_cols if c not in obj_df.columns]
    if missing:
        raise ValueError(f"Missing object columns in {obj_path}: {missing}")

    # --- solar elevation
    pv_df["solar_elevation"] = 90 - pv_df["solar_zenith"]

    # --- small speedup: precompute object angular width + margin once
    # margin = 5 degrees as in your original code
    obj_cache = []
    for _, r in obj_df.iterrows():
        half_width = degrees(atan((r["width_m"] / 2.0) / r["distance_m"]))
        obj_cache.append({
            "object_id": r["object_id"],
            "azimuth_deg": float(r["azimuth_deg"]),
            "height_m": float(r["height_m"]),
            "distance_m": float(r["distance_m"]),
            "shading_intensity": float(r["shading_intensity"]),
            "angular_half_width_plus_margin": float(half_width + 5.0),
        })

    # --- compute shading
    results = pv_df.apply(lambda r: compute_shading_for_row(r, obj_cache), axis=1)
    pv_df["shading_loss"] = results.map(lambda x: x[0]).astype(float)
    pv_df["contrib_objects"] = results.map(lambda x: ";".join(map(str, x)) if x[1] else "")

    # --- factors & adjusted power
    pv_df["shading_factor"] = 1.0 - pv_df["shading_loss"]
    pv_df["P_actual_new_kW"] = pv_df["P_ac"] * pv_df["shading_factor"]

    # Energy (kWh). If you KNOW your data are hourly, this equals power.
    # If you prefer, compute dt in hours:
    #   dt_hr = pv_df["timestamp_local"].diff().dt.total_seconds().div(3600).fillna(1.0).clip(lower=0.0, upper=2.0)
    #   pv_df["E_kWh_new"] = pv_df["P_actual_new_kW"] * dt_hr
    pv_df["E_kWh_new"] = pv_df["P_actual_new_kW"]

    # --- output columns
    cols = [
        "timestamp_local", "solar_azimuth", "solar_zenith", "solar_elevation",
        "gti" if "gti" in pv_df.columns else None,
        "shading_loss", "shading_factor", "contrib_objects", "weather_type",
        "P_ac", "P_actual_new_kW", "E_kWh_new",
    ]
    cols = [c for c in cols if c is not None]

    # --- save
    pv_df.to_csv(out_path, index=False, columns=cols)
    print(f"Saved: {out_path}  (rows: {len(pv_df)})")

print("\n✅ Done. Check the 'shading' folder for one CSV per state.")
