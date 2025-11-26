#!/usr/bin/env python3
"""
compute_shading.py
 
Compute site-level shading impact on existing PV generation timeseries using nearby objects.
 
Usage:
    python compute_shading.py --pv pv_result.csv --objects state_object.csv --out shaded_output.csv
 
If --out is omitted the output will be written next to the PV input with suffix "_shaded.csv".
"""
 
from pathlib import Path
import argparse
import os
import math
import pandas as pd
import numpy as np
from math import atan, tan, radians, degrees
 
# ----------------------------
# Helpers
# ----------------------------
def build_obj_cache(obj_df: pd.DataFrame, margin_deg: float = 5.0):
    """
    Precompute object angular half-width plus margin and return list of dicts for fast iteration.
    Input obj_df must contain: object_id, azimuth_deg, height_m, distance_m, width_m, shading_intensity
    """
    cache = []
    for _, r in obj_df.iterrows():
        width = float(r["width_m"])
        dist = float(r["distance_m"])
        # avoid division by zero
        if dist <= 0:
            half_width_deg = 90.0
        else:
            half_width_deg = degrees(atan((width / 2.0) / max(dist, 1e-9)))
        cache.append({
            "object_id": str(r["object_id"]),
            "azimuth_deg": float(r["azimuth_deg"]) % 360.0,
            "height_m": float(r["height_m"]),
            "distance_m": float(dist),
            "shading_intensity": float(r["shading_intensity"]),
            "angular_half_width_plus_margin": float(half_width_deg + margin_deg),
            # store width if needed
            "width_m": float(width),
        })
    return cache
 
def angular_difference(a_deg, b_deg):
    """Return smallest absolute angular difference between two bearings (0..180)."""
    diff = abs((a_deg - b_deg + 180) % 360 - 180)
    return diff
 
def compute_shading_for_row(row: pd.Series, obj_cache: list, max_total: float = 0.95):
    """
    For one timestamp row, compute total shading and list of contributing object ids.
    Expects row to contain:
      - solar_azimuth (deg) OR azimuth
      - solar_elevation (deg) OR elevation OR solar_zenith (deg, then elevation = 90 - zenith)
      - P_ac (AC power) or P_dc
    Returns: (shading_loss (0..max_total), list_of_object_ids)
    """
    # pick azimuth value
    sun_az = None
    for cname in ("solar_azimuth", "azimuth", "sun_azimuth"):
        if cname in row and not pd.isna(row[cname]):
            sun_az = float(row[cname])
            break
    if sun_az is None:
        return 0.0, []
 
    # pick elevation
    solar_elev = None
    for cname in ("solar_elevation", "elevation", "solar_elevation_deg"):
        if cname in row and not pd.isna(row[cname]):
            solar_elev = float(row[cname])
            break
    # if only zenith is present, use 90 - zenith
    if solar_elev is None:
        if "solar_zenith" in row and not pd.isna(row["solar_zenith"]):
            solar_elev = 90.0 - float(row["solar_zenith"])
        elif "zenith" in row and not pd.isna(row["zenith"]):
            solar_elev = 90.0 - float(row["zenith"])
 
    # if sun is below horizon or elevation missing, no shading on PV (we treat it as zero)
    if solar_elev is None or solar_elev <= 0:
        return 0.0, []
 
    shaded_ids = []
    total = 0.0
 
    # For each object, check whether object is aligned and whether shadow length covers distance
    for obj in obj_cache:
        diff = angular_difference(sun_az, obj["azimuth_deg"])
        # if object is roughly in sun direction (within half width + margin)
        if diff <= obj["angular_half_width_plus_margin"]:
            # compute shadow length at ground given object height and sun elevation
            # avoid division by zero for near-zero elevation (sun at horizon -> very long shadow)
            tan_elev = math.tan(math.radians(solar_elev))
            if tan_elev <= 1e-6:
                shadow_length = float("inf")
            else:
                shadow_length = obj["height_m"] / tan_elev
            # if shadow length longer than distance to object, object will cast shadow onto site
            if shadow_length >= obj["distance_m"]:
                total += obj["shading_intensity"]
                shaded_ids.append(obj["object_id"])
 
    # clamp to max_total
    total = min(total, max_total)
    return float(total), shaded_ids
 
# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Compute shading-adjusted PV from PV timeseries + objects list.")
    p.add_argument("--pv", required=True, help="Path to PV timeseries CSV (your pv_result.csv)")
    p.add_argument("--objects", required=True, help="Path to objects CSV (your <state>_object.csv)")
    p.add_argument("--out", required=False, help="Output CSV path. If omitted writes <pv_basename>_shaded.csv")
    p.add_argument("--session", action="store_true", help="If --out not provided, use _session_shaded suffix")
    args = p.parse_args()
 
    pv_path = Path(args.pv)
    obj_path = Path(args.objects)
 
    if not pv_path.exists():
        raise SystemExit(f"PV file not found: {pv_path}")
    if not obj_path.exists():
        raise SystemExit(f"Object file not found: {obj_path}")
 
    # load
    pv_df = pd.read_csv(pv_path)
    obj_df = pd.read_csv(obj_path)
 
    # timestamps: accept timestamp_utc or timestamp_local -> normalize to pandas datetime
    if "timestamp_utc" in pv_df.columns:
        ts_col = "timestamp_utc"
    elif "timestamp_local" in pv_df.columns:
        ts_col = "timestamp_local"
    else:
        # try common alternatives
        possible = [c for c in pv_df.columns if "time" in c or "timestamp" in c]
        if possible:
            ts_col = possible[0]
        else:
            raise SystemExit("No timestamp column found in PV file (expected timestamp_utc or timestamp_local).")
 
    pv_df[ts_col] = pd.to_datetime(pv_df[ts_col], utc=True, errors="coerce")
 
    if pv_df[ts_col].isna().any():
        raise SystemExit("Some timestamps could not be parsed - check CSV timestamp format.")
 
    # normalize column names: ensure we have solar azimuth + elevation or zenith
    # Accept either 'azimuth' or 'solar_azimuth' for sun direction, 'elevation'/'solar_elevation'/'zenith'
    # If zenith present we will use elevation = 90 - zenith
    # If user has 'azimuth' in degrees but values outside 0..360 normalize them
    for az in ("solar_azimuth", "azimuth"):
        if az in pv_df.columns:
            pv_df["solar_azimuth"] = pv_df[az].astype(float) % 360.0
            break
 
    # elevation / zenith handling - keep both if present
    if "solar_elevation" not in pv_df.columns:
        if "elevation" in pv_df.columns:
            pv_df["solar_elevation"] = pv_df["elevation"].astype(float)
        elif "solar_zenith" in pv_df.columns:
            pv_df["solar_elevation"] = 90.0 - pv_df["solar_zenith"].astype(float)
        elif "zenith" in pv_df.columns:
            pv_df["solar_elevation"] = 90.0 - pv_df["zenith"].astype(float)
        else:
            # can't compute elevation -> try to compute from cos_theta/cosine geometry? if none, set NaN
            pv_df["solar_elevation"] = np.nan
 
    # ensure power columns exist; prefer P_ac, fallback P_dc etc.
    power_col = None
    for c in ("P_ac", "P_AC", "PAC", "P_dc", "P_DC", "P_DC_kW"):
        if c in pv_df.columns:
            power_col = c
            break
    if power_col is None:
        raise SystemExit("No power column found in PV file (expected P_ac or P_dc).")
 
    # Validate objects file columns
    needed_obj_cols = ["object_id", "width_m", "distance_m", "azimuth_deg", "height_m", "shading_intensity"]
    missing = [c for c in needed_obj_cols if c not in obj_df.columns]
    if missing:
        raise SystemExit(f"Missing columns in objects CSV: {missing}")
 
    # Build cache
    obj_cache = build_obj_cache(obj_df)
 
    # compute shading row-wise
    print("Computing shading for each row... (this may take a while for large files)")
    # apply row-wise
    def _row_apply(r):
        loss, ids = compute_shading_for_row(r, obj_cache)
        return pd.Series({"shading_loss": loss, "contrib_objects_list": ids})
 
    shading = pv_df.apply(_row_apply, axis=1)
 
    pv_df = pd.concat([pv_df, shading], axis=1)
 
    # stringify contributing objects
    pv_df["contrib_objects"] = pv_df["contrib_objects_list"].apply(lambda L: ";".join(L) if isinstance(L, list) and L else "")
 
    pv_df["shading_factor"] = 1.0 - pv_df["shading_loss"]
 
    # compute adjusted power
    # prefer P_ac if present, else P_dc
    pv_df["P_actual_new_kW"] = pv_df[power_col].astype(float) * pv_df["shading_factor"]
    # energy column E_kWh_new equal to power if data is hourly; otherwise user should adjust externally
    pv_df["E_kWh_new"] = pv_df["P_actual_new_kW"]
 
    # choose output path
    if args.out:
        out_path = Path(args.out)
    else:
        suffix = "_session_shaded.csv" if args.session else "_shaded.csv"
        out_path = pv_path.parent / (pv_path.stem + suffix)
 
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    # choose columns to write (keep many original columns + new ones)
    keep_cols = list(pv_df.columns)  # by default include all; you can restrict if needed
    # but ensure new cols are near the end in a nice order
    order_cols = [ts_col, "solar_azimuth", "solar_elevation", "solar_zenith" if "solar_zenith" in pv_df.columns else "zenith"]
    # filter existing
    order_cols = [c for c in order_cols if c in pv_df.columns]
    # append power & new fields
    extras = [power_col, "shading_loss", "shading_factor", "contrib_objects", "P_actual_new_kW", "E_kWh_new"]
    for c in extras:
        if c in pv_df.columns and c not in order_cols:
            order_cols.append(c)
    # finally add any remaining columns
    remaining = [c for c in pv_df.columns if c not in order_cols]
    final_cols = order_cols + remaining
 
    pv_df.to_csv(out_path, index=False, columns=final_cols)
    print(f"Saved shaded results to: {out_path}  (rows: {len(pv_df)})")
 
if __name__ == "__main__":
    main()