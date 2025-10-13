import os
import glob
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from math import atan, tan, radians, degrees

# ----------------------------
# Defaults (used if --base/--pv/--objects not provided)
# ----------------------------
DEFAULT_BASE = "/workspaces/MAD--PV-Generation"
PV_DIR_DEFAULT  = f"{DEFAULT_BASE}/pv_generation"
OBJ_DIR_DEFAULT = f"{DEFAULT_BASE}/object"
OUT_DIR_DEFAULT = f"{DEFAULT_BASE}/shading"

# ----------------------------
# Helpers to derive state key from filenames
# pv files: pv_generation_<state>.csv
# obj files: <state>_object.csv
# ----------------------------
def state_key_from_pv(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("pv_generation_", "", 1)

def state_key_from_obj(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("_object", "", 1)

def discover_by_state(base_dir: Path, state: str) -> tuple[Path, Path, Path]:
    """Return (pv_csv, obj_csv, out_dir) using a conventional layout under base_dir."""
    pv_dir  = base_dir / "pv_generation"
    obj_dir = base_dir / "object"
    out_dir = base_dir / "shading"
    pv_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    pv_csv  = pv_dir  / f"pv_generation_{state}.csv"
    obj_csv = obj_dir / f"{state}_object.csv"
    return pv_csv, obj_csv, out_dir

def compute_shading_for_row(row, obj_df_cached):
    """
    Returns (total_shading, contrib_list)
    total_shading in [0..0.95], contrib_list = list of object_ids
    """
    sun_az = row.get("solar_azimuth", np.nan)
    solar_zen = row.get("solar_zenith", np.nan)
    gti = row.get("gti", np.nan)

    # derive elevation if not precomputed
    solar_elev = row.get("solar_elevation", None)
    if solar_elev is None or pd.isna(solar_elev):
        if pd.isna(solar_zen):
            return 0.0, []
        solar_elev = 90 - float(solar_zen)

    if pd.isna(solar_elev) or solar_elev <= 0:
        return 0.0, []
    if pd.isna(gti) or gti == 0:
        # if you want to allow shading even when gti is NaN, remove this guard
        return 0.0, []

    shaded_ids = []
    total = 0.0

    for obj in obj_df_cached:
        # azimuth difference (0..180)
        diff = abs(float(sun_az) - obj["azimuth_deg"])
        if diff > 180:
            diff = 360 - diff

        # object roughly aligned with the sun?
        if diff <= obj["angular_half_width_plus_margin"]:
            # shadow length at ground
            sl = obj["height_m"] / max(tan(radians(solar_elev)), 1e-6)
            if sl >= obj["distance_m"]:
                total += obj["shading_intensity"]
                shaded_ids.append(obj["object_id"])

    return min(total, 0.95), shaded_ids

def build_obj_cache(obj_df: pd.DataFrame):
    """Precompute angular half-width (+5° margin) and other fields."""
    cache = []
    for _, r in obj_df.iterrows():
        width = float(r["width_m"])
        dist  = float(r["distance_m"])
        half_width_deg = degrees(atan((width / 2.0) / max(dist, 1e-6)))
        cache.append({
            "object_id": str(r["object_id"]),
            "azimuth_deg": float(r["azimuth_deg"]),
            "height_m": float(r["height_m"]),
            "distance_m": float(dist),
            "shading_intensity": float(r["shading_intensity"]),
            "angular_half_width_plus_margin": float(half_width_deg + 5.0),
        })
    return cache

def main():
    parser = argparse.ArgumentParser(description="Compute shading-adjusted PV from PV + object CSVs.")
    parser.add_argument("--state", help="State key to process (e.g. 'texas')", required=False)
    parser.add_argument("--base", help="Base folder containing pv_generation/, object/, shading/", required=False)
    parser.add_argument("--pv", help="Path to pv_generation_<state>.csv", required=False)
    parser.add_argument("--objects", help="Path to <state>_object.csv", required=False)
    parser.add_argument("--out", help="Output CSV path (if omitted, will write under shading/)", required=False)
    parser.add_argument("--session", action="store_true", help="When --out not provided, write *_session.csv")
    args = parser.parse_args()

    # Resolve base and default dirs
    if args.base:
        base_dir = Path(args.base).resolve()
    else:
        base_dir = Path(DEFAULT_BASE).resolve()

    pv_dir  = Path(PV_DIR_DEFAULT if not args.base else base_dir / "pv_generation")
    obj_dir = Path(OBJ_DIR_DEFAULT if not args.base else base_dir / "object")
    out_dir = Path(OUT_DIR_DEFAULT if not args.base else base_dir / "shading")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine state & input files
    state = args.state

    # If explicit pv/objects paths given, use them
    pv_csv  = Path(args.pv).resolve() if args.pv else None
    obj_csv = Path(args.objects).resolve() if args.objects else None

    # Infer state from explicit paths if needed
    if state is None:
        if pv_csv is not None:
            state = state_key_from_pv(str(pv_csv))
        elif obj_csv is not None:
            state = state_key_from_obj(str(obj_csv))

    # If still no state, but base provided or defaults present, try to discover one state (or error)
    if state is None:
        # If neither file is given, but we’re being called from UI, we should error clearly.
        raise SystemExit("No --state provided and could not infer from --pv/--objects paths.")

    # If paths not provided, discover from conventional layout
    if pv_csv is None or obj_csv is None:
        pv_csv2, obj_csv2, out_dir2 = discover_by_state(base_dir, state)
        if pv_csv is None:
            pv_csv = pv_csv2
        if obj_csv is None:
            obj_csv = obj_csv2
        # out_dir is still out_dir (from args/base), we only use out_dir2 for path idea

    if not pv_csv.exists():
        raise SystemExit(f"PV file not found: {pv_csv}")
    if not obj_csv.exists():
        raise SystemExit(f"Object file not found: {obj_csv}")

    # Determine output path
    if args.out:
        out_csv = Path(args.out).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        # default naming under shading/
        suffix = "_session.csv" if args.session else ".csv"
        out_csv = out_dir / f"{state}_pv_result{suffix}"

    print(f"=== Processing state: {state} ===")
    print(f"PV:   {pv_csv}")
    print(f"OBJ:  {obj_csv}")
    print(f"OUT:  {out_csv}")

    # ---------- Load inputs ----------
    pv_df  = pd.read_csv(pv_csv)
    obj_df = pd.read_csv(obj_csv)

    # Timestamp
    if "timestamp_local" not in pv_df.columns:
        raise SystemExit(f"'timestamp_local' column not found in {pv_csv}")
    pv_df["timestamp_local"] = pd.to_datetime(pv_df["timestamp_local"], errors="coerce")

    # Required PV columns
    needed_pv_cols = ["solar_azimuth", "solar_zenith", "P_ac"]
    missing = [c for c in needed_pv_cols if c not in pv_df.columns]
    if missing:
        raise SystemExit(f"Missing PV columns in {pv_csv}: {missing}")

    # Required object columns
    needed_obj_cols = ["object_id", "width_m", "distance_m", "azimuth_deg", "height_m", "shading_intensity"]
    missing = [c for c in needed_obj_cols if c not in obj_df.columns]
    if missing:
        raise SystemExit(f"Missing object columns in {obj_csv}: {missing}")

    # Solar elevation
    if "solar_elevation" not in pv_df.columns:
        pv_df["solar_elevation"] = 90 - pv_df["solar_zenith"]

    # Build object cache
    obj_cache = build_obj_cache(obj_df)

    # Compute shading
    results = pv_df.apply(lambda r: compute_shading_for_row(r, obj_cache), axis=1)
    pv_df["shading_loss"] = results.map(lambda x: x[0]).astype(float)
    pv_df["contrib_objects"] = results.map(lambda x: ";".join(map(str, x)) if x[1] else "")

    # Factors & adjusted power
    pv_df["shading_factor"] = 1.0 - pv_df["shading_loss"]
    pv_df["P_actual_new_kW"] = pv_df["P_ac"] * pv_df["shading_factor"]

    # Energy (kWh). If you know your data are hourly, this equals power.
    pv_df["E_kWh_new"] = pv_df["P_actual_new_kW"]

    # Output columns
    cols = [
        "timestamp_local", "solar_azimuth", "solar_zenith", "solar_elevation",
        "gti" if "gti" in pv_df.columns else None,
        "weather_type" if "weather_type" in pv_df.columns else None,
        "shading_loss", "shading_factor", "contrib_objects",
        "P_ac", "P_actual_new_kW", "E_KWh_new" if "E_KWh_new" in pv_df.columns else None,  # keep existing if present
        "E_kWh_new",
    ]
    cols = [c for c in cols if c is not None and c in pv_df.columns]

    # Save
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pv_df.to_csv(out_csv, index=False, columns=cols)
    print(f"Saved: {out_csv}  (rows: {len(pv_df)})")
    print("✅ Done.")

if __name__ == "__main__":
    main()
