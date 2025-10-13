import os
import sys
import time
import math
import uuid
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from streamlit_folium import st_folium
import folium

from data_loader import load_state, get_states_and_base, build_paths, resolve_base
from kpi_utils import annualize_monthly, shading_kpis

# ---------------------------------------------------------------------
# Streamlit Page
# ---------------------------------------------------------------------
st.set_page_config(page_title="PV + Shading + Tariff Explorer", layout="wide")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _secrets_optional():
    try:
        return st.secrets
    except Exception:
        return None

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    df.columns = [str(c).strip() for c in df.columns]
    return df

def guess_time_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(k in lc for k in ("timestamp", "time", "date", "datetime")):
            return c
    return None

def parse_ts_column(df: pd.DataFrame, col: str | None):
    if df is None or col is None or col not in df.columns:
        return None
    s = df[col]

    # Already datetime?
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            return s.dt.tz_convert(None)
        except Exception:
            try:
                return s.dt.tz_localize(None)
            except Exception:
                return pd.to_datetime(s, errors="coerce")

    s_str = s.astype(str).str.strip().replace({"": np.nan})

    ts = pd.to_datetime(s_str, errors="coerce", utc=False, infer_datetime_format=True)
    if pd.api.types.is_datetime64_any_dtype(ts) and ts.notna().any():
        return ts

    ts2 = pd.to_datetime(s_str.str.replace("Z", "", regex=False), errors="coerce", utc=True)
    if ts2.notna().any():
        try:
            return ts2.dt.tz_convert(None)
        except Exception:
            return ts2.dt.tz_localize(None)

    num = pd.to_numeric(s_str, errors="coerce")
    if num.notna().any():
        median = num.dropna().median()
        if median > 1e11:
            ts3 = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
        else:
            ts3 = pd.to_datetime(num, unit="s", errors="coerce", utc=True)
        if ts3.notna().any():
            try:
                return ts3.dt.tz_convert(None)
            except Exception:
                return ts3.dt.tz_localize(None)
    return None

# --- geo helpers ------------------------------------------------------
def meters_to_latlon_offsets(lat_deg: float, dx_m: float, dy_m: float):
    lat_rad = math.radians(lat_deg)
    dlat = dy_m / 111_320.0
    dlon = dx_m / (111_320.0 * math.cos(lat_rad) if abs(math.cos(lat_rad)) > 1e-6 else 1e-6)
    return dlat, dlon

def rotate_xy_clockwise(x_m: float, y_m: float, azimuth_deg: float):
    th = math.radians(azimuth_deg)
    xr = x_m * math.cos(th) + y_m * math.sin(th)
    yr = -x_m * math.sin(th) + y_m * math.cos(th)
    return xr, yr

def make_pv_polygon(lat_center: float, lon_center: float, length_m: float = 10.0, width_m: float = 5.0, azimuth_deg: float = 180.0):
    L = length_m / 2.0
    W = width_m / 2.0
    corners_xy = [(-W, -L), ( W, -L), ( W,  L), (-W,  L)]
    ring = []
    for (x, y) in corners_xy:
        xr, yr = rotate_xy_clockwise(x, y, azimuth_deg)
        dlat, dlon = meters_to_latlon_offsets(lat_center, xr, yr)
        ring.append([lon_center + dlon, lat_center + dlat])
    ring.append(ring[0])
    return ring

# ---------------------------------------------------------------------
# PV centers per site (lat, lon, tz optional)
# ---------------------------------------------------------------------
SITE_CENTERS = {
    "california":        (37.390026, -122.08123,  "America/Los_Angeles"),
    "north_carolinas":   (35.759573,  -79.019300, "America/New_York"),
    "texas":             (29.760077,  -95.370111, "America/Chicago"),
    "north_dakota":      (46.539175, -102.868223, "America/Chicago"),
    "colorado":          (39.306108, -102.269356, "America/Denver"),
    "michigan":          (45.421402,  -83.81833,  "America/Detroit"),
    "maine":             (44.952297,  -67.660831, "America/New_York"),
    "washington":        (47.606139, -122.332848, "America/Los_Angeles"),
    "missouri":          (36.083959,  -89.829251, "America/Chicago"),
    "nevada":            (41.947679, -116.098709, "America/Los_Angeles"),
    "florida":           (25.761680,  -80.191179, "America/New_York"),
}

# ---------------------------------------------------------------------
# Session keys (per state)
# ---------------------------------------------------------------------
def _objs_key(state):        return f"objs_df_{state}"
def _objs_dirty_key(state):  return f"objs_dirty_{state}"
def _obj_tmp_key(state):     return f"obj_tmp_path_{state}"
def _shad_df_key(state):     return f"shading_df_session_{state}"
def _shad_path_key(state):   return f"shading_tmp_path_{state}"
def _pv_df_key(state):       return f"pv_df_session_{state}"
def _pv_path_key(state):     return f"pv_tmp_path_{state}"

def _normalize_objects_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["object_id","object_type","latitude","longitude","height_m","shading_intensity"])
    df = df.copy()
    for col in ["latitude","longitude","height_m","shading_intensity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "object_id" in df.columns:
        df["object_id"] = df["object_id"].astype(str)
    return df

def _find_shading_script() -> Path | None:
    here = Path(__file__).resolve().parent
    cands = [
        here / "shading.py",
        here.parent / "shading.py",
        here.parent / "main" / "shading.py",
        here.parent.parent / "shading.py",
    ]
    for p in cands:
        if p.exists():
            return p
    return None

# ---- robust runner: returns actual shading + (optional) pv file it finds ----
def run_shading_for_state(state: str, base_dir: Path, objects_csv: Path, pv_csv: Path, out_csv: Path):
    """
    Try multiple CLI patterns. If expected out_csv isn't found, search likely dirs.
    Return: (ok: bool, message: str, shading_path: Path|None, pv_path: Path|None)
    """
    script = _find_shading_script()
    if script is None:
        return False, "shading.py not found. Place it next to this app or at the repo root.", None, None

    out_csv = Path(out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    search_dirs = [
        out_csv.parent,
        script.parent,
        Path(base_dir) / "shading",
        Path(base_dir) / "pv_generation",
        Path(base_dir),
    ]

    attempts = [
        [sys.executable, str(script), "--state", state, "--objects", str(objects_csv), "--pv", str(pv_csv), "--out", str(out_csv)],
        [sys.executable, str(script), "--base", str(base_dir), "--state", state, "--out", str(out_csv)],
    ]

    def _find_outputs(since_ts: float):
        cand_sh, cand_pv = [], []
        for d in search_dirs:
            if not d.exists():
                continue
            for p in d.glob("*.csv"):
                try:
                    if p.stat().st_mtime >= since_ts - 1:
                        name = p.name.lower()
                        if state.lower() in name:
                            score = 0
                            if "session" in name: score += 3
                            if "result"  in name: score += 2
                            if "shade"   in name or "shad" in name: score += 2
                            if "pv"      in name: score += 1
                            tup = (p.stat().st_mtime, score, p)
                            if any(k in name for k in ["shad", "shade", "result"]) and "pv_generation" not in name:
                                cand_sh.append(tup)
                            if any(k in name for k in ["pv", "pv_gen", "pv_generation"]):
                                cand_pv.append(tup)
                except Exception:
                    pass

        def _pick(cands):
            if not cands: return None
            cands.sort(key=lambda t: (t[1], t[0]), reverse=True)
            return cands[0][2]

        return _pick(cand_sh), _pick(cand_pv)

    tried = []
    start_all = time.time()
    for args in attempts:
        start = time.time()
        try:
            cp = subprocess.run(args, capture_output=True, text=True, timeout=900, cwd=str(script.parent))
            dur = time.time() - start
            tried.append(f"TRIED: {' '.join(args)}\n -> exit {cp.returncode} in {dur:.1f}s\nSTDERR:\n{cp.stderr or '(empty)'}\nSTDOUT:\n{cp.stdout or '(empty)'}")
            if cp.returncode == 0:
                if out_csv.exists():
                    sh_found, pv_found = out_csv, None
                    _, pv_found = _find_outputs(start)
                    return True, f"shading.py OK in {dur:.1f}s (wrote expected shading).", sh_found, pv_found
                sh_found, pv_found = _find_outputs(start)
                if sh_found or pv_found:
                    return True, f"shading.py OK in {dur:.1f}s (found outputs).", sh_found, pv_found
        except Exception as e:
            tried.append(f"TRIED: {' '.join(args)}\n -> Exception: {e}")

    sh_found, pv_found = _find_outputs(start_all)
    if sh_found or pv_found:
        return True, "shading.py appears to have produced files (found via search).", sh_found, pv_found

    return False, "All CLI patterns failed or no output CSV was created.\n\n" + "\n\n".join(tried), None, None

def make_bill_waterfall(baseline: float, buy: float, sell: float, net: float) -> go.Figure:
    vals = [baseline or 0.0, -(buy or 0.0), -(sell or 0.0), net or 0.0]
    measures = ["absolute", "relative", "relative", "total"]
    fig = go.Figure(go.Waterfall(
        name="Bill", orientation="v", measure=measures,
        x=["Baseline (no PV)", "Energy bought", "Export credit", "Net bill"],
        y=vals, connector={"line": {"dash": "dot", "width": 1}},
        text=[f"{v:,.0f}" for v in vals], textposition="outside"
    ))
    fig.update_layout(title="Annual Bill Breakdown (Estimated)", showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig

# ---------------------------------------------------------------------
# Discover + load data
# ---------------------------------------------------------------------
STATES, DATA_BASE = get_states_and_base(_secrets_optional())
if not STATES:
    st.error("No datasets found. Set PV_DATA_BASE env var or place this app next to your 'main' data folder.")
    st.stop()

# Sidebar controls
st.sidebar.title("Controls")
default_idx = STATES.index("texas") if "texas" in STATES else 0
state = st.sidebar.selectbox("Site / State", STATES, index=min(default_idx, len(STATES)-1))
show_map = st.sidebar.checkbox("Show Map", value=True)
show_uncertainty = st.sidebar.checkbox("Show Uncertainty Bands (if available)", value=False)

# Load data for selected state
data, debug = load_state(state, _secrets_optional())

# Normalize columns
for k in ("monthly_net", "hourly_net", "pv_generation", "objects", "shading"):
    if data.get(k) is not None:
        data[k] = normalize_cols(data[k])

df_month        = data["monthly_net"]
df_hour         = data["hourly_net"]
df_pv_original  = data["pv_generation"]
df_shad_original= data["shading"]
df_objs_original= _normalize_objects_df(data["objects"])

# Session-scoped objects (start from ORIGINAL, but never overwrite originals)
objs_session_key  = _objs_key(state)
dirty_session_key = _objs_dirty_key(state)
obj_tmp_key       = _obj_tmp_key(state)

if objs_session_key not in st.session_state:
    st.session_state[objs_session_key] = df_objs_original.copy()
if dirty_session_key not in st.session_state:
    st.session_state[dirty_session_key] = False
if obj_tmp_key not in st.session_state:
    st.session_state[obj_tmp_key] = None

df_objs = st.session_state[objs_session_key]

# Session-scoped shading & pv (ACTIVE overrides)
shad_df_key  = _shad_df_key(state)
shad_tmp_key = _shad_path_key(state)
pv_df_key    = _pv_df_key(state)
pv_tmp_key   = _pv_path_key(state)
for k in (shad_df_key, pv_df_key, shad_tmp_key, pv_tmp_key):
    if k not in st.session_state:
        st.session_state[k] = None

df_shad_active = st.session_state[shad_df_key] if st.session_state[shad_df_key] is not None else df_shad_original
df_pv_active   = st.session_state[pv_df_key]   if st.session_state[pv_df_key]   is not None else df_pv_original

with st.sidebar.expander("Data paths (debug)", expanded=False):
    st.write(f"Base: {debug['base']}")
    st.json(debug["exists"])
    if st.session_state[obj_tmp_key]:
        st.caption(f"Session object file: {st.session_state[obj_tmp_key]}")
    if st.session_state[shad_tmp_key]:
        st.caption(f"Session shading file: {st.session_state[shad_tmp_key]}")
    if st.session_state[pv_tmp_key]:
        st.caption(f"Session PV file: {st.session_state[pv_tmp_key]}")

# Advanced: choose timestamp columns (ACTIVE shading / ACTIVE pv)
with st.sidebar.expander("Advanced • Timestamp columns", expanded=False):
    mon_guess = guess_time_col(df_month) if df_month is not None else None
    hr_guess  = guess_time_col(df_hour) if df_hour is not None else None
    sh_guess  = guess_time_col(df_shad_active) if df_shad_active is not None else None
    pv_guess  = guess_time_col(df_pv_active) if df_pv_active is not None else None

    mon_col = st.selectbox("Monthly time column",
        options=(list(df_month.columns) if df_month is not None else []),
        index=(list(df_month.columns).index(mon_guess) if (df_month is not None and mon_guess in df_month.columns) else 0) if (df_month is not None and len(df_month.columns)>0) else None,
        key="mon_ts_col"
    ) if df_month is not None else None

    hr_col = st.selectbox("Hourly time column",
        options=(list(df_hour.columns) if df_hour is not None else []),
        index=(list(df_hour.columns).index(hr_guess) if (df_hour is not None and hr_guess in df_hour.columns) else 0) if (df_hour is not None and len(df_hour.columns)>0) else None,
        key="hr_ts_col"
    ) if df_hour is not None else None

    sh_col = st.selectbox("Shading time column (ACTIVE)",
        options=(list(df_shad_active.columns) if df_shad_active is not None else []),
        index=(list(df_shad_active.columns).index(sh_guess) if (df_shad_active is not None and sh_guess in df_shad_active.columns) else 0) if (df_shad_active is not None and len(df_shad_active.columns)>0) else None,
        key="sh_ts_col"
    ) if df_shad_active is not None else None

    pv_col = st.selectbox("PV generation time column (ACTIVE)",
        options=(list(df_pv_active.columns) if df_pv_active is not None else []),
        index=(list(df_pv_active.columns).index(pv_guess) if (df_pv_active is not None and pv_guess in df_pv_active.columns) else 0) if (df_pv_active is not None and len(df_pv_active.columns)>0) else None,
        key="pv_ts_col"
    ) if df_pv_active is not None else None

# Parse timestamps
ts_month = parse_ts_column(df_month, mon_col) if df_month is not None else None
ts_hour  = parse_ts_column(df_hour,  hr_col)  if df_hour  is not None else None
ts_shad  = parse_ts_column(df_shad_active,  sh_col) if df_shad_active is not None else None
ts_pv    = parse_ts_column(df_pv_active,    pv_col) if df_pv_active   is not None else None

# ---------------------------------------------------------------------
# Sidebar session actions: Recompute + End session
# ---------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Session / Recompute")

def _session_paths(base, state):
    paths  = build_paths(base, state)
    obj_orig = paths["objects"]
    pv_orig  = paths["pv_generation"]
    sh_orig  = paths["shading"]
    # session file locations
    obj_sess = obj_orig.with_name(obj_orig.stem + "_session.csv")     # .../object/<state>_object_session.csv
    sh_sess  = sh_orig.with_name(sh_orig.stem + "_session.csv")       # .../shading/<state>_pv_result_session.csv
    pv_sess  = pv_orig.with_name(pv_orig.stem + "_session.csv")       # .../pv_generation/pv_generation_<state>_session.csv (if produced)
    return obj_orig, pv_orig, sh_orig, obj_sess, pv_sess, sh_sess

if st.sidebar.button("Recompute shading now"):
    base = resolve_base(_secrets_optional())
    obj_orig, pv_orig, sh_orig, obj_sess, pv_sess, sh_sess = _session_paths(Path(base), state)

    # 1) Write **SESSION OBJECTS** file only (do not touch originals)
    try:
        obj_sess.parent.mkdir(parents=True, exist_ok=True)
        df_objs.to_csv(obj_sess, index=False)
        st.session_state[obj_tmp_key] = str(obj_sess)
    except Exception as e:
        st.error(f"Could not write session objects CSV:\n{e}")
    # 2) Call shading.py using session objects; write shading to session output
    ok, msg, sh_actual, pv_actual = run_shading_for_state(
        state=state, base_dir=Path(base),
        objects_csv=Path(obj_sess),                 # <-- session objects!
        pv_csv=Path(pv_orig),                       # PV original (or provide pv_sess if you also session PV)
        out_csv=Path(sh_sess),                      # shading session path
    )
    if not ok:
        st.error(msg)
    else:
        st.success(msg)
        # Load shading session
        try:
            target_sh = Path(sh_actual or sh_sess)
            df_new_sh = pd.read_csv(target_sh)
            st.session_state[shad_df_key]  = normalize_cols(df_new_sh)
            st.session_state[shad_tmp_key] = str(target_sh)
        except Exception as e:
            st.error(f"Recompute OK, but failed to read shading CSV:\n{e}")

        # Load PV session (if produced)
        try:
            if pv_actual:
                target_pv = Path(pv_actual)
            elif Path(pv_sess).exists():
                target_pv = pv_sess
            else:
                target_pv = None
            if target_pv:
                df_new_pv = pd.read_csv(target_pv)
                st.session_state[pv_df_key]  = normalize_cols(df_new_pv)
                st.session_state[pv_tmp_key] = str(target_pv)
                st.toast(f"Loaded session PV: {Path(target_pv).name}")
        except Exception as e:
            st.warning(f"No PV session file loaded: {e}")

        st.toast("Session data active.")
        st.rerun()

if st.sidebar.button("End session (clean temp)"):
    # Delete session shading + PV + OBJECT files if they exist
    for key in (shad_tmp_key, pv_tmp_key, obj_tmp_key):
        tmp_path = st.session_state.get(key)
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            st.warning(f"Could not delete temp file {tmp_path}: {e}")

    # Clear session overrides & dirty flags for this state
    for k in (_objs_key(state), _objs_dirty_key(state),
              _shad_df_key(state), _shad_path_key(state),
              _pv_df_key(state), _pv_path_key(state),
              _obj_tmp_key(state)):
        if k in st.session_state:
            del st.session_state[k]

    st.toast("Session cleared; reloading original datasets.")
    st.rerun()

# ---------------------------------------------------------------------
# KPIs (use ACTIVE shading df)
# ---------------------------------------------------------------------
ak = annualize_monthly(df_month)
sk = shading_kpis(df_shad_active)

# Fallback KPIs from hourly if monthly is missing
if (not ak) and (df_hour is not None) and (len(df_hour) > 0):
    load_kwh   = float(pd.to_numeric(df_hour.get("Load_E_kWh", df_hour.get("Load_kW", 0)), errors="coerce").fillna(0).sum())
    pv_kwh     = float(pd.to_numeric(df_hour.get("PV_E_kWh",   df_hour.get("PV_P_kW", 0)), errors="coerce").fillna(0).sum())
    export_kwh = float(pd.to_numeric(df_hour.get("Export_kWh", 0), errors="coerce").fillna(0).sum())
    self_consumed = max(pv_kwh - export_kwh, 0.0)
    ak = {
        "annual_pv_kwh": pv_kwh,
        "annual_load_kwh": load_kwh,
        "annual_import_kwh": float(pd.to_numeric(df_hour.get("Import_kWh", 0), errors="coerce").fillna(0).sum()),
        "annual_export_kwh": export_kwh,
        "net_bill": 0.0,
        "baseline_bill_est": 0.0,
        "savings_est": 0.0,
        "self_consumption_pct": (self_consumed / pv_kwh * 100) if pv_kwh > 0 else 0.0,
        "self_sufficiency_pct": (self_consumed / load_kwh * 100) if load_kwh > 0 else 0.0,
        "avg_buy_rate": 0.0,
    }

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Annual PV (kWh)", f"{ak.get('annual_pv_kwh',0):,.0f}")
c2.metric("Annual Load (kWh)", f"{ak.get('annual_load_kwh',0):,.0f}")
c3.metric("Self-consumption (%)", f"{ak.get('self_consumption_pct',0):.1f}%")
c4.metric("Savings (est)", f"${ak.get('savings_est',0):,.0f}")
c5.metric("Avg Shading Loss", (f"{sk.get('avg_shading_loss_pct',0):.1f}%" if sk.get('avg_shading_loss_pct') is not None else "—"))

st.markdown("---")

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_overview, tab_map, tab_energy, tab_bill, tab_exports, tab_data = st.tabs(
    ["Overview", "Map & Shading", "Energy Profiles", "Billing", "Export/Import", "Data"]
)

# -------------------- Overview --------------------------
with tab_overview:
    st.subheader(f"Summary for {state.title()}")
    if df_month is not None and not df_month.empty:
        m = df_month.copy()
        tsm = parse_ts_column(m, guess_time_col(m)) if 'mon_ts_col' not in st.session_state else ts_month
        if tsm is None:
            st.warning("Could not parse a timestamp column in monthly data. Choose it in the sidebar (Advanced • Timestamp columns).")
        else:
            m["month"] = tsm.dt.strftime("%Y-%m")
            ycols = [c for c in ["PV_E_kWh", "Load_E_kWh"] if c in m.columns]
            if ycols:
                fig = px.bar(m, x="month", y=ycols, barmode="group", title="Monthly Energy")
                st.plotly_chart(fig, use_container_width=True)

            buy  = pd.to_numeric(m.get("Buy_$", 0), errors="coerce").fillna(0).sum()
            sell = pd.to_numeric(m.get("Sell_$", 0), errors="coerce").fillna(0).sum()
            net  = pd.to_numeric(m.get("NetBill_$", 0), errors="coerce").fillna(0).sum()
            baseline = ak.get("baseline_bill_est", 0.0)
            figw = make_bill_waterfall(baseline, buy, sell, net)
            st.plotly_chart(figw, use_container_width=True)
    else:
        st.info("Monthly net metering file not found for this state.")

# -------------------- Map & Shading ---------------------
with tab_map:
    st.subheader("PV Site & Shading Objects")

    # Center
    if state in SITE_CENTERS:
        center_lat, center_lon, _tz = SITE_CENTERS[state]
    else:
        if not df_objs.empty:
            center_lat = df_objs["latitude"].dropna().mean()
            center_lon = df_objs["longitude"].dropna().mean()
        else:
            center_lat, center_lon = 37.3894, -122.0839

    with st.expander("PV array settings", expanded=False):
        colA, colB, colC = st.columns(3)
        pv_length_m = colA.number_input("Array length (m)", min_value=2.0, max_value=200.0, value=12.0, step=1.0)
        pv_width_m  = colB.number_input("Array width (m)",  min_value=1.0, max_value=200.0, value=6.0,  step=1.0)
        pv_az_deg   = colC.number_input("Azimuth (°)", min_value=0.0, max_value=359.0, value=180.0, step=1.0,
                                        help="0=N, 90=E, 180=S, 270=W")

    pv_ring = make_pv_polygon(center_lat, center_lon, length_m=pv_length_m, width_m=pv_width_m, azimuth_deg=pv_az_deg)
    mode = st.radio("Mode", ["View", "Edit / Add"], horizontal=True)

    if st.session_state[dirty_session_key]:
        st.markdown("<span style='background:#fff3cd;color:#664d03;padding:4px 8px;border:1px solid #ffe69c;border-radius:6px;'>Unsaved changes (session)</span>", unsafe_allow_html=True)

    if mode == "View":
        layers = []
        pv_feature = [{
            "type": "Feature",
            "properties": {"name": "PV Array", "length_m": pv_length_m, "width_m": pv_width_m, "azimuth_deg": pv_az_deg},
            "geometry": {"type": "Polygon", "coordinates": [pv_ring]},
        }]
        layers.append(
            pdk.Layer(
                "PolygonLayer", pv_feature,
                get_polygon="geometry.coordinates[0]",
                stroked=True, filled=True,
                get_line_color=[0,180,255], line_width_min_pixels=2,
                get_fill_color=[0,180,255,60], pickable=True,
            )
        )
        if not df_objs.empty and {"latitude","longitude"}.issubset(df_objs.columns):
            dfv = df_objs.copy()
            type_colors = {"Pole":[255,180,0], "Building":[255,99,132], "Tree":[50,205,50]}
            dfv["__fill__"] = dfv.get("object_type", pd.Series(index=dfv.index)).map(type_colors)
            dfv["__fill__"] = dfv["__fill__"].apply(lambda v: v if isinstance(v, (list, tuple, np.ndarray)) else [200,200,200])
            r = pd.to_numeric(dfv.get("height_m", 3.0), errors="coerce").fillna(3.0).clip(1, 20)
            dfv["__radius__"] = (r * 1.0).astype(float)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer", dfv,
                    get_position=["longitude","latitude"], radius_units="meters",
                    get_radius="__radius__", get_fill_color="__fill__",
                    get_line_color=[0,0,0,160], line_width_min_pixels=1, pickable=True,
                )
            )
        deck = pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=18, pitch=45),
            layers=layers,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            tooltip={"html":"<b>{name}</b><br/>Type:{object_type}<br/>Height:{height_m}m"}
        )
        st.pydeck_chart(deck)

        legend_items = [("PV Array","#00B4FF"), ("Pole","#FFB400"), ("Building","#FF6384"), ("Tree","#32CD32")]
        legend_html = """
        <style>.legend-box{display:flex;gap:10px;background:rgba(0,0,0,0.55);border:1px solid rgba(255,255,255,0.2);
        padding:10px 12px;border-radius:8px;width:fit-content;margin-top:8px;}
        .legend-item{display:flex;align-items:center;gap:8px;color:#fff;font-size:0.9rem;}
        .legend-swatch{width:14px;height:14px;border-radius:3px;border:1px solid rgba(255,255,255,0.6);}</style>
        <div class="legend-box">%s</div>
        """ % ("".join(f'<div class="legend-item"><span class="legend-swatch" style="background:{c}"></span>{n}</div>' for n,c in legend_items))
        st.markdown(legend_html, unsafe_allow_html=True)

    else:
        # Folium edit/add UI
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=19, tiles="CartoDB positron")
        folium.Polygon(
            locations=[[lat, lon] for lon, lat in pv_ring],
            color="#00B4FF", weight=2, fill=True, fill_opacity=0.25, tooltip="PV Array"
        ).add_to(fmap)

        if not df_objs.empty:
            for _, row in df_objs.iterrows():
                oid = str(row.get("object_id", ""))
                ot  = str(row.get("object_type", "Unknown"))
                lat = float(row.get("latitude", center_lat))
                lon = float(row.get("longitude", center_lon))
                folium.Marker(
                    location=[lat, lon],
                    tooltip=f"{ot} ({oid})" if oid else ot,
                    popup=folium.Popup(html=f"{oid}" if oid else ot, max_width=200),
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(fmap)

        fol_ret = st_folium(fmap, height=520, width=None, returned_objects=["last_clicked"], key=f"folium_click_{state}")
        clicked = (fol_ret or {}).get("last_clicked")

        st.markdown("#### Edit or add objects")
        c1, c2, c3 = st.columns([1,1,1])

        # ADD
        with c1.expander("Add new object", expanded=False):
            if clicked:
                st.info(f"Clicked lat/lon: {clicked['lat']:.7f}, {clicked['lng']:.7f}")
            new_oid = st.text_input("Object ID", value=f"OBJ_{uuid.uuid4().hex[:6].upper()}")
            new_type = st.selectbox("Type", ["Pole", "Building", "Tree", "Other"], index=0)
            new_height = st.number_input("Height (m)", min_value=0.0, value=3.0, step=0.5)
            new_intensity = st.slider("Shading intensity (0-1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            new_lat = st.number_input("Latitude", value=(clicked["lat"] if clicked else center_lat), format="%.7f")
            new_lon = st.number_input("Longitude", value=(clicked["lng"] if clicked else center_lon), format="%.7f")
            if st.button("Add object", key="btn_add_obj"):
                new_row = {
                    "object_id": str(new_oid),
                    "object_type": new_type,
                    "latitude": float(new_lat),
                    "longitude": float(new_lon),
                    "height_m": float(new_height),
                    "shading_intensity": float(new_intensity),
                }
                st.session_state[objs_session_key] = pd.concat([df_objs, pd.DataFrame([new_row])], ignore_index=True)
                st.session_state[dirty_session_key] = True
                st.success("Added in session. Use 'Recompute shading' to see impact.")
                st.rerun()

        # MOVE
        with c2.expander("Move existing object", expanded=False):
            if df_objs.empty or "object_id" not in df_objs.columns:
                st.info("No objects loaded.")
            else:
                sel_oid = st.selectbox("Select object", options=sorted(map(str, df_objs["object_id"])))
                row = df_objs[df_objs["object_id"].astype(str) == sel_oid].iloc[0]
                st.write(f"Current: {float(row['latitude']):.7f}, {float(row['longitude']):.7f}")
                if clicked:
                    st.info(f"Clicked lat/lon: {clicked['lat']:.7f}, {clicked['lng']:.7f}")
                new_lat2 = st.number_input("New latitude", value=float(row["latitude"]), format="%.7f", key="mv_lat")
                new_lon2 = st.number_input("New longitude", value=float(row["longitude"]), format="%.7f", key="mv_lon")
                if st.button("Move object here", key="btn_move_obj"):
                    df2 = df_objs.copy()
                    idx = df2.index[df2["object_id"].astype(str) == sel_oid][0]
                    df2.at[idx, "latitude"]  = float(new_lat2)
                    df2.at[idx, "longitude"] = float(new_lon2)
                    st.session_state[objs_session_key] = df2
                    st.session_state[dirty_session_key] = True
                    st.success("Moved in session. Use 'Recompute shading' to see impact.")
                    st.rerun()

        # DELETE
        with c3.expander("Delete object", expanded=False):
            if df_objs.empty or "object_id" not in df_objs.columns:
                st.info("No objects loaded.")
            else:
                del_oid = st.selectbox("Select object to delete", options=sorted(map(str, df_objs["object_id"])), key="del_oid")
                if st.button("Delete selected object", key="btn_del_obj"):
                    df2 = df_objs[df_objs["object_id"].astype(str) != del_oid].copy()
                    st.session_state[objs_session_key] = df2
                    st.session_state[dirty_session_key] = True
                    st.success(f"Deleted in session. Use 'Recompute shading' to see impact.")
                    st.rerun()

        # Save session objects (temp) and optional persist
        colL, colR = st.columns([1,1])
        if colL.button("Save session objects (temp only)"):
            base = resolve_base(_secrets_optional())
            _, _, _, obj_sess, _, _ = _session_paths(Path(base), state)
            try:
                obj_sess.parent.mkdir(parents=True, exist_ok=True)
                st.session_state[objs_session_key].to_csv(obj_sess, index=False)
                st.session_state[obj_tmp_key] = str(obj_sess)
                st.success(f"Session objects written: {obj_sess}")
            except Exception as e:
                st.error(f"Failed to write session objects:\n{e}")

        if colR.button("Persist to original (optional)"):
            st.warning("Original object CSV will be overwritten.")
            base = resolve_base(_secrets_optional())
            paths = build_paths(base, state)
            obj_orig = paths["objects"]
            try:
                obj_orig.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = str(obj_orig) + ".tmp"
                st.session_state[objs_session_key].to_csv(tmp_path, index=False)
                if os.path.exists(obj_orig):
                    os.replace(tmp_path, obj_orig)
                else:
                    os.rename(tmp_path, obj_orig)
                st.session_state[dirty_session_key] = False
                st.success(f"Persisted to: {obj_orig}")
            except Exception as e:
                st.error(f"Failed to persist original:\n{e}")

    # Shading loss chart (ACTIVE)
    st.markdown("**Shading loss over time (ACTIVE)**")
    if df_shad_active is not None and not df_shad_active.empty:
        sh = df_shad_active.copy()
        ts_active = ts_shad
        if ts_active is not None and "shading_loss" in sh.columns:
            sh["__ts__"] = ts_active
            sh["shading_loss"] = pd.to_numeric(sh["shading_loss"], errors="coerce")
            daily = sh.groupby(sh["__ts__"].dt.date)["shading_loss"].mean().reset_index()
            daily.columns = ["date","avg_shading_loss"]
            fig = px.line(daily, x="date", y="avg_shading_loss", title="Average Daily Shading Loss (Active)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Active shading file has no parseable timestamp or no 'shading_loss' column.")
    else:
        st.caption("No active shading data.")

# -------------------- Energy Profiles -------------------
with tab_energy:
    st.subheader("Energy Production & Load")
    if df_hour is not None and not df_hour.empty:
        h = df_hour.copy()
        if ts_hour is None:
            st.warning("Could not parse a timestamp column in hourly data. Choose it in the sidebar (Advanced • Timestamp columns).")
        else:
            h["__ts__"] = ts_hour
            ycols = [c for c in ["PV_P_kW", "Load_kW"] if c in h.columns]
            if ycols:
                fig = px.line(h, x="__ts__", y=ycols, title="Power (kW)")
                st.plotly_chart(fig, use_container_width=True)

            value_col = "PV_E_kWh" if "PV_E_kWh" in h.columns else ("PV_P_kW" if "PV_P_kW" in h.columns else None)
            if value_col:
                pv = h[["__ts__", value_col]].copy()
                pv["date"] = pv["__ts__"].dt.date
                pv["hour"] = pv["__ts__"].dt.hour
                heat = pv.pivot_table(index="hour", columns="date", values=value_col, aggfunc="sum")
                st.dataframe(heat)
                st.caption("Tip: replace with a heatmap chart if desired.")
            else:
                st.info("PV column not found for heatmap (looked for PV_E_kWh / PV_P_kW).")
    else:
        st.info("Hourly net file not found for this state.")

# -------------------- Billing ---------------------------
with tab_bill:
    st.subheader("Billing & Savings")
    if df_month is not None and not df_month.empty:
        m = df_month.copy()
        if ts_month is None:
            st.warning("Could not parse a timestamp column in monthly data. Choose it in the sidebar (Advanced • Timestamp columns).")
        else:
            m["month"] = ts_month.dt.strftime("%Y-%m")
            cols = [c for c in ["Buy_$", "Sell_$", "NetBill_$"] if c in m.columns]
            if cols:
                fig = px.bar(m, x="month", y=cols, barmode="group", title="Monthly Costs")
                st.plotly_chart(fig, use_container_width=True)

            table_cols = ["month"]
            table_cols += [c for c in ["Import_kWh", "Export_kWh", "Buy_$", "Sell_$", "NetBill_$"] if c in m.columns]
            st.dataframe(m[table_cols])
    else:
        st.info("Monthly net metering file not found for this state.")

# -------------------- Export / Import -------------------
with tab_exports:
    st.subheader("Import vs Export (Hourly)")
    if df_hour is not None and not df_hour.empty:
        h = df_hour.copy()
        if ts_hour is None:
            st.warning("Could not parse a timestamp column in hourly data. Choose it in the sidebar (Advanced • Timestamp columns).")
        else:
            h["__ts__"] = ts_hour
            ycols = [c for c in ["Import_kWh", "Export_kWh"] if c in h.columns]
            if ycols:
                fig = px.area(h, x="__ts__", y=ycols, title="Import/Export Energy (kWh)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Hourly file lacks Import_kWh / Export_kWh columns.")
    else:
        st.info("Hourly net file not found for this state.")

# -------------------- Data (debug) ----------------------
with tab_data:
    st.subheader("Raw Data Snapshots")

    if df_pv_active is not None and not df_pv_active.empty:
        st.write("PV Generation (ACTIVE sample)")
        st.dataframe(df_pv_active.head(200))
        if st.session_state[pv_df_key] is not None:
            st.caption(f"Showing session file: {st.session_state[pv_tmp_key]}")
        else:
            st.caption("Showing original PV generation csv.")
    else:
        st.caption("No PV generation csv found or it is empty.")

    if df_shad_active is not None and not df_shad_active.empty:
        st.write("Shading Result (ACTIVE sample)")
        st.dataframe(df_shad_active.head(200))
        if st.session_state[shad_df_key] is not None:
            st.caption(f"Showing session file: {st.session_state[shad_tmp_key]}")
        else:
            st.caption("Showing original shading csv.")
    else:
        st.caption("No shading result csv found or it is empty.")

    if df_objs is not None and not df_objs.empty:
        st.write("Objects (session copy; originals untouched)")
        st.dataframe(df_objs)
        if st.session_state[obj_tmp_key]:
            st.caption(f"Session objects file: {st.session_state[obj_tmp_key]}")
    else:
        st.caption("No object csv found or it is empty.")

    if df_month is not None and not df_month.empty:
        st.write("Monthly Net Metering")
        st.dataframe(df_month)
    else:
        st.caption("No monthly net csv found or it is empty.")

    if df_hour is not None and not df_hour.empty:
        st.write("Hourly Net Metering (sample)")
        st.dataframe(df_hour.head(500))
    else:
        st.caption("No hourly net csv found or it is empty.")
