# streamlit_app.py
import os
import sys
import time
import math
import uuid
import subprocess
from pathlib import Path
from datetime import timedelta

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

from assistant import run_ai_chatbot

# make assistant importable if the app is inside pv_ui_streamlit_app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from assistant.assistant_tab import render as render_assistant

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
def get_site_center(state: str, df_objs: pd.DataFrame | None):
    """Return (center_lat, center_lon) for a state. Falls back to object mean or a default."""
    if state in SITE_CENTERS:
        return SITE_CENTERS[state][0], SITE_CENTERS[state][1]
    if df_objs is not None and not df_objs.empty and {"latitude","longitude"}.issubset(df_objs.columns):
        lat = pd.to_numeric(df_objs["latitude"], errors="coerce").dropna()
        lon = pd.to_numeric(df_objs["longitude"], errors="coerce").dropna()
        if len(lat) and len(lon):
            return float(lat.mean()), float(lon.mean())
    # default fallback (Mountain View-ish)
    return 37.3894, -122.0839

DEFAULT_WIDTH_BY_TYPE = {"Pole": 1.0, "Tree": 3.0, "Building": 10.0, "Other": 2.0}

def latlon_to_dxdy_meters(lat0: float, lon0: float, lat1: float, lon1: float):
    lat_rad = math.radians(lat0)
    dy = (lat1 - lat0) * 111_320.0
    dx = (lon1 - lon0) * 111_320.0 * (math.cos(lat_rad) if abs(math.cos(lat_rad)) > 1e-6 else 0.0)
    return dx, dy

def bearing_deg_from_dxdy(dx_m: float, dy_m: float):
    # 0°=North, 90°=East …
    return (math.degrees(math.atan2(dx_m, dy_m)) + 360.0) % 360.0

def enrich_objects_with_derived(df: pd.DataFrame, center_lat: float, center_lon: float) -> pd.DataFrame:
    """Ensure distance_m & azimuth_deg from lat/lon. Do NOT fill width_m automatically."""
    if df is None or df.empty:
        return df
    df2 = df.copy()

    # Ensure columns exist
    for col in ["width_m", "distance_m", "azimuth_deg"]:
        if col not in df2.columns:
            df2[col] = np.nan

    # Compute distance/azimuth from lat/lon to PV center
    for i, row in df2.iterrows():
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        if pd.notna(lat) and pd.notna(lon):
            dx, dy = latlon_to_dxdy_meters(center_lat, center_lon, float(lat), float(lon))
            df2.at[i, "distance_m"] = float(math.hypot(dx, dy))
            df2.at[i, "azimuth_deg"] = float(bearing_deg_from_dxdy(dx, dy))

    # Numeric cleanup
    for c in ["width_m", "distance_m", "azimuth_deg", "height_m", "shading_intensity"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    return df2

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
# --- net-metering session keys ---
def _net_hour_df_key(state):    return f"net_hour_df_session_{state}"
def _net_month_df_key(state):   return f"net_month_df_session_{state}"
def _net_hour_path_key(state):  return f"net_hour_tmp_path_{state}"
def _net_month_path_key(state): return f"net_month_tmp_path_{state}"

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

def _find_pvnet_script() -> Path | None:
    here = Path(__file__).resolve().parent
    cands = [
        # original name
        here / "pv_net_metering.py",
        here.parent / "pv_net_metering.py",
        here.parent / "main" / "pv_net_metering.py",
        here.parent.parent / "pv_net_metering.py",
        # your renamed script
        here / "consumption.py",
        here.parent / "consumption.py",
        here.parent / "main" / "consumption.py",
        here.parent.parent / "consumption.py",
    ]
    for p in cands:
        if p.exists():
            return p
    return None

def run_net_metering_for_state(state: str, base_dir: Path, use_session: bool=True):
    """
    Call pv_net_metering.py (or consumption.py) for a single state.
    Returns (ok, message, hourly_csv_path|None, monthly_csv_path|None)
    """
    script = _find_pvnet_script()
    if script is None:
        return False, "pv_net_metering.py / consumption.py not found.", None, None

    in_dir   = Path(base_dir) / "shading"
    out_dir  = Path(base_dir) / "net_metering"
    load_dir = Path(base_dir) / "loads"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [sys.executable, str(script), "--state", state,
            "--in", str(in_dir), "--out", str(out_dir), "--loads", str(load_dir)]
    if use_session:
        args.append("--session")

    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=900, cwd=str(script.parent))
        if cp.returncode != 0:
            return False, f"pv_net_metering.py failed (code {cp.returncode}).\nSTDERR:\n{cp.stderr}", None, None
    except Exception as e:
        return False, f"pv_net_metering.py exception: {e}", None, None

    suf = "_session" if use_session else ""
    hourly = out_dir / f"{state}_hourly_net{suf}.csv"
    monthly = out_dir / f"{state}_monthly_net{suf}.csv"
    return (hourly.exists() or monthly.exists()), "pv_net_metering.py OK.", (str(hourly) if hourly.exists() else None), (str(monthly) if monthly.exists() else None)

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

import plotly.graph_objects as go

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def make_bill_waterfall(baseline: float, buy: float, sell: float, net: float) -> go.Figure:
    # sanitize
    baseline = baseline or 0.0
    buy      = buy or 0.0
    sell     = sell or 0.0
    net      = net or 0.0
    savings  = max(baseline - net, 0.0)
    pct_save = (savings / baseline * 100.0) if baseline > 0 else 0.0

    # colors (supported buckets for waterfall)
    cost_color    = "rgba(220, 76, 70, 0.9)"   # decreasing
    credit_color  = "rgba(60, 170, 90, 0.9)"   # increasing
    total_color   = "rgba(160,160,160, 0.95)"  # totals
    savings_color = "rgba(56, 149, 211, 1.0)"  # indicator accent

    # --- figure with 2 panes: waterfall + indicator "card"
    # fig = make_subplots(
    #     rows=1, cols=2,
    #     column_widths=[0.72, 0.28],
    #     specs=[[{"type": "xy"}, {"type": "domain"}]],
    #     horizontal_spacing=0.08,
    #     subplot_titles=("Bill Breakdown", "Savings")
    # )

    # (1) Waterfall trace (left)
    wf = go.Waterfall(
        name="Bill",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Baseline (no PV)", "Energy bought", "Export credit", "Net bill"],
        y=[baseline, -buy, -sell, net],
        connector={"line": {"dash": "dot", "width": 1}},
        text=[f"${v:,.0f}" for v in [baseline, -buy, -sell, net]],
        textposition="outside",
        decreasing={"marker": {"color": cost_color}},
        increasing={"marker": {"color": credit_color}},
        totals={"marker": {"color": total_color}},
        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"
    )
    fig.add_trace(wf, row=1, col=1)

    # Add a faint reference line at the baseline level, helps readability
    fig.add_hline(y=baseline, line_width=1, line_dash="dot",
                  line_color="rgba(200,200,200,0.6)", row=1, col=1)

    # (2) Big savings indicator (right)
    ind = go.Indicator(
        mode="number+gauge+delta",
        value=savings,
        number={"prefix": "$", "font": {"size": 48, "color": savings_color}},
        delta={"reference": 0, "relative": False, "increasing": {"color": savings_color}},
        title={"text": f"<b>Savings</b><br><span style='font-size:14px'>{pct_save:.1f}% of baseline</span>"},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, max(baseline, savings) or 1]},
            "bar": {"color": savings_color},
            "threshold": {
                "line": {"color": total_color, "width": 1},
                "thickness": 0.6, "value": savings
            },
            "bgcolor": "rgba(0,0,0,0)"
        },
        domain={"x": [0, 1], "y": [0.1, 0.9]}
    )
    fig.add_trace(ind, row=1, col=2)

    # Layout polish
    fig.update_yaxes(title="$", row=1, col=1)
    fig.update_layout(
        title="Bill Breakdown & Savings",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        bargap=0.25,
    )

    # Helpful label near the net step showing computed savings
    fig.add_annotation(
        x=0.88, y=savings, xref="paper", yref="y1",
        text=f"Savings: ${savings:,.0f}",
        font=dict(color=savings_color, size=12),
        showarrow=False
    )
    return fig

# ---------------------------------------------------------------------
# Discover + load data
# ---------------------------------------------------------------------
STATES, DATA_BASE = get_states_and_base(_secrets_optional())
if not STATES:
    st.error("No datasets found. Set PV_DATA_BASE env var or place this app next to your 'main' data folder.")
    st.stop()

st.sidebar.title("Shady")

def panel_label(s: str) -> str:
    # shows "Texas - Panel", "California - Panel", etc.
    return f"{s.title()} - Panel 1"

default_idx = STATES.index("texas") if "texas" in STATES else 0

state = st.sidebar.selectbox(
    "PV Panel",
    STATES,                                # underlying values stay the raw state keys
    index=min(default_idx, len(STATES)-1),
    format_func=panel_label,               # only changes the display text
    key="pv_panel_select",
)

# optional: use this pretty name in headers/captions
selected_panel_name = panel_label(state)
# st.subheader(f"Summary for {selected_panel_name}")


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

net_hour_key   = _net_hour_df_key(state)
net_month_key  = _net_month_df_key(state)
net_hour_path  = _net_hour_path_key(state)
net_month_path = _net_month_path_key(state)

for k in (net_hour_key, net_month_key, net_hour_path, net_month_path):
    if k not in st.session_state:
        st.session_state[k] = None

# Use session net-metering files if they exist
df_month_active = st.session_state[net_month_key] if st.session_state[net_month_key] is not None else df_month
df_hour_active  = st.session_state[net_hour_key]  if st.session_state[net_hour_key]  is not None else df_hour

# Downstream code expects df_month/df_hour names
df_month = df_month_active
df_hour  = df_hour_active

if objs_session_key not in st.session_state:
    st.session_state[objs_session_key] = df_objs_original.copy()
if dirty_session_key not in st.session_state:
    st.session_state[dirty_session_key] = False
if obj_tmp_key not in st.session_state:
    st.session_state[obj_tmp_key] = None

df_objs = st.session_state[objs_session_key]

center_lat, center_lon = get_site_center(state, df_objs)
# After df_objs is created and center_lat/center_lon are known:
if not df_objs.empty:
    df2 = df_objs.copy()

    # Fill width_m defaults if missing (for legacy rows in original CSVs)
    if "width_m" not in df2.columns:
        df2["width_m"] = np.nan
    if "object_type" in df2.columns:
        need_w = df2["width_m"].isna()
        df2.loc[need_w, "width_m"] = df2.loc[need_w, "object_type"].map(DEFAULT_WIDTH_BY_TYPE).fillna(2.0)

    # Compute distance/azimuth where missing
    for i, r in df2.iterrows():
        lat = pd.to_numeric(r.get("latitude"), errors="coerce")
        lon = pd.to_numeric(r.get("longitude"), errors="coerce")
        if pd.notna(lat) and pd.notna(lon):
            dx, dy = latlon_to_dxdy_meters(center_lat, center_lon, float(lat), float(lon))
            if "distance_m" not in df2.columns or pd.isna(r.get("distance_m")):
                df2.at[i, "distance_m"] = float(math.hypot(dx, dy))
            if "azimuth_deg" not in df2.columns or pd.isna(r.get("azimuth_deg")):
                df2.at[i, "azimuth_deg"] = float(bearing_deg_from_dxdy(dx, dy))

    st.session_state[objs_session_key] = df2
    df_objs = df2

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

# -------- No sidebar timestamp pickers: auto-pick columns safely --------
def _safe_pick_time_col(df):
    if df is None or df.empty:
        return None
    g = guess_time_col(df)
    if g in (None, "") or g not in df.columns:
        for c in df.columns:
            try:
                s = pd.to_datetime(df[c], errors="coerce", utc=True)
                if s.notna().any():
                    return c
            except Exception:
                pass
        return df.columns[0]
    return g

# Auto-pick columns
mon_col = _safe_pick_time_col(df_month)         if df_month is not None else None
hr_col  = _safe_pick_time_col(df_hour)          if df_hour  is not None else None
sh_col  = _safe_pick_time_col(df_shad_active)   if df_shad_active is not None else None
pv_col  = _safe_pick_time_col(df_pv_active)     if df_pv_active   is not None else None

# Parse timestamps
ts_month = parse_ts_column(df_month, mon_col) if df_month is not None else None
ts_hour  = parse_ts_column(df_hour,  hr_col)  if df_hour  is not None else None
ts_shad  = parse_ts_column(df_shad_active,  sh_col) if df_shad_active is not None else None
ts_pv    = parse_ts_column(df_pv_active,    pv_col) if df_pv_active   is not None else None
# ------------------ Global time range (date inputs, no slider) ------------------
from datetime import datetime, time as dtime

# st.markdown("### Time Range")

def _bounds_from_any(df_list, ts_list):
    """Find min/max timestamp across all available dataframes."""
    mins, maxs = [], []
    for df, ts in zip(df_list, ts_list):
        if df is not None and not df.empty and ts is not None and pd.to_datetime(ts, errors="coerce").notna().any():
            mins.append(pd.Timestamp(pd.to_datetime(ts).min()))
            maxs.append(pd.Timestamp(pd.to_datetime(ts).max()))
    if not mins or not maxs:
        # fallback to 'today' if nothing is available
        today = pd.Timestamp("today").normalize()
        return today, today
    return min(mins), max(maxs)

# Collect existing parsed timestamps (you already compute these above)
all_min, all_max = _bounds_from_any(
    [df_hour, df_month, df_shad_active, df_pv_active],
    [ts_hour, ts_month, ts_shad,       ts_pv]
)

# Defaults: last 12 months (clamped to data bounds)
def_start_default = max(all_min.normalize(), (all_max - pd.DateOffset(years=1)).normalize())
def_end_default   = all_max.normalize()

# Read stored values (so tabs share the same dates)
gkey_start = "GLOBAL_DATE_START"
gkey_end   = "GLOBAL_DATE_END"
if gkey_start not in st.session_state:
    st.session_state[gkey_start] = def_start_default.date()
if gkey_end not in st.session_state:
    st.session_state[gkey_end] = def_end_default.date()

st.sidebar.markdown("---")
st.sidebar.subheader("Time Range")
 
start_date = st.sidebar.date_input(
    "Start date",
    value=st.session_state[gkey_start],
    min_value=all_min.date(),
    max_value=all_max.date(),
    key="date_input_start"
)
end_date = st.sidebar.date_input(
    "End date",
    value=st.session_state[gkey_end],
    min_value=all_min.date(),
    max_value=all_max.date(),
    key="date_input_end"
)

# Guard: swap if user picks end < start
if end_date < start_date:
    st.warning("End date is before start date. Adjusting to match start date.")
    end_date = start_date

# Store back so other tabs see the same range
st.session_state[gkey_start] = start_date
st.session_state[gkey_end]   = end_date

# Convert to timestamps that cover whole days
g_start = pd.Timestamp(datetime.combine(start_date, dtime.min))
g_end   = pd.Timestamp(datetime.combine(end_date,   dtime.max))

st.sidebar.caption(f"Active range: **{g_start.date()} → {g_end.date()}**")

# Small helper used later (unchanged if you already have it)
def filter_df_by_global(df, ts, start_ts, end_ts):
    """
    Return a copy of df filtered to [start_ts, end_ts] and always include
    a datetime column '__ts__' aligned to the returned rows.
    """
    if df is None or df.empty or ts is None:
        return None  # signal unusable

    s = pd.to_datetime(ts, errors="coerce")
    if s.isna().all():
        return None

    mask = (s >= start_ts) & (s <= end_ts)
    out = df.loc[mask].copy()

    # attach helper timestamp column aligned by index
    out["__ts__"] = s.loc[mask].values
    return out


# ---- render the single, global time range slider ----
# g_start, g_end = render_global_time_filter(ts_hour, ts_month)

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
    obj_sess = obj_orig.with_name(obj_orig.stem + "_session.csv")
    sh_sess  = sh_orig.with_name(sh_orig.stem + "_session.csv")
    pv_sess  = pv_orig.with_name(pv_orig.stem + "_session.csv")
    return obj_orig, pv_orig, sh_orig, obj_sess, pv_sess, sh_sess

if st.sidebar.button("Recompute shading now"):
    base = resolve_base(_secrets_optional())
    obj_orig, pv_orig, sh_orig, obj_sess, pv_sess, sh_sess = _session_paths(Path(base), state)

    # 1) write session objects file only
    try:
        obj_sess.parent.mkdir(parents=True, exist_ok=True)
        center_lat, center_lon = get_site_center(state, st.session_state[objs_session_key])
        objs_enriched = enrich_objects_with_derived(st.session_state[objs_session_key], center_lat, center_lon)

        # Enforce width_m > 0
        if "width_m" not in objs_enriched.columns or objs_enriched["width_m"].fillna(0).le(0).any():
            st.error("All objects must have a positive width_m to run shading. Please add width for missing objects.")
            st.stop()

        objs_enriched.to_csv(obj_sess, index=False)
        st.session_state[obj_tmp_key] = str(obj_sess)
    except Exception as e:
        st.error(f"Could not write session objects CSV:\n{e}")

    # 2) run shading
    ok, msg, sh_actual, pv_actual = run_shading_for_state(
        state=state, base_dir=Path(base),
        objects_csv=Path(obj_sess),
        pv_csv=Path(pv_orig),
        out_csv=Path(sh_sess),
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

        # 3) recompute net-metering (session)
        ok2, msg2, hour_path, mon_path = run_net_metering_for_state(
            state=state, base_dir=Path(debug["base"]), use_session=True
        )
        if not ok2:
            st.warning(msg2)
        else:
            st.toast("Net-metering recomputed (session).")
            try:
                if hour_path and os.path.exists(hour_path):
                    df_hour_sess = pd.read_csv(hour_path)
                    st.session_state[net_hour_key]  = normalize_cols(df_hour_sess)
                    st.session_state[net_hour_path] = hour_path
                if mon_path and os.path.exists(mon_path):
                    df_month_sess = pd.read_csv(mon_path)
                    st.session_state[net_month_key]  = normalize_cols(df_month_sess)
                    st.session_state[net_month_path] = mon_path
            except Exception as e:
                st.warning(f"Loaded shading session, but failed to load net-metering CSVs: {e}")

        st.toast("Session data active.")
        st.rerun()

if st.sidebar.button("End session (clean temp)"):
    for key in (
        _obj_tmp_key(state),
        _shad_path_key(state),
        _pv_path_key(state),
        _net_hour_path_key(state),
        _net_month_path_key(state),
    ):
        tmp_path = st.session_state.get(key)
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e:
            st.warning(f"Could not delete temp file {tmp_path}: {e}")

    for k in (
        _objs_key(state), _objs_dirty_key(state),
        _shad_df_key(state), _shad_path_key(state),
        _pv_df_key(state), _pv_path_key(state),
        _net_hour_df_key(state), _net_month_df_key(state),
        _net_hour_path_key(state), _net_month_path_key(state),
        _obj_tmp_key(state),
    ):
        if k in st.session_state:
            del st.session_state[k]

    st.toast("Session cleared; reloading original datasets.")
    st.rerun()

# ---------------------------------------------------------------------
# KPIs (now driven by the GLOBAL time window) — with working savings
# ---------------------------------------------------------------------

Hf = filter_df_by_global(df_hour,  ts_hour,  g_start, g_end)
Mf = filter_df_by_global(df_month, ts_month, g_start, g_end)
Sf = filter_df_by_global(df_shad_active, ts_shad, g_start, g_end)

# Early guards so later code never crashes:
if Mf is not None and "__ts__" not in Mf.columns:
    Mf = None
if Hf is not None and "__ts__" not in Hf.columns:
    Hf = None


def _num(s):
    if s is None:
        return 0.0
    return float(pd.to_numeric(s, errors="coerce").fillna(0).sum())

def _infer_rates(Mf, Hf):
    """
    Try to get (buy_rate, sell_rate) in $/kWh.
    Prefer monthly totals; if those fall through, return (None, None).
    """
    if Mf is not None and not Mf.empty:
        imp = _num(Mf.get("Import_kWh"))
        exp = _num(Mf.get("Export_kWh"))
        buy_d = _num(Mf.get("Buy_$"))
        sell_d = _num(Mf.get("Sell_$"))
        buy_r  = (buy_d / imp)  if imp  > 0 and buy_d  > 0 else None
        sell_r = (sell_d / exp) if exp > 0 and sell_d > 0 else None
        if buy_r is not None or sell_r is not None:
            return buy_r, sell_r
    return None, None

def _kpis(Mf, Hf):
    # Energy totals (prefer monthly in-window, else hourly in-window)
    if Mf is not None and not Mf.empty:
        pv_kwh   = _num(Mf.get("PV_E_kWh"))
        load_kwh = _num(Mf.get("Load_E_kWh"))
        imp_kwh  = _num(Mf.get("Import_kWh"))
        exp_kwh  = _num(Mf.get("Export_kWh"))

        buy_r, sell_r = _infer_rates(Mf, Hf)

        # Net bill from file if present; else compute from kWh × rates
        if "NetBill_$" in (Mf.columns if Mf is not None else []):
            net_bill = _num(Mf.get("NetBill_$"))
        else:
            net_bill = None
            if buy_r is not None or sell_r is not None:
                br = 0.0 if buy_r  is None else buy_r
                sr = 0.0 if sell_r is None else sell_r
                net_bill = imp_kwh * br - exp_kwh * sr

        # Baseline = load without PV at buy rate
        baseline = None
        if buy_r is not None:
            baseline = load_kwh * buy_r

        # Savings = baseline − net bill
        savings = None
        if baseline is not None and net_bill is not None:
            savings = baseline - net_bill

        self_consumed = max(pv_kwh - exp_kwh, 0.0) if pv_kwh > 0 else 0.0
        sc_pct = (self_consumed / pv_kwh * 100.0) if pv_kwh > 0 else 0.0
        ss_pct = (self_consumed / load_kwh * 100.0) if load_kwh > 0 else 0.0

        return {
            "annual_pv_kwh": pv_kwh,
            "annual_load_kwh": load_kwh,
            "annual_import_kwh": imp_kwh,
            "annual_export_kwh": exp_kwh,
            "net_bill": 0.0 if net_bill is None else float(net_bill),
            "baseline_bill_est": 0.0 if baseline is None else float(baseline),
            "savings_est": 0.0 if savings is None else float(savings),
            "self_consumption_pct": sc_pct,
            "self_sufficiency_pct": ss_pct,
            "avg_buy_rate": 0.0 if buy_r is None else float(buy_r),
        }

    # Fallback: hourly window (needs inferred rates; if none, savings remains 0)
    if Hf is not None and not Hf.empty:
        load_kwh   = _num(Hf.get("Load_E_kWh", Hf.get("Load_kW")))
        pv_kwh     = _num(Hf.get("PV_E_kWh",   Hf.get("PV_P_kW")))
        imp_kwh    = _num(Hf.get("Import_kWh"))
        exp_kwh    = _num(Hf.get("Export_kWh"))

        buy_r, sell_r = _infer_rates(Mf, Hf)  # still try to infer from monthly if available

        baseline = load_kwh * buy_r if buy_r is not None else None
        net_bill = None
        if buy_r is not None or sell_r is not None:
            br = 0.0 if buy_r  is None else buy_r
            sr = 0.0 if sell_r is None else sell_r
            net_bill = imp_kwh * br - exp_kwh * sr
        savings = (baseline - net_bill) if (baseline is not None and net_bill is not None) else None

        self_consumed = max(pv_kwh - exp_kwh, 0.0) if pv_kwh > 0 else 0.0
        sc_pct = (self_consumed / pv_kwh * 100.0) if pv_kwh > 0 else 0.0
        ss_pct = (self_consumed / load_kwh * 100.0) if load_kwh > 0 else 0.0

        return {
            "annual_pv_kwh": pv_kwh,
            "annual_load_kwh": load_kwh,
            "annual_import_kwh": imp_kwh,
            "annual_export_kwh": exp_kwh,
            "net_bill": 0.0 if net_bill is None else float(net_bill),
            "baseline_bill_est": 0.0 if baseline is None else float(baseline),
            "savings_est": 0.0 if savings is None else float(savings),
            "self_consumption_pct": sc_pct,
            "self_sufficiency_pct": ss_pct,
            "avg_buy_rate": 0.0 if buy_r is None else float(buy_r),
        }

    return {}

ak = _kpis(Mf, Hf)

# Average shading loss in-window
if Sf is not None and not Sf.empty and "shading_loss" in Sf.columns:
    # handle either 0–1 or 0–100 data
    vals = pd.to_numeric(Sf["shading_loss"], errors="coerce").dropna()
    if not vals.empty:
        mx = vals.max()
        avg_shade = float(vals.mean() * (100.0 if mx <= 1.5 else 1.0))
    else:
        avg_shade = float("nan")
else:
    avg_shade = float("nan")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total PV (kWh)", f"{ak.get('annual_pv_kwh',0):,.0f}")
c2.metric("Total Load (kWh)", f"{ak.get('annual_load_kwh',0):,.0f}")
c3.metric("Self-consumption (%)", f"{ak.get('self_consumption_pct',0):.1f}%")
c4.metric("Savings (est)", f"${ak.get('savings_est',0):,.0f}")
c5.metric("Avg Shading Loss", ("—" if np.isnan(avg_shade) else f"{avg_shade:.1f}%"))

st.markdown("---")

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_overview, tab_map, tab_assistant = st.tabs(
    ["Overview", "Map & Shading", "Assistant"]
)

# -------------------- Overview --------------------------
with tab_overview:
    st.subheader(f"Summary for {state.title()}")

    # ==== ENERGY (moved here) ==========================================
    st.markdown("### Energy")
    gran_energy = st.radio(
        "Granularity (Energy)",
        ["Hourly", "Daily", "Monthly", "Annual"],
        index=2,  # default Monthly
        horizontal=True,
        key="overview_energy_granularity",
    )

    # Build filtered once
    Hf = filter_df_by_global(df_hour,  ts_hour,  g_start, g_end)
    Mf = filter_df_by_global(df_month, ts_month, g_start, g_end)

    def _resample_hourly(df_hourly, freq):
        if df_hourly is None or df_hourly.empty:
            return None
        cols = [c for c in ["PV_E_kWh", "Load_E_kWh"] if c in df_hourly.columns]
        if not cols:
            return None
        out = df_hourly.set_index("__ts__")[cols].resample(freq).sum(min_count=1).reset_index()
        return out

    if gran_energy == "Hourly":
        if Hf is not None and not Hf.empty:
            ycols = [c for c in ["PV_E_kWh", "Load_E_kWh"] if c in Hf.columns] or \
                    [c for c in ["PV_P_kW", "Load_kW"] if c in Hf.columns]
            if ycols:
                title = "Power (Hourly, kW)" if any(c.endswith("_kW") for c in ycols) else "Energy (Hourly, kWh)"
                fig = px.line(Hf, x="__ts__", y=ycols, title=title)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly data available.")

    elif gran_energy == "Daily":
        daily = _resample_hourly(Hf, "D")
        if daily is not None and not daily.empty:
            daily["date"] = daily["__ts__"].dt.date
            fig = px.bar(daily, x="date", y=[c for c in ["PV_E_kWh", "Load_E_kWh"] if c in daily.columns],
                         barmode="group", title="Energy (Daily)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cannot build daily aggregation.")

    elif gran_energy == "Monthly":
        if Mf is not None and not Mf.empty and "__ts__" in Mf.columns:
            mm = Mf.copy()
            cols = [c for c in ["PV_E_kWh", "Load_E_kWh"] if c in mm.columns]
            if cols:
                mm["month"] = mm["__ts__"].dt.strftime("%Y-%m")
                fig = px.bar(mm, x="month", y=cols, barmode="group", title="Energy (Monthly)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Monthly file lacks PV_E_kWh / Load_E_kWh columns.")
        else:
            monthly = _resample_hourly(Hf, "MS")
            if monthly is not None and not monthly.empty:
                monthly["month"] = monthly["__ts__"].dt.strftime("%Y-%m")
                fig = px.bar(monthly, x="month", y=[c for c in ["PV_E_kWh","Load_E_kWh"] if c in monthly.columns],
                             barmode="group", title="Energy (Monthly)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available to build monthly view.")

    else:  # Annual
        annual_df = None
        if Mf is not None and not Mf.empty:
            cols = [c for c in ["PV_E_kWh","Load_E_kWh"] if c in Mf.columns]
            if cols:
                annual_df = Mf.set_index("__ts__")[cols].resample("YS").sum(min_count=1).reset_index()
        if annual_df is None and Hf is not None and not Hf.empty:
            cols = [c for c in ["PV_E_kWh","Load_E_kWh"] if c in Hf.columns]
            if cols:
                monthly = Hf.set_index("__ts__")[cols].resample("MS").sum(min_count=1)
                annual_df = monthly.resample("YS").sum(min_count=1).reset_index()
        if annual_df is not None and not annual_df.empty:
            annual_df["year"] = annual_df["__ts__"].dt.year
            fig = px.bar(annual_df, x="year", y=[c for c in ["PV_E_kWh","Load_E_kWh"] if c in annual_df.columns],
                         barmode="group", title="Energy (Annual)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available to build annual view.")

    # ==== BILLING (moved here) ========================================
    st.markdown("### Billing & Savings")
    gran_bill = st.radio(
        "Granularity (Billing)",
        ["Hourly", "Daily", "Monthly", "Annual"],
        index=2,  # default Monthly
        horizontal=True,
        key="overview_billing_granularity",
    )

    # infer rates using monthly slice if present
    def _infer_rates(dfm):
        if dfm is None or dfm.empty: return (None, None)
        tot_import = pd.to_numeric(dfm.get("Import_kWh", 0), errors="coerce").sum()
        tot_export = pd.to_numeric(dfm.get("Export_kWh", 0), errors="coerce").sum()
        tot_buy    = pd.to_numeric(dfm.get("Buy_$", 0), errors="coerce").sum()
        tot_sell   = pd.to_numeric(dfm.get("Sell_$", 0), errors="coerce").sum()
        buy_rate  = float(tot_buy / tot_import) if tot_import and tot_import > 0 else None
        sell_rate = float(tot_sell / tot_export) if tot_export and tot_export > 0 else None
        return buy_rate, sell_rate

    buy_rate, sell_rate = _infer_rates(Mf if Mf is not None else df_month)

    if gran_bill == "Hourly":
        if Hf is None or Hf.empty:
            st.info("No hourly data.")
        else:
            for c in ["Import_kWh", "Export_kWh"]:
                if c not in Hf.columns: Hf[c] = 0.0
            if "Buy_$" not in Hf.columns and buy_rate is not None:
                Hf["Buy_$"] = pd.to_numeric(Hf["Import_kWh"], errors="coerce").fillna(0.0) * buy_rate
            if "Sell_$" not in Hf.columns and sell_rate is not None:
                Hf["Sell_$"] = pd.to_numeric(Hf["Export_kWh"], errors="coerce").fillna(0.0) * sell_rate
            cols = [c for c in ["Buy_$", "Sell_$"] if c in Hf.columns]
            if cols:
                Hf["NetBill_$"] = Hf.get("Buy_$", 0) - Hf.get("Sell_$", 0)
                fig = px.line(Hf, x="__ts__", y=cols + (["NetBill_$"] if "NetBill_$" in Hf.columns else []),
                              title="Billing (Hourly)")
                st.plotly_chart(fig, use_container_width=True)

    elif gran_bill == "Daily":
        if Hf is None or Hf.empty:
            st.info("No hourly data.")
        else:
            for c in ["Import_kWh", "Export_kWh"]:
                if c not in Hf.columns: Hf[c] = 0.0
            daily = Hf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("D").sum(min_count=1)
            if buy_rate is not None:
                daily["Buy_$"] = daily["Import_kWh"] * buy_rate
            if sell_rate is not None:
                daily["Sell_$"] = daily["Export_kWh"] * sell_rate
            if "Buy_$" in daily.columns or "Sell_$" in daily.columns:
                daily["NetBill_$"] = daily.get("Buy_$", 0) - daily.get("Sell_$", 0)
            daily = daily.reset_index()
            daily["date"] = daily["__ts__"].dt.date
            cols = [c for c in ["Buy_$", "Sell_$", "NetBill_$"] if c in daily.columns]
            if cols:
                fig = px.bar(daily, x="date", y=cols, barmode="group", title="Billing (Daily)")
                st.plotly_chart(fig, use_container_width=True)

    elif gran_bill == "Monthly":
        if Mf is None or Mf.empty:
            st.info("No monthly data.")
        else:
            Ms = Mf.copy()
            Ms["month"] = Ms["__ts__"].dt.strftime("%Y-%m")
            cols = [c for c in ["Buy_$", "Sell_$", "NetBill_$"] if c in Ms.columns]
            if cols:
                fig = px.bar(Ms, x="month", y=cols, barmode="group", title="Billing (Monthly)")
                st.plotly_chart(fig, use_container_width=True)

            # Waterfall + savings card
            # buy  = pd.to_numeric(Ms.get("Buy_$", 0), errors="coerce").fillna(0).sum()
            # sell = pd.to_numeric(Ms.get("Sell_$", 0), errors="coerce").fillna(0).sum()
            # net  = pd.to_numeric(Ms.get("NetBill_$", 0), errors="coerce").fillna(0).sum()
            # baseline = ak.get("baseline_bill_est", 0.0)
            # figw = make_bill_waterfall(baseline, buy, sell, net)
            # st.plotly_chart(figw, use_container_width=True)

    else:  # Annual
        if Mf is None or Mf.empty:
            st.info("No monthly data to build annual billing.")
        else:
            Y = Mf.set_index("__ts__")[[c for c in ["Buy_$","Sell_$","NetBill_$"] if c in Mf.columns]].resample("YS").sum(min_count=1).reset_index()
            Y["year"] = Y["__ts__"].dt.year
            fig = px.bar(Y, x="year", y=[c for c in ["Buy_$","Sell_$","NetBill_$"] if c in Y.columns],
                         barmode="group", title="Billing (Annual)")
            st.plotly_chart(fig, use_container_width=True)

    # ==== IMPORT / EXPORT (moved here) =================================
    st.markdown("### Import vs Export")
    gran_ie = st.radio(
        "Granularity (Import/Export)",
        ["Hourly", "Daily", "Monthly", "Annual"],
        index=0,
        horizontal=True,
        key="overview_imp_exp_granularity",
    )

    if Hf is None or Hf.empty:
        st.info("No hourly data in selected range for import/export.")
    else:
        for c in ["Import_kWh", "Export_kWh"]:
            if c not in Hf.columns: Hf[c] = 0.0

        if gran_ie == "Hourly":
            ycols = [c for c in ["Import_kWh", "Export_kWh"] if c in Hf.columns]
            if ycols:
                fig = px.area(Hf, x="__ts__", y=ycols, title="Import/Export Energy (Hourly kWh)")
                st.plotly_chart(fig, use_container_width=True)

        elif gran_ie == "Daily":
            daily = Hf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("D").sum(min_count=1).reset_index()
            daily["date"] = daily["__ts__"].dt.date
            fig = px.area(daily, x="date", y=["Import_kWh", "Export_kWh"], title="Import/Export Energy (Daily kWh)")
            st.plotly_chart(fig, use_container_width=True)

        elif gran_ie == "Monthly":
            if Mf is not None and not Mf.empty and all(c in Mf.columns for c in ["Import_kWh", "Export_kWh"]):
                Ms = Mf.copy()
                Ms["month"] = Ms["__ts__"].dt.strftime("%Y-%m")
                fig = px.area(Ms, x="month", y=["Import_kWh", "Export_kWh"], title="Import/Export Energy (Monthly kWh)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                mon = Hf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("MS").sum(min_count=1).reset_index()
                mon["month"] = mon["__ts__"].dt.strftime("%Y-%m")
                fig = px.area(mon, x="month", y=["Import_kWh", "Export_kWh"], title="Import/Export Energy (Monthly kWh)")
                st.plotly_chart(fig, use_container_width=True)

        else:  # Annual
            if Mf is not None and not Mf.empty and all(c in Mf.columns for c in ["Import_kWh", "Export_kWh"]):
                Y = Mf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("YS").sum(min_count=1).reset_index()
            else:
                Y = Hf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("YS").sum(min_count=1).reset_index()
            Y["year"] = Y["__ts__"].dt.year
            fig = px.bar(Y, x="year", y=["Import_kWh", "Export_kWh"], barmode="group", title="Import/Export Energy (Annual kWh)")
            st.plotly_chart(fig, use_container_width=True)


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

    if st.session_state[_objs_dirty_key(state)]:
        st.markdown(
            "<span style='background:#fff3cd;color:#664d03;padding:4px 8px;border:1px solid #ffe69c;border-radius:6px;'>"
            "Unsaved changes (session)</span>",
            unsafe_allow_html=True
        )

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
            new_height = st.number_input("Height (m)", min_value=0.0, value=3.0, step=0.1)
            new_width  = st.number_input("Width (m)",  min_value=0.0, value=2.0, step=0.1, help="Required by shading model")
            new_intensity = st.slider("Shading intensity (0–1)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

            new_lat = st.number_input("Latitude",  value=(clicked["lat"] if clicked else center_lat),  format="%.7f")
            new_lon = st.number_input("Longitude", value=(clicked["lng"] if clicked else center_lon), format="%.7f")

            if st.button("Add object", key="btn_add_obj"):
                dx, dy = latlon_to_dxdy_meters(center_lat, center_lon, float(new_lat), float(new_lon))
                new_distance = float(math.hypot(dx, dy))
                new_azimuth  = float(bearing_deg_from_dxdy(dx, dy))

                if new_width <= 0:
                    st.error("Width (m) must be > 0.")
                else:
                    new_row = {
                        "object_id": str(new_oid),
                        "object_type": new_type,
                        "latitude": float(new_lat),
                        "longitude": float(new_lon),
                        "height_m": float(new_height),
                        "width_m": float(new_width),
                        "shading_intensity": float(new_intensity),
                        "distance_m": new_distance,
                        "azimuth_deg": new_azimuth,
                    }
                    st.session_state[objs_session_key] = pd.concat(
                        [df_objs, pd.DataFrame([new_row])], ignore_index=True
                    )
                    st.session_state[dirty_session_key] = True
                    st.success("Added to session. Use 'Recompute shading' to see impact.")
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

                    dx, dy = latlon_to_dxdy_meters(center_lat, center_lon, float(new_lat2), float(new_lon2))
                    df2.at[idx, "distance_m"] = float(math.hypot(dx, dy))
                    df2.at[idx, "azimuth_deg"] = float(bearing_deg_from_dxdy(dx, dy))

                    if "width_m" not in df2.columns or pd.isna(df2.at[idx, "width_m"]):
                        ot = str(df2.at[idx, "object_type"]) if "object_type" in df2.columns else "Other"
                        df2.at[idx, "width_m"] = float(DEFAULT_WIDTH_BY_TYPE.get(ot, 2.0))

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
                center_lat, center_lon = get_site_center(state, st.session_state[objs_session_key])
                objs_enriched = enrich_objects_with_derived(st.session_state[objs_session_key], center_lat, center_lon)
                objs_enriched.to_csv(obj_sess, index=False)
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

    # Shading loss chart (ACTIVE) — filtered by global window if possible
    st.markdown("**Shading loss over time (ACTIVE)**")
    Sf = filter_df_by_global(df_shad_active, ts_shad, g_start, g_end)
    if Sf is not None and not Sf.empty:
        if "shading_loss" in Sf.columns:
            Sf["shading_loss"] = pd.to_numeric(Sf["shading_loss"], errors="coerce")
            daily = Sf.set_index("__ts__")["shading_loss"].resample("D").mean().reset_index()
            daily["date"] = daily["__ts__"].dt.date
            fig = px.line(daily, x="date", y="shading_loss", title="Average Daily Shading Loss (Active)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Active shading file has no 'shading_loss' column.")
    else:
        st.caption("No active shading data in selected window.")

# -------------------- Energy Profiles -------------------
# with tab_energy:
#     st.subheader("Energy Production & Load")

#     gran_energy = st.radio(
#         "Granularity",
#         ["Hourly", "Daily", "Monthly", "Annual"],
#         index=0,  # default Hourly
#         horizontal=True,
#         key="energy_granularity",
#     )

#     Hf = filter_df_by_global(df_hour, ts_hour, g_start, g_end)

#     if Hf is None or Hf.empty:
#         st.info("No hourly data in selected global window.")
#     else:
#         if gran_energy == "Hourly":
#             ycols = [c for c in ["PV_P_kW","Load_kW"] if c in Hf.columns] or \
#                     [c for c in ["PV_E_kWh","Load_E_kWh"] if c in Hf.columns]
#             if ycols:
#                 title = "Power (Hourly, kW)" if any(c.endswith("_kW") for c in ycols) else "Energy (Hourly, kWh)"
#                 fig = px.line(Hf, x="__ts__", y=ycols, title=title)
#                 st.plotly_chart(fig, use_container_width=True)

            # value_col = "PV_E_kWh" if "PV_E_kWh" in Hf.columns else ("PV_P_kW" if "PV_P_kW" in Hf.columns else None)
            # if value_col and not Hf.empty:
            #     pv = Hf[["__ts__", value_col]].copy()
            #     pv["date"] = pv["__ts__"].dt.date
            #     pv["hour"] = pv["__ts__"].dt.hour
            #     heat = pv.pivot_table(index="hour", columns="date", values=value_col, aggfunc="sum")
            #     st.dataframe(heat)
            #     st.caption("Hourly PV heatmap (filtered to global window).")

        # elif gran_energy == "Daily":
        #     cols = [c for c in ["PV_E_kWh","Load_E_kWh"] if c in Hf.columns]
        #     if cols:
        #         daily = Hf.set_index("__ts__")[cols].resample("D").sum(min_count=1).reset_index()
        #         daily["date"] = daily["__ts__"].dt.date
        #         fig = px.bar(daily, x="date", y=cols, barmode="group", title="Energy (Daily)")
        #         st.plotly_chart(fig, use_container_width=True)

        # elif granularity == "Daily":
        #     daily = _resample_hourly(Hf, "D")
        #     if daily is not None and not daily.empty:
        #         daily["date"] = daily["__ts__"].dt.date
        #         fig = px.bar(daily, x="date", y=[c for c in ["PV_E_kWh", "Load_E_kWh"] if c in daily.columns],
        #                     barmode="group", title="Energy (Daily)")
        #         st.plotly_chart(fig, use_container_width=True)
        #     else:
        #         st.info("Cannot build daily aggregation (need hourly net file).")

        # elif gran_energy == "Monthly":
        #     cols = [c for c in ["PV_E_kWh","Load_E_kWh"] if c in Hf.columns]
        #     if cols:
        #         mon = Hf.set_index("__ts__")[cols].resample("MS").sum(min_count=1).reset_index()
        #         mon["month"] = mon["__ts__"].dt.strftime("%Y-%m")
        #         fig = px.bar(mon, x="month", y=cols, barmode="group", title="Energy (Monthly)")
        #         st.plotly_chart(fig, use_container_width=True)

        # else:  # Annual
        #     cols = [c for c in ["PV_E_kWh","Load_E_kWh"] if c in Hf.columns]
        #     if cols:
        #         mon = Hf.set_index("__ts__")[cols].resample("MS").sum(min_count=1)
        #         yr  = mon.resample("YS").sum(min_count=1).reset_index()
        #         yr["year"] = yr["__ts__"].dt.year
        #         fig = px.bar(yr, x="year", y=cols, barmode="group", title="Energy (Annual)")
        #         st.plotly_chart(fig, use_container_width=True)

# -------------------- Billing ---------------------------
# with tab_bill:
#     st.subheader("Billing & Savings")

#     gran_bill = st.radio(
#         "Granularity",
#         ["Hourly", "Daily", "Monthly", "Annual"],
#         index=2,  # default Monthly
#         horizontal=True,
#         key="billing_granularity",
#     )

#     Hf = filter_df_by_global(df_hour,  ts_hour,  g_start, g_end)
#     Mf = filter_df_by_global(df_month, ts_month, g_start, g_end)

#     def _infer_rates(dfm):
#         if dfm is None or dfm.empty: return (None, None)
#         tot_import = pd.to_numeric(dfm.get("Import_kWh", 0), errors="coerce").sum()
#         tot_export = pd.to_numeric(dfm.get("Export_kWh", 0), errors="coerce").sum()
#         tot_buy = pd.to_numeric(dfm.get("Buy_$", 0), errors="coerce").sum()
#         tot_sell = pd.to_numeric(dfm.get("Sell_$", 0), errors="coerce").sum()
#         buy_rate  = float(tot_buy / tot_import) if tot_import and tot_import > 0 else None
#         sell_rate = float(tot_sell / tot_export) if tot_export and tot_export > 0 else None
#         return buy_rate, sell_rate

#     buy_rate, sell_rate = _infer_rates(Mf if Mf is not None else df_month)

#     if gran_bill == "Hourly":
#         if Hf is None or Hf.empty:
#             st.info("No hourly data in selected global window.")
#         else:
#             for c in ["Import_kWh", "Export_kWh"]:
#                 if c not in Hf.columns: Hf[c] = 0.0

#             if "Buy_$" not in Hf.columns and buy_rate is not None:
#                 Hf["Buy_$"] = pd.to_numeric(Hf["Import_kWh"], errors="coerce").fillna(0.0) * buy_rate
#             if "Sell_$" not in Hf.columns and sell_rate is not None:
#                 Hf["Sell_$"] = pd.to_numeric(Hf["Export_kWh"], errors="coerce").fillna(0.0) * sell_rate

#             cols = [c for c in ["Buy_$", "Sell_$"] if c in Hf.columns]
#             if cols:
#                 Hf["NetBill_$"] = Hf.get("Buy_$", 0) - Hf.get("Sell_$", 0)
#                 fig = px.line(Hf, x="__ts__", y=cols + (["NetBill_$"] if "NetBill_$" in Hf.columns else []),
#                               title="Billing (Hourly)")
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.info("Cannot compute Hourly costs (Buy_$ / Sell_$ not found and rates unavailable).")

#     elif gran_bill == "Daily":
#         if Hf is None or Hf.empty:
#             st.info("No hourly data to build Daily billing.")
#         else:
#             for c in ["Import_kWh", "Export_kWh"]:
#                 if c not in Hf.columns: Hf[c] = 0.0

#             daily = Hf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("D").sum(min_count=1)
#             if buy_rate is not None:
#                 daily["Buy_$"] = daily["Import_kWh"] * buy_rate
#             if sell_rate is not None:
#                 daily["Sell_$"] = daily["Export_kWh"] * sell_rate
#             if "Buy_$" in daily.columns or "Sell_$" in daily.columns:
#                 daily["NetBill_$"] = daily.get("Buy_$", 0) - daily.get("Sell_$", 0)

#             daily = daily.reset_index()
#             daily["date"] = daily["__ts__"].dt.date
#             cols = [c for c in ["Buy_$", "Sell_$", "NetBill_$"] if c in daily.columns]
#             if cols:
#                 fig = px.bar(daily, x="date", y=cols, barmode="group", title="Billing (Daily)")
#                 st.plotly_chart(fig, use_container_width=True)

#     elif gran_bill == "Monthly":
#         if Mf is None or Mf.empty:
#             st.info("No monthly data in selected global window.")
#         else:
#             Ms = Mf.copy()
#             Ms["month"] = Ms["__ts__"].dt.strftime("%Y-%m")
#             cols = [c for c in ["Buy_$", "Sell_$", "NetBill_$"] if c in Ms.columns]
#             if cols:
#                 fig = px.bar(Ms, x="month", y=cols, barmode="group", title="Billing (Monthly)")
#                 st.plotly_chart(fig, use_container_width=True)

#     else:  # Annual
#         if Mf is None or Mf.empty:
#             st.info("No monthly data to build annual billing.")
#         else:
#             Y = Mf.set_index("__ts__")[[c for c in ["Buy_$","Sell_$","NetBill_$"] if c in Mf.columns]].resample("YS").sum(min_count=1).reset_index()
#             Y["year"] = Y["__ts__"].dt.year
#             fig = px.bar(Y, x="year", y=[c for c in ["Buy_$","Sell_$","NetBill_$"] if c in Y.columns],
#                          barmode="group", title="Billing (Annual)")
#             st.plotly_chart(fig, use_container_width=True)

# # -------------------- Export / Import -------------------
# with tab_exports:
#     st.subheader("Import vs Export")

#     gran_ie = st.radio(
#         "Granularity",
#         ["Hourly", "Daily", "Monthly", "Annual"],
#         index=0,  # default Hourly
#         horizontal=True,
#         key="imp_exp_granularity",
#     )

#     Hf = filter_df_by_global(df_hour,  ts_hour,  g_start, g_end)
#     Mf = filter_df_by_global(df_month, ts_month, g_start, g_end)

#     if Hf is None or Hf.empty:
#         st.info("No hourly data in selected global window.")
#     else:
#         for c in ["Import_kWh", "Export_kWh"]:
#             if c not in Hf.columns:
#                 Hf[c] = 0.0

#         if gran_ie == "Hourly":
#             ycols = [c for c in ["Import_kWh", "Export_kWh"] if c in Hf.columns]
#             if ycols:
#                 fig = px.area(Hf, x="__ts__", y=ycols, title="Import/Export Energy (Hourly kWh)")
#                 st.plotly_chart(fig, use_container_width=True)

#         elif gran_ie == "Daily":
#             daily = Hf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("D").sum(min_count=1).reset_index()
#             daily["date"] = daily["__ts__"].dt.date
#             fig = px.area(daily, x="date", y=["Import_kWh", "Export_kWh"], title="Import/Export Energy (Daily kWh)")
#             st.plotly_chart(fig, use_container_width=True)

#         elif gran_ie == "Monthly":
#             if Mf is not None and not Mf.empty and all(c in Mf.columns for c in ["Import_kWh", "Export_kWh"]):
#                 Ms = Mf.copy()
#                 Ms["month"] = Ms["__ts__"].dt.strftime("%Y-%m")
#                 fig = px.area(Ms, x="month", y=["Import_kWh", "Export_kWh"], title="Import/Export Energy (Monthly kWh)")
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 mon = Hf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("MS").sum(min_count=1).reset_index()
#                 mon["month"] = mon["__ts__"].dt.strftime("%Y-%m")
#                 fig = px.area(mon, x="month", y=["Import_kWh", "Export_kWh"], title="Import/Export Energy (Monthly kWh)")
#                 st.plotly_chart(fig, use_container_width=True)

#         else:  # Annual
#             if Mf is not None and not Mf.empty and all(c in Mf.columns for c in ["Import_kWh", "Export_kWh"]):
#                 Y = Mf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("YS").sum(min_count=1).reset_index()
#             else:
#                 Y = Hf.set_index("__ts__")[["Import_kWh", "Export_kWh"]].resample("YS").sum(min_count=1).reset_index()

#             Y["year"] = Y["__ts__"].dt.year
#             fig = px.bar(Y, x="year", y=["Import_kWh", "Export_kWh"], barmode="group", title="Import/Export Energy (Annual kWh)")
#             st.plotly_chart(fig, use_container_width=True)

# # -------------------- Data (debug) ----------------------
# with tab_data:
#     st.subheader("Raw Data Snapshots")

#     if df_pv_active is not None and not df_pv_active.empty:
#         st.write("PV Generation (ACTIVE sample)")
#         st.dataframe(df_pv_active.head(200))
#         if st.session_state[pv_df_key] is not None:
#             st.caption(f"Showing session file: {st.session_state[pv_tmp_key]}")
#         else:
#             st.caption("Showing original PV generation csv.")
#     else:
#         st.caption("No PV generation csv found or it is empty.")

#     if df_shad_active is not None and not df_shad_active.empty:
#         st.write("Shading Result (ACTIVE sample)")
#         st.dataframe(df_shad_active.head(200))
#         if st.session_state[shad_df_key] is not None:
#             st.caption(f"Showing session file: {st.session_state[shad_tmp_key]}")
#         else:
#             st.caption("Showing original shading csv.")
#     else:
#         st.caption("No shading result csv found or it is empty.")

#     if df_objs is not None and not df_objs.empty:
#         st.write("Objects (session copy; originals untouched)")
#         st.dataframe(df_objs)
#         if st.session_state[obj_tmp_key]:
#             st.caption(f"Session objects file: {st.session_state[obj_tmp_key]}")
#     else:
#         st.caption("No object csv found or it is empty.")

#     if df_month is not None and not df_month.empty:
#         st.write("Monthly Net Metering")
#         st.dataframe(df_month)
#     else:
#         st.caption("No monthly net csv found or it is empty.")

#     if df_hour is not None and not df_hour.empty:
#         st.write("Hourly Net Metering (sample)")
#         st.dataframe(df_hour.head(500))
#     else:
#         st.caption("No hourly net csv found or it is empty.")

# -------------------- Assistant -------------------------
with tab_assistant:
    run_ai_chatbot()
