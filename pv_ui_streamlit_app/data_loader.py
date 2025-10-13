from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd

EXPECTED_SUBFOLDERS = ["net_metering", "pv_generation", "shading", "object"]

def _looks_like_base(p: Path) -> bool:
    try:
        return p.exists() and all((p / sub).exists() for sub in EXPECTED_SUBFOLDERS)
    except Exception:
        return False

def resolve_base(st_secrets: Optional[dict] = None) -> Path:
    # 1) Environment variable
    env_base = os.getenv("PV_DATA_BASE", "").strip('"').strip("'")
    if env_base:
        p = Path(env_base).expanduser().resolve()
        if _looks_like_base(p):
            return p

    # 2) Streamlit secrets (optional)
    if st_secrets is not None:
        try:
            sec = st_secrets.get("DATA_BASE", "")
            if sec:
                p = Path(sec).expanduser().resolve()
                if _looks_like_base(p):
                    return p
        except Exception:
            pass

    # 3) Relative to app
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "main",  # ../main
        here / "main",         # ./main
        Path.cwd(),            # current working dir
    ]
    for p in candidates:
        if _looks_like_base(p):
            return p

    # Fallback: parent folder (UI will show error and stop)
    return here.parent

def build_paths(base: Path, state: str) -> Dict[str, Path]:
    return {
        "hourly_net":   base / "net_metering" / f"{state}_hourly_net.csv",
        "monthly_net":  base / "net_metering" / f"{state}_monthly_net.csv",
        "pv_generation":base / "pv_generation" / f"pv_generation_{state}.csv",
        "objects":      base / "object"        / f"{state}_object.csv",
        "shading":      base / "shading"       / f"{state}_pv_result.csv",
    }

def load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None

def detect_states(base: Path) -> List[str]:
    states = set()
    try:
        for p in (base / "net_metering").glob("*_hourly_net.csv"):
            states.add(p.stem.replace("_hourly_net", ""))
    except Exception:
        pass
    try:
        for p in (base / "net_metering").glob("*_monthly_net.csv"):
            states.add(p.stem.replace("_monthly_net", ""))
    except Exception:
        pass
    try:
        for p in (base / "pv_generation").glob("pv_generation_*.csv"):
            states.add(p.stem.replace("pv_generation_", ""))
    except Exception:
        pass
    try:
        for p in (base / "shading").glob("*_pv_result.csv"):
            states.add(p.stem.replace("_pv_result", ""))
    except Exception:
        pass
    try:
        for p in (base / "object").glob("*_object.csv"):
            states.add(p.stem.replace("_object", ""))
    except Exception:
        pass
    return sorted(states)

def load_state(state: str, st_secrets: Optional[dict] = None) -> Tuple[Dict[str, Optional[pd.DataFrame]], Dict]:
    base = resolve_base(st_secrets)
    paths = build_paths(base, state)
    data = {
        "hourly_net":   load_csv(paths["hourly_net"]),
        "monthly_net":  load_csv(paths["monthly_net"]),
        "pv_generation":load_csv(paths["pv_generation"]),
        "objects":      load_csv(paths["objects"]),
        "shading":      load_csv(paths["shading"]),
    }
    debug = {
        "base": str(base),
        "exists": {k: f"{v} {'✓' if v.exists() else '✗'}" for k, v in paths.items()},
    }
    return data, debug

def get_states_and_base(st_secrets: Optional[dict] = None):
    base = resolve_base(st_secrets)
    return detect_states(base), base
