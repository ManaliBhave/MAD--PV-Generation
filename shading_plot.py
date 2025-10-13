import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Settings
# ----------------------------
IN_DIR = "shading"
OUT_DIR = "plot"  # keep as you used in your run; change to "plots" if you prefer
os.makedirs(OUT_DIR, exist_ok=True)

PLOT_LAST_DAYS = 60       # None for full period
ROLLING_HOURS = 6         # visual smoothing; 0/None to disable
DAILY_FROM_ENERGY = True  # prefer E_kWh_new daily sums if present

# ----------------------------
# Helpers
# ----------------------------
def load_csv(path):
    df = pd.read_csv(path)

    # accept either timestamp_local or period_end (fallback)
    ts_col = "timestamp_local" if "timestamp_local" in df.columns else "period_end"
    if ts_col not in df.columns:
        raise ValueError(f"Neither 'timestamp_local' nor 'period_end' in {path}")

    # IMPORTANT: parse with utc=True to avoid mixed-tz object dtype
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.rename(columns={ts_col: "ts"})
    return df

def subset_last_days(df, days):
    if not days:
        return df
    end = df["ts"].max()
    start = end - pd.Timedelta(days=days)
    return df[df["ts"] >= start].copy()

def pick_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None

# ----------------------------
# Discover files
# ----------------------------
files = sorted(glob.glob(os.path.join(IN_DIR, "*_pv_result.csv")))
if not files:
    raise SystemExit(f"No files found in {IN_DIR}/")

print(f"Found {len(files)} file(s). Saving plots to '{OUT_DIR}'")

# ----------------------------
# Per-state processing
# ----------------------------
for fpath in files:
    state = os.path.splitext(os.path.basename(fpath))[0].replace("_pv_result", "")
    print(f"- {state}")

    df = load_csv(fpath)

    col_p_ac  = pick_col(df, ["P_ac", "P_AC", "Pac"])
    col_p_new = pick_col(df, ["P_actual_new_kW", "P_new_kW", "P_after_kW"])
    col_e     = pick_col(df, ["E_kWh_new", "Energy_kWh"])
    if col_p_ac is None or col_p_new is None:
        print(f"  ! Skipping (missing power columns) in {fpath}")
        continue

    # restrict window
    dd = subset_last_days(df, PLOT_LAST_DAYS)

    # optional smoothing (note: 'h' not 'H')
    if ROLLING_HOURS and ROLLING_HOURS > 1:
        dds = dd.set_index("ts")
        dds[f"{col_p_ac}_sm"]  = dds[col_p_ac].rolling(f"{ROLLING_HOURS}h", min_periods=1).mean()
        dds[f"{col_p_new}_sm"] = dds[col_p_new].rolling(f"{ROLLING_HOURS}h", min_periods=1).mean()
        dds = dds.reset_index()
        y1, y2 = f"{col_p_ac}_sm", f"{col_p_new}_sm"
    else:
        dds = dd
        y1, y2 = col_p_ac, col_p_new

    # ---------- Plot 1: Power time series ----------
    plt.figure(figsize=(13, 5))
    plt.plot(dds["ts"], dds[y1], label="Actual Power (P_ac)")
    plt.plot(dds["ts"], dds[y2], label="Post-Shading Power")
    plt.xlabel("Time (UTC)")  # timestamps were converted to UTC
    plt.ylabel("Power (kW)")
    span = f"last {PLOT_LAST_DAYS} days" if PLOT_LAST_DAYS else "full period"
    plt.title(f"{state.capitalize()}: Actual vs Post-Shading Power — {span}")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, f"{state}_power_compare.png")
    plt.savefig(out1, dpi=160)
    plt.close()

    # ---------- Plot 2: Daily energy ----------
    dfe = df.set_index("ts")
    if not isinstance(dfe.index, pd.DatetimeIndex):
        dfe.index = pd.to_datetime(dfe.index, utc=True, errors="coerce")

    if DAILY_FROM_ENERGY and col_e is not None:
        # ensure numeric
        dfe[col_e] = pd.to_numeric(dfe[col_e], errors="coerce")
        daily = dfe[[col_e]].resample("D").sum(min_count=1)
        daily.rename(columns={col_e: "E_post_kWh"}, inplace=True)
        daily["E_actual_kWh"] = dfe[col_p_ac].resample("D").sum(min_count=1)
    else:
        daily = pd.DataFrame({
            "E_actual_kWh": dfe[col_p_ac].resample("D").sum(min_count=1),
            "E_post_kWh":   dfe[col_p_new].resample("D").sum(min_count=1),
        })

    daily = daily.dropna(how="all")

    plt.figure(figsize=(12, 5))
    plt.plot(daily.index, daily["E_actual_kWh"], label="Daily Energy (Actual)")
    plt.plot(daily.index, daily["E_post_kWh"], label="Daily Energy (Post-Shading)")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Energy (kWh/day)")
    plt.title(f"{state.capitalize()}: Daily Energy — Actual vs Post-Shading")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # out2 = os.path.join(OUT_DIR, f"{state}_daily_energy.png")
    # plt.savefig(out2, dpi=160)
    # plt.close()

print("✅ All plots generated.")
