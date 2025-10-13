# plot_net_metering.py  (fixed)
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN_DIR = "net_metering"
OUT_DIR = "net_metering_plots"
os.makedirs(OUT_DIR, exist_ok=True)

def load_csv(path, parse_index=True):
    df = pd.read_csv(path)
    if parse_index:
        ts_col = df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.set_index(ts_col).sort_index()
    return df

def state_name_from(prefix):
    return os.path.basename(prefix).split("_", 1)[0]

def safe_get(df, name, fallback=0.0):
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(fallback, index=df.index)

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

daily_files   = sorted(glob.glob(os.path.join(IN_DIR, "*_daily_net.csv")))
monthly_files = sorted(glob.glob(os.path.join(IN_DIR, "*_monthly_net.csv")))
summary_path  = os.path.join(IN_DIR, "summary_all_states.csv")

if not daily_files:
    raise SystemExit(f"No *_daily_net.csv files found in {IN_DIR}")
if not monthly_files:
    raise SystemExit(f"No *_monthly_net.csv files found in {IN_DIR}")

# Optional: savings bar chart if present
if os.path.exists(summary_path):
    summary = pd.read_csv(summary_path).sort_values("savings_$", ascending=False)
    plt.figure(figsize=(12,5))
    plt.bar(summary["state"], summary["savings_$"])
    plt.title("Total Savings by State")
    plt.ylabel("Savings ($)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "summary_savings_by_state.png"), dpi=160)
    plt.close()

def plot_daily(state, df):
    df = ensure_numeric(df, ["Import_kWh","Export_kWh","PV_E_kWh","Load_E_kWh","Buy_$","Sell_$","NetBill_$"])

    plt.figure(figsize=(13,5))
    plt.plot(df.index, safe_get(df, "Import_kWh"), label="Import (kWh)")
    plt.plot(df.index, safe_get(df, "Export_kWh"), label="Export (kWh)")
    plt.title(f"{state.capitalize()}: Daily Import vs Export Energy")
    plt.xlabel("Date"); plt.ylabel("Energy (kWh)")
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{state}_daily_import_export.png"), dpi=160)
    plt.close()

    if "NetBill_$" in df.columns:
        plt.figure(figsize=(13,5))
        plt.plot(df.index, df["NetBill_$"], label="Net Bill ($)")
        plt.title(f"{state.capitalize()}: Daily Net Bill")
        plt.xlabel("Date"); plt.ylabel("$")
        plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{state}_daily_netbill.png"), dpi=160)
        plt.close()

def plot_monthly(state, df):
    df = ensure_numeric(df, ["Buy_$","Sell_$","NetBill_$","Import_kWh","Export_kWh","PV_E_kWh","Load_E_kWh"])

    plt.figure(figsize=(12,5))
    plt.plot(df.index, safe_get(df,"Buy_$"),  label="Buy from Grid ($)")
    plt.plot(df.index, safe_get(df,"Sell_$"), label="Sell to Grid ($)")
    if "NetBill_$" in df.columns:
        plt.plot(df.index, df["NetBill_$"], label="Net Bill ($)")
    plt.title(f"{state.capitalize()}: Monthly Cost/Credit")
    plt.xlabel("Month"); plt.ylabel("$")
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{state}_monthly_costs.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(12,5))
    plt.plot(df.index, safe_get(df,"Import_kWh"), label="Import (kWh)")
    plt.plot(df.index, safe_get(df,"Export_kWh"), label="Export (kWh)")
    plt.title(f"{state.capitalize()}: Monthly Import/Export Energy")
    plt.xlabel("Month"); plt.ylabel("Energy (kWh)")
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{state}_monthly_import_export.png"), dpi=160)
    plt.close()

def compute_metrics(state, daily_df, monthly_df):
    pv_kwh   = pd.to_numeric(daily_df.get("PV_E_kWh",   pd.Series(0, index=daily_df.index)), errors="coerce").sum()
    load_kwh = pd.to_numeric(daily_df.get("Load_E_kWh", pd.Series(0, index=daily_df.index)), errors="coerce").sum()
    exp_kwh  = pd.to_numeric(daily_df.get("Export_kWh", pd.Series(0, index=daily_df.index)), errors="coerce").sum()
    imp_kwh  = pd.to_numeric(daily_df.get("Import_kWh", pd.Series(0, index=daily_df.index)), errors="coerce").sum()

    buy_usd  = pd.to_numeric(monthly_df.get("Buy_$",      pd.Series(0, index=monthly_df.index)), errors="coerce").sum()
    sell_usd = pd.to_numeric(monthly_df.get("Sell_$",     pd.Series(0, index=monthly_df.index)), errors="coerce").sum()
    net_usd  = pd.to_numeric(monthly_df.get("NetBill_$",  pd.Series(0, index=monthly_df.index)), errors="coerce").sum()

    solar_fraction   = 0.0 if load_kwh == 0 else (load_kwh - imp_kwh) / load_kwh
    export_fraction  = 0.0 if pv_kwh   == 0 else exp_kwh / pv_kwh
    self_consumption = 1 - export_fraction
    eff_cost_per_kWh = np.nan if load_kwh == 0 else net_usd / load_kwh

    return {
        "state": state,
        "total_PV_kWh": pv_kwh,
        "total_Load_kWh": load_kwh,
        "total_Export_kWh": exp_kwh,
        "total_Import_kWh": imp_kwh,
        "solar_fraction_of_load": round(float(solar_fraction), 3),
        "export_fraction_of_PV": round(float(export_fraction), 3),
        "self_consumption_ratio": round(float(self_consumption), 3),
        "total_buy_$": float(buy_usd),
        "total_sell_$": float(sell_usd),
        "total_net_bill_$": float(net_usd),
        "effective_cost_$_per_kWh_of_load": None if np.isnan(eff_cost_per_kWh) else round(float(eff_cost_per_kWh), 4),
    }

# gather per-state metrics
daily_map   = {state_name_from(p): p for p in daily_files}
monthly_map = {state_name_from(p): p for p in monthly_files}
states = sorted(set(daily_map) & set(monthly_map))

metrics = []
for state in states:
    ddf = load_csv(daily_map[state],   parse_index=True)
    mdf = load_csv(monthly_map[state], parse_index=True)
    plot_daily(state, ddf)
    plot_monthly(state, mdf)
    metrics.append(compute_metrics(state, ddf, mdf))

metrics_df = pd.DataFrame(metrics).sort_values("state")
metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

print("✅ Plots written to:", OUT_DIR)
print("   • summary_savings_by_state.png (if summary file present)")
print("   • <state>_daily_import_export.png")
print("   • <state>_daily_netbill.png")
print("   • <state>_monthly_costs.png")
print("   • <state>_monthly_import_export.png")
print("📄 Metrics table: net_metering_plots/metrics_summary.csv")
