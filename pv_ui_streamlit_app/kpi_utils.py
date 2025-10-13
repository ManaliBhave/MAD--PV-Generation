import numpy as np
import pandas as pd

def safe_sum(series):
    try:
        return float(pd.to_numeric(series, errors="coerce").dropna().sum())
    except Exception:
        return 0.0

def annualize_monthly(df_monthly: pd.DataFrame):
    if df_monthly is None or len(df_monthly) == 0:
        return {}
    # Sum across available months
    cols = [c for c in df_monthly.columns if str(c) != "timestamp_local"]
    totals = {col: safe_sum(df_monthly[col]) for col in cols}

    load_kwh   = totals.get("Load_E_kWh", 0.0)
    pv_kwh     = totals.get("PV_E_kWh", 0.0)
    import_kwh = totals.get("Import_KWh", totals.get("Import_kWh", 0.0))
    export_kwh = totals.get("Export_KWh", totals.get("Export_kWh", 0.0))
    buy_cost   = totals.get("Buy_$", 0.0)
    sell_credit= totals.get("Sell_$", 0.0)
    net_bill   = totals.get("NetBill_$", 0.0)

    avg_buy_rate = (buy_cost / import_kwh) if import_kwh > 0 else 0.0
    baseline_bill = load_kwh * avg_buy_rate
    savings = baseline_bill - net_bill

    self_consumed_kwh = max(pv_kwh - export_kwh, 0.0)
    self_consumption = (self_consumed_kwh / pv_kwh * 100.0) if pv_kwh > 0 else 0.0
    self_sufficiency = (self_consumed_kwh / load_kwh * 100.0) if load_kwh > 0 else 0.0

    return {
        "annual_pv_kwh": pv_kwh,
        "annual_load_kwh": load_kwh,
        "annual_import_kwh": import_kwh,
        "annual_export_kwh": export_kwh,
        "net_bill": net_bill,
        "baseline_bill_est": baseline_bill,
        "savings_est": savings,
        "self_consumption_pct": self_consumption,
        "self_sufficiency_pct": self_sufficiency,
        "avg_buy_rate": avg_buy_rate,
    }

def shading_kpis(df_shading: pd.DataFrame):
    if df_shading is None or df_shading.empty:
        return {"avg_shading_loss_pct": None}
    if "shading_loss" not in df_shading.columns:
        return {"avg_shading_loss_pct": None}
    s = pd.to_numeric(df_shading["shading_loss"], errors="coerce")
    if s.dropna().empty:
        return {"avg_shading_loss_pct": None}
    factor = 100.0 if s.max(skipna=True) <= 1.5 else 1.0
    return {"avg_shading_loss_pct": float(s.mean(skipna=True) * factor)}
