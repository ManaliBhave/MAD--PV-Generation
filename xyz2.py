import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------
# PARAMETERS (same as before)
# ----------------------------
 
panel_area = 10.0                 # m²
module_efficiency = 0.18          # 18%
temp_coeff = -0.004               # -0.4% per °C
NOCT = 45                         # °C
inverter_eff = 0.96               # 96%
panel_tilt = 30                   # degrees
panel_azimuth = 180               # degrees
# 0.25 = 15-min interval, 1.0 = hourly
interval_hours = 1.0              
 
 
# ----------------------------
# LOAD MERGED OPEN-METEO FILE
# ----------------------------
 
df = pd.read_csv("merged_open_meteo.csv")
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
df = df.set_index("timestamp_utc")
 
 
# ----------------------------
# 1. Solar elevation
# ----------------------------
 
if "zenith" in df.columns:
    df["elevation"] = 90 - df["zenith"]
else:
    # if zenith not available → compute via pvlib
    import pvlib
    lat, lon = 52.52, 13.41  # adjust as needed
    solpos = pvlib.solarposition.get_solarposition(df.index, lat, lon)
    df["zenith"] = solpos["zenith"]
    df["azimuth"] = solpos["azimuth"]
    df["elevation"] = 90 - df["zenith"]
 

# ----------------------------
# 2. Angle of incidence
# ----------------------------
 
def calc_cos_theta(zenith, azimuth_sun, tilt, azimuth_panel):
    z = np.radians(zenith)
    a_sun = np.radians(azimuth_sun)
    a_panel = np.radians(azimuth_panel)
    b = np.radians(tilt)
    cos_theta = (
        np.cos(z) * np.cos(b)
        + np.sin(z) * np.sin(b) * np.cos(a_sun - a_panel)
    )
    return np.clip(cos_theta, 0, 1)

df["cos_theta"] = calc_cos_theta(
    df["zenith"], df["azimuth"], panel_tilt, panel_azimuth
)
 
 
# ----------------------------
# 3. POA irradiance (E_poa)
# ----------------------------
 
df["E_poa"] = np.where(
    df["gti"].notna() & (df["gti"] > 0),
    df["gti"],
    df["dni"] * df["cos_theta"]
    + df["dhi"] * ((1 + np.cos(np.radians(panel_tilt))) / 2) 
    + df["ghi"] * df.get("albedo", 0.2) * ((1 - np.cos(np.radians(panel_tilt))) / 2)
)
 
 
# ----------------------------
# 4. Cell temperature (NOCT)
# ----------------------------
 
df["T_cell"] = df["air_temp"] + (df["E_poa"] / 800) * (NOCT - 20)
 
 
# ----------------------------
# 5. Temperature correction
# ----------------------------
 
df["temp_factor"] = 1 + temp_coeff * (df["T_cell"] - 25)
 
 
# ----------------------------
# 6. DC Power (kW)
# ----------------------------
 
df["P_dc"] = (
    df["E_poa"]
    * panel_area
    * module_efficiency 
    * df["temp_factor"] 
) / 1000.0
 
 
# ---------------------------- 
# 7. AC Power (kW) 
# ----------------------------
 
df["P_ac"] = df["P_dc"] * inverter_eff
 
 
# ---------------------------- 
# 8. Include soiling losses 
# ----------------------------
 
if "snow_soiling_rooftop" in df.columns: 
    df["P_actual"] = df["P_ac"] * (1 - df["snow_soiling_rooftop"] / 100) 
else:
    df["P_actual"] = df["P_ac"]
 
 
# ----------------------------
# 9. Energy produced (kWh)
# ----------------------------
 
df["E_kWh"] = df["P_actual"] * interval_hours
 
 
# ----------------------------
# SAVE OUTPUT
# ----------------------------
 
df.to_csv("pv_generation_results.csv")
print("\nSaved → pv_generation_results.csv")
print(df[["E_poa", "T_cell", "P_actual", "E_kWh"]].tail(20))
 
# ============================================================
#                 VISUALIZE THE PV GENERATION
# ============================================================
 
plt.figure(figsize=(18, 6))
plt.plot(df.index, df["P_actual"], label="PV Power (kW)", color='orange')
plt.title("PV Power Output - Hourly")
plt.xlabel("Time")
plt.ylabel("kW")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pv_power_plot.png", dpi=200)
print("Saved → pv_power_plot.png")
 
 
plt.figure(figsize=(18, 6))
df["E_kWh"].cumsum().plot(color='green', label="Cumulative Energy (kWh)")
plt.title("Cumulative PV Energy Production")
plt.xlabel("Time")
plt.ylabel("kWh")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pv_cumulative_energy.png", dpi=200)
print("Saved → pv_cumulative_energy.png")
 
print("\n--- PV CALCULATION + VISUALIZATION DONE ---")