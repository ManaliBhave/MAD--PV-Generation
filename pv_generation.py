import pandas as pd
import numpy as np

# ----------------------------
# PARAMETERS (customize here)
# ----------------------------
panel_area = 10.0                # m²
module_efficiency = 0.18         # 18%
temp_coeff = -0.004              # per °C
NOCT = 45                        # Nominal Operating Cell Temp (°C)
inverter_eff = 0.96              # 96%
# other_losses = 0.08              # 8% losses (wiring, mismatch, etc.)
panel_tilt = 30                  # degrees (tilt of PV)
panel_azimuth = 180              # south-facing in northern hemisphere

# ----------------------------
# LOAD DATA
# ----------------------------
# Example: Replace 'your_file.csv' with your dataset
df = pd.read_csv("solcast_5yr_hybrid_forecast.csv")

# Ensure correct data types
df['period_end'] = pd.to_datetime(df['period_end'])

# ----------------------------
# CALCULATIONS
# ----------------------------

# 1. Compute solar elevation angle
df['elevation'] = 90 - df['zenith']

# 2. Compute angle of incidence θ (radians)
def calc_cos_theta(zenith, azimuth_sun, tilt, azimuth_panel):
    # convert to radians
    z = np.radians(zenith)
    a_sun = np.radians(azimuth_sun)
    a_panel = np.radians(azimuth_panel)
    b = np.radians(tilt)
    cos_theta = np.cos(z) * np.cos(b) + np.sin(z) * np.sin(b) * np.cos(a_sun - a_panel)
    return np.clip(cos_theta, 0, 1)

df['cos_theta'] = calc_cos_theta(df['zenith'], df['azimuth'], panel_tilt, panel_azimuth)

# 3. Determine Plane-of-Array (POA) Irradiance (E_poa)
# Prefer measured 'gti' if available, else compute from components
df['E_poa'] = np.where(
    df['gti'] > 0,
    df['gti'],
    df['dni'] * df['cos_theta'] + df['dhi'] * ((1 + np.cos(np.radians(panel_tilt))) / 2)
    + df['ghi'] * df['albedo'] * ((1 - np.cos(np.radians(panel_tilt))) / 2)
)

# 4. Compute module temperature (NOCT model)
df['T_cell'] = df['air_temp'] + (df['E_poa'] / 800) * (NOCT - 20)

# 5. Temperature correction factor
df['temp_factor'] = 1 + temp_coeff * (df['T_cell'] - 25)

# 6. DC Power (kW)
df['P_dc'] = (df['E_poa'] * panel_area * module_efficiency * df['temp_factor']) / 1000

# 7. AC Power after inverter
df['P_ac'] = df['P_dc'] * inverter_eff

# 8. Apply soiling, shading, and other losses
df['P_actual'] = df['P_ac'] * (1 - df['snow_soiling_rooftop']/100) #* (1 - other_losses)

# 9. Energy generated in the interval (kWh)
# If each record is 15-min interval → multiply power (kW) * 0.25 hours
interval_hours = 0.25  # Change to 0.25 if your data is 15-minute intervals
df['E_kWh'] = df['P_actual'] * interval_hours

# ----------------------------
# OUTPUT RESULTS
# ----------------------------
print(df[['period_end', 'E_poa', 'T_cell', 'P_actual', 'E_kWh']].head())

# Optionally save to file
df.to_csv("pv_generation_results_5y.csv", index=False)
print("\nSaved results → pv_generation_results_5y.csv")
 