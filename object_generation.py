import pandas as pd
import numpy as np
import math
import random
 
# -----------------------------
# CONFIGURATION
# -----------------------------
CENTER_LAT = 25.76168
CENTER_LON = -80.191179
NUM_OBJECTS = 10
NUM_PV_METERS = 10
np.random.seed(42)
 
# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def random_azimuth():
    return np.random.uniform(0, 360)
 
def random_distance():
    # distance in meters from PV panel
    return np.random.uniform(5, 50)
 
def offset_coordinates(lat, lon, distance_m, azimuth_deg):
    # approximate conversion for small distances
    earth_radius = 6378137
    d_lat = (distance_m * math.cos(math.radians(azimuth_deg))) / earth_radius
    d_lon = (distance_m * math.sin(math.radians(azimuth_deg))) / (earth_radius * math.cos(math.radians(lat)))
    return lat + math.degrees(d_lat), lon + math.degrees(d_lon)
 
def random_object_type():
    return np.random.choice(["Tree", "Building", "Pole"])
 
def shading_intensity(obj_type):
    return {"Building": 0.7, "Tree": 0.4, "Pole": 0.3}[obj_type]
 
# -----------------------------
# 1️⃣ Generate Object Data
# -----------------------------
objects = []
for i in range(NUM_OBJECTS):
    obj_type = random_object_type()
    height = np.random.uniform(3, 30) if obj_type != "Pole" else np.random.uniform(5, 12)
    width = np.random.uniform(2, 15) if obj_type != "Pole" else np.random.uniform(0.3, 1)
    azimuth = random_azimuth()
    distance = random_distance()
    lat, lon = offset_coordinates(CENTER_LAT, CENTER_LON, distance, azimuth)
    shade_intensity = shading_intensity(obj_type)
 
    objects.append({
        "object_id": f"OBJ_{i+1}",
        "object_type": obj_type,
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "height_m": round(height, 2),
        "width_m": round(width, 2),
        "azimuth_deg": round(azimuth, 1),
        "distance_m": round(distance, 1),
        "shading_intensity": shade_intensity
    })
 
df_objects = pd.DataFrame(objects)
df_objects.to_csv("object/florida_object.csv", index=False)
 
# -----------------------------
# 2️⃣ Generate PV Meter Data
# -----------------------------
# pv_data = []
# for i in range(NUM_PV_METERS):
#     pv_id = f"PV_{i+1}"
#     tilt = np.random.uniform(10, 40)
#     azimuth = 180  # south-facing (typical in Northern Hemisphere)
#     irradiance = np.random.uniform(400, 1000)  # W/m²
#     panel_efficiency = np.random.uniform(0.15, 0.22)
#     panel_area = np.random.uniform(5, 25)  # m²
#     ideal_generation = irradiance * panel_area * panel_efficiency / 1000  # kW
#     shading_loss = np.random.uniform(0, 0.2)
#     actual_generation = ideal_generation * (1 - shading_loss)
 
#     pv_data.append({
#         "pv_id": pv_id,
#         "latitude": CENTER_LAT + np.random.uniform(-0.0005, 0.0005),
#         "longitude": CENTER_LON + np.random.uniform(-0.0005, 0.0005),
#         "tilt_deg": round(tilt, 2),
#         "azimuth_deg": azimuth,
#         "irradiance_wm2": round(irradiance, 2),
#         "panel_efficiency": round(panel_efficiency, 3),
#         "panel_area_m2": round(panel_area, 2),
#         "ideal_generation_kw": round(ideal_generation, 3),
#         "shading_loss_factor": round(shading_loss, 3),
#         "actual_generation_kw": round(actual_generation, 3)
#     })
 
# df_pv = pd.DataFrame(pv_data)
# df_pv.to_csv("mock_pv_meters.csv", index=False)
 
# -----------------------------
# OUTPUT
# -----------------------------
print("✅ Generated Files:")
print(" - mock_objects.csv (surrounding objects)")
# print(" - mock_pv_meters.csv (PV panels and generation data)")
print("\nSample object data:\n", df_objects.head())
# print("\nSample PV data:\n", df_pv.head())
