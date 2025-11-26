import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
 
def main(csv_file):
    print(f"[INFO] Loading: {csv_file}")
    df = pd.read_csv(csv_file)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
 
    # ---------------------------------------------------------
    # 1. SHADING LOSS VS TIME
    # ---------------------------------------------------------
 
    fig1 = px.line(
        df,
        x="timestamp_utc",
        y="shading_loss",
        title="Shading Loss Over Time",
        labels={"shading_loss": "Shading Loss (fraction)"},
        markers=True
    )
 
    fig1.write_html("shading_vs_time.html")
    print("Saved → shading_vs_time.html")
 
    # ---------------------------------------------------------
    # 2. POWER COMPARISON: P_ac vs P_actual_new_kW
    # ---------------------------------------------------------
 
    fig2 = go.Figure()
 
    fig2.add_trace(go.Scatter(
        x=df["timestamp_utc"], y=df["P_ac"],
        mode='lines', name="Original P_ac", line=dict(color="blue")
    ))
 
    fig2.add_trace(go.Scatter(
        x=df["timestamp_utc"], y=df["P_actual_new_kW"],
        mode='lines', name="Shading-adjusted P_actual", line=dict(color="red")
    ))
 
    fig2.update_layout(
        title="PV Power Before vs After Shading",
        xaxis_title="Time",
        yaxis_title="Power (kW)"
    )
 
    fig2.write_html("power_comparison.html")
    print("Saved → power_comparison.html")
 
    # ---------------------------------------------------------
    # 3. ENERGY (kWh) OVER TIME
    # ---------------------------------------------------------
 
    fig3 = px.line(
        df,
        x="timestamp_utc",
        y="E_kWh_new",
        title="PV Energy Generated per Interval",
        labels={"E_kWh_new": "Energy (kWh)"},
        markers=True,
    )
 
    fig3.write_html("pv_energy.html")
    print("Saved → pv_energy.html")
 
    # ---------------------------------------------------------
    # 4. OBJECT CONTRIBUTIONS (Shading Objects)
    #     Count number of objects contributing at each timestamp
    # ---------------------------------------------------------
 
    def count_objects(x):
        if isinstance(x, str) and x.strip():
            return len(x.split(";"))
        return 0
 
    df["num_objects"] = df["contrib_objects"].apply(count_objects)
 
    fig4 = px.bar(
        df,
        x="timestamp_utc",
        y="num_objects",
        title="Number of Shading Objects Affecting PV Over Time",
        labels={"num_objects": "# of Objects Shading"},
    )
 
    fig4.write_html("shading_objects.html")
    print("Saved → shading_objects.html")
    print("\n🎉 All visualizations generated successfully!")
    print("Files created:")
    print(" - shading_vs_time.html")
    print(" - power_comparison.html")
    print(" - pv_energy.html")
    print(" - shading_objects.html")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to pv_shaded.csv")
    args = parser.parse_args()
    main(args.file)