import os
import pandas as pd
import plotly.express as px
import streamlit as st
import google.generativeai as genai
from pathlib import Path
import re
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
 
 
# --------------------------------------------------------------------
# 🌟 GEMINI 2.5 FLASH SETUP
# --------------------------------------------------------------------
GEMINI_API_KEY = "AIzaSyBkigkItuqaRN2bSpnz5J2ZqwFHBwV0350"  # ⚠️ Replace with your key if needed
 
# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
 
# ✅ Use latest Gemini 2.5 Flash model (supports generate_content)
MODEL_ID = "gemini-2.5-flash"
model = genai.GenerativeModel(MODEL_ID)
 
 
def safe_read_csv(path, **kwargs):
    df = pd.read_csv(path, **kwargs)
    for col in df.columns:
        lc = str(col).lower()
        if any(k in lc for k in ["time", "date", "timestamp", "datetime"]):
            try:
                s = pd.to_datetime(df[col], errors="coerce", utc=True)
                if pd.api.types.is_datetime64tz_dtype(s):
                    s = s.dt.tz_convert(None)
                df[col] = s
            except Exception:
                pass
    return df
 
# --------------------------------------------------------------------
# 📂 Auto-load all CSVs
# --------------------------------------------------------------------
def load_all_data(base_dir: Path = Path.cwd()) -> dict[str, pd.DataFrame]:
    subfolders = ["net_metering", "pv_generation", "object", "shading", "data"]
    data_dict = {}
    for folder in subfolders:
        folder_path = base_dir / folder
        if folder_path.exists():
            csv_files = list(folder_path.glob("*.csv"))
            for csv_file in csv_files:
                try:
                    df = safe_read_csv(csv_file)  # ✅ replaced here
                    data_dict[f"{folder}/{csv_file.name}"] = df
                except Exception as e:
                    st.warning(f"⚠️ Could not read {csv_file.name}: {e}")
        else:
            st.warning(f"❌ Folder not found: {folder_path}")
    return data_dict
 
 
# --------------------------------------------------------------------
# 💬 Gemini AI Helper
# --------------------------------------------------------------------
def ask_ai(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return getattr(response, "text", str(response))
    except Exception as e:
        return f"❌ Model error: {e}"
 
 
# --------------------------------------------------------------------
# 🤖 MAIN CHATBOT FUNCTION
# --------------------------------------------------------------------
def run_ai_chatbot():
    st.set_page_config(page_title="⚡ Shady Data Chat Assistant", page_icon="🌞", layout="wide")
 
    st.title("⚡ Shady Data Chat Assistant")
    st.caption("Chat with your solar datasets — ask questions and get visual or analytical answers instantly.")
 
    # ✅ Adjusted base directory (parent folder containing data folders)
    base_dir = Path(__file__).resolve().parents[1]
    all_data = load_all_data(base_dir)
 
    if not all_data:
        st.error(f"❌ No CSV files found under {base_dir}. Check your folder paths.")
        st.stop()
    else:
        st.success(f"✅ Loaded {len(all_data)} CSV files successfully from {base_dir}.")
 
    # 🕒 Optional: show example time columns
    for name, df in all_data.items():
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                sample_val = df[col].dropna().astype(str).head(1).to_list()
                st.write(f"🕒 {name} → Column: {col} → Example: {sample_val}")
 
    # Store chat conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "chart" in msg:
                st.plotly_chart(msg["chart"], use_container_width=True)
 
    # Chat input
    user_input = st.chat_input("Ask something about your solar data...")
 
    if not user_input:
        return
 
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
 
    # --------------------------------------------------------------------
    # 🧠 Dynamically select relevant datasets based on user's question
    # --------------------------------------------------------------------
    sample_text = ""
 
    if "pv" in user_input.lower() or "generation" in user_input.lower():
        relevant = [(name, df) for name, df in all_data.items() if "pv_generation" in name.lower()]
    elif "net" in user_input.lower() or "metering" in user_input.lower():
        relevant = [(name, df) for name, df in all_data.items() if "net_metering" in name.lower()]
    elif "shade" in user_input.lower():
        relevant = [(name, df) for name, df in all_data.items() if "shading" in name.lower()]
    elif "object" in user_input.lower():
        relevant = [(name, df) for name, df in all_data.items() if "object" in name.lower()]
    else:
        relevant = list(all_data.items())  # fallback: include all
 
    # 🔹 Include ALL 10 sites (no limitation)
    for name, df in relevant:
        sample_text += f"### {name}\n{df.head(5).to_csv(index=False)}\n\n"
 
    # 💡 Optional improvement — provide Gemini full dataset overview
    file_summary = "\n".join([f"{name}: {list(df.columns)}" for name, df in all_data.items()])
 
    # --------------------------------------------------------------------
    # 🧠 AI context with file overview + dataset samples
    # --------------------------------------------------------------------
    context = (
        "You are an AI data analyst for a solar PV and energy monitoring system.\n"
        "You have access to multiple datasets from folders like pv_generation, net_metering, shading, object, and data.\n"
        "Each folder contains data from multiple sites. Use this to analyze trends across all 10 sites.\n\n"
        "Each folder represents:\n"
        "  • pv_generation → PV_E_kWh, timestamp_local (energy produced)\n"
        "  • net_metering → Net_Import_kWh, Net_Export_kWh (grid exchange)\n"
        "  • shading → shading percentage vs time\n"
        "  • object → nearby obstruction metrics\n"
        "  • data → weather/environment readings.\n\n"
        "You can answer analytical and comparative questions using these datasets.\n"
        "When visualizing data:\n"
        "  • Always produce valid Python code inside ```python\n"
        "  • Define a Plotly figure named `fig`\n"
        "  • Do not call fig.show() or print anything.\n\n"
        f"Here is a list of all available files and their columns:\n{file_summary}\n\n"
        f"Dataset samples (showing first few rows per file):\n{sample_text}\n\n"
        f"User question:\n{user_input}\n"
    )
 
    # --------------------------------------------------------------------
    # 🔥 Ask Gemini
    # --------------------------------------------------------------------
    ai_reply = ask_ai(context)
 
    with st.chat_message("assistant"):
        st.markdown(ai_reply)
 
        # Extract Python code
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", ai_reply, re.DOTALL)
        code_to_run = code_blocks[0].strip() if code_blocks else ""
 
        # --------------------------------------------------------------------
        # 🔧 Fix CSV paths in AI code
        # --------------------------------------------------------------------
        subfolders = ["pv_generation", "net_metering", "object", "shading", "data"]
 
        def find_csv_in_subfolders(filename):
            for folder in subfolders:
                possible_path = base_dir / folder / filename
                if possible_path.exists():
                    return str(Path(folder) / filename)
            for folder in subfolders:
                folder_path = base_dir / folder
                if folder_path.exists():
                    for possible in folder_path.rglob("*.csv"):
                        if possible.name.lower() == filename.lower():
                            return str(possible.relative_to(base_dir))
            st.warning(f"⚠️ File '{filename}' not found under {base_dir}.")
            return filename
 
 
        def fix_csv_paths(code: str) -> str:
            """
            - Rewrites pd.read_csv('...') to absolute paths under base_dir (as today).
            - Also replaces pd.read_csv( with safe_read_csv( to enforce datetime parsing.
            """
            # First, rewrite explicit file paths to absolute ones
            pattern = r"pd\.read_csv\(['\"]([^'\"]+)['\"]\)"
            matches = re.findall(pattern, code)
 
            for fname in matches:
                found_path = None
 
                # Already absolute?
                if os.path.isabs(fname):
                    continue
 
                # Check each known data folder and raw relative
                for folder in ["pv_generation", "net_metering", "object", "shading", "data"]:
                    possible_path = base_dir / fname
                    if possible_path.exists():
                        found_path = possible_path
                        break
 
                    possible_path = base_dir / folder / Path(fname).name
                    if possible_path.exists():
                        found_path = possible_path
                        break
 
                if found_path:
                    safe_path = str(found_path.resolve()).replace("\\", "/")
                    code = re.sub(
                        rf"pd\.read_csv\(['\"]{re.escape(fname)}['\"]\)",
                        f"pd.read_csv(r'{safe_path}')",
                        code
                    )
                    st.info(f"📂 Fixed path for '{fname}' → {safe_path}")
                else:
                    st.warning(f"⚠️ Could not find CSV: {fname}")
 
            # Second, enforce our safe reader (handles datetime coercion)
            code = re.sub(r"\bpd\.read_csv\(", "safe_read_csv(", code)
 
            return code
 
        code_to_run = fix_csv_paths(code_to_run)
 
        # --------------------------------------------------------------------
        # 🧩 Merge all datasets & enrich with Location
        # --------------------------------------------------------------------
        df_list = []
        for name, df in all_data.items():
            location = Path(name).stem.split("_")[0].capitalize()
            df["Location"] = location
            df["source"] = name
            df_list.append(df)
 
        if not df_list:
            st.error("⚠️ No CSV data found to combine.")
            st.stop()
 
        df_all = pd.concat(df_list, ignore_index=True)
 
        if isinstance(df_all.index, pd.DatetimeIndex):
            df_all = df_all.reset_index()
 
        # Ensure timestamp_local stays a column even if index was datetime
        if "timestamp_local" not in df_all.columns:
            time_like_cols = [c for c in df_all.columns if "time" in c.lower() or "date" in c.lower()]
            if not time_like_cols and isinstance(df_all.index, pd.DatetimeIndex):
                df_all["timestamp_local"] = df_all.index
            elif time_like_cols:
                df_all.rename(columns={time_like_cols[0]: "timestamp_local"}, inplace=True)
 
        # 🕒 Clean datetime columns safely
        def clean_datetime_columns(df):
            for col in df.columns:
                if any(x in col.lower() for x in ["time", "date", "timestamp"]):
                    try:
                        # Convert string columns to datetime
                        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
 
                        # If timezone-aware, drop timezone for Plotly compatibility
                        if pd.api.types.is_datetime64tz_dtype(df[col]):
                            df[col] = df[col].dt.tz_convert(None)
                    except Exception as e:
                        st.warning(f"⚠️ Could not parse datetime column '{col}': {e}")
            return df
 
        df_all = clean_datetime_columns(df_all)
 
        def pick_time_col(frame: pd.DataFrame):
            for c in frame.columns:
                if any(k in c.lower() for k in ["timestamp", "time", "date"]):
                    try:
                        pd.to_datetime(frame[c], errors="raise")
                        return c
                    except Exception:
                        pass
            return None
 
        # --------------------------------------------------------------------
        # 🚀 Execute AI-generated chart code
        # --------------------------------------------------------------------
        if code_to_run:
            try:
                exec_env = {
                    "px": px,
                    "pd": pd,
                    "st": st,
                    "df": df_all,
                    "go": go,
                    "np": np,
                    "plt": plt,
                    "pick_time_col": pick_time_col,
                    "clean_datetime_columns": clean_datetime_columns,  # <-- add this helper
                    "safe_read_csv": safe_read_csv,
                }
                exec(code_to_run, exec_env)
                fig = exec_env.get("fig")
 
                if isinstance(fig, go.Figure):
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("✅ AI-generated chart.")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "Chart rendered.", "chart": fig}
                    )
                else:
                    st.warning("⚠️ The model did not create a valid Plotly figure named `fig`.")
            except Exception as e:
                st.warning(f"⚠️ Chart execution error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Execution error: {e}"}
                )
        else:
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
 