# streamlit app file
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# Use the direct asset URL, not the release page URL.
# Replace OWNER, REPO, TAG, and FILE_NAME with your real release values.
MODEL_URL = "https://github.com/sotoantonio/Cooler-Power-Predict-streamlit/releases/download/V.1.0.0/ml_power_model_1.pkl"
MODEL_PATH = Path(__file__).with_name("ml_power_model_1.pkl")

def download_model():
    response = requests.get(MODEL_URL, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type:
        raise ValueError(
            "The model URL returned HTML instead of the pickle file. "
            "Use the GitHub release asset download URL that contains /releases/download/."
        )

    model_bytes = response.content
    if len(model_bytes) < 1024:
        raise ValueError("Downloaded model file is unexpectedly small. Check the release asset URL.")

    MODEL_PATH.write_bytes(model_bytes)


@st.cache_resource
def load_model():
    # First try local file if it already exists.
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            # Remove corrupted/stale file and fetch a clean copy.
            MODEL_PATH.unlink(missing_ok=True)

    download_model()
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(
        f"Failed to load the model from the GitHub release asset: "
        f"{type(exc).__name__}: {exc}"
    )
    st.caption(f"Checked URL: {MODEL_URL}")
    st.stop()

st.title("Cooling System Power Simulator")
st.sidebar.header("System Controls")
# --- Inputs ---
fan1 = st.sidebar.slider("Tower Fan 1", 0.0, 100.0, 50.0)
fan2 = st.sidebar.slider("Tower Fan 2", 0.0, 100.0, 50.0)
fan3 = st.sidebar.slider("Tower Fan 3", 0.0, 100.0, 50.0)
fan4 = st.sidebar.slider("Tower Fan 4", 0.0, 100.0, 50.0)

chiller1 = st.sidebar.selectbox("Chiller 1 On", [0, 1])
chiller2 = st.sidebar.selectbox("Chiller 2 On", [0, 1])
chiller3 = st.sidebar.selectbox("Chiller 3 On", [0, 1])

pumpA = st.sidebar.selectbox("Pump A On", [0, 1])
pumpB = st.sidebar.selectbox("Pump B On", [0, 1])

secondary_pump = st.sidebar.slider("Secondary Pump Power", 0.0, 500.0, 100.0)

load = st.sidebar.slider("Cooling Load", 0.0, 1000.0, 500.0)
db_temp = st.sidebar.slider("Dry Bulb Temp", 0.0, 120.0, 75.0)
wb_temp = st.sidebar.slider("Wet Bulb Temp", 0.0, 100.0, 65.0)

# --- Create input array ---
input_data = np.array([[fan1, fan2, fan3, fan4,
                        chiller1, chiller2, chiller3,
                        pumpA, pumpB,
                        secondary_pump,
                        load, db_temp, wb_temp]])

# Predict
prediction = model.predict(input_data)[0]

# --- Output ---
st.subheader("Predicted System Power")
st.metric(label="Total Power (kW)", value=f"{prediction:.2f}")

# Optional: show inputs
if st.checkbox("Show Input Data"):
    df = pd.DataFrame(input_data, columns=[
        "Fan1","Fan2","Fan3","Fan4",
        "Ch1","Ch2","Ch3",
        "PumpA","PumpB",
        "SecPump","Load","DB","WB"
    ])
    st.dataframe(df)

st.write("### System Insight")

if prediction > 1000:
    st.warning("⚠️ High energy usage detected")
elif prediction < 500:
    st.success("✅ Efficient operating range")
