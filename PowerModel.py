#streamlit app file
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import requests

url= "https://drive.google.com/uc?export=download&id=1eIZGPVBif2YciB3MBGsoB--gd8Hw6UEu"

@st.cache_resource
def load_model():
    response = requests.get(url)
    with open("model.pkl", "wb") as f:
        f.write(response.content)
    return joblib.load("model.pkl")

model = load_model()

st.title("Cooling System Power Simulator")

# Load trained model
model = joblib.load("ml_power_model_1.pkl")
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