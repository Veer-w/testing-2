# app.py

import streamlit as st
import pandas as pd
import time
import joblib
from ml_model import SyntheticDataGenerator
import altair as alt

# Load the trained model
model = joblib.load("predictive_model.pkl")

sensor_names = ["Temperature Sensor", "Pressure Sensor", "Vibration Sensor", "Humidity Sensor"]
generator = SyntheticDataGenerator(sensor_names)

if "sensor_data" not in st.session_state:
    st.session_state.sensor_data = pd.DataFrame(columns=["timestamp"] + sensor_names)

# Function to update sensor data
def update_sensor_data(new_data):
    st.session_state.sensor_data = pd.concat([st.session_state.sensor_data, pd.DataFrame([new_data])], ignore_index=True)

st.sidebar.title("Sensor Thresholds")
thresholds = {sensor: st.sidebar.slider(sensor, 0, 100, 70) for sensor in sensor_names}

st.title("Predictive Maintenance Dashboard")
st.subheader("Real-time Sensor Data")

chart_placeholders = {sensor: st.empty() for sensor in sensor_names}
alert_placeholders = {sensor: st.empty() for sensor in sensor_names}

# Function to create Altair line chart
def create_line_chart(data, sensor):
    chart = alt.Chart(data).mark_line().encode(
        x='timestamp:T',
        y=f'{sensor}:Q'
    ).properties(
        width=600,
        height=300,
        title=f'{sensor} Readings'
    )
    return chart

# Main loop to generate and update sensor data
while True:
    sensor_data = generator.generate_data()
    update_sensor_data(sensor_data)
    
    # Convert timestamp to datetime
    st.session_state.sensor_data['timestamp'] = pd.to_datetime(st.session_state.sensor_data['timestamp'])

    # Prepare the data for prediction
    sensor_df = pd.DataFrame([sensor_data])  # Single data point as DataFrame
    sensor_features = sensor_df[sensor_names]

    # Make a prediction with the model
    prediction = model.predict(sensor_features)[0]

    # Display sensor charts and alerts
    for sensor in sensor_names:
        chart_placeholders[sensor].altair_chart(create_line_chart(st.session_state.sensor_data, sensor))

        # Display prediction alert if the model predicts failure
        if prediction == 1:
            alert_placeholders[sensor].error(f"Warning! Model predicts {sensor} may soon exceed the threshold.")
        elif sensor_data[sensor] > thresholds[sensor]:
            alert_placeholders[sensor].error(f"{sensor} has crossed the threshold with a reading of {sensor_data[sensor]:.2f}")
        else:
            alert_placeholders[sensor].empty()  

    time.sleep(1)
