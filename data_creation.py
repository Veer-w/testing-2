# train_model.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume `sensor_data` is a DataFrame with sensor values and a target label for failure
# For simplicity, here’s a small example. Replace this with your actual dataset.
sensor_data = pd.DataFrame({
    'Temperature Sensor': [22, 80, 30, 90, 15],
    'Pressure Sensor': [30, 85, 40, 92, 18],
    'Vibration Sensor': [20, 90, 35, 88, 12],
    'Humidity Sensor': [25, 88, 33, 85, 17],
    'Failure': [0, 1, 0, 1, 0]  # Label indicating failure (1) or normal (0)
})

X = sensor_data.drop(columns=['Failure'])
y = sensor_data['Failure']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, "predictive_model.pkl")
print("Model trained and saved as 'predictive_model.pkl'")
