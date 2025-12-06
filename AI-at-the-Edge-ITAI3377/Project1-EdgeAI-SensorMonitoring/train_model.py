import pandas as pd
import numpy as np
import tensorflow as tf
from model import model as mod

# Load CSV
df = pd.read_csv("../data/sensor_readings.csv")
X = df[['temp','humidity','vibration']].values

# Simulate labels: 0=normal, 1=anomaly
y = np.zeros(len(X))
y[-5:] = 1  # last 5 rows are anomalies

# Create model
ml_model = mod.create_model(input_shape=(3,))
ml_model.fit(X, y, epochs=50, batch_size=4)

# Save TF model
ml_model.save("anomaly_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(ml_model)
tflite_model = converter.convert()
with open("anomaly_model.tflite","wb") as f:
    f.write(tflite_model)

print("TFLite model created: anomaly_model.tflite")
