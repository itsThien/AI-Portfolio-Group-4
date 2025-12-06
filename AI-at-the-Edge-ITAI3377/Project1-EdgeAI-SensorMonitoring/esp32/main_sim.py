import numpy as np
import tensorflow as tf
from sensor import SensorReader
import time

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="../model/anomaly_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

reader = SensorReader()

while True:
    sample = np.array(reader.read(), dtype=np.float32).reshape(1,3)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])[0][0]
    status = "Anomaly" if result > 0.5 else "Normal"
    print(f"Sensor: {sample.flatten()}, Model: {result:.2f}, Status: {status}")
    time.sleep(2)
