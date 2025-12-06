#include "Arduino.h"
#include "TensorFlowLite.h"
#include "anomaly_model_data.h" // TFLite model converted to C array

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>

#define DHTPIN 14
#define DHTTYPE DHT22

DHT dht(DHTPIN, DHTTYPE);

// TensorFlow Lite globals
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

void setup() {
  Serial.begin(115200);
  dht.begin();

  // Initialize TFLite Micro
  static tflite::MicroErrorReporter error_reporter;
  static tflite::MicroMutableOpResolver<5> resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, tensor_arena_size, &error_reporter);

  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  float vibration = 0.01; // placeholder

  input->data.f[0] = temp;
  input->data.f[1] = hum;
  input->data.f[2] = vibration;

  interpreter->Invoke();

  float anomaly = output->data.f[0];
  Serial.print("Temp: "); Serial.print(temp);
  Serial.print(", Hum: "); Serial.print(hum);
  Serial.print(", Vibration: "); Serial.print(vibration);
  Serial.print(", Status: ");
  Serial.println(anomaly>0.5?"Anomaly":"Normal");

  delay(2000);
}
