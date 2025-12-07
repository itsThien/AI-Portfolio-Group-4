import random

class SensorReader:
    def read(self):
        temp = round(random.uniform(20,30),1)
        hum = round(random.uniform(40,60),1)
        vibration = round(random.uniform(0,0.05),3)
        return [temp, hum, vibration]
