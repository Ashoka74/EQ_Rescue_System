
import streamlit as st
from google.generativeai.types import content_types
from collections.abc import Iterable
import time
from sodapy import Socrata
import json
from IPython.display import display
from IPython.display import Markdown
# import ABC and abstract
from abc import ABC, abstractmethod
from enum import Enum
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from typing import List, Tuple, Optional, Dict, Any

from utils import GeminiConfig


gemini_api = 'AIzaSyAEALAXiaE1HcD8qcN1duY4OtmUDfYqquk'
model_path = 'models/gemini-1.5-flash'
response_type = 'application/json'

class SupportType(Enum):
    PSYCHOLOGICAL = 'psychological'
    PHYSICAL = 'physical'
    EMOTIONAL = 'emotional'

class DisasterType(Enum):
    EARTHQUAKE = 'earthquake'
    WILDFIRE = 'wildfire'
    FLOOD = 'flood'
    TSUNAMI = 'tsunami'
    BOMB = 'bomb'

class InformType(Enum):
    ROAD_BLOCKS = 'road_blocks'
    FRIENDS_STATUS = 'friend_status'
    HELP_STATUS = 'help_status'
    QUEUE_POSITION = 'queue_pos'


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

class Sensor:
    TYPE_ACCELEROMETER = 1
    TYPE_GYROSCOPE = 4
    TYPE_GRAVITY = 9

class SensorEvent:
    def __init__(self, sensor: Sensor, values: List[float], timestamp: int):
        self.sensor = sensor
        self.values = values
        self.timestamp = timestamp

class SensorManager:
    def __init__(self):
        self.sensors = []
        self.listeners = []

    def getDefaultSensor(self, sensor_type: int) -> Sensor:
        # Simulated method to get a default sensor
        return Sensor()

    def registerListener(self, listener, sensor: Sensor, sampling_period_us: int):
        self.listeners.append((listener, sensor, sampling_period_us))

    def unregisterListener(self, listener):
        self.listeners = [l for l in self.listeners if l[0] != listener]

class SensorEventListener:
    def onSensorChanged(self, event: SensorEvent):
        pass

    def onAccuracyChanged(self, sensor: Sensor, accuracy: int):
        pass


class GenerateResponse(GeminiConfig):
    def __init__(self, img):
        super().__init__(gemini_api, model_path, response_type)
        self.response = self.model.generate_content(img)

    def to_markdown(self):
        return to_markdown(self.response.text)


class SupportAgent:
    def __init__(self, user_id, support_type: SupportType, disaster_type: DisasterType, aim):
        self.user_id = user_id
        self.support_type = support_type
        self.disaster_type = disaster_type
        self.aim = aim
        self.conversation = []

    def generate_prompt(self):
        return f'You are a {self.support_type.value} support agent. Your aim is to {self.aim} for a victim of a {self.disaster_type.value}. This is the history of conversation {self.conversation}'

    
class InformAgent:
    def __init__(self, user_id, information_type: InformType, disaster_type: DisasterType, aim):
        self.user_id = user_id
        self.information_type = information_type
        self.disaster_type = disaster_type
        self.aim = aim
        self.conversation = []

    def generate_prompt(self):
        return f'You are a {self.information_type.value} inform agent. Your aim is to {self.aim} for a victim of a {self.disaster_type.value}. This is the history of conversation {self.conversation}'



def get_accelerometer_data(sampling_period_us: int, duration_s: float) -> List[Tuple[float, float, float]]:
    sensor_manager = SensorManager()
    accelerometer = sensor_manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    data = []

    class AccelerometerListener(SensorEventListener):
        def onSensorChanged(self, event):
            data.append((event.values[0], event.values[1], event.values[2]))

    listener = AccelerometerListener()
    sensor_manager.registerListener(listener, accelerometer, sampling_period_us)

    time.sleep(duration_s)

    sensor_manager.unregisterListener(listener)
    return data


def get_gyroscope_data(sampling_period_us: int, duration_s: float) -> List[Tuple[float, float, float]]:
    sensor_manager = SensorManager()
    gyroscope = sensor_manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    data = []

    class GyroscopeListener(SensorEventListener):
        def onSensorChanged(self, event):
            data.append((event.values[0], event.values[1], event.values[2]))

    listener = GyroscopeListener()
    sensor_manager.registerListener(listener, gyroscope, sampling_period_us)

    time.sleep(duration_s)

    sensor_manager.unregisterListener(listener)
    return 


def detect_significant_motion(threshold: float, duration_s: float) -> bool:
    sensor_manager = SensorManager()
    accelerometer = sensor_manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    motion_detected = False

    class MotionListener(SensorEventListener):
        def onSensorChanged(self, event):
            nonlocal motion_detected
            acceleration = (event.values[0]**2 + event.values[1]**2 + event.values[2]**2)**0.5
            if acceleration > threshold:
                motion_detected = True

    listener = MotionListener()
    sensor_manager.registerListener(listener, accelerometer, SensorManager.SENSOR_DELAY_NORMAL)

    time.sleep(duration_s)

    sensor_manager.unregisterListener(listener)
    return motion_detected
import math


def get_device_orientation(sampling_period_us: int) -> Tuple[float, float, float]:
    sensor_manager = SensorManager()
    accelerometer = sensor_manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    magnetometer = sensor_manager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    
    accel_data = [0, 0, 0]
    mag_data = [0, 0, 0]
    orientation = [0, 0, 0]

    class OrientationListener(SensorEventListener):
        def onSensorChanged(self, event):
            if event.sensor.getType() == Sensor.TYPE_ACCELEROMETER:
                accel_data[:] = event.values
            elif event.sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD:
                mag_data[:] = event.values
            
            if all(accel_data) and all(mag_data):
                R = [[0]*3 for _ in range(3)]
                I = [[0]*3 for _ in range(3)]
                SensorManager.getRotationMatrix(R, I, accel_data, mag_data)
                SensorManager.getOrientation(R, orientation)

    listener = OrientationListener()
    sensor_manager.registerListener(listener, accelerometer, sampling_period_us)
    sensor_manager.registerListener(listener, magnetometer, sampling_period_us)

    time.sleep(0.5)  # Wait for a short time to get a reading

    sensor_manager.unregisterListener(listener)
    # Convert radians to degrees
    return tuple(math.degrees(angle) for angle in orientation)
