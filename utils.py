
import streamlit as st
from google.generativeai.types import content_types
from collections.abc import Iterable
import time
from sodapy import Socrata
import json
from IPython.display import display
from IPython.display import Markdown
from geopy.geocoders import Nominatim

# import ABC and abstract
from abc import ABC, abstractmethod
from enum import Enum

import pathlib
import textwrap
import requests

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
from typing import List, Tuple, Optional, Dict, Any



class GeminiConfig:
    def __init__(self, gemini_api, model_path, response_type):
        self.gemini_api = gemini_api
        self.model_path = model_path
        self.response_type = response_type

        self.generation_config = genai.GenerationConfig(response_mime_type=self.response_type)
        genai.configure(api_key=self.gemini_api)
        self.model = genai.GenerativeModel(self.model_path, generation_config=self.generation_config)
        self.safety = [
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            }
        ]


def tool_config_from_mode(mode: str, fns: Iterable[str] = ()):
    """Create a tool config with the specified function calling mode."""
    return content_types.to_tool_config(
        {"function_calling_config": {"mode": mode, "allowed_function_names": fns}}
    )


victim_info_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        'personal_info': genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'name': genai.protos.Schema(type=genai.protos.Type.STRING),
                'age': genai.protos.Schema(type=genai.protos.Type.STRING),
                'gender': genai.protos.Schema(type=genai.protos.Type.STRING),
                'location': genai.protos.Schema(type=genai.protos.Type.STRING),
            }
        ),
        'medical_info': genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'injuries': genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
                'medical_conditions': genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
                'medications': genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
            }
        ),
        'situation': genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'disaster_type': genai.protos.Schema(type=genai.protos.Type.STRING),
                'immediate_needs': genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
                'safety_status': genai.protos.Schema(type=genai.protos.Type.STRING),
            }
        ),
        'contact_info': genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'phone': genai.protos.Schema(type=genai.protos.Type.STRING),
                'emergency_contact': genai.protos.Schema(type=genai.protos.Type.STRING),
            }
        ),
        'resources': genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'food_water': genai.protos.Schema(type=genai.protos.Type.STRING),
                'shelter': genai.protos.Schema(type=genai.protos.Type.STRING),
                'communication_devices': genai.protos.Schema(type=genai.protos.Type.ARRAY, items=genai.protos.Schema(type=genai.protos.Type.STRING)),
            }
        ),
    }
)

parameters_schema = genai.protos.Schema(
    type=genai.protos.Type.OBJECT,
    properties={
        'victim_info': genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'personal_info': victim_info_schema.properties['personal_info'],
                'medical_info': victim_info_schema.properties['medical_info'],
                'situation': victim_info_schema.properties['situation'],
                'contact_info': victim_info_schema.properties['contact_info'],
                'resources': victim_info_schema.properties['resources'],
            }
        )
    }
)

add_victim_info = genai.protos.FunctionDeclaration(
    name="add_victim_info",
    description=textwrap.dedent("""
        Adds victim information to the database.
    """),
    parameters=parameters_schema
)


def get_ip():
    response = requests.get('https://api.ipify.org?format=json')
    return response.json()["ip"]

def get_location_from_ip(ip_address):
    response = requests.get(f'https://ipapi.co/{ip_address}/json/').json()
    location_data = {
        "latitude": response.get("latitude"),
        "longitude": response.get("longitude"),
    }
    return location_data

def get_user_location_fn(lat, lon):
    geolocator = Nominatim(user_agent="streamlit_geolocation_app")
    location = geolocator.reverse(f"{lat},{lon}")
    return location

def process_location(user_location):
    try:
        if user_location:
            latitude = user_location.get("latitude")
            longitude = user_location.get("longitude")
            loc_req = get_user_location_fn(latitude, longitude)
            city = loc_req.raw['address'].get('city', 'N/A')
            county = loc_req.raw['address'].get('county', 'N/A')
            state = loc_req.raw['address'].get('state', 'N/A')
            country = loc_req.raw['address'].get('country', 'N/A')
            return f"City: {city}, County: {county}, State: {state}, Country: {country}"
        else:
            return "No location data received."
    except Exception as e:
        return f"Error: {str(e)}"