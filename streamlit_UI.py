
import streamlit as st
from google.generativeai.types import content_types
from collections.abc import Iterable
import time
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

from LLM.function_calling.geolocation_data import GeolocationService
from LLM.function_calling.rescue_data import get_rescue_data
from LLM.function_calling.sensor_data import get_device_orientation, detect_significant_motion
from utils import GeminiConfig
import os
import dotenv
dotenv.load_dotenv()

geolocation_service = GeolocationService('test')

def get_location() -> str:
    try:
        return geolocation_service.get_location()
    except Exception as e:
        return f"Error occurred: {e}"
    

def generate_response_(user_input):
    response = chat.send_message(
        user_input
    )
    try:
        response.text
        return response.text
    except:
        response.candidates[0].content.parts[0].text
        return response.candidates[0].content.parts[0].text

def generate_response(user_input):
    with st.status("Running function...") as status_text:
        response = chat.send_message(user_input)
        if response.candidates[0].content.parts[0].function_call:
            function_name = response.candidates[0].content.parts[0].function_call.name
            function_args = response.candidates[0].content.parts[0].function_call.args
            result = globals()[function_name](**function_args)
            #if 'victim_info' in function_name:
            st.session_state.victim_info = result
                # insert in json instead of replacing it 
            response = chat.send_message(f"Make into a readable format: {result}")
            return response.text
        else:
            return response.text


gemini_api = os.getenv("gemini_api")
model_path = 'models/gemini-1.5-flash'  
response_type = 'application/json'
config = GeminiConfig(gemini_api, model_path, response_type)
instruction = "You are a post-disaster bot. Help victim while collecting valuable data for intervention teams. Only return JSON output when calling functions."
model = genai.GenerativeModel(
    "models/gemini-1.5-flash", tools=[get_location, get_device_orientation, detect_significant_motion, get_rescue_data], system_instruction=instruction, safety_settings=config.safety)
chat = model.start_chat()


# App title
st.set_page_config(page_title="Natural Hazard Rescue Bot 💬", layout="wide", page_icon="⚠️")
st._config.set_option(f'theme.base' ,"dark" )
st._config.set_option(f'theme.backgroundColor' ,"black" )


one, two, three = st.columns([.1, .9, .1])
with two:
    st.title("💬 Natural Hazard Rescue App⚠️")
    st.write("This bot is designed to help victims of natural disasters by providing support and information. It can also collect valuable data for intervention teams.\n\nTo get started, simply type your query in the chat box on the left.")
    # add optional functions to call
    st.write("### Optional Functions\n\n- `get_location()`: Get the current location of the user.\n- `get_device_orientation(sampling_period_us: int)`: Get the device orientation.\n- `detect_significant_motion(threshold: float, duration_s: float)`: Detect significant motion.\n- `get_rescue_data(incident_number: Optional[int], time: Optional[str])`: Fetch rescue data for a specific incident number and time.\n- `add_victim_info(victim_info: Dict[str, Any])`: Add victim information to the database.")
    # get some colors 
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Hugging Face Credentials


left, middle, right = st.columns([.5,.1,.4])
with left:
    with st.container(height=820, border=True):

        if "prompt" not in st.session_state:
                st.session_state.prompt = ''
        if st.session_state['prompt']: 
            st.session_state.messages.append({"role": "user", "content": st.session_state['prompt']})
            # try:
            #     response_ = generate_response_(st.session_state['prompt'])
            # except:
            response_ = generate_response_(st.session_state['prompt'])
            st.session_state.messages.append({"role": "assistant", "content": response_})
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

with right:
    if "victim_info" not in st.session_state:
        st.session_state.victim_info = {
            "personal_info": {
                "name": "",
                "age": "",
                "gender": "",
                "location": "",
            },
            "medical_info": {
                "injuries": [],
                "medical_conditions": [],
                "medications": [],
            },
            "situation": {
                "disaster_type": "",
                "immediate_needs": [],
                "safety_status": "",
            },
            "contact_info": {
                "phone": "",
                "emergency_contact": "",
            },
            "resources": {
                "food_water": "",
                "shelter": "",
                "communication_devices": [],
            }
        }
    else:
        st.write("Victim Info:\n\n", st.session_state.victim_info)    

st.session_state['prompt'] = st.chat_input("Enter Query here")

