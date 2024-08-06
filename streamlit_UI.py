
import streamlit as st
from google.generativeai.types import content_types
from collections.abc import Iterable
import time
import json
from IPython.display import display
from IPython.display import Markdown
import pandas as pd

import re
import requests 

# import ABC and abstract
from abc import ABC, abstractmethod
from enum import Enum

import pathlib
import textwrap
from geopy.geocoders import Nominatim
from IPython.display import Image, display, Audio, Markdown
from streamlit_geolocation import streamlit_geolocation
from audiorecorder import audiorecorder

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
from typing import List, Tuple, Optional, Dict, Any

from LLM.function_calling.geolocation_data import GeolocationService
from LLM.function_calling.rescue_data import get_rescue_data
from LLM.function_calling.vital_data import update_victim_json
from utils import GeminiConfig, get_ip, get_location_from_ip, get_user_location_fn, process_location
import os
import dotenv
dotenv.load_dotenv()

import io
from io import BytesIO
from pydub import AudioSegment
import base64
from faster_whisper import WhisperModel
model_size = "large-v2"

def get_user_location():
    location = streamlit_geolocation()
    loc_df = pd.DataFrame({'latitude': [location['latitude']], 'longitude': [location['longitude']]})
    st.map(loc_df, size=(800, 600))

def get_user_location_fn(lat,lon):
    # Fetch the user's IP address
    #ip_address = requests.get('https://api.ipify.org').text
    
    # Use Nominatim service to get location data
    geolocator = Nominatim(user_agent="geoapiExercises")            
    location = geolocator.reverse(f"{lat},{lon}")
    # Display
    return location


def process_location(user_location):
    try:
        if user_location:
            latitude = user_location.get("latitude")
            longitude = user_location.get("longitude")
            loc_req = get_user_location_fn(latitude, longitude)
            city = loc_req.raw['address']['city']
            county = loc_req.raw['address']['county']
            state = loc_req.raw['address']['state']
            country = loc_req.raw['address']['country']
            return f"{city}" 
        else:
            return "No location data received."
    except Exception as e:
        return f"Error: {str(e)}"



import numpy as np
def get_audio(audio):
    if audio is not None:
        audio_buffer = io.BytesIO()
        audio.export(audio_buffer, format="wav", parameters=["-ar", str(16000)])
        audio_array = np.frombuffer(audio_buffer.getvalue()[44:], dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = model_audio.transcribe(audio_array)
        text = ' '.join([segment.text for segment in segments])
        return text
    else:
        return None

def text_to_speech_elevenlabs(text):
    text = ''.join(text)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
    elevenlab_api = os.getenv('elevenlabs_api')
    headers = {
    "accept": "audio/mpeg",
    "xi-api-key": elevenlab_api,
    "Content-Type": "application/json",
    }

    data = {"text": text}

    response = requests.post(url, headers = headers, json=data)


    if response.status_code == 200:
        print(response.status_code)
        return response.content
    else:
        print(f"Error: {response.status_code}, {response.content}")
        return None


def text_to_speech_openai(text):
    client_tts = OpenAI(api_key=OPENAI_API_KEY)
    text = ''.join(text)
    response = client_tts.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )

    try:
        return response.content
    except Exception as e:
        print(f'Error : {e}')
        

def on_click_play_audio_button(text):
    # remove all inbetween '```' 
    if '```' in text:
        text = text.split('```')[-1]
    print(text)
    audio_content = text_to_speech_elevenlabs(text)
    # audio_content = text_to_speech_openai(text)

    if audio_content is not None:
        audio_content = io.BytesIO(audio_content)
        audio_content.seek(0) #AttributeError: 'bytes' object has no attribute 'seek'
        # to solve:
        audio_base_64 = base64.b64encode(audio_content.read()).decode("utf-8")
        st.audio(audio_content, format='audio/mpeg', autoplay=True)
    

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

victim_template = {"personal_info": {
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


if "victim_template" not in st.session_state:
    st.session_state['victim_template'] = victim_template

if "prompt" not in st.session_state:
        st.session_state.prompt = None

if "victim_info" not in st.session_state:
        st.session_state.victim_info = victim_template

geolocation_service = GeolocationService(os.getenv('geolocator_api'))

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
        return response.text
    except:
        response.candidates[0].content.parts[0].text
        if 'victim' in response.candidates[0].content.parts[0].function_call.name:
            json_victim = json.loads(response.candidates[0].content.parts[0].text.replace('```', ''))
            st.session_state.victim_info = json_victim
            return response.candidates[0].content.parts[0].text
        for part in response.parts:
            if fn := part.function_call:
                args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
                st.error(f"{fn.name}({args})")

        return response.candidates[0].content.parts[0].text


# def generate_response(user_input):
#     with st.status("Running function...") as status_text:
#         response = chat.send_message(user_input)
#         if response.candidates[0].content.parts[0].function_call:
#             function_name = response.candidates[0].content.parts[0].function_call.name
#             function_args = response.candidates[0].content.parts[0].function_call.args
#             result = globals()[function_name](**function_args)
#             st.write(result)
#             st.session_state.victim_info = result
#             response = chat.send_message(f"Make into a readable format: {result}")
#             return response.text
#         else:
#             return response.text

def generate_response(user_input):
        response = chat.send_message(user_input)
        if response.candidates[0].content.parts[0].function_call:
            function_name = response.candidates[0].content.parts[0].function_call.name
            function_args = response.candidates[0].content.parts[0].function_call.args
            with st.status(f"Runing function {function_name}...") as status_text:
                # Parse the JSON string into a dictionary
                try:
                    function_args_dict = json.loads(function_args)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {function_args}")
                    return "Error processing function arguments."

                result = globals()[function_name](**function_args_dict)
                #status_text.update(result)
                st.session_state.victim_info = result
                #response = chat.send_message(f"Make into a readable format: {result}")
                return response.text
        else:
            return response.text


gemini_api = os.getenv("gemini_api")
model_path = 'models/gemini-1.5-flash'  
response_type = 'application/json'
config = GeminiConfig(gemini_api, model_path, response_type)
instruction = "You are a post-disaster bot. Help victim while collecting valuable data for intervention teams. Only return JSON output when calling functions. Eventually, your aim is to complete this json: {st.session_state.victim_template}"
model = genai.GenerativeModel(
    "models/gemini-1.5-flash", tools=[get_rescue_data, get_location], system_instruction=instruction, safety_settings=config.safety)
chat = model.start_chat(enable_automatic_function_calling=True)


# App title
st.set_page_config(page_title="Natural Hazard Rescue Bot 💬", layout="wide", page_icon="⚠️")
st._config.set_option(f'theme.base' ,"dark" )
st._config.set_option(f'theme.backgroundColor' ,"black" )


@st.cache_resource
def load_model(size=model_size):
    return WhisperModel(model_size, device="cuda", compute_type="float16")

model_audio = load_model()




if 'user_location' not in st.session_state:
        ip = get_ip()
        st.session_state.user_location = get_location_from_ip(ip)


one, two, three = st.columns([.1, .9, .1])
with two:
    st.title("💬 Natural Hazard Rescue App⚠️")
    st.write("This bot is designed to help victims of natural disasters by providing support and information. It can also collect valuable data for intervention teams.\n\nTo get started, simply type your query in the chat box on the left.")
    # add optional functions to call
    #st.write("### Optional Functions\n\n- `get_location()`: Get the current location of the user.\n- `get_device_orientation(sampling_period_us: int)`: Get the device orientation.\n- `detect_significant_motion(threshold: float, duration_s: float)`: Detect significant motion.\n- `get_rescue_data(incident_number: Optional[int], time: Optional[str])`: Fetch rescue data for a specific incident number and time.\n- `add_victim_info(victim_info: Dict[str, Any])`: Add victim information to the database.")
    # get some colors 
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Hugging Face Credentials


def fix_json(json_string):
    # Remove any leading/trailing whitespace and 'json' prefix
    json_string = json_string.strip()
    json_string = re.sub(r'^json\s*', '', json_string, flags=re.IGNORECASE)
    
    # Remove any trailing commas before closing braces or brackets
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    
    # Add missing commas between key-value pairs
    json_string = re.sub(r'"\s*}\s*"', '", "', json_string)
    json_string = re.sub(r'"\s*]\s*"', '"], "', json_string)
    
    # Ensure all keys are properly quoted
    json_string = re.sub(r'([{,])\s*(\w+):', r'\1 "\2":', json_string)
    
    # Replace single quotes with double quotes
    json_string = json_string.replace("'", '"')
    
    # Ensure proper formatting for empty strings and arrays
    json_string = re.sub(r':\s*""', ': ""', json_string)
    json_string = re.sub(r':\s*\[\]', ': []', json_string)
    
    # Try to parse the JSON
    try:
        parsed_json = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

left, middle, right = st.columns([.5,.1,.4])
with left:
    with st.container(height=820, border=True):

        if "messages" not in st.session_state:
                st.session_state.messages = []
        left_, right_ = st.columns([.9, .1])
        with right_:
            audio = audiorecorder("🎤", "stop", show_visualizer=False) # make an icon to stop recording emoji : 
        with left_:
            st.session_state['prompt'] = st.chat_input("Enter Query here")

        if len(audio) > 0:
            st.session_state['prompt'] = get_audio(audio)
        if st.session_state['prompt']: 
            st.session_state.messages.append({"role": "user", "content": st.session_state['prompt']})
            try:
                response_ = generate_response_(st.session_state['prompt'])
            except Exception as e:
                st.warning(f"Error occurred: {e}")
                response_ = 'Sorry, I could not understand the request, can you repeat please?'
            st.session_state.messages.append({"role": "assistant", "content": response_})
            on_click_play_audio_button(response_)
            if '```json' in response_:
                try:
                    answer = update_victim_json(new_infos=response_.replace('```json\n', '').replace('\n```', ''))
                    st.session_state['victim_info'] = json.loads(answer)
                except Exception as e:
                    try:
                        st.error(f"{answer}\n\nError occurred: {e}")
                        answer = update_victim_json(new_infos=response_.replace('```json\n', '').replace('\n```', '').replace("'", '"'))
                        st.session_state['victim_info'] = json.loads(answer)
                    except Exception as f:
                        answer = fix_json(update_victim_json(new_infos=response_))
                        st.session_state['victim_info'] = json.loads(answer)
                        st.error(f"{answer}\n\nError occurred: {f}")

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

with right:
    st.write("Victim Info:\n\n", st.session_state.victim_info)    


