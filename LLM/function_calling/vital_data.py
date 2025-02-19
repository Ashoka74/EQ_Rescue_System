from typing import List, Tuple, Optional, Dict, Any
import streamlit as st
import google.generativeai as genai
import json
import os
import dotenv
from utils import victim_info_schema

dotenv.load_dotenv()

gemini_api = os.getenv("gemini_api")


def update_victim_json(new_infos: Optional[Dict[str, Any]]):
    victim_template = st.session_state.get('victim_template', {})
    history_infos = st.session_state.get('victim_info', {})
    print(victim_template)
    prompt = f"Update the JSON structure: {victim_template}\n\n with accurate informations based on history: {history_infos}\n\n and new informations: {new_infos}\n\n. Output should be a JSON file. Leave blank (e.g.""), when there is no information. Output:"
    genai.configure(api_key=gemini_api)
    model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type":"application/json"})
    response = model.generate_content(prompt)
    return response.text