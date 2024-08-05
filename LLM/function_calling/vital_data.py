from typing import List, Tuple, Optional, Dict, Any
import genai
import streamlit as st
from streamlit_UI import gemini_api


def update_victim_json(new_infos: Optional[Dict[str, Any]]):
    victim_template = st.session_state.get('victim_info', {})
    print(victim_template)
    prompt = f"Update the JSON structure: {victim_template}\n\n with accurate information: {new_infos}\n\n. Output should be a JSON file. Output:"
    genai.configure(api_key=gemini_api)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    # parse json
    return response.text