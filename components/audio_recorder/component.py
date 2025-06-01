import streamlit as st
from . import audio_recorder
import base64
import tempfile
import os

def create_audio_recorder_component():
    """Create professional audio recorder with direct data return"""
    
    # Create the recorder component
    audio_data = audio_recorder(key="professional_recorder")
    
    # Handle returned audio data
    if audio_data is not None and audio_data.get("audioData"):
        return process_audio_data(audio_data["audioData"])
    
    return None

def process_audio_data(base64_audio):
    """Process base64 audio data and return file path"""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(base64_audio)
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix=".webm")
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        return temp_path
        
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return None
