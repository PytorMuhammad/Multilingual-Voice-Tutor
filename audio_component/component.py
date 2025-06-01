import os
import streamlit as st
import streamlit.components.v1 as components

# Create component
_component_func = components.declare_component(
    "audio_recorder",
    path=os.path.join(os.path.dirname(__file__), "frontend"),
)

def audio_recorder(key=None):
    """
    Custom audio recorder component that returns audio data directly to Python.
    
    Returns:
        dict: {"audio_data": base64_string, "sample_rate": int} or None
    """
    component_value = _component_func(key=key)
    return component_value
