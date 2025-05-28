#!/bin/bash
export STREAMLIT_SERVER_PORT=${PORT:-8080}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

streamlit run multilingual_voice_tutor_enhanced.py
