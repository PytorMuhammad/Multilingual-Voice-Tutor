#!/usr/bin/env python3
import os
import sys

# NUCLEAR OPTION: Clear ALL Streamlit env vars
env_keys_to_remove = [k for k in os.environ.keys() if 'STREAMLIT' in k or 'streamlit' in k.lower()]
for key in env_keys_to_remove:
    del os.environ[key]

# Get port from Railway
port = os.environ.get('PORT', '8080')

# Run Streamlit with explicit settings
os.system(f'streamlit run tutor_app.py --server.port {port} --server.address 0.0.0.0 --server.headless true --server.enableCORS false')
