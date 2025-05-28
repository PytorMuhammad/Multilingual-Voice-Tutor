#!/usr/bin/env python3
import os
import sys
import subprocess

# Nuclear option - clear ALL Streamlit env vars
streamlit_vars = [k for k in os.environ.keys() if k.startswith('STREAMLIT_')]
for var in streamlit_vars:
    del os.environ[var]

# Get port from Railway
port = os.environ.get('PORT', '8080')

# Create streamlit config
config_dir = os.path.expanduser('~/.streamlit')
os.makedirs(config_dir, exist_ok=True)

with open(f'{config_dir}/config.toml', 'w') as f:
    f.write(f"""
[server]
port = {port}
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false
""")

# Run streamlit
os.execv(sys.executable, [sys.executable, '-m', 'streamlit', 'run', 'multilingual_voice_tutor_enhanced.py'])
