import os
import subprocess
import sys

# Set the port from Railway's PORT env var, default to 8080
port = os.environ.get('PORT', '8080')

# Clear any problematic Streamlit env vars
if 'STREAMLIT_SERVER_PORT' in os.environ:
    del os.environ['STREAMLIT_SERVER_PORT']

# Run Streamlit with explicit arguments
cmd = [
    sys.executable, '-m', 'streamlit', 'run', 
    'multilingual_voice_tutor_enhanced.py',
    '--server.port', port,
    '--server.address', '0.0.0.0',
    '--server.headless', 'true',
    '--server.enableCORS', 'false',
    '--server.enableXsrfProtection', 'false'
]

subprocess.run(cmd)
