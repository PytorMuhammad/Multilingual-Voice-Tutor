#!/usr/bin/env python3
"""
Railway-compatible launcher for the multilingual voice tutor
"""
import os
import sys
import subprocess

def main():
    # Get port from Railway (defaults to 8080)
    port = os.environ.get('PORT', '8080')
    
    # Clear any problematic environment variables
    env_vars_to_clear = [
        'STREAMLIT_SERVER_PORT',
        'STREAMLIT_SERVER_ADDRESS', 
        'STREAMLIT_SERVER_HEADLESS',
        'STREAMLIT_SERVER_ENABLE_CORS',
        'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'
    ]
    
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
    
    # Launch the app
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        'tutor_app.py',  # Your renamed main file
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    print(f"Starting app on port {port}")
    print(f"Command: {' '.join(cmd)}")
    
    # Execute
    os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main()
