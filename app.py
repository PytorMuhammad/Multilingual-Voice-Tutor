import os
import subprocess
import threading
import time
from flask import Flask, redirect

app = Flask(__name__)

# Global variable to track Streamlit process
streamlit_process = None

def start_streamlit():
    """Start Streamlit in background"""
    global streamlit_process
    
    # Clear problematic env vars completely
    for key in list(os.environ.keys()):
        if 'STREAMLIT' in key:
            del os.environ[key]
    
    # Start Streamlit on fixed port 8501
    cmd = [
        'python', '-m', 'streamlit', 'run', 'tutor_app.py',
        '--server.port', '8501',
        '--server.address', '127.0.0.1',
        '--server.headless', 'true'
    ]
    
    streamlit_process = subprocess.Popen(cmd)
    print("Streamlit started on port 8501")

@app.route('/')
def home():
    return redirect('http://127.0.0.1:8501')

@app.route('/<path:path>')
def proxy(path):
    return redirect(f'http://127.0.0.1:8501/{path}')

if __name__ == '__main__':
    # Start Streamlit in background
    threading.Thread(target=start_streamlit, daemon=True).start()
    
    # Wait a bit for Streamlit to start
    time.sleep(3)
    
    # Start Flask on Railway's port
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
