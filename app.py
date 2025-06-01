#!/usr/bin/env python3
import os
import sys
import subprocess

# NUCLEAR OPTION: Clear ALL Streamlit env vars
env_keys_to_remove = [k for k in os.environ.keys() if 'STREAMLIT' in k or 'streamlit' in k.lower()]
for key in env_keys_to_remove:
    del os.environ[key]

# Get port from Railway
port = os.environ.get('PORT', '8080')

# Check if React component is built
def check_component_build():
    build_path = "components/audio_recorder/frontend/build"
    if not os.path.exists(build_path):
        print("❌ React component not built!")
        print("Building component now...")
        
        try:
            # Try to build the component if Node.js is available
            frontend_path = "components/audio_recorder/frontend"
            if os.path.exists(frontend_path):
                os.chdir(frontend_path)
                subprocess.run(["npm", "install"], check=True)
                subprocess.run(["npm", "run", "build"], check=True)
                os.chdir("../../..")
                print("✅ Component built successfully!")
            else:
                print("⚠️ Frontend path not found, component will use fallback mode")
        except Exception as e:
            print(f"⚠️ Could not build component: {e}")
            print("App will run with fallback HTML5 recorder")
        
    else:
        print("✅ React component build found!")

# Check component before starting
check_component_build()

# Run Streamlit with explicit settings
print(f"🚀 Starting Streamlit on port {port}")
os.system(f'streamlit run tutor_app.py --server.port {port} --server.address 0.0.0.0 --server.headless true --server.enableCORS false')
