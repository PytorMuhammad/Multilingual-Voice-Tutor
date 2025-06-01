#!/usr/bin/env python3
import os

# Get port from Railway
port = os.environ.get('PORT', '5000')

# Run Flask app
os.system(f'python flask_app.py')
