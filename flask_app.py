from flask import Flask, request, render_template, jsonify, send_file
import tempfile
import asyncio
import os
import soundfile as sf
import numpy as np
import noisereduce as nr
from scipy import signal

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio_file.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Apply 500% amplification
        audio, sample_rate = sf.read(temp_path)
        amplified_audio = audio * 5.0
        
        # Prevent clipping
        max_val = np.max(np.abs(amplified_audio))
        if max_val > 0.95:
            amplified_audio = amplified_audio * (0.95 / max_val)
        
        # Enhanced processing
        enhanced_audio = nr.reduce_noise(y=amplified_audio.flatten(), sr=sample_rate)
        
        # High-pass filter
        nyquist = sample_rate / 2
        low_cutoff = 80 / nyquist
        b, a = signal.butter(2, low_cutoff, btype='high')
        filtered_audio = signal.filtfilt(b, a, enhanced_audio)
        
        # Save enhanced audio
        enhanced_path = tempfile.mktemp(suffix=".wav")
        sf.write(enhanced_path, filtered_audio, sample_rate)
        
        # Process (you'll need to adapt your existing functions)
        # text, audio_path = process_with_apis(enhanced_path)
        
        # Clean up
        os.unlink(temp_path)
        
        return jsonify({
            'success': True,
            'message': 'Audio processed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
