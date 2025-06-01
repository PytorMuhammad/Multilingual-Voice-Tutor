from flask import Flask, render_template, request, jsonify, send_file
import os
import asyncio
import tempfile
import soundfile as sf
import numpy as np
from werkzeug.utils import secure_filename
import logging

# Import your existing functions
from tutor_app import (
    amplify_audio_500_percent, 
    process_voice_input_pronunciation_enhanced,
    process_multilingual_text_seamless
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys from environment
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            temp_path = tempfile.mktemp(suffix=".wav")
            file.save(temp_path)
            
            # Load and amplify audio (500% boost)
            audio, sample_rate = sf.read(temp_path)
            amplified_audio_data = (audio.reshape(-1, 1), sample_rate)
            enhanced_audio_data = amplify_audio_500_percent(amplified_audio_data)
            
            # Save enhanced version
            enhanced_path = tempfile.mktemp(suffix=".wav")
            enhanced_audio, enhanced_sr = enhanced_audio_data
            sf.write(enhanced_path, enhanced_audio, enhanced_sr)
            
            # Process with enhanced pipeline
            result = asyncio.run(process_voice_input_pronunciation_enhanced(enhanced_path))
            text, audio_path, stt_latency, llm_latency, tts_latency = result
            
            # Clean up
            os.unlink(temp_path)
            os.unlink(enhanced_path)
            
            if text and audio_path:
                return jsonify({
                    'success': True,
                    'transcribed_text': text,
                    'audio_url': f'/download_audio/{os.path.basename(audio_path)}',
                    'latency': {
                        'stt': stt_latency,
                        'llm': llm_latency, 
                        'tts': tts_latency,
                        'total': stt_latency + llm_latency + tts_latency
                    }
                })
            else:
                return jsonify({'error': 'Processing failed'}), 500
                
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_audio/<filename>')
def download_audio(filename):
    # Serve generated audio files
    return send_file(f"/tmp/{filename}", as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
