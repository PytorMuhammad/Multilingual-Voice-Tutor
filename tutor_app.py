import os
import queue
import time
import tempfile
import logging
import json
import asyncio
# Add these imports after the existing import statements
from scipy import signal
import scipy.signal
import queue
import threading
import re
import streamlit.components.v1
import uuid
import base64
from pathlib import Path
from datetime import datetime
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from io import BytesIO
import noisereduce as nr  # For noise reduction

# Web interface and async handling
import streamlit as st
import httpx
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av

# Audio processing
from scipy.io import wavfile
import sounddevice as sd
import soundfile as sf
import librosa
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("multilingual_voice_tutor")

# ----------------------------------------------------------------------------------
# CONFIGURATION SECTION
# ----------------------------------------------------------------------------------

# Secrets and API keys
if 'api_keys_initialized' not in st.session_state:
    st.session_state.api_keys_initialized = False
    st.session_state.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

# API endpoints
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
OPENAI_API_URL = "https://api.openai.com/v1"

# Default voice settings
if 'voice_settings' not in st.session_state:
    st.session_state.voice_settings = {
        "default": {
            "stability": 0.6,
            "similarity_boost": 0.7
        },
        "cs": {  # Czech settings
            "stability": 0.65,  # Higher stability for more consistent Czech
            "similarity_boost": 0.6  # Lower similarity to reduce German influence
        },
        "de": {  # German settings
            "stability": 0.55,
            "similarity_boost": 0.7
        }
    }

# Default voice selection
if 'elevenlabs_voice_id' not in st.session_state:
    st.session_state.elevenlabs_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice - better for multilingual

# Whisper speech recognition config
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = "medium"
    st.session_state.whisper_local_model = None

# Language distribution preference
if 'language_distribution' not in st.session_state:
    st.session_state.language_distribution = {
        "cs": 50,  # Czech percentage
        "de": 50   # German percentage
    }

# Language preference for response
if 'response_language' not in st.session_state:
    st.session_state.response_language = "both"  # Options: "cs", "de", "both"

# Language codes and settings
SUPPORTED_LANGUAGES = {
    "cs": {"name": "Czech", "confidence_threshold": 0.65},
    "de": {"name": "German", "confidence_threshold": 0.65}
}

# Performance monitoring
if 'performance_metrics' not in st.session_state:
    st.session_state.performance_metrics = {
        "stt_latency": [],
        "llm_latency": [],
        "tts_latency": [],
        "total_latency": [],
        "api_calls": {"whisper": 0, "openai": 0, "elevenlabs": 0}
    }

# Conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Progress and status tracking
if 'message_queue' not in st.session_state:
    st.session_state.message_queue = queue.Queue()

# Audio session variables
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
    st.session_state.last_audio_output = None

def check_system_dependencies():
    """Check and install system dependencies for audio processing"""
    try:
        # Check if ffmpeg is available
        import subprocess
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.warning("⚠️ Audio processing may be limited. Installing dependencies...")
        # Railway should handle this via nixpacks
    
    return True
# Add this new function after the import statements (around line 200)
import streamlit.components.v1 as components

def create_audio_recorder_component():
    """Create HTML5 audio recorder component"""
    html_code = """
    <div style="padding: 20px; border: 2px solid #ff4b4b; border-radius: 10px; text-align: center; background-color: #f0f2f6;">
        <div id="status" style="font-size: 18px; margin-bottom: 15px; font-weight: bold;">🎤 Ready to Record</div>
        
        <button id="recordBtn" onclick="toggleRecording()" 
                style="background: #ff4b4b; color: white; border: none; padding: 15px 30px; 
                       border-radius: 25px; cursor: pointer; font-size: 16px; font-weight: bold; margin: 5px;">
            🔴 START RECORDING
        </button>
        
        <button id="previewBtn" onclick="playPreview()" disabled
                style="background: #00cc88; color: white; border: none; padding: 15px 30px; 
                       border-radius: 25px; cursor: pointer; font-size: 16px; font-weight: bold; margin: 5px;">
            🔊 PREVIEW
        </button>
        
        <button id="processBtn" onclick="processAudio()" disabled
                style="background: #ff6600; color: white; border: none; padding: 15px 30px; 
                       border-radius: 25px; cursor: pointer; font-size: 16px; font-weight: bold; margin: 5px;">
            ⚡ PROCESS
        </button>
        
        <div id="timer" style="font-size: 14px; margin-top: 10px; color: #666;">00:00</div>
        <audio id="audioPreview" controls style="width: 100%; margin-top: 15px; display: none;"></audio>
        <div id="audioData" style="display: none;"></div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let recordingTime = 0;
        let timerInterval;
        let recordedBlob = null;

        // Initialize when page loads
        window.onload = function() {
            initializeRecorder();
        };

        async function initializeRecorder() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    } 
                });
                
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = function() {
                    recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    
                    // Enable preview and process buttons
                    document.getElementById('previewBtn').disabled = false;
                    document.getElementById('processBtn').disabled = false;
                    
                    // Update status
                    document.getElementById('status').innerHTML = '✅ Recording Complete - Ready to Preview/Process';
                    
                    // Create audio URL for preview
                    const audioUrl = URL.createObjectURL(recordedBlob);
                    const audioPreview = document.getElementById('audioPreview');
                    audioPreview.src = audioUrl;
                    audioPreview.style.display = 'block';
                };
                
                document.getElementById('status').innerHTML = '🎤 Microphone Ready - Click START to Record';
                
            } catch (error) {
                document.getElementById('status').innerHTML = '❌ Microphone access denied';
                console.error('Error accessing microphone:', error);
            }
        }

        function toggleRecording() {
            const recordBtn = document.getElementById('recordBtn');
            const statusDiv = document.getElementById('status');
            
            if (!isRecording) {
                // Start recording
                audioChunks = [];
                recordingTime = 0;
                isRecording = true;
                
                recordBtn.innerHTML = '⏹️ STOP RECORDING';
                recordBtn.style.background = '#666';
                statusDiv.innerHTML = '🔴 RECORDING - Speak clearly in Czech or German';
                
                // Disable other buttons
                document.getElementById('previewBtn').disabled = true;
                document.getElementById('processBtn').disabled = true;
                document.getElementById('audioPreview').style.display = 'none';
                
                // Start timer
                timerInterval = setInterval(updateTimer, 1000);
                
                // Start recording
                mediaRecorder.start(1000); // Collect data every second
                
            } else {
                // Stop recording
                isRecording = false;
                mediaRecorder.stop();
                
                recordBtn.innerHTML = '🔄 NEW RECORDING';
                recordBtn.style.background = '#ff4b4b';
                
                // Stop timer
                clearInterval(timerInterval);
            }
        }

        function updateTimer() {
            recordingTime++;
            const minutes = Math.floor(recordingTime / 60);
            const seconds = recordingTime % 60;
            document.getElementById('timer').innerHTML = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }

        function playPreview() {
            const audioPreview = document.getElementById('audioPreview');
            audioPreview.play();
            document.getElementById('status').innerHTML = '🔊 Playing Preview';
        }

        async function processAudio() {
            if (recordedBlob) {
                document.getElementById('status').innerHTML = '⚡ Processing audio...';
                document.getElementById('processBtn').disabled = true;
                
                // Convert blob to base64
                const reader = new FileReader();
                reader.onloadend = function() {
                    const base64Data = reader.result.split(',')[1];
                    
                    // Store in hidden div for Streamlit to access
                    document.getElementById('audioData').innerHTML = base64Data;
                    
                    // Trigger Streamlit rerun by dispatching custom event
                    window.parent.postMessage({
                        type: 'audio_recorded',
                        data: base64Data
                    }, '*');
                    
                    document.getElementById('status').innerHTML = '✅ Audio sent for processing!';
                };
                reader.readAsDataURL(recordedBlob);
            }
        }

        // Reset function for new recording
        function resetRecorder() {
            audioChunks = [];
            recordedBlob = null;
            recordingTime = 0;
            isRecording = false;
            
            document.getElementById('recordBtn').innerHTML = '🔴 START RECORDING';
            document.getElementById('recordBtn').style.background = '#ff4b4b';
            document.getElementById('previewBtn').disabled = true;
            document.getElementById('processBtn').disabled = false;
            document.getElementById('audioPreview').style.display = 'none';
            document.getElementById('status').innerHTML = '🎤 Ready for New Recording';
            document.getElementById('timer').innerHTML = '00:00';
            
            clearInterval(timerInterval);
        }
    </script>
    """
    
    # Return the component with a unique height
    return components.html(html_code, height=300)

# Add these functions after the create_audio_recorder_component function

def process_html5_audio_data(base64_audio_data):
    """Process base64 audio data from HTML5 recorder"""
    try:
        import base64
        import io
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(base64_audio_data)
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix=".webm")
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # Convert webm to wav for processing
        wav_path = convert_webm_to_wav(temp_path)
        
        # Apply 500% amplification
        amplified_path = amplify_recorded_audio(wav_path)
        
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if wav_path != amplified_path and os.path.exists(wav_path):
            os.unlink(wav_path)
            
        return amplified_path
        
    except Exception as e:
        logger.error(f"HTML5 audio processing error: {str(e)}")
        return None

def convert_webm_to_wav(webm_path):
    """Convert WebM audio to WAV format"""
    try:
        from pydub import AudioSegment
        
        # Load WebM audio
        audio = AudioSegment.from_file(webm_path, format="webm")
        
        # Convert to WAV
        wav_path = tempfile.mktemp(suffix=".wav")
        audio.export(wav_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        
        return wav_path
        
    except Exception as e:
        logger.error(f"WebM to WAV conversion error: {str(e)}")
        # Fallback: try to use the original file
        return webm_path

def amplify_recorded_audio(audio_path):
    """Apply 500% amplification to recorded audio"""
    try:
        # Load audio
        audio, sample_rate = sf.read(audio_path)
        
        # Apply 500% amplification
        amplified_audio = audio * 5.0
        
        # Prevent clipping
        max_val = np.max(np.abs(amplified_audio))
        if max_val > 0.95:
            amplified_audio = amplified_audio * (0.95 / max_val)
        
        # Apply noise reduction
        try:
            enhanced_audio = nr.reduce_noise(y=amplified_audio, sr=sample_rate)
        except:
            enhanced_audio = amplified_audio
        
        # Save enhanced audio
        enhanced_path = tempfile.mktemp(suffix=".wav")
        sf.write(enhanced_path, enhanced_audio, sample_rate)
        
        return enhanced_path
        
    except Exception as e:
        logger.error(f"Audio amplification error: {str(e)}")
        return audio_path

async def process_html5_recorded_voice(audio_path):
    """Process HTML5 recorded voice through the enhanced pipeline"""
    try:
        # Process with the existing enhanced pipeline
        text, audio_output_path, stt_latency, llm_latency, tts_latency = await process_voice_input_pronunciation_enhanced(audio_path)
        
        # Store results in session state
        if text:
            st.session_state.last_text_input = text
        if audio_output_path:
            st.session_state.last_audio_output = audio_output_path
        
        # Show results
        total_latency = stt_latency + llm_latency + tts_latency
        st.success(f"✅ **HTML5 Voice Processing Complete!** ({total_latency:.2f}s)")
        
        return True
        
    except Exception as e:
        st.error(f"HTML5 voice processing error: {str(e)}")
        logger.error(f"HTML5 voice processing error: {str(e)}")
        return False

def get_recorded_audio_data():
    """Get audio data from the HTML component"""
    # This will be called after the component processes audio
    # In practice, we'll handle this through session state
    pass
# Add these enhanced audio capture functions after the import statements
# Replace the existing capture_enhanced_audio function with this simplified version
def capture_enhanced_audio():
    """Simplified audio capture that works with the frame-based system"""
    try:
        if st.session_state.audio_frames and len(st.session_state.audio_frames) > 10:
            return process_audio_frames(st.session_state.audio_frames)
        return None
    except Exception as e:
        logger.error(f"Enhanced audio capture error: {str(e)}")
        return None

def amplify_audio_500_percent(audio_data):
    """Amplify audio by 500% for better pronunciation detection"""
    try:
        audio, sample_rate = audio_data
        
        # Apply 500% amplification (5x boost)
        amplified_audio = audio * 5.0
        
        # Prevent clipping while maintaining pronunciation clarity
        max_val = np.max(np.abs(amplified_audio))
        if max_val > 0.95:  # Prevent clipping
            amplified_audio = amplified_audio * (0.95 / max_val)
        
        # Apply additional preprocessing for pronunciation clarity
        # High-pass filter to remove low-frequency noise
        from scipy import signal
        nyquist = sample_rate / 2
        low_cutoff = 80 / nyquist  # Remove very low frequencies
        b, a = signal.butter(2, low_cutoff, btype='high')
        filtered_audio = signal.filtfilt(b, a, amplified_audio.flatten())
        
        return (filtered_audio.reshape(-1, 1), sample_rate)
    except Exception as e:
        logger.error(f"Audio amplification error: {str(e)}")
        return audio_data

def preview_audio_enhanced(audio_data):
    """Preview recorded audio with enhanced playback"""
    try:
        if audio_data is None:
            st.error("No audio data to preview")
            return
            
        # Save audio for preview
        temp_preview_path = save_audio_for_preview(audio_data)
        if temp_preview_path:
            st.session_state.audio_preview_path = temp_preview_path
            
            # Display audio player
            with open(temp_preview_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/wav")
                st.success("🔊 **Audio Preview Ready** - You can hear your recording above")
        
    except Exception as e:
        st.error(f"Preview error: {str(e)}")

def save_audio_for_preview(audio_data):
    """Save audio data for preview playback"""
    try:
        audio, sample_rate = audio_data
        
        # Create temporary file for preview
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, audio, sample_rate)
        
        return temp_path
    except Exception as e:
        logger.error(f"Save preview error: {str(e)}")
        return None

def enhanced_recording_interface():
    """Enhanced real-time recording interface"""
    # Initialize recorder if needed
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    
    # Start recording if not already started
    if st.session_state.is_recording and not st.session_state.audio_recorder.recording:
        st.session_state.audio_recorder.start_recording()
    
    # Show live recording feedback
    if st.session_state.is_recording:
        # Create animated recording indicator
        import time
        for i in range(3):
            time.sleep(0.5)
            st.empty()

def process_audio_frames(audio_frames):
    """Process captured audio frames into usable audio data with 500% amplification"""
    try:
        if not audio_frames or len(audio_frames) < 10:
            logger.error("Insufficient audio frames captured")
            return None
        
        # Convert frames to audio data
        sample_rate = 16000
        audio_data = []
        
        for frame in audio_frames:
            # Convert frame to numpy array
            sound = frame.to_ndarray()
            if sound.size > 0:
                audio_data.append(sound)
        
        if not audio_data:
            logger.error("No valid audio data in frames")
            return None
        
        # Combine all audio data
        combined_audio = np.concatenate(audio_data, axis=0)
        
        # Apply 500% amplification immediately
        amplified_audio = combined_audio * 5.0
        
        # Prevent clipping while maintaining clarity
        max_val = np.max(np.abs(amplified_audio))
        if max_val > 0.95:
            amplified_audio = amplified_audio * (0.95 / max_val)
        
        # Apply noise reduction and enhancement
        try:
            # Remove noise
            enhanced_audio = nr.reduce_noise(y=amplified_audio.flatten(), sr=sample_rate)
            
            # Apply high-pass filter for clarity
            from scipy import signal
            nyquist = sample_rate / 2
            low_cutoff = 80 / nyquist
            b, a = signal.butter(2, low_cutoff, btype='high')
            filtered_audio = signal.filtfilt(b, a, enhanced_audio)
            
            return (filtered_audio.reshape(-1, 1), sample_rate)
            
        except Exception as e:
            logger.warning(f"Advanced processing failed, using basic amplification: {str(e)}")
            return (amplified_audio, sample_rate)
        
    except Exception as e:
        logger.error(f"Audio frame processing error: {str(e)}")
        return None

def display_audio_preview():
    """Display audio preview with enhanced playback"""
    try:
        if not st.session_state.recorded_audio_data:
            st.error("No audio data to preview")
            return
        
        # Save audio for preview
        audio, sample_rate = st.session_state.recorded_audio_data
        
        # Create temporary file for preview
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, audio, sample_rate)
        
        # Display audio player
        with open(temp_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
            st.success("🔊 **Audio Preview** - You can hear your enhanced recording above")
        
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass
            
    except Exception as e:
        st.error(f"Preview error: {str(e)}")

def process_recorded_voice():
    """Process the recorded voice with enhanced pipeline"""
    try:
        if not st.session_state.recorded_audio_data:
            st.error("No audio data to process")
            return
        
        # Save audio to temporary file
        audio, sample_rate = st.session_state.recorded_audio_data
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, audio, sample_rate)
        
        # Process with enhanced pipeline
        text, audio_path, stt_latency, llm_latency, tts_latency = asyncio.run(
            process_voice_input_pronunciation_enhanced(temp_path)
        )
        
        # Store results
        if text:
            st.session_state.last_text_input = text
        if audio_path:
            st.session_state.last_audio_output = audio_path
        
        # Show results
        total_latency = stt_latency + llm_latency + tts_latency
        st.success(f"✅ **Voice Processing Complete!** ({total_latency:.2f}s)")
        
        # Reset recording state
        st.session_state.recording_state = 'idle'
        st.session_state.recorded_audio_data = None
        st.session_state.audio_frames = []
        
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        logger.error(f"Voice processing error: {str(e)}")
        
def process_pronunciation_enhanced_audio():
    """Process recorded audio with pronunciation focus"""
    try:
        if not st.session_state.recorded_audio_data:
            st.error("No audio data to process")
            return
            
        # Save enhanced audio to file
        audio_file_path = save_enhanced_audio_file(st.session_state.recorded_audio_data)
        
        if audio_file_path:
            # Process with enhanced pipeline
            text, audio_path, stt_latency, llm_latency, tts_latency = asyncio.run(
                process_voice_input_pronunciation_enhanced(audio_file_path)
            )
            
            # Store results
            if text:
                st.session_state.last_text_input = text
            st.session_state.last_audio_output = audio_path
            
            # Show results
            total_latency = stt_latency + llm_latency + tts_latency
            st.success(f"✅ **Pronunciation-Enhanced Processing Complete!** ({total_latency:.2f}s)")
            
            # Clean up
            os.unlink(audio_file_path)
            if st.session_state.audio_preview_path:
                try:
                    os.unlink(st.session_state.audio_preview_path)
                except:
                    pass
    except Exception as e:
        st.error(f"Processing error: {str(e)}")

def save_enhanced_audio_file(audio_data):
    """Save enhanced audio data to file for processing"""
    try:
        audio, sample_rate = audio_data
        
        # Create temporary file
        temp_path = tempfile.mktemp(suffix=".wav")
        sf.write(temp_path, audio, sample_rate)
        
        return temp_path
    except Exception as e:
        logger.error(f"Save enhanced audio error: {str(e)}")
        return None

def process_uploaded_audio_enhanced(uploaded_file):
    """Process uploaded audio with 500% amplification"""
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_file_path = tmp_file.name
        
        # Load and amplify audio
        audio, sample_rate = sf.read(audio_file_path)
        amplified_audio_data = (audio.reshape(-1, 1), sample_rate)
        enhanced_audio_data = amplify_audio_500_percent(amplified_audio_data)
        
        # Save enhanced version
        enhanced_path = save_enhanced_audio_file(enhanced_audio_data)
        
        # Process with enhanced pipeline
        text, audio_path, stt_latency, llm_latency, tts_latency = asyncio.run(
            process_voice_input_pronunciation_enhanced(enhanced_path)
        )
        
        # Store results
        if text:
            st.session_state.last_text_input = text
        st.session_state.last_audio_output = audio_path
        
        # Show results
        total_latency = stt_latency + llm_latency + tts_latency
        st.success(f"✅ **Enhanced Upload Processing Complete!** ({total_latency:.2f}s)")
        
        # Clean up
        os.unlink(audio_file_path)
        if enhanced_path:
            os.unlink(enhanced_path)
            
    except Exception as e:
        st.error(f"Upload processing error: {str(e)}")
        
# Enhanced voice processing functions to fix transcription issues
# Add these functions to your tutor_app.py file

import streamlit.components.v1 as components
import time
import asyncio
import tempfile
import os
import base64

def create_enhanced_audio_recorder_component():
    """Enhanced HTML5 audio recorder with automatic processing"""
    html_code = """
    <div style="padding: 20px; border: 2px solid #ff4b4b; border-radius: 10px; text-align: center; background-color: #f0f2f6;">
        <div id="status" style="font-size: 18px; margin-bottom: 15px; font-weight: bold;">🎤 Ready to Record</div>
        
        <button id="recordBtn" onclick="toggleRecording()" 
                style="background: #ff4b4b; color: white; border: none; padding: 15px 30px; 
                       border-radius: 25px; cursor: pointer; font-size: 16px; font-weight: bold; margin: 5px;">
            🔴 START RECORDING
        </button>
        
        <button id="previewBtn" onclick="playPreview()" disabled
                style="background: #00cc88; color: white; border: none; padding: 15px 30px; 
                       border-radius: 25px; cursor: pointer; font-size: 16px; font-weight: bold; margin: 5px;">
            🔊 PREVIEW YOUR VOICE
        </button>
        
        <div id="processingStatus" style="font-size: 16px; margin: 10px 0; color: #666; display: none;">
            ⚡ Auto-processing your voice...
        </div>
        
        <div id="timer" style="font-size: 14px; margin-top: 10px; color: #666;">00:00</div>
        <audio id="audioPreview" controls style="width: 100%; margin-top: 15px; display: none;"></audio>
        
        <!-- Hidden elements for data transfer -->
        <input type="hidden" id="audioDataField" value="" />
        <input type="hidden" id="processingTriggered" value="false" />
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let recordingTime = 0;
        let timerInterval;
        let recordedBlob = null;

        // Initialize when page loads
        window.onload = function() {
            initializeRecorder();
        };

        async function initializeRecorder() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    } 
                });
                
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = function() {
                    recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    
                    // Enable preview button
                    document.getElementById('previewBtn').disabled = false;
                    
                    // Create audio URL for preview
                    const audioUrl = URL.createObjectURL(recordedBlob);
                    const audioPreview = document.getElementById('audioPreview');
                    audioPreview.src = audioUrl;
                    audioPreview.style.display = 'block';
                    
                    // AUTOMATIC PROCESSING: Convert to base64 and trigger processing
                    autoProcessRecording();
                };
                
                document.getElementById('status').innerHTML = '🎤 Microphone Ready - Click START to Record';
                
            } catch (error) {
                document.getElementById('status').innerHTML = '❌ Microphone access denied';
                console.error('Error accessing microphone:', error);
            }
        }

        function toggleRecording() {
            const recordBtn = document.getElementById('recordBtn');
            const statusDiv = document.getElementById('status');
            
            if (!isRecording) {
                // Start recording
                audioChunks = [];
                recordingTime = 0;
                isRecording = true;
                
                recordBtn.innerHTML = '⏹️ STOP RECORDING';
                recordBtn.style.background = '#666';
                statusDiv.innerHTML = '🔴 RECORDING - Speak clearly in Czech or German';
                
                // Hide processing status and reset
                document.getElementById('processingStatus').style.display = 'none';
                document.getElementById('processingTriggered').value = 'false';
                document.getElementById('previewBtn').disabled = true;
                document.getElementById('audioPreview').style.display = 'none';
                
                // Start timer
                timerInterval = setInterval(updateTimer, 1000);
                
                // Start recording
                mediaRecorder.start(1000);
                
            } else {
                // Stop recording
                isRecording = false;
                mediaRecorder.stop();
                
                recordBtn.innerHTML = '🔄 NEW RECORDING';
                recordBtn.style.background = '#ff4b4b';
                
                statusDiv.innerHTML = '✅ Recording Complete - Processing automatically...';
                
                // Stop timer
                clearInterval(timerInterval);
            }
        }

        function updateTimer() {
            recordingTime++;
            const minutes = Math.floor(recordingTime / 60);
            const seconds = recordingTime % 60;
            document.getElementById('timer').innerHTML = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }

        function playPreview() {
            const audioPreview = document.getElementById('audioPreview');
            audioPreview.play();
            document.getElementById('status').innerHTML = '🔊 Playing Your Recording';
        }

        async function autoProcessRecording() {
            if (recordedBlob && document.getElementById('processingTriggered').value === 'false') {
                // Mark as processing to avoid double-processing
                document.getElementById('processingTriggered').value = 'true';
                
                // Show processing status
                document.getElementById('processingStatus').style.display = 'block';
                document.getElementById('status').innerHTML = '⚡ Auto-Processing Your Voice...';
                
                // Convert blob to base64
                const reader = new FileReader();
                reader.onloadend = function() {
                    const base64Data = reader.result.split(',')[1];
                    
                    // Store in hidden field for Streamlit to access
                    document.getElementById('audioDataField').value = base64Data;
                    
                    // Trigger Streamlit state update
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: {
                            audio_data: base64Data,
                            timestamp: Date.now(),
                            processing: true
                        }
                    }, '*');
                    
                    document.getElementById('status').innerHTML = '✅ Voice sent for transcription and response generation!';
                };
                reader.readAsDataURL(recordedBlob);
            }
        }

        // Reset function for new recording
        function resetRecorder() {
            audioChunks = [];
            recordedBlob = null;
            recordingTime = 0;
            isRecording = false;
            
            document.getElementById('recordBtn').innerHTML = '🔴 START RECORDING';
            document.getElementById('recordBtn').style.background = '#ff4b4b';
            document.getElementById('previewBtn').disabled = true;
            document.getElementById('audioPreview').style.display = 'none';
            document.getElementById('processingStatus').style.display = 'none';
            document.getElementById('status').innerHTML = '🎤 Ready for New Recording';
            document.getElementById('timer').innerHTML = '00:00';
            document.getElementById('processingTriggered').value = 'false';
            
            clearInterval(timerInterval);
        }
    </script>
    """
    
    # Return the component with callback to handle returned data
    return components.html(html_code, height=350, key="enhanced_audio_recorder")

def handle_auto_voice_processing():
    """Handle automatic voice processing after recording stops"""
    
    # Check if we have new audio data to process
    if 'audio_processing_queue' not in st.session_state:
        st.session_state.audio_processing_queue = []
    
    if 'last_processed_timestamp' not in st.session_state:
        st.session_state.last_processed_timestamp = 0
    
    # Get audio recorder component data
    recorder_data = st.session_state.get('enhanced_audio_recorder')
    
    if recorder_data and isinstance(recorder_data, dict):
        audio_data = recorder_data.get('audio_data')
        timestamp = recorder_data.get('timestamp', 0)
        processing_flag = recorder_data.get('processing', False)
        
        # Only process if we have new data
        if (audio_data and 
            timestamp > st.session_state.last_processed_timestamp and 
            processing_flag):
            
            st.session_state.last_processed_timestamp = timestamp
            
            # Show processing status
            with st.spinner("🔄 Transcribing and generating response..."):
                # Process the audio data
                success = process_base64_audio_automatically(audio_data)
                
                if success:
                    st.success("✅ Voice processed successfully!")
                    st.balloons()  # Celebration for successful processing
                else:
                    st.error("❌ Error processing voice. Please try again.")
            
            # Force refresh to show results
            st.rerun()

def process_base64_audio_automatically(base64_audio_data):
    """Process base64 audio data automatically"""
    try:
        # Step 1: Convert base64 to audio file
        st.info("🎧 Converting and enhancing audio...")
        audio_file_path = convert_base64_to_audio_file(base64_audio_data)
        
        if not audio_file_path:
            st.error("Failed to convert audio data")
            return False
        
        # Step 2: Apply 500% amplification for better transcription
        st.info("🔊 Applying 500% amplification for clarity...")
        amplified_path = amplify_recorded_audio(audio_file_path)
        
        # Step 3: Process through the enhanced pipeline
        st.info("⚡ Processing through AI pipeline...")
        
        # Run async processing
        result = asyncio.run(process_enhanced_voice_pipeline(amplified_path))
        
        # Step 4: Display results
        if result:
            text_input, audio_output, stt_latency, llm_latency, tts_latency = result
            
            # Store results for display
            st.session_state.last_text_input = text_input
            st.session_state.last_audio_output = audio_output
            
            # Show latency info
            total_latency = stt_latency + llm_latency + tts_latency
            
            # Display success message with metrics
            st.success(f"""
            ✅ **Voice Processing Complete!**
            
            🎯 **Transcribed:** {text_input}
            
            ⏱️ **Performance:**
            - Speech-to-Text: {stt_latency:.2f}s
            - AI Response: {llm_latency:.2f}s  
            - Text-to-Speech: {tts_latency:.2f}s
            - **Total Time: {total_latency:.2f}s**
            """)
            
            return True
        else:
            st.error("❌ Failed to process voice through AI pipeline")
            return False
        
    except Exception as e:
        st.error(f"❌ Error in automatic processing: {str(e)}")
        logger.error(f"Automatic processing error: {str(e)}")
        return False
    
    finally:
        # Clean up temporary files
        try:
            if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
            if 'amplified_path' in locals() and amplified_path != audio_file_path and os.path.exists(amplified_path):
                os.unlink(amplified_path)
        except:
            pass

def convert_base64_to_audio_file(base64_data):
    """Convert base64 audio data to temporary file"""
    try:
        # Decode base64 data
        audio_bytes = base64.b64decode(base64_data)
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix=".webm")
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # Convert webm to wav for better processing
        wav_path = convert_webm_to_wav(temp_path)
        
        # Clean up webm file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return wav_path
        
    except Exception as e:
        logger.error(f"Base64 conversion error: {str(e)}")
        return None

async def process_enhanced_voice_pipeline(audio_file_path):
    """Process voice through the enhanced pipeline with better error handling"""
    try:
        # Use the existing enhanced processing function
        result = await process_voice_input_pronunciation_enhanced(audio_file_path)
        
        if result and len(result) == 5:
            text_input, audio_output, stt_latency, llm_latency, tts_latency = result
            
            if text_input and text_input.strip():
                return result
            else:
                logger.error("No text transcribed from audio")
                return None
        else:
            logger.error("Invalid result from voice processing pipeline")
            return None
            
    except Exception as e:
        logger.error(f"Enhanced voice pipeline error: {str(e)}")
        return None

def display_voice_results():
    """Display both preview and generated audio results"""
    
    # Display transcribed text
    if hasattr(st.session_state, 'last_text_input') and st.session_state.last_text_input:
        st.subheader("📝 What You Said:")
        st.info(st.session_state.last_text_input)
    
    # Display AI response text
    if (hasattr(st.session_state, 'conversation_history') and 
        st.session_state.conversation_history):
        
        last_exchange = st.session_state.conversation_history[-1]
        if 'assistant_response' in last_exchange:
            st.subheader("🤖 AI Tutor Response:")
            st.success(last_exchange['assistant_response'])
    
    # Display generated audio
    if hasattr(st.session_state, 'last_audio_output') and st.session_state.last_audio_output:
        st.subheader("🔊 AI Generated Speech:")
        
        # Display audio player
        audio_bytes = display_audio(st.session_state.last_audio_output, autoplay=True)
        
        if audio_bytes:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    label="💾 Download AI Response",
                    data=audio_bytes,
                    file_name="ai_tutor_response.mp3",
                    mime="audio/mp3"
                )
            with col2:
                if st.button("🔄 Clear Results"):
                    # Clear results for new recording
                    if hasattr(st.session_state, 'last_text_input'):
                        del st.session_state.last_text_input
                    if hasattr(st.session_state, 'last_audio_output'):
                        del st.session_state.last_audio_output
                    st.rerun()

# ENHANCED MAIN UI FUNCTION - Replace the voice input section in your main() function

def enhanced_voice_input_section():
    """Enhanced voice input section with automatic processing"""
    
    st.subheader("🎤 Professional Voice Recording")
    
    # Check API keys
    keys_set = (
        st.session_state.elevenlabs_api_key and 
        st.session_state.openai_api_key
    )
    
    if not keys_set:
        st.warning("⚠️ Please set both API keys in the sidebar first")
        return
    
    # Instructions
    st.info("""
    **🎯 Automatic Voice Processing:**
    1. 🔴 Click START RECORDING and speak clearly in Czech/German
    2. ⏹️ Click STOP when finished  
    3. 🎧 Your voice will be processed automatically
    4. 🔊 Listen to both your recording and AI response
    
    **✅ Features:** 500% amplification, automatic transcription, instant AI response
    """)
    
    # Enhanced audio recorder
    create_enhanced_audio_recorder_component()
    
    # Handle automatic processing
    handle_auto_voice_processing()
    
    # Display results
    display_voice_results()
    
    # Alternative upload for testing
    st.markdown("---")
    st.write("**📁 Alternative: Upload Audio File for Testing**")
    
    uploaded_file = st.file_uploader(
        "Upload audio file", 
        type=['wav', 'mp3', 'webm', 'ogg'],
        key="manual_upload_test"
    )
    
    if uploaded_file:
        if st.button("🔄 Process Uploaded File", type="secondary"):
            with st.spinner("Processing uploaded audio..."):
                # Process uploaded file
                temp_path = tempfile.mktemp(suffix=".wav")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Apply amplification
                amplified_path = amplify_recorded_audio(temp_path)
                
                # Process
                result = asyncio.run(process_enhanced_voice_pipeline(amplified_path))
                
                if result:
                    text_input, audio_output, stt_latency, llm_latency, tts_latency = result
                    st.session_state.last_text_input = text_input
                    st.session_state.last_audio_output = audio_output
                    
                    total_latency = stt_latency + llm_latency + tts_latency
                    st.success(f"✅ Upload processed in {total_latency:.2f}s")
                    st.rerun()
                
                # Clean up
                try:
                    os.unlink(temp_path)
                    if amplified_path != temp_path:
                        os.unlink(amplified_path)
                except:
                    pass
# ----------------------------------------------------------------------------------
# SPEECH RECOGNITION (STT) SECTION - IMPROVED FOR BETTER ACCURACY
# ----------------------------------------------------------------------------------

class AudioRecorder:
    """Class for recording and processing audio input"""
    
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000  # Whisper prefers 16kHz
        
    def start_recording(self):
        """Start recording audio"""
        self.recording = True
        self.audio_data = []
        
        def record_thread():
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self._audio_callback):
                while self.recording:
                    time.sleep(0.1)
        
        self.thread = threading.Thread(target=record_thread)
        self.thread.start()
        return True
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio data"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_data.append(indata.copy())
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        if not self.recording:
            return None
            
        self.recording = False
        self.thread.join()
        
        if not self.audio_data:
            return None
            
        # Combine all audio chunks
        audio = np.concatenate(self.audio_data, axis=0)
        
        # Reset for next recording
        self.audio_data = []
        
        return audio, self.sample_rate
    
    def save_audio(self, audio_data, filename="recorded_audio.wav"):
        """Save audio data to file"""
        if audio_data is None:
            return None
            
        audio, sample_rate = audio_data
        sf.write(filename, audio, sample_rate)
        return filename

    def convert_to_mp3(self, audio_data, output_filename="recorded_audio.mp3"):
        """Convert recorded audio to MP3 format"""
        if audio_data is None:
            return None
            
        wav_file = self.save_audio(audio_data, "temp_recording.wav")
        audio = AudioSegment.from_wav(wav_file)
        audio.export(output_filename, format="mp3")
        
        # Remove temporary file
        os.remove(wav_file)
        
        return output_filename

    def enhance_audio_quality(self, audio_data):
        """Enhance audio quality for better transcription, including noise reduction"""
        if audio_data is None:
            return None
            
        # Handle both tuple and file path inputs
        if isinstance(audio_data, tuple):
            audio, sample_rate = audio_data
            # Save to temporary file
            temp_wav = "temp_enhance.wav"
            sf.write(temp_wav, audio, sample_rate)
        else:
            # audio_data is a file path
            temp_wav = audio_data
        
        try:
            # Load with pydub for processing
            audio_segment = AudioSegment.from_wav(temp_wav)
            
            # Normalize volume
            normalized_audio = audio_segment.normalize()
            
            # Remove silence at beginning and end
            trimmed_audio = self._trim_silence(normalized_audio)
            
            # Convert to numpy array for noise reduction
            samples = np.array(trimmed_audio.get_array_of_samples()).astype(np.float32)
            
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(y=samples, sr=self.sample_rate)
            
            # Convert back to AudioSegment
            reduced_audio = AudioSegment(
                reduced_noise.astype(np.int16).tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            
            # Export enhanced audio
            enhanced_wav = "enhanced_recording.wav"
            reduced_audio.export(enhanced_wav, format="wav")
            
            # Clean up temp file if we created it
            if isinstance(audio_data, tuple) and os.path.exists(temp_wav):
                os.remove(temp_wav)
                
            return enhanced_wav
            
        except Exception as e:
            logger.error(f"Audio enhancement error: {str(e)}")
            return temp_wav if os.path.exists(temp_wav) else None
        
    def _trim_silence(self, audio_segment, silence_threshold=-50, min_silence_len=300):
        """Remove silence from beginning and end of recording"""
        # Split on silence
        chunks = split_on_silence(
            audio_segment, 
            min_silence_len=min_silence_len,
            silence_thresh=silence_threshold,
            keep_silence=100  # Keep 100ms of silence
        )
        
        # If no chunks found, return original
        if not chunks:
            return audio_segment
            
        # Combine chunks
        combined_audio = chunks[0]
        for chunk in chunks[1:]:
            combined_audio += chunk
            
        return combined_audio

class SpeechRecognizer:
    """Class for speech recognition and language detection with improved accuracy"""
    
    def __init__(self, model_name="medium"):
        """Initialize with the specified Whisper model"""
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """Load the Whisper model if not already loaded"""
        if st.session_state.whisper_local_model is None:
            with st.spinner(f"Loading Whisper model '{self.model_name}'..."):
                st.session_state.whisper_local_model = whisper.load_model(self.model_name)
        
        self.model = st.session_state.whisper_local_model
        return self.model
    
    def transcribe_audio(self, audio_file, language=None):
        """Transcribe audio using Whisper model with improved options"""
        start_time = time.time()
        
        try:
            # Load model if needed
            self.load_model()
            
            # Transcribe with appropriate options for better Czech/German detection
            options = {
                "task": "transcribe",
                "verbose": False,
                "beam_size": 5,         # Increased from default of 5
                "best_of": 5,           # Increased from default of 5
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Temperature fallback
                "without_timestamps": False,  # Include timestamps for better segmentation
                "word_timestamps": True      # Get word-level timestamps
            }
            
            # If we know the language already, specify it
            if language in ["cs", "de"]:
                options["language"] = language
            
            # Run transcription
            result = self.model.transcribe(audio_file, **options)
            
            # Calculate latency and update metrics
            latency = time.time() - start_time
            st.session_state.performance_metrics["stt_latency"].append(latency)
            st.session_state.performance_metrics["api_calls"]["whisper"] += 1
            
            return {
                "text": result["text"],
                "language": result["language"],
                "segments": result["segments"],
                "latency": latency
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return {
                "text": "",
                "language": None,
                "error": str(e),
                "latency": time.time() - start_time
            }
    
    def detect_and_mark_languages(self, transcription):
        """Detect language segments and mark with appropriate tags - improved version"""
        if not transcription or "text" not in transcription:
            return ""
            
        text = transcription["text"]
        detected_language = transcription.get("language")
        segments = transcription.get("segments", [])
        
        # If no segments, return the text with detected language markup
        if not segments or len(segments) <= 1:
            if detected_language in SUPPORTED_LANGUAGES:
                return f"[{detected_language}] {text}"
            return text
        
        # Advanced language detection using multiple features
        return self._advanced_language_marking(segments, detected_language)
    
    def _advanced_language_marking(self, segments, default_language):
        """Advanced language marking using multiple detection techniques"""
        # Initialize
        marked_text = ""
        current_language = None
        buffer = []
        
        # Process each segment with advanced language detection
        for segment in segments:
            segment_text = segment["text"].strip()
            
            # Skip empty segments
            if not segment_text:
                continue
                
            # Get language confidence for this segment
            segment_language = self._detect_segment_language_advanced(segment_text, default_language)
            
            # If language changes or confidence is high, mark the language
            if segment_language != current_language:
                # Flush buffer if we have content
                if buffer:
                    # Add previous language marker
                    if current_language in SUPPORTED_LANGUAGES:
                        marked_text += f"[{current_language}] "
                    
                    # Add buffered text
                    marked_text += " ".join(buffer) + " "
                    buffer = []
                
                # Set new language
                current_language = segment_language
                
                # Start new buffer with this segment
                buffer.append(segment_text)
            else:
                # Same language, add to buffer
                buffer.append(segment_text)
        
        # Flush any remaining buffer
        if buffer:
            if current_language in SUPPORTED_LANGUAGES:
                marked_text += f"[{current_language}] "
            marked_text += " ".join(buffer)
        
        return marked_text.strip()
    
    def _detect_segment_language_advanced(self, text, default_language):
        """Enhanced language detection using multiple signals"""
        # First check: character frequency analysis
        czech_chars = set("áčďéěíňóřšťúůýž")
        german_chars = set("äöüß")
        
        text_lower = text.lower()
        
        # Count special characters
        czech_char_count = sum(1 for char in text_lower if char in czech_chars)
        german_char_count = sum(1 for char in text_lower if char in german_chars)
        
        # Character-based confidence
        char_confidence = 0
        char_language = None
        
        if czech_char_count > german_char_count:
            char_language = "cs"
            char_confidence = min(1.0, czech_char_count / max(len(text) * 0.1, 1))
        elif german_char_count > czech_char_count:
            char_language = "de"
            char_confidence = min(1.0, german_char_count / max(len(text) * 0.1, 1))
        
        # Second check: vocabulary analysis
        # Czech-specific common words (expanded list)
        czech_words = {
            "jsem", "jsi", "je", "jsou", "byl", "byla", "bylo", "být", "budu", 
            "máme", "mám", "prosím", "děkuji", "ahoj", "dobrý", "dobře", "ano", "ne",
            "já", "ty", "on", "ona", "my", "vy", "oni", "den", "noc", "chci", "dnes",
            "zítra", "včera", "tady", "tam", "proč", "kde", "kdy", "jak", "co", "kdo",
            "to", "ten", "ta", "mít", "jít", "dělat", "vidět", "slyšet", "vědět"
        }
        
        # German-specific common words (expanded list)
        german_words = {
            "ich", "du", "er", "sie", "es", "wir", "ihr", "sind", "ist", "bin",
            "habe", "haben", "hatte", "war", "gewesen", "bitte", "danke", "gut", "ja", "nein",
            "der", "die", "das", "ein", "eine", "zu", "von", "mit", "für", "auf",
            "wenn", "aber", "oder", "und", "nicht", "auch", "so", "wie", "was", "wo",
            "wann", "wer", "warum", "möchte", "kann", "muss", "soll", "darf", "will"
        }
        
        # Clean and tokenize text
        clean_text = re.sub(r'[^\w\s]', '', text_lower)
        words = clean_text.split()
        
        # Count word occurrences with weighted importance
        czech_word_count = sum(1 for word in words if word in czech_words)
        german_word_count = sum(1 for word in words if word in german_words)
        
        # Word-based confidence
        word_confidence = 0
        word_language = None
        
        if czech_word_count > german_word_count:
            word_language = "cs"
            word_confidence = min(1.0, czech_word_count / max(len(words) * 0.2, 1))
        elif german_word_count > czech_word_count:
            word_language = "de"
            word_confidence = min(1.0, german_word_count / max(len(words) * 0.2, 1))
        
        # Combine evidence with weighted approach
        # Characters are more reliable in these languages due to unique letters
        if char_language and word_language:
            # If both agree, that's strongest
            if char_language == word_language:
                return char_language
            # If they disagree, go with the higher confidence
            elif char_confidence > word_confidence:
                return char_language
            else:
                return word_language
        # If only one has evidence
        elif char_language:
            return char_language
        elif word_language:
            return word_language
        
        # Fall back to default language
        return default_language

async def transcribe_with_api(audio_file, api_key):
    """Enhanced transcription with pronunciation focus for Czech/German"""
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            # Open file in binary mode
            with open(audio_file, "rb") as f:
                file_content = f.read()
            
            # Prepare the multipart form data
            files = {
                "file": (os.path.basename(audio_file), file_content, "audio/wav")
            }
            
            # ENHANCED: Pronunciation-focused settings for Czech/German
            data = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "temperature": "0.0",  # LOWEST temperature for consistent pronunciation
                "language": None,  # Let Whisper auto-detect between cs/de
                "prompt": "This audio contains Czech and German speech. Focus on accurate pronunciation and phonetic understanding. Common Czech words: ahoj, děkuji, prosím, dobrý den. Common German words: hallo, danke, bitte, guten tag."  # Pronunciation hints
            }
            
            # Send the request with pronunciation-enhanced settings
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # ENHANCED: Apply pronunciation-based post-processing
                enhanced_result = enhance_pronunciation_transcription(result)
                
                # Calculate latency and update metrics
                latency = time.time() - start_time
                st.session_state.performance_metrics["stt_latency"].append(latency)
                st.session_state.performance_metrics["api_calls"]["whisper"] += 1
                
                return enhanced_result
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {
                    "text": "",
                    "language": None,
                    "error": f"API error: {response.status_code} - {response.text}",
                    "latency": time.time() - start_time
                }
    
    except Exception as e:
        logger.error(f"Enhanced transcription API error: {str(e)}")
        return {
            "text": "",
            "language": None,
            "error": str(e),
            "latency": time.time() - start_time
        }

def enhance_pronunciation_transcription(result):
    """Post-process transcription for better pronunciation understanding"""
    try:
        text = result.get("text", "")
        language = result.get("language", "auto")
        segments = result.get("segments", [])
        
        # Apply pronunciation-based corrections
        enhanced_text = apply_pronunciation_corrections(text, language)
        
        # Enhance segments with pronunciation markers
        enhanced_segments = []
        for segment in segments:
            enhanced_segment = segment.copy()
            enhanced_segment["text"] = apply_pronunciation_corrections(
                segment.get("text", ""), language
            )
            enhanced_segments.append(enhanced_segment)
        
        return {
            "text": enhanced_text,
            "language": language,
            "segments": enhanced_segments,
            "latency": result.get("latency", 0),
            "pronunciation_enhanced": True
        }
        
    except Exception as e:
        logger.error(f"Pronunciation enhancement error: {str(e)}")
        return result

def apply_pronunciation_corrections(text, language):
    """Apply pronunciation-based corrections for Czech/German"""
    if not text:
        return text
    
    # Czech pronunciation corrections
    czech_corrections = {
        # Common mispronunciations to correct pronunciations
        "dziekuji": "děkuji",
        "prosiem": "prosím", 
        "dobri": "dobrý",
        "ahoy": "ahoj",
        "yak": "jak",
        "tschechisch": "česky",
        "ano": "ano",
        "ne": "ne"
    }
    
    # German pronunciation corrections
    german_corrections = {
        "ich": "ich",
        "das": "das", 
        "ist": "ist",
        "und": "und",
        "haben": "haben",
        "sein": "sein",
        "werden": "werden",
        "können": "können",
        "müssen": "müssen",
        "guten tag": "guten Tag",
        "auf wiedersehen": "auf Wiedersehen"
    }
    
    # Apply corrections based on detected language or overall context
    corrected_text = text
    
    # Apply Czech corrections if Czech content detected
    if language == "cs" or any(word in text.lower() for word in ["ahoj", "děkuji", "prosím"]):
        for wrong, correct in czech_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
    
    # Apply German corrections if German content detected  
    if language == "de" or any(word in text.lower() for word in ["hallo", "guten", "danke"]):
        for wrong, correct in german_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

# ----------------------------------------------------------------------------------
# LANGUAGE MODEL (LLM) SECTION - ENHANCED FOR BETTER LANGUAGE CONTROL
# ----------------------------------------------------------------------------------

async def generate_llm_response(prompt, system_prompt=None, api_key=None):
    """Generate a response using the OpenAI GPT model with enhanced language control"""
    if not api_key:
        api_key = st.session_state.openai_api_key
        
    if not api_key:
        logger.error("OpenAI API key not provided")
        return {
            "response": "Error: OpenAI API key not configured. Please set it in the sidebar.",
            "latency": 0
        }
    
    start_time = time.time()
    
    # Set up the conversation messages
    messages = []
    
    # Add custom system prompt with language distribution preferences
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        # Create dynamic system prompt based on language preferences
        response_language = st.session_state.response_language
        
        if response_language == "both":
            # Get distribution preferences
            cs_percent = st.session_state.language_distribution["cs"]
            de_percent = st.session_state.language_distribution["de"]
            
            system_content = (
                f"You are a multilingual AI language tutor specializing in Czech and German. "
                f"Respond to the user using both languages with approximately {cs_percent}% Czech and {de_percent}% German. "
                f"Always use appropriate language markers [cs] and [de] to indicate language sections. "
                f"Keep your responses educational, helpful, and natural. "
                f"Teach proper grammar, vocabulary, and pronunciation, and correct errors when appropriate."
            )
        elif response_language == "cs":
            system_content = (
                "You are a Czech language tutor. Always respond in Czech only, with the [cs] marker. "
                "Keep your responses educational, helpful, and natural. "
                "Teach proper grammar, vocabulary, and pronunciation, and correct errors when appropriate."
            )
        elif response_language == "de":
            system_content = (
                "You are a German language tutor. Always respond in German only, with the [de] marker. "
                "Keep your responses educational, helpful, and natural. "
                "Teach proper grammar, vocabulary, and pronunciation, and correct errors when appropriate."
            )
        else:
            # Default to matching the input language
            system_content = (
                "You are a multilingual AI language tutor specializing in Czech and German. "
                "Respond to the user in the same language they used. "
                "If they used language markers like [cs] or [de], maintain those markers in your response. "
                "Keep your responses educational, helpful, and natural. "
                "Teach proper grammar, vocabulary, and pronunciation, and correct errors when appropriate."
            )
            
        messages.append({"role": "system", "content": system_content})
    
    # Add previous conversation history for context
    for exchange in st.session_state.conversation_history[-5:]:  # Last 5 exchanges
        if "user_input" in exchange:
            messages.append({"role": "user", "content": exchange["user_input"]})
        if "assistant_response" in exchange:
            messages.append({"role": "assistant", "content": exchange["assistant_response"]})
    
    # Add the current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENAI_API_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",  # Use faster model for lower latency
                    "messages": messages,
                    "temperature": 0.5,
                    "max_tokens": 400
                },
                timeout=30.0
            )
            
            # Calculate latency and update metrics
            latency = time.time() - start_time
            st.session_state.performance_metrics["llm_latency"].append(latency)
            st.session_state.performance_metrics["api_calls"]["openai"] += 1
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                # Ensure language markers are preserved and added if missing
                response_text = preserve_language_markers(prompt, response_text)
                
                return {
                    "response": response_text,
                    "latency": latency,
                    "tokens": result.get("usage", {})
                }
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return {
                    "response": f"Error: {response.status_code}",
                    "error": response.text,
                    "latency": latency
                }
    
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        return {
            "response": f"Error: {str(e)}",
            "latency": time.time() - start_time
        }

def add_language_markers(text):
    """Add language markers to text based on language detection"""
    # Split into sentences for language detection
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    result = ""
    current_language = None
    
    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        # Detect language
        sentence_language = detect_primary_language(sentence)
        
        # Add language marker if different from current
        if sentence_language != current_language and sentence_language in ["cs", "de"]:
            result += f"[{sentence_language}] "
            current_language = sentence_language
        
        result += sentence + " "
    
    # If no language was detected, fall back to Czech
    if current_language is None:
        return f"[cs] {text}"
        
    return result.strip()

def preserve_language_markers(input_text, response_text):
    """ENHANCED: Handle auto language detection mode"""
    
    # Check if response already has markers
    if re.search(r'\[([a-z]{2})\]', response_text):
        return fix_language_markers(response_text)
    
    response_language = st.session_state.response_language
    
    # NEW: Auto mode - detect from input and respond accordingly
    if response_language == "auto":
        detected_distribution = auto_detect_language_distribution(input_text)
        return apply_auto_distribution(response_text, detected_distribution)
    
    # Existing logic for other modes...
    input_markers = re.findall(r'\[([a-z]{2})\]', input_text)
    
    if input_markers:
        dominant_input_language = input_markers[0]
        if len(set(input_markers)) == 1:
            return f"[{dominant_input_language}] {response_text}"
        return add_language_markers_from_input(response_text, input_markers)
    
    if response_language == "both":
        return apply_distribution_settings(response_text)
    elif response_language in ["cs", "de"]:
        return f"[{response_language}] {response_text}"
    
    return add_language_markers(response_text)

def apply_auto_distribution(text, distribution):
    """Apply auto-detected language distribution"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    
    cs_percent = distribution["cs"]
    de_percent = distribution["de"]
    
    total_sentences = len(sentences)
    cs_sentences = int(total_sentences * (cs_percent / 100))
    
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            if i < cs_sentences:
                result += f"[cs] {sentence} "
            else:
                result += f"[de] {sentence} "
    
    return result.strip()

def add_language_markers_from_input(text, input_languages):
    """Add markers based on input languages, not distribution"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 1:
        return f"[{input_languages[0]}] {text}"
    
    result = ""
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            # Alternate between input languages
            lang_idx = i % len(input_languages)
            result += f"[{input_languages[lang_idx]}] {sentence} "
    
    return result.strip()

def auto_detect_language_distribution(input_text):
    """Auto-detect language distribution from user input"""
    
    # Remove any existing language markers for clean detection
    clean_text = re.sub(r'\[[a-z]{2}\]', '', input_text).strip()
    
    # Split into words for analysis
    words = re.findall(r'\b\w+\b', clean_text.lower())
    
    if not words:
        return {"cs": 50, "de": 50}  # Default if no words
    
    # Czech indicators
    czech_chars = set("áčďéěíňóřšťúůýž")
    czech_words = {
        "jsem", "jsi", "je", "jsou", "ahoj", "dobrý", "dobře", "ano", "ne",
        "já", "ty", "on", "ona", "prosím", "děkuji", "jak", "co", "kde", "kdy"
    }
    
    # German indicators  
    german_chars = set("äöüß")
    german_words = {
        "ich", "du", "er", "sie", "ist", "bin", "habe", "haben", "gut", "ja", "nein",
        "der", "die", "das", "ein", "eine", "und", "wie", "was", "wo", "wann"
    }
    
    # Count evidence
    czech_evidence = 0
    german_evidence = 0
    
    # Character-based evidence
    for char in clean_text.lower():
        if char in czech_chars:
            czech_evidence += 2
        elif char in german_chars:
            german_evidence += 2
    
    # Word-based evidence (stronger weight)
    for word in words:
        if word in czech_words:
            czech_evidence += 3
        elif word in german_words:
            german_evidence += 3
    
    # Calculate percentages
    total_evidence = czech_evidence + german_evidence
    
    if total_evidence == 0:
        return {"cs": 50, "de": 50}  # Default if unclear
    
    cs_percent = int((czech_evidence / total_evidence) * 100)
    de_percent = 100 - cs_percent
    
    # Ensure minimum 20% for each language to maintain bilingual nature
    if cs_percent < 20:
        cs_percent = 20
        de_percent = 80
    elif de_percent < 20:
        de_percent = 20
        cs_percent = 80
    
    return {"cs": cs_percent, "de": de_percent}

def apply_distribution_settings(text):
    """Only apply distribution when no input language specified"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = ""
    
    cs_percent = st.session_state.language_distribution["cs"]
    de_percent = st.session_state.language_distribution["de"]
    
    total_sentences = len(sentences)
    cs_sentences = int(total_sentences * (cs_percent / 100))
    
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            if i < cs_sentences:
                result += f"[cs] {sentence} "
            else:
                result += f"[de] {sentence} "
    
    return result.strip()

def fix_language_markers(text):
    """Fix invalid or missing language markers"""
    # Pattern to match language markers
    marker_pattern = r'\[([a-z]{2})\]'
    
    # Find all markers
    markers = re.findall(marker_pattern, text)
    
    # Replace invalid markers with valid ones
    for marker in markers:
        if marker not in ["cs", "de"]:
            # Replace with a valid marker based on surrounding text
            text = text.replace(f"[{marker}]", "[cs]")
    
    return text

def process_multilingual_text_seamless(text, detect_language=True):
    """IMPROVED: Process text with consistent audio quality"""
    
    segments = parse_language_segments_advanced(text)
    
    if len(segments) <= 1:
        return process_multilingual_text(text, detect_language)
    
    # Collect all audio segments first
    audio_segments = []
    language_codes = []
    total_time = 0
    
    for i, segment in enumerate(segments):
        if not segment["text"].strip():
            continue
            
        processed_text = prepare_text_for_seamless_transition(
            segment["text"], 
            segment["language"],
            is_first=(i == 0),
            is_last=(i == len(segments)-1),
            prev_lang=segments[i-1]["language"] if i > 0 else None
        )
        
        audio_data, generation_time = generate_speech_seamless(
            processed_text, 
            language_code=segment["language"],
            context={
                "position": i,
                "total_segments": len(segments),
                "prev_language": segments[i-1]["language"] if i > 0 else None,
                "next_language": segments[i+1]["language"] if i < len(segments)-1 else None
            }
        )
        
        if audio_data:
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            audio_segments.append(audio_segment)
            language_codes.append(segment["language"])
            total_time += generation_time
    
    if not audio_segments:
        return None, 0
    
    # CRITICAL: Normalize all segments for consistent quality
    normalized_segments = normalize_audio_segments(audio_segments, language_codes)
    
    # Combine with smooth transitions
    combined_audio = normalized_segments[0]
    
    for i in range(1, len(normalized_segments)):
        # Apply smooth crossfade
        crossfade_duration = 50  # ms
        combined_audio = combined_audio.append(
            normalized_segments[i], 
            crossfade=crossfade_duration
        )
    
    # Final quality enhancement
    combined_audio = enhance_multilingual_audio_final(combined_audio)
    
    # Save with consistent quality
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        combined_audio.export(
            temp_file.name, 
            format="mp3", 
            bitrate="192k",
            parameters=["-ac", "1", "-ar", "22050", "-af", "volume=0.8"]  # Consistent volume
        )
        return temp_file.name, total_time
        

def parse_language_segments_advanced(text):
    """Advanced parsing that preserves context for better transitions"""
    segments = []
    
    # Split by language markers but preserve context
    parts = re.split(r'(\[[a-z]{2}\])', text)
    
    current_language = None
    current_text = ""
    
    for part in parts:
        if re.match(r'\[[a-z]{2}\]', part):
            # Save previous segment
            if current_text.strip():
                segments.append({
                    "text": current_text.strip(),
                    "language": current_language
                })
            
            # Set new language
            current_language = part[1:-1]  # Remove brackets
            current_text = ""
        else:
            current_text += part
    
    # Add final segment
    if current_text.strip():
        segments.append({
            "text": current_text.strip(),
            "language": current_language
        })
    
    # Detect language for unmarked segments
    for segment in segments:
        if segment["language"] is None:
            segment["language"] = detect_primary_language(segment["text"])
    
    return segments

def prepare_text_for_seamless_transition(text, language, is_first, is_last, prev_lang):
    """Prepare text for seamless language transitions"""
    
    # Add micro-pauses at language boundaries
    if not is_first and prev_lang and prev_lang != language:
        text = f"<break time='100ms'/>{text}"
    
    # Language-specific pronunciation hints
    if language == "cs":
        # Czech pronunciation optimization
        text = optimize_czech_pronunciation(text)
    elif language == "de": 
        # German pronunciation optimization
        text = optimize_german_pronunciation(text)
    
    return text

def generate_speech_seamless(text, language_code, context):
    """Generate speech optimized for seamless multilingual transitions"""
    
    voice_id = st.session_state.elevenlabs_voice_id
    api_key = st.session_state.elevenlabs_api_key
    
    # CRITICAL: Use consistent voice with language-specific fine-tuning
    model_id = "eleven_multilingual_v2"  # Best for seamless switching
    
    # Context-aware voice settings for seamless transitions
    voice_settings = get_contextual_voice_settings(language_code, context)
    
    # Enhanced text with SSML for better pronunciation
    enhanced_text = add_pronunciation_markup(text, language_code)
    
    data = {
        "text": enhanced_text,
        "model_id": model_id,
        "voice_settings": voice_settings,
        "apply_text_normalization": "auto"  # Better for mixed content
    }
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json", 
        "xi-api-key": api_key
    }
    
    try:
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            json=data,
            headers=headers,
            timeout=20
        )
        
        if response.status_code == 200:
            return BytesIO(response.content), 0.5  # Faster processing
        else:
            return None, 0
            
    except Exception as e:
        logger.error(f"Seamless TTS error: {str(e)}")
        return None, 0

def optimize_czech_pronunciation(text):
    """Optimize text for better Czech pronunciation"""
    # Add stress markers for difficult Czech words
    czech_stress_words = {
        "děkuji": "děkuji",  # Already stressed correctly
        "prosím": "prosím",
        "můžeme": "můžeme"
    }
    
    for word, stressed in czech_stress_words.items():
        text = text.replace(word, stressed)
    
    return text

def optimize_german_pronunciation(text):
    """Optimize text for better German pronunciation"""
    # Add pronunciation hints for German
    german_pronunciation = {
        "ich": "ikh",  # Better pronunciation hint
        "nicht": "nikht"
    }
    
    for word, hint in german_pronunciation.items():
        if word in text.lower():
            text = text.replace(word, f"{word}")  # Keep original but will be processed
    
    return text

def add_pronunciation_markup(text, language_code):
    """Add SSML markup for better pronunciation"""
    
    # Basic SSML wrapper
    if language_code == "cs":
        return f'<speak><lang xml:lang="cs">{text}</lang></speak>'
    elif language_code == "de":
        return f'<speak><lang xml:lang="de">{text}</lang></speak>'
    else:
        return text

def apply_language_transition_blend(audio_segment, prev_lang, current_lang):
    """Apply audio processing for smoother language transitions"""
    
    # Normalize volume for consistent transitions
    normalized = audio_segment.normalize()
    
    # Add slight fade-in for smoother start after language switch
    fade_duration = 30  # ms
    faded = normalized.fade_in(fade_duration)
    
    return faded

def enhance_multilingual_audio_final(combined_audio):
    """Final enhancement for multilingual audio"""
    
    # Normalize overall volume
    normalized = combined_audio.normalize()
    
    # Apply gentle compression for consistency
    compressed = normalized.compress_dynamic_range(threshold=-20.0, ratio=2.0)
    
    # Slight reverb for naturalness (if needed)
    # In production, you might add subtle reverb here
    
    return compressed

def get_contextual_voice_settings(language_code, context):
    """Get voice settings optimized for context and seamless transitions"""
    
    base_settings = {
        "stability": 0.75,
        "similarity_boost": 0.85,
        "style": 0.6,
        "use_speaker_boost": True
    }
    
    # Adjust based on position and language transitions
    if context["position"] > 0:  # Not first segment
        base_settings["stability"] += 0.1  # More stable for consistency
    
    # Language-specific adjustments
    if language_code == "cs":
        base_settings.update({
            "stability": 0.80,
            "similarity_boost": 0.90,
            "style": 0.7
        })
    elif language_code == "de":
        base_settings.update({
            "stability": 0.75,
            "similarity_boost": 0.85,
            "style": 0.65
        })
    
    return base_settings

def detect_primary_language(text):
    """Detect the primary language of a text with improved accuracy"""
    # Czech-specific characters
    czech_chars = set("áčďéěíňóřšťúůýž")
    
    # German-specific characters
    german_chars = set("äöüß")
    
    # Count language-specific characters
    text_lower = text.lower()
    czech_count = sum(1 for char in text_lower if char in czech_chars)
    german_count = sum(1 for char in text_lower if char in german_chars)
    
    # Czech-specific words
    czech_words = {
        "jsem", "jsi", "je", "jsou", "byl", "byla", "bylo", "být", "budu", 
        "máme", "mám", "prosím", "děkuji", "ahoj", "dobrý", "dobře", "ano", "ne",
        "já", "ty", "on", "ona", "my", "vy", "oni", "den", "noc", "chci", "dnes",
        "zítra", "včera", "tady", "tam", "proč", "kde", "kdy", "jak", "co", "kdo",
        "to", "ten", "ta", "mít", "jít", "dělat", "vidět", "slyšet", "vědět"
    }
    
    # German-specific words
    german_words = {
        "ich", "du", "er", "sie", "es", "wir", "ihr", "sind", "ist", "bin",
        "habe", "haben", "hatte", "war", "gewesen", "bitte", "danke", "gut", "ja", "nein",
        "der", "die", "das", "ein", "eine", "zu", "von", "mit", "für", "auf",
        "wenn", "aber", "oder", "und", "nicht", "auch", "so", "wie", "was", "wo",
        "wann", "wer", "warum", "möchte", "kann", "muss", "soll", "darf", "will"
    }
    
    # Count word occurrences
    words = re.findall(r'\b\w+\b', text_lower)
    czech_word_count = sum(1 for word in words if word in czech_words)
    german_word_count = sum(1 for word in words if word in german_words)
    
    # Improved scoring system with weighted metrics
    czech_evidence = czech_count * 2 + czech_word_count * 3
    german_evidence = german_count * 2 + german_word_count * 3
    
    # Add grammatical structure evidence
    # Czech typically has words with these endings
    czech_endings = ["ová", "ský", "ští", "íme", "áme", "íte", "áte", "ní", "ou"]
    # German typically has these structures
    german_structures = ["ich ", "du ", "wir ", "sie ", "haben ", "sein ", "werden "]
    
    for ending in czech_endings:
        if ending in text_lower:
            czech_evidence += 1
            
    for structure in german_structures:
        if structure in text_lower:
            german_evidence += 1
    
    # Determine primary language
    if czech_evidence > german_evidence and czech_evidence > 0:
        return "cs"
    elif german_evidence > czech_evidence and german_evidence > 0:
        return "de"
    
    # If unable to determine, use default based on distribution preference
    if st.session_state.language_distribution["cs"] >= st.session_state.language_distribution["de"]:
        return "cs"
    else:
        return "de"

# ----------------------------------------------------------------------------------
# TEXT-TO-SPEECH (TTS) SECTION - ENHANCED FOR ACCENT ISOLATION
# ----------------------------------------------------------------------------------

def get_voices():
    """Fetch available voices from ElevenLabs API with robust error handling"""
    api_key = st.session_state.elevenlabs_api_key
    if not api_key or not isinstance(api_key, str) or not api_key.strip():
        st.error("ElevenLabs API key is missing or invalid. Please set it in the sidebar.")
        return []
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }
    try:
        response = requests.get(f"{ELEVENLABS_API_URL}/voices", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "voices" in data:
                return data["voices"]
            else:
                st.error("No 'voices' key in ElevenLabs API response. Full response: " + str(data))
                return []
        else:
            st.error(f"Failed to get voices: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching voices: {e}")
        return []

def generate_speech(text, language_code=None, voice_id=None):
    """Generate speech using ElevenLabs API with improved error handling and language-specific optimization"""
    if not text or text.strip() == "":
        logger.error("Empty text provided to generate_speech")
        return None, 0
        
    if not voice_id:
        voice_id = st.session_state.elevenlabs_voice_id
        
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        logger.error("ElevenLabs API key not provided")
        return None, 0
    
    # Check cache first
    cache_key = f"{text}_{language_code}_{voice_id}"
    if hasattr(st.session_state, 'tts_cache') and cache_key in st.session_state.tts_cache:
        return st.session_state.tts_cache[cache_key]
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
# Use MULTILINGUAL model for native-quality pronunciation
    model_id = "eleven_multilingual_v2"  # CRITICAL: This fixes accent issues
    
    # HIGH QUALITY voice settings for native-like pronunciation
    if language_code == "cs":  # Czech
        voice_settings = {
            "stability": 0.85,        # HIGH for consistent Czech pronunciation
            "similarity_boost": 0.95,  # VERY HIGH for native Czech quality
            "style": 0.8,             # HIGH for natural Czech expression
            "use_speaker_boost": True  # ENABLE for better quality
        }
    elif language_code == "de":  # German
        voice_settings = {
            "stability": 0.80,        # HIGH for consistent German pronunciation  
            "similarity_boost": 0.90,  # VERY HIGH for native German quality
            "style": 0.75,            # HIGH for natural German expression
            "use_speaker_boost": True  # ENABLE for better quality
        }
    else:
        voice_settings = {
            "stability": 0.85,
            "similarity_boost": 0.90,
            "style": 0.8,
            "use_speaker_boost": True
        }
    
    # Simplified text optimization
    optimized_text = text.strip()
    
    data = {
        "text": optimized_text,
        "model_id": model_id,
        "voice_settings": voice_settings
    }
    
    start_time = time.time()
    
    try:
        # Reduced retries and timeout
        max_retries = 2
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    json=data,
                    headers=headers,
                    timeout=15  # Reduced timeout
                )
                
                if response.status_code == 200:
                    break
                    
                if response.status_code in [429, 500, 502, 503, 504]:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    break
                    
            except requests.exceptions.RequestException as e:
                time.sleep(retry_delay * (attempt + 1))
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            content = response.content
            if len(content) < 100:
                return None, generation_time
                
            # Cache the result
            if not hasattr(st.session_state, 'tts_cache'):
                st.session_state.tts_cache = {}
            st.session_state.tts_cache[cache_key] = (BytesIO(content), generation_time)
            
            return BytesIO(content), generation_time
        else:
            return None, generation_time
    
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        return None, time.time() - start_time

def optimize_text_for_language(text, language_code):
    """Optimize text for specific language pronunciation"""
    # Simplified optimization for demo - in production, would use more sophisticated SSML
    if language_code == "cs":
        # Add pauses after punctuation for Czech
        text = re.sub(r'([.!?])', r'\1...', text)
        
        # Slow down slightly for more accurate Czech pronunciation
        text = f"{text}"
    elif language_code == "de":
        # German optimization
        # Add slight emphasis to compound words
        text = re.sub(r'([A-ZÄÖÜa-zäöüß]{6,})', r' \1 ', text)
    
    return text

def enhance_audio_for_language(audio_segment, language_code):
    """Apply language-specific enhancements to the audio"""
    try:
        # Basic enhancements for both languages
        enhanced = audio_segment.normalize()
        
        if language_code == "cs":
            # Czech typically has more varied pitch - enhance slightly
            # In production, would use more complex DSP transformations
            enhanced = enhanced.normalize()
        elif language_code == "de":
            # German tends to have more emphasis - boost low frequencies slightly
            # This is a simple approximation - real production would use filters
            enhanced = enhanced.normalize()
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing audio: {str(e)}")
        return audio_segment  # Return original on error

def process_multilingual_text(text, detect_language=True):
    """Process text with language markers and generate audio with accent isolation"""
    # Parse language segments
    segments = []
    
    if detect_language:
        # Split by markers if they exist [de] for German, [cs] for Czech
        parts = text.split("[")
        
        for i, part in enumerate(parts):
            if i == 0 and part:  # First part without language tag
                segments.append({"text": part, "language": None})
                continue
                
            if not part:
                continue
                
            if "]" in part:
                lang_code, content = part.split("]", 1)
                if lang_code.lower() in ["cs", "de"]:
                    segments.append({"text": content, "language": lang_code.lower()})
                else:
                    detected = detect_primary_language(content)
                    segments.append({"text": content, "language": detected})
            else:
                segments.append({"text": part, "language": None})
    else:
        segments = [{"text": text, "language": None}]
    
    # Process each segment
    combined_audio = AudioSegment.silent(duration=0)
    total_time = 0
    
    for segment in segments:
        if not segment["text"].strip():
            continue
            
        # Generate audio for segment
        audio_data, generation_time = generate_speech(
            segment["text"], 
            language_code=segment["language"]
        )
        
        if audio_data:
            # Load audio data with minimal processing
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            combined_audio += audio_segment
            total_time += generation_time
    
    # Save to temporary file with minimal processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        combined_audio.export(temp_file.name, format="mp3", bitrate="128k")  # Reduced bitrate for speed
        return temp_file.name, total_time
    

def normalize_audio_segments(audio_segments, target_language_codes):
    """Normalize volume and speed across language segments"""
    normalized_segments = []
    
    # Calculate target volume from first segment
    if audio_segments:
        target_volume = audio_segments[0].dBFS
    
    for i, (segment, lang_code) in enumerate(zip(audio_segments, target_language_codes)):
        # Normalize volume to consistent level
        volume_normalized = segment.normalize(headroom=0.1)
        
        # Ensure consistent dBFS across segments
        if abs(volume_normalized.dBFS - target_volume) > 3:  # More than 3dB difference
            volume_normalized = volume_normalized + (target_volume - volume_normalized.dBFS)
        
        # Normalize speed (playback rate consistency)
        speed_normalized = normalize_speech_speed(volume_normalized, lang_code)
        
        normalized_segments.append(speed_normalized)
    
    return normalized_segments

def normalize_speech_speed(audio_segment, language_code):
    """Ensure consistent speech speed across languages"""
    # Language-specific speed normalization
    if language_code == "cs":
        # Czech tends to be spoken slightly faster, slow down by 5%
        return audio_segment._spawn(audio_segment.raw_data, overrides={
            "frame_rate": int(audio_segment.frame_rate * 0.95)
        }).set_frame_rate(audio_segment.frame_rate)
    elif language_code == "de":
        # German is usually well-paced, keep normal
        return audio_segment
    
    return audio_segment

# ----------------------------------------------------------------------------------
# END-TO-END PIPELINE - ENHANCED FOR LATENCY OPTIMIZATION
# ----------------------------------------------------------------------------------

async def process_voice_input_enhanced(audio_file):
    """Enhanced voice processing with OpenAI API and auto language detection"""
    pipeline_start_time = time.time()
    
    try:
        # Step 1: Enhanced Audio Preprocessing
        st.session_state.message_queue.put("🎧 Cleaning and enhancing audio...")
        
        # Enhance audio quality
        recorder = AudioRecorder()
        enhanced_audio_file = recorder.enhance_audio_quality(audio_file)
        if enhanced_audio_file and os.path.exists(enhanced_audio_file):
            audio_file = enhanced_audio_file
        
        # Step 2: OpenAI Whisper API Transcription
        st.session_state.message_queue.put("🎯 Transcribing with OpenAI...")
        
        transcription = await asyncio.wait_for(
            transcribe_with_api(audio_file, st.session_state.openai_api_key),
            timeout=30.0
        )
        
        if not transcription or not transcription.get("text"):
            st.session_state.message_queue.put("❌ No speech detected")
            return None, None, 0, 0, 0
        
        # Step 3: Auto Language Distribution Detection
        user_input = transcription["text"].strip()
        st.session_state.message_queue.put(f"📝 Transcribed: {user_input}")
        
        # Detect language distribution from user input for auto mode
        detected_distribution = None
        if st.session_state.response_language == "auto":
            detected_distribution = auto_detect_language_distribution(user_input)
            st.session_state.message_queue.put(f"🔍 Auto-detected: {detected_distribution['cs']}% Czech, {detected_distribution['de']}% German")
        
        # Step 4: Generate Response with OpenAI
        st.session_state.message_queue.put("🤖 Generating intelligent response...")
        
        # Create enhanced system prompt for auto mode
        if st.session_state.response_language == "auto" and detected_distribution:
            cs_percent = detected_distribution["cs"]
            de_percent = detected_distribution["de"]
            system_prompt = (
                f"You are a multilingual AI language tutor. The user spoke {cs_percent}% Czech and {de_percent}% German. "
                f"Respond naturally using the same language distribution: {cs_percent}% Czech and {de_percent}% German. "
                f"Always use language markers [cs] and [de] to indicate language sections. "
                f"Be conversational and educational."
            )
        else:
            # Use existing logic for other modes
            system_prompt = None
        
        llm_result = await generate_llm_response(user_input, system_prompt)
        
        if "error" in llm_result:
            st.session_state.message_queue.put(f"❌ Response generation failed: {llm_result.get('error')}")
            return user_input, None, transcription.get("latency", 0), 0, 0
        
        response_text = llm_result["response"]
        st.session_state.message_queue.put(f"💬 Generated: {response_text}")
        
        # Step 5: High-Quality Voice Synthesis
        st.session_state.message_queue.put("🎵 Generating natural speech...")
        audio_path, tts_latency = process_multilingual_text_seamless(response_text)
        
        # Calculate total latency
        total_latency = time.time() - pipeline_start_time
        st.session_state.performance_metrics["total_latency"].append(total_latency)
        
        # Update conversation history
        st.session_state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": response_text,
            "latency": {
                "stt": transcription.get("latency", 0),
                "llm": llm_result.get("latency", 0),
                "tts": tts_latency,
                "total": total_latency
            }
        })
        
        st.session_state.message_queue.put(f"✅ Complete! Total time: {total_latency:.2f}s")
        
        # Clean up enhanced audio file
        if enhanced_audio_file and enhanced_audio_file != audio_file:
            try:
                os.unlink(enhanced_audio_file)
            except:
                pass
        
        return user_input, audio_path, transcription.get("latency", 0), llm_result.get("latency", 0), tts_latency
        
    except asyncio.TimeoutError:
        st.session_state.message_queue.put("⏰ Processing timed out")
        return None, None, 0, 0, 0
    except Exception as e:
        logger.error(f"Enhanced voice processing error: {str(e)}")
        st.session_state.message_queue.put(f"❌ Error: {str(e)}")
        return None, None, 0, 0, 0

async def process_voice_input_pronunciation_enhanced(audio_file):
    """Enhanced voice processing focusing on pronunciation accuracy"""
    pipeline_start_time = time.time()
    
    try:
        # Step 1: Enhanced Audio Preprocessing with 500% boost
        st.session_state.message_queue.put("🔊 Amplifying audio for pronunciation clarity...")
        
        # Apply additional audio enhancement for pronunciation
        enhanced_audio_file = enhance_audio_for_pronunciation(audio_file)
        if enhanced_audio_file and os.path.exists(enhanced_audio_file):
            audio_file = enhanced_audio_file
        
        # Step 2: Pronunciation-Enhanced Transcription
        st.session_state.message_queue.put("🎯 Analyzing pronunciation patterns...")
        
        transcription = await asyncio.wait_for(
            transcribe_with_api(audio_file, st.session_state.openai_api_key),
            timeout=30.0
        )
        
        if not transcription or not transcription.get("text"):
            st.session_state.message_queue.put("❌ No clear pronunciation detected")
            return None, None, 0, 0, 0
        
        # Step 3: Pronunciation-Based Language Understanding
        user_input = transcription["text"].strip()
        pronunciation_context = extract_pronunciation_context(transcription)
        
        st.session_state.message_queue.put(f"📝 Pronunciation Analysis: {user_input}")
        st.session_state.message_queue.put(f"🎭 Detected Context: {pronunciation_context}")
        
        # Step 4: Generate Response Based on Pronunciation Understanding
        st.session_state.message_queue.put("🤖 Generating pronunciation-aware response...")
        
        # Create pronunciation-enhanced system prompt
        system_prompt = create_pronunciation_aware_prompt(user_input, pronunciation_context)
        
        llm_result = await generate_llm_response(user_input, system_prompt)
        
        if "error" in llm_result:
            st.session_state.message_queue.put(f"❌ Response generation failed: {llm_result.get('error')}")
            return user_input, None, transcription.get("latency", 0), 0, 0
        
        response_text = llm_result["response"]
        st.session_state.message_queue.put(f"💬 Pronunciation-Aware Response: {response_text}")
        
        # Step 5: High-Quality Voice Synthesis
        st.session_state.message_queue.put("🎵 Generating accent-free speech...")
        audio_path, tts_latency = process_multilingual_text_seamless(response_text)
        
        # Calculate total latency
        total_latency = time.time() - pipeline_start_time
        st.session_state.performance_metrics["total_latency"].append(total_latency)
        
        # Update conversation history with pronunciation context
        st.session_state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "pronunciation_context": pronunciation_context,
            "assistant_response": response_text,
            "latency": {
                "stt": transcription.get("latency", 0),
                "llm": llm_result.get("latency", 0),
                "tts": tts_latency,
                "total": total_latency
            }
        })
        
        st.session_state.message_queue.put(f"✅ Pronunciation-Enhanced Processing Complete! ({total_latency:.2f}s)")
        
        # Clean up
        if enhanced_audio_file and enhanced_audio_file != audio_file:
            try:
                os.unlink(enhanced_audio_file)
            except:
                pass
        
        return user_input, audio_path, transcription.get("latency", 0), llm_result.get("latency", 0), tts_latency
        
    except Exception as e:
        logger.error(f"Pronunciation-enhanced processing error: {str(e)}")
        st.session_state.message_queue.put(f"❌ Error: {str(e)}")
        return None, None, 0, 0, 0

def enhance_audio_for_pronunciation(audio_file):
    """Additional audio enhancement specifically for pronunciation clarity"""
    try:
        # Load audio
        audio, sample_rate = sf.read(audio_file)
        
        # Apply multiple enhancement techniques
        # 1. Spectral subtraction for noise reduction
        enhanced_audio = nr.reduce_noise(y=audio, sr=sample_rate)
        
        # 2. Dynamic range compression for consistent volume
        enhanced_audio = np.tanh(enhanced_audio * 2.0)  # Soft limiting
        
        # 3. High-frequency emphasis for consonant clarity
        from scipy import signal
        # Pre-emphasis filter to boost high frequencies (important for consonants)
        pre_emphasis = 0.97
        emphasized_audio = np.append(enhanced_audio[0], enhanced_audio[1:] - pre_emphasis * enhanced_audio[:-1])
        
        # Save enhanced audio
        enhanced_path = tempfile.mktemp(suffix=".wav")
        sf.write(enhanced_path, emphasized_audio, sample_rate)
        
        return enhanced_path
        
    except Exception as e:
        logger.error(f"Pronunciation enhancement error: {str(e)}")
        return audio_file

def extract_pronunciation_context(transcription):
    """Extract pronunciation context for better understanding"""
    try:
        text = transcription.get("text", "")
        language = transcription.get("language", "unknown")
        segments = transcription.get("segments", [])
        
        context = {
            "primary_language": language,
            "confidence_level": "high" if transcription.get("pronunciation_enhanced") else "medium",
            "detected_patterns": [],
            "language_switches": 0
        }
        
        # Analyze pronunciation patterns
        if segments:
            prev_lang = None
            for segment in segments:
                segment_text = segment.get("text", "")
                detected_lang = detect_primary_language(segment_text)
                
                if prev_lang and prev_lang != detected_lang:
                    context["language_switches"] += 1
                
                prev_lang = detected_lang
        
        # Add pronunciation difficulty markers
        czech_difficult = ["ř", "ň", "ť", "ď", "ů", "ě"]
        german_difficult = ["ü", "ä", "ö", "ß", "ch", "sch"]
        
        for char in czech_difficult:
            if char in text.lower():
                context["detected_patterns"].append(f"Czech_{char}")
                
        for pattern in german_difficult:
            if pattern in text.lower():
                context["detected_patterns"].append(f"German_{pattern}")
        
        return context
        
    except Exception as e:
        logger.error(f"Context extraction error: {str(e)}")
        return {"primary_language": "unknown", "confidence_level": "low"}

def create_pronunciation_aware_prompt(user_input, pronunciation_context):
    """Create system prompt that considers pronunciation context"""
    
    primary_lang = pronunciation_context.get("primary_language", "unknown")
    confidence = pronunciation_context.get("confidence_level", "medium")
    patterns = pronunciation_context.get("detected_patterns", [])
    switches = pronunciation_context.get("language_switches", 0)
    
    system_prompt = f"""You are a multilingual AI language tutor specializing in Czech and German pronunciation.

PRONUNCIATION ANALYSIS:
- Primary detected language: {primary_lang}
- Pronunciation confidence: {confidence}
- Detected pronunciation patterns: {', '.join(patterns) if patterns else 'None'}
- Language switches detected: {switches}

The user's input shows pronunciation characteristics that suggest they may have spoken:
{user_input}

RESPONSE INSTRUCTIONS:
1. Respond naturally based on the INTENDED meaning, not just literal text
2. If pronunciation suggests Czech, respond with appropriate Czech content marked [cs]
3. If pronunciation suggests German, respond with appropriate German content marked [de]
4. If mixed languages detected, use both languages appropriately marked
5. Be supportive and educational about pronunciation
6. Correct any obvious pronunciation-based misunderstandings gently

Focus on the communicative intent behind the pronunciation rather than perfect transcription accuracy."""

    return system_prompt
        
async def process_text_input(text):
    """Process text input through the LLM → TTS pipeline with language distribution control"""
    pipeline_start_time = time.time()
    
    # Step 1: Generate response with LLM with language control
    st.session_state.message_queue.put("Generating language tutor response...")
    
    # Set up custom prompt based on language preferences
    response_language = st.session_state.response_language
    language_distribution = st.session_state.language_distribution
    
    # Create prompt with custom instructions
    if response_language == "both":
        cs_percent = language_distribution["cs"]
        de_percent = language_distribution["de"]
        system_prompt = (
            f"You are a multilingual AI language tutor. Respond with approximately {cs_percent}% Czech and {de_percent}% German. "
            f"Always use language markers [cs] and [de] to indicate language."
        )
    elif response_language in ["cs", "de"]:
        system_prompt = f"You are a language tutor. Respond only in {response_language} with [{response_language}] markers."
    else:
        system_prompt = None
    
    # Generate the LLM response with appropriate system prompt
    llm_result = await generate_llm_response(text, system_prompt)
    
    if "error" in llm_result:
        st.session_state.message_queue.put(f"Error generating response: {llm_result.get('error')}")
        return None, llm_result.get("latency", 0), 0
    
    response_text = llm_result["response"]
    st.session_state.message_queue.put(f"Generated response: {response_text}")
    
    # Step 2: Text-to-Speech with accent isolation
    st.session_state.message_queue.put("Generating speech with accent isolation...")
    audio_path, tts_latency = process_multilingual_text_seamless(response_text)
    
    # Calculate total latency
    total_latency = time.time() - pipeline_start_time
    st.session_state.performance_metrics["total_latency"].append(total_latency)
    
    # Update conversation history
    st.session_state.conversation_history.append({
        "timestamp": datetime.now().isoformat(),
        "user_input": text,
        "assistant_response": response_text,
        "latency": {
            "stt": 0,
            "llm": llm_result.get("latency", 0),
            "tts": tts_latency,
            "total": total_latency
        }
    })
    
    st.session_state.message_queue.put(f"Complete pipeline executed in {total_latency:.2f} seconds")
    
    return audio_path, llm_result.get("latency", 0), tts_latency

# ----------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------------

def display_audio(audio_path, autoplay=False):
    """Display audio in Streamlit with improved error handling"""
    if not audio_path:
        logger.error("No audio path provided")
        return None
        
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None
        
    try:
        # Check if file is valid and has content
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            return None
            
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            
            # Use native Streamlit audio component
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
            
            # Return audio bytes for download button
            return audio_bytes
    except Exception as e:
        logger.error(f"Error displaying audio: {str(e)}")
        return None

def calculate_average_latency(latency_list, recent_count=5):
    """Calculate average latency from most recent measurements"""
    if not latency_list:
        return 0
        
    recent = latency_list[-min(recent_count, len(latency_list)):]
    return sum(recent) / len(recent)

def update_status():
    """Update status display from message queue"""
    status_text = ""
    while True:
        try:
            message = st.session_state.message_queue.get_nowait()
            status_text += message + "\n"
            st.session_state.status_area.text_area("Processing Log", value=status_text, height=200)
        except queue.Empty:
            break

def encode_audio_to_base64(audio_path):
    """Encode audio file to base64 for embedding in HTML"""
    if not audio_path or not os.path.exists(audio_path):
        return None
        
    with open(audio_path, "rb") as audio_file:
        encoded = base64.b64encode(audio_file.read()).decode()
        return encoded

# ----------------------------------------------------------------------------------
# STREAMLIT UI - ENHANCED WITH LANGUAGE CONTROL OPTIONS
# ----------------------------------------------------------------------------------



# REPLACE the voice input section in your main() function with this enhanced version

def main():
    """Main application entry point - UPDATED VERSION"""
    # Page configuration - ONLY ONCE!
    st.set_page_config(
        page_title="Multilingual AI Voice Tutor",
        page_icon="🎙️",
        layout="wide"
    )
    
    # Application title
    st.title("🎙️ Multilingual AI Voice Tutor")
    st.subheader("Advanced Czech ↔ German Language Switching with Auto-Processing")
    
    # Status area for progress updates
    if 'status_area' not in st.session_state:
        st.session_state.status_area = st.empty()
    
    # Initialize session state for auto-processing
    if 'audio_processing_queue' not in st.session_state:
        st.session_state.audio_processing_queue = []
    if 'last_processed_timestamp' not in st.session_state:
        st.session_state.last_processed_timestamp = 0
    
    # Sidebar configuration (keep your existing sidebar code here)
    with st.sidebar:
        st.header("Configuration")
        
        # API keys section
        st.subheader("API Keys")
        
        elevenlabs_key = st.text_input(
            "ElevenLabs API Key", 
            value=st.session_state.elevenlabs_api_key,
            type="password",
            help="Required for text-to-speech"
        )
        
        openai_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key,
            type="password",
            help="Required for speech recognition and language understanding"
        )
        
        if st.button("💾 Save API Keys"):
            st.session_state.elevenlabs_api_key = elevenlabs_key
            st.session_state.openai_api_key = openai_key
            st.session_state.api_keys_initialized = True
            st.success("✅ API keys saved successfully!")
        
        # Rest of your sidebar configuration...
        # (Keep all your existing sidebar code - voice settings, language preferences, etc.)
    
    # MAIN INTERACTION AREA - UPDATED
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("🎯 Input")
        
        # Input type selection
        input_type = st.radio(
            "Select Input Method", 
            ["🎤 Voice Input", "📝 Text Input"], 
            horizontal=True
        )
        
        if input_type == "📝 Text Input":
            # Text input section (keep your existing text input code)
            st.subheader("Text Input")
            st.write("Use [cs] to mark Czech text and [de] to mark German text.")
            
            # Demo scenarios (keep your existing demo examples)
            demo_scenarios = {
                "Basic Greetings": (
                    "[de] Guten Tag! Wie geht es Ihnen heute? [cs] Dobrý den! Jak se dnes máte?"
                ),
                "Language Learning": (
                    "[de] Ich lerne jetzt Deutsch und Tschechisch. [cs] Učím se německy a česky."
                ),
                "Custom Input": ""
            }
            
            selected_scenario = st.selectbox("Demo Examples", options=list(demo_scenarios.keys()))
            
            text_input = st.text_area(
                "Edit or enter new text",
                value=demo_scenarios[selected_scenario],
                height=150
            )
            
            if st.button("🔄 Process Text", type="primary"):
                if text_input:
                    with st.spinner("Processing text input..."):
                        audio_path, llm_latency, tts_latency = asyncio.run(process_text_input(text_input))
                        st.session_state.last_text_input = text_input
                        st.session_state.last_audio_output = audio_path
                        total_latency = llm_latency + tts_latency
                        st.success(f"✅ Text processed in {total_latency:.2f} seconds")
                        st.rerun()
        
        else:
            # ENHANCED VOICE INPUT SECTION
            enhanced_voice_input_section()
    
    with col2:
        st.header("📊 Output & Results")
        
        # Check if we have any results to display
        has_text_result = hasattr(st.session_state, 'last_text_input') and st.session_state.last_text_input
        has_conversation = hasattr(st.session_state, 'conversation_history') and st.session_state.conversation_history
        has_audio_result = hasattr(st.session_state, 'last_audio_output') and st.session_state.last_audio_output
        
        if has_text_result or has_conversation or has_audio_result:
            
            # Transcribed/Input text
            if has_text_result:
                st.subheader("📝 Input Text")
                st.text_area(
                    "Your input with language markers",
                    value=st.session_state.last_text_input,
                    height=100,
                    disabled=True,
                    key="display_input_text"
                )
            
            # AI Generated response text
            if has_conversation:
                last_exchange = st.session_state.conversation_history[-1]
                if 'assistant_response' in last_exchange:
                    st.subheader("🤖 AI Tutor Response")
                    st.text_area(
                        "Generated response text",
                        value=last_exchange['assistant_response'],
                        height=120,
                        disabled=True,
                        key="display_response_text"
                    )
            
            # Generated audio
            if has_audio_result:
                st.subheader("🔊 Generated Speech")
                
                # Display audio player
                audio_bytes = display_audio(st.session_state.last_audio_output, autoplay=False)
                
                if audio_bytes:
                    # Action buttons
                    col_a, col_b, col_c = st.columns([1, 1, 1])
                    
                    with col_a:
                        st.download_button(
                            label="💾 Download Audio",
                            data=audio_bytes,
                            file_name="multilingual_tutor_response.mp3",
                            mime="audio/mp3"
                        )
                    
                    with col_b:
                        if st.button("▶️ Play Audio"):
                            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    
                    with col_c:
                        if st.button("🗑️ Clear Results"):
                            # Clear all results
                            for key in ['last_text_input', 'last_audio_output']:
                                if hasattr(st.session_state, key):
                                    delattr(st.session_state, key)
                            st.rerun()
            
            # Performance metrics for last processing
            if has_conversation:
                last_exchange = st.session_state.conversation_history[-1]
                latency = last_exchange.get('latency', {})
                
                st.subheader("⚡ Performance Metrics")
                
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                
                with col_p1:
                    st.metric("STT", f"{latency.get('stt', 0):.2f}s")
                with col_p2:
                    st.metric("LLM", f"{latency.get('llm', 0):.2f}s")
                with col_p3:
                    st.metric("TTS", f"{latency.get('tts', 0):.2f}s")
                with col_p4:
                    st.metric("Total", f"{latency.get('total', 0):.2f}s")
        
        else:
            # No results yet - show instructions
            st.info("""
            **🎯 Ready for Voice or Text Input!**
            
            **Voice Mode:**
            1. Click START RECORDING
            2. Speak in Czech or German  
            3. Click STOP (auto-processing begins)
            4. Listen to your recording and AI response
            
            **Text Mode:**
            - Use [cs] and [de] markers for language switching
            - Click Process Text for AI response
            
            **Results will appear here after processing!**
            """)
    
    # Conversation History Section
    if st.session_state.conversation_history:
        st.markdown("---")
        st.header("📚 Conversation History")
        
        # Show recent conversations
        recent_conversations = st.session_state.conversation_history[-3:]  # Last 3
        
        for i, exchange in enumerate(reversed(recent_conversations)):
            with st.expander(f"💬 Exchange {len(recent_conversations)-i} - {exchange.get('timestamp', 'Unknown')[:19]}"):
                
                # User input
                st.markdown("**👤 You said:**")
                st.code(exchange.get('user_input', 'No input'), language="text")
                
                # AI response
                st.markdown("**🤖 AI Tutor:**")
                st.code(exchange.get('assistant_response', 'No response'), language="text")
                
                # Performance metrics
                latency = exchange.get('latency', {})
                st.caption(f"⏱️ STT: {latency.get('stt', 0):.2f}s | LLM: {latency.get('llm', 0):.2f}s | TTS: {latency.get('tts', 0):.2f}s | Total: {latency.get('total', 0):.2f}s")
    
    # Status and Debug Section
    st.markdown("---")
    st.header("🔧 System Status")
    
    # API Keys Status
    col_s1, col_s2 = st.columns([1, 1])
    
    with col_s1:
        st.success("✅ ElevenLabs API" if st.session_state.elevenlabs_api_key else "❌ ElevenLabs API")
    with col_s2:
        st.success("✅ OpenAI API" if st.session_state.openai_api_key else "❌ OpenAI API")
    
    # Overall system status
    if st.session_state.elevenlabs_api_key and st.session_state.openai_api_key:
        st.success("🚀 **System Ready** - Voice processing enabled!")
    else:
        st.warning("⚠️ **Setup Required** - Please configure API keys in sidebar")
    
    # Processing status area
    st.session_state.status_area = st.empty()
    update_status()
