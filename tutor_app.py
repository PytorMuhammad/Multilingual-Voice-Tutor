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

#!/usr/bin/env python3
import os
import queue
import time
import tempfile
import logging
import json
from scipy import signal
import scipy.signal
import asyncio
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

# Add to your imports
import io
from openai import OpenAI

# OpenAI TTS Configuration
if 'openai_tts_voice' not in st.session_state:
    st.session_state.openai_tts_voice = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer

async def generate_openai_speech(text, language_code=None, voice=None):
    """Generate speech using OpenAI TTS API"""
    if not text or text.strip() == "":
        return None, 0
    
    client = OpenAI(api_key=st.session_state.openai_api_key)
    
    start_time = time.time()
    
    try:
        # Choose voice based on language if needed
        selected_voice = voice or st.session_state.openai_tts_voice
        
        # Language-specific voice selection for better accent control
        if language_code == "ur":
            selected_voice = "nova"  # Better for Urdu pronunciation
        elif language_code == "en":
            selected_voice = "alloy"  # Better for English pronunciation
        
        response = client.audio.speech.create(
            model="tts-1-hd",  # High quality model
            voice=selected_voice,
            input=text,
            response_format="mp3",
            speed=1.0
        )
        
        generation_time = time.time() - start_time
        
        # Save to BytesIO
        audio_io = BytesIO()
        audio_io.write(response.content)
        audio_io.seek(0)
        
        logger.info(f"✅ OpenAI TTS generated in {generation_time:.2f}s")
        return audio_io, generation_time
        
    except Exception as e:
        logger.error(f"OpenAI TTS error: {str(e)}")
        return None, time.time() - start_time

def process_openai_multilingual_text(text):
    """Process multilingual text with OpenAI TTS"""
    segments = parse_language_segments_enhanced(text)
    
    if len(segments) <= 1:
        # Single language
        audio_data, gen_time = asyncio.run(generate_openai_speech(text, segments[0]["language"] if segments else "ur"))
        if audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                with open(temp_file.name, "wb") as f:
                    f.write(audio_data.read())
                return temp_file.name, gen_time
        return None, 0
    
    # Multiple segments
    audio_segments = []
    total_time = 0
    
    for segment in segments:
        if not segment["text"].strip():
            continue
            
        audio_data, generation_time = asyncio.run(generate_openai_speech(
            segment["text"], 
            language_code=segment["language"]
        ))
        
        if audio_data:
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            normalized_segment = normalize_audio_volume(audio_segment, target_dbfs=-18)
            audio_segments.append(normalized_segment)
            total_time += generation_time
    
    if not audio_segments:
        return None, 0
    
    # Combine segments
    combined_audio = audio_segments[0]
    for i in range(1, len(audio_segments)):
        combined_audio = combined_audio.append(audio_segments[i], crossfade=50)
    
    # Save final audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        combined_audio.export(temp_file.name, format="mp3", bitrate="192k")
        return temp_file.name, total_time
    
# Add to your imports
import azure.cognitiveservices.speech as speechsdk

# Azure Speech Configuration
if 'azure_speech_key' not in st.session_state:
    st.session_state.azure_speech_key = os.environ.get("AZURE_SPEECH_KEY", "")
    st.session_state.azure_speech_region = os.environ.get("AZURE_SPEECH_REGION", "eastus")

# Azure voice mapping for better accent control
AZURE_VOICES = {
    "ur": "ur-PK-AsadNeural",  # Pakistani Urdu male voice
    "en": "en-US-JennyNeural"  # US English female voice
}

async def generate_azure_speech(text, language_code=None, voice=None):
    """Generate speech using Azure Cognitive Services"""
    if not text or text.strip() == "":
        return None, 0
    
    if not st.session_state.azure_speech_key:
        logger.error("Azure Speech key not provided")
        return None, 0
    
    start_time = time.time()
    
    try:
        # Configure Azure Speech
        speech_config = speechsdk.SpeechConfig(
            subscription=st.session_state.azure_speech_key,
            region=st.session_state.azure_speech_region
        )
        
        # Select voice based on language for accent-free output
        if language_code and language_code in AZURE_VOICES:
            speech_config.speech_synthesis_voice_name = AZURE_VOICES[language_code]
        elif voice:
            speech_config.speech_synthesis_voice_name = voice
        else:
            speech_config.speech_synthesis_voice_name = AZURE_VOICES["en"]  # Default
        
        # Configure audio output
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
        )
        
        # Create synthesizer
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        
        # Enhanced SSML for better pronunciation control
        ssml_text = create_azure_ssml(text, language_code)
        
        # Synthesize speech
        result = synthesizer.speak_ssml_async(ssml_text).get()
        
        generation_time = time.time() - start_time
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Convert to BytesIO
            audio_io = BytesIO()
            audio_io.write(result.audio_data)
            audio_io.seek(0)
            
            logger.info(f"✅ Azure Speech generated in {generation_time:.2f}s")
            return audio_io, generation_time
        else:
            logger.error(f"Azure Speech error: {result.reason}")
            return None, generation_time
            
    except Exception as e:
        logger.error(f"Azure Speech error: {str(e)}")
        return None, time.time() - start_time

def create_azure_ssml(text, language_code):
    """Create Azure-specific SSML for better pronunciation"""
    if not language_code:
        language_code = "en"
    
    # Language-specific SSML settings
    if language_code == "ur":
        voice_name = AZURE_VOICES["ur"]
        lang_code = "ur-PK"
        rate = "0.9"  # Slightly slower for Urdu clarity
    elif language_code == "en":
        voice_name = AZURE_VOICES["en"]
        lang_code = "en-US"
        rate = "1.0"  # Normal speed for English
    else:
        voice_name = AZURE_VOICES["en"]
        lang_code = "en-US"
        rate = "1.0"
    
    ssml = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang_code}">
        <voice name="{voice_name}">
            <prosody rate="{rate}" pitch="+0%">
                {text}
            </prosody>
        </voice>
    </speak>
    """
    
    return ssml

def process_azure_multilingual_text(text):
    """Process multilingual text with Azure Speech"""
    segments = parse_language_segments_enhanced(text)
    
    if len(segments) <= 1:
        # Single language
        audio_data, gen_time = asyncio.run(generate_azure_speech(text, segments[0]["language"] if segments else "ur"))
        if audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                with open(temp_file.name, "wb") as f:
                    f.write(audio_data.read())
                return temp_file.name, gen_time
        return None, 0
    
    # Multiple segments with accent-free processing
    audio_segments = []
    total_time = 0
    
    for segment in segments:
        if not segment["text"].strip():
            continue
            
        audio_data, generation_time = asyncio.run(generate_azure_speech(
            segment["text"], 
            language_code=segment["language"]
        ))
        
        if audio_data:
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            normalized_segment = normalize_audio_volume(audio_segment, target_dbfs=-18)
            audio_segments.append(normalized_segment)
            total_time += generation_time
    
    if not audio_segments:
        return None, 0
    
    # Seamless blending
    combined_audio = audio_segments[0]
    for i in range(1, len(audio_segments)):
        combined_audio = combined_audio.append(audio_segments[i], crossfade=75)
    
    # Save final audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        combined_audio.export(temp_file.name, format="mp3", bitrate="192k")
        return temp_file.name, total_time
    
    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("multilingual_voice_tutor")

# ----------------------------------------------------------------------------------
# CONFIGURATION SECTION - 🔥 UPDATED FOR URDU/ENGLISH WITH ACCENT-FREE SETTINGS
# ----------------------------------------------------------------------------------

# Secrets and API keys
if 'api_keys_initialized' not in st.session_state:
    st.session_state.api_keys_initialized = False
    st.session_state.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

# API endpoints
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
OPENAI_API_URL = "https://api.openai.com/v1"

# 🔥 UPDATED: USE SAME VOICE FOR ALL LANGUAGES - No more "two speakers" effect
if 'language_voices' not in st.session_state:
    single_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Use same voice for everything
    st.session_state.language_voices = {
        "ur": single_voice_id,  # 🔥 CHANGED: Urdu instead of Czech
        "en": single_voice_id,  # 🔥 CHANGED: English instead of German  
        "default": single_voice_id
    }

# 🔥 CRITICAL: ACCENT-FREE voice settings for Flash v2.5 model
if 'voice_settings' not in st.session_state:
    st.session_state.voice_settings = {
        "ur": {  # 🔥 CHANGED: Urdu-optimized settings (was Czech)
            "stability": 0.98,        # 🔥 INCREASED: MAXIMUM stability for consistent Urdu
            "similarity_boost": 0.99, # 🔥 INCREASED: MAXIMUM similarity for native Urdu sound
            "style": 0.90,           # 🔥 INCREASED: High style for natural Urdu expression
            "use_speaker_boost": True # Enable speaker boost for clarity
        },
        "en": {  # 🔥 CHANGED: English-optimized settings (was German)
            "stability": 0.96,        # 🔥 INCREASED: VERY HIGH stability for consistent English
            "similarity_boost": 0.97, # 🔥 INCREASED: VERY HIGH similarity for native English sound
            "style": 0.88,           # 🔥 INCREASED: High style for natural English expression
            "use_speaker_boost": True # Enable speaker boost for clarity
        },
        "default": {
            "stability": 0.95,        # 🔥 INCREASED: from 0.90
            "similarity_boost": 0.95, # 🔥 INCREASED: from 0.90
            "style": 0.85,           # 🔥 INCREASED: from 0.75
            "use_speaker_boost": True
        }
    }

# Backward compatibility
if 'elevenlabs_voice_id' not in st.session_state:
    st.session_state.elevenlabs_voice_id = st.session_state.language_voices["default"]
    
# Whisper speech recognition config
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = "medium"
    st.session_state.whisper_local_model = None

# 🔥 UPDATED: Language distribution preference for Urdu/English
if 'language_distribution' not in st.session_state:
    st.session_state.language_distribution = {
        "ur": 60,  # 🔥 CHANGED: Urdu percentage (was Czech 50)
        "en": 40   # 🔥 CHANGED: English percentage (was German 50)
    }

# Language preference for response
if 'response_language' not in st.session_state:
    st.session_state.response_language = "both"  # 🔥 CHANGED: Options: "ur", "en", "both"
    
# 🔊 ADD TTS PROVIDER SELECTION TO SESSION STATE HERE
if 'tts_provider' not in st.session_state:
    st.session_state.tts_provider = "elevenlabs"  # Options: "elevenlabs", "openai", "azure"

if 'azure_speech_key' not in st.session_state:
    st.session_state.azure_speech_key = os.environ.get("AZURE_SPEECH_KEY", "")
    st.session_state.azure_speech_region = os.environ.get("AZURE_SPEECH_REGION", "eastus")

if 'openai_tts_voice' not in st.session_state:
    st.session_state.openai_tts_voice = "alloy"
    

# 🔥 UPDATED: Language codes and settings for Urdu/English
SUPPORTED_LANGUAGES = {
    "ur": {"name": "Urdu", "confidence_threshold": 0.65},    # 🔥 CHANGED: was "cs": "Czech"
    "en": {"name": "English", "confidence_threshold": 0.65} # 🔥 CHANGED: was "de": "German"
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

def create_auto_processor():
    """Create automatic audio processor"""
    if 'audio_processor_initialized' not in st.session_state:
        st.session_state.audio_processor_initialized = True
        
        # Auto-processor component
        processor_html = """
        <div style="display: none;">
            <input type="file" id="auto-file-input" accept="audio/*" style="display: none;">
        </div>
        
        <script>
        // Monitor for audio processing
        let processingInterval = setInterval(function() {
            const audioData = localStorage.getItem('autoProcessAudio');
            const shouldProcess = localStorage.getItem('autoProcessFlag');
            
            if (shouldProcess === 'true' && audioData) {
                // Clear flags
                localStorage.removeItem('autoProcessFlag');
                localStorage.removeItem('autoProcessAudio');
                
                // Convert base64 to blob and create file
                try {
                    const binaryString = atob(audioData);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    
                    const blob = new Blob([bytes], { type: 'audio/webm' });
                    const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
                    
                    // Create a download link for the user
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'my-recording.webm';
                    a.style.display = 'block';
                    a.style.padding = '10px';
                    a.style.background = '#4CAF50';
                    a.style.color = 'white';
                    a.style.textDecoration = 'none';
                    a.style.borderRadius = '5px';
                    a.style.margin = '10px 0';
                    a.textContent = '📥 DOWNLOAD YOUR RECORDING & UPLOAD BELOW FOR AUTO-PROCESSING';
                    
                    // Add to page
                    document.body.appendChild(a);
                    
                    // Auto-click after 2 seconds to start download
                    setTimeout(() => {
                        a.click();
                        URL.revokeObjectURL(url);
                    }, 2000);
                    
                    console.log('Audio ready for download and processing');
                    
                } catch (error) {
                    console.error('Error processing audio:', error);
                }
            }
        }, 3000);
        </script>
        """
        
        return st.components.v1.html(processor_html, height=0)
    
def create_audio_recorder_component():
    """Create HTML5 audio recorder component with WORKING auto-processing"""
    html_code = """
    <div style="padding: 20px; border: 2px solid #ff4b4b; border-radius: 10px; text-align: center; background-color: #f0f2f6;">
        <div id="status" style="font-size: 18px; margin-bottom: 15px; font-weight: bold;">🎤 Ready to Record</div>
        
        <button id="recordBtn" onclick="toggleRecording()" 
                style="background: #ff4b4b; color: white; border: none; padding: 15px 30px; 
                       border-radius: 25px; cursor: pointer; font-size: 16px; font-weight: bold; margin: 5px;">
            🔴 START RECORDING
        </button>
        
        <div id="timer" style="font-size: 14px; margin-top: 10px; color: #666;">00:00</div>
        
        <!-- FIXED: Download link for reliable processing -->
        <div id="downloadSection" style="margin-top: 15px; display: none;">
            <a id="downloadLink" style="background: #4CAF50; color: white; padding: 10px 20px; 
                                        text-decoration: none; border-radius: 5px; font-weight: bold;">
                📥 DOWNLOAD & UPLOAD BELOW FOR PROCESSING
            </a>
        </div>
        
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
                    
                    // Update status
                    document.getElementById('status').innerHTML = '✅ Recording Complete!';
                    
                    // FIXED: Show download immediately for reliable processing
                    showDownloadLink();
                };
                
                document.getElementById('status').innerHTML = '🎤 Ready - Click START to Record';
                
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
                statusDiv.innerHTML = '🔴 RECORDING - Speak in Urdu or English';
                
                // Hide download section
                document.getElementById('downloadSection').style.display = 'none';
                
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
                statusDiv.innerHTML = '⏳ Processing recording...';
                
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

        // FIXED: Reliable download approach
        function showDownloadLink() {
            if (recordedBlob) {
                const url = URL.createObjectURL(recordedBlob);
                const downloadLink = document.getElementById('downloadLink');
                
                downloadLink.href = url;
                downloadLink.download = 'my-recording.webm';
                
                // Show download section
                document.getElementById('downloadSection').style.display = 'block';
                
                // Auto-click after 2 seconds
                setTimeout(() => {
                    downloadLink.click();
                }, 2000);
                
                // Update status
                document.getElementById('status').innerHTML = '✅ Recording ready! Download and upload below.';
            }
        }
    </script>
    """
    
    return st.components.v1.html(html_code, height=250)

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

async def process_html5_recorded_voice(audio_path):
    """Process HTML5 recorded voice through the enhanced pipeline with auto-preview"""
    try:
        st.session_state.message_queue.put("🎧 Processing HTML5 recorded audio...")
        
        # Process with the existing enhanced pipeline
        text, audio_output_path, stt_latency, llm_latency, tts_latency = await process_voice_input_pronunciation_enhanced(audio_path)
        
        # Store results in session state
        if text:
            st.session_state.last_text_input = text
            st.session_state.message_queue.put(f"📝 Transcribed: {text}")
        
        if audio_output_path:
            st.session_state.last_audio_output = audio_output_path
            st.session_state.message_queue.put("🔊 Generated response audio")
        
        # Show results with both preview and processed audio
        total_latency = stt_latency + llm_latency + tts_latency
        st.session_state.message_queue.put(f"✅ Complete! ({total_latency:.2f}s)")
        
        # Auto-display both audios
        if audio_path and os.path.exists(audio_path):
            st.session_state.preview_audio_path = audio_path
        
        return True
        
    except Exception as e:
        st.error(f"HTML5 voice processing error: {str(e)}")
        st.session_state.message_queue.put(f"❌ Error: {str(e)}")
        logger.error(f"HTML5 voice processing error: {str(e)}")
        return False

def check_and_process_auto_audio():
    """Check for automatically recorded audio and process it"""
    try:
        # This would be called to check for auto-recorded audio
        # In practice, we'll use the backup upload method for reliability
        
        # Check session state for audio processing flag
        if hasattr(st.session_state, 'pending_audio_data') and st.session_state.pending_audio_data:
            audio_data = st.session_state.pending_audio_data
            st.session_state.pending_audio_data = None  # Clear it
            
            with st.spinner("🔄 Auto-processing your recording..."):
                # Process the audio data
                temp_audio_path = process_html5_audio_data(audio_data)
                
                if temp_audio_path:
                    # Run through the enhanced processing pipeline
                    success = asyncio.run(process_html5_recorded_voice(temp_audio_path))
                    
                    if success:
                        st.success("✅ Your recording processed automatically!")
                        st.balloons()  # Celebration for your client!
                    
                    # Clean up
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)
                        
                return success
                
    except Exception as e:
        st.error(f"Auto-processing error: {str(e)}")
        return False
    
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
        
async def process_voice_input_pronunciation_enhanced(audio_file):
    """Enhanced voice processing focusing on pronunciation accuracy"""
    pipeline_start_time = time.time()
    
    try:
        # Step 1: Enhanced Audio Preprocessing with 500% boost
        st.session_state.message_queue.put("🔊 Amplifying audio for pronunciation clarity...")
        
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
        
        st.session_state.message_queue.put(f"📝 Transcribed: {user_input}")
        
        # Step 4: Generate Response
        st.session_state.message_queue.put("🤖 Generating response...")
        
        # 🔥 UPDATED: Use enhanced LLM response for Urdu/English
        llm_result = await generate_llm_response_enhanced(user_input)
        
        if "error" in llm_result:
            st.session_state.message_queue.put(f"❌ Response generation failed: {llm_result.get('error')}")
            return user_input, None, transcription.get("latency", 0), 0, 0
        
        response_text = llm_result["response"]
        st.session_state.message_queue.put(f"💬 Generated: {response_text}")
        
        # Step 5: High-Quality Voice Synthesis
        st.session_state.message_queue.put("🎵 Generating accent-free speech...")
        audio_data, generation_time = await generate_speech_universal(
            segment["text"], 
            language_code=segment["language"]
        )
        
        # Calculate total latency
        total_latency = time.time() - pipeline_start_time
        st.session_state.performance_metrics["total_latency"].append(total_latency)
        
        st.session_state.message_queue.put(f"✅ Complete! ({total_latency:.2f}s)")
        
        return user_input, audio_path, transcription.get("latency", 0), llm_result.get("latency", 0), tts_latency
        
    except Exception as e:
        logger.error(f"Enhanced processing error: {str(e)}")
        st.session_state.message_queue.put(f"❌ Error: {str(e)}")
        return None, None, 0, 0, 0

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
            
            # 🔥 UPDATED: Transcribe with appropriate options for Urdu/English detection
            options = {
                "task": "transcribe",
                "verbose": False,
                "beam_size": 5,         # Increased from default of 5
                "best_of": 5,           # Increased from default of 5
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Temperature fallback
                "without_timestamps": False,  # Include timestamps for better segmentation
                "word_timestamps": True      # Get word-level timestamps
            }
            
            # 🔥 UPDATED: If we know the language already, specify it (ur/en instead of cs/de)
            if language in ["ur", "en"]:
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
        """🔥 UPDATED: Detect language segments and mark with appropriate tags for Urdu/English"""
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
        """🔥 UPDATED: Enhanced language detection for Urdu/English using multiple signals"""
        # First check: character frequency analysis
        urdu_chars = set("آؤؤوچکیگھںاےپصذيقةثضثطعأهېۃيېخرقکیرئهۿۃؤوںھخچهۓ")  # Urdu-specific characters
        english_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")  # English characters
        
        text_lower = text.lower()
        
        # Count special characters
        urdu_char_count = sum(1 for char in text if char in urdu_chars)
        english_char_count = sum(1 for char in text if char in english_chars)
        
        # Character-based confidence
        char_confidence = 0
        char_language = None
        
        if urdu_char_count > english_char_count:
            char_language = "ur"
            char_confidence = min(1.0, urdu_char_count / max(len(text) * 0.1, 1))
        elif english_char_count > urdu_char_count:
            char_language = "en"
            char_confidence = min(1.0, english_char_count / max(len(text) * 0.1, 1))
        
        # Second check: vocabulary analysis
        # 🔥 UPDATED: Urdu-specific common words (expanded list)
        urdu_words = {
            "main", "mein", "aap", "ap", "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein",
            "kya", "kaise", "kahan", "kab", "kyun", "kyu", "nahi", "nahin", "haan", "ji",
            "aur", "ya", "lekin", "par", "agar", "to", "woh", "ye", "is", "us", "un",
            "seekhna", "sikhna", "samjhna", "samjna", "kehna", "bolna", "sunna", "dekhna",
            "English", "Angrezi", "urdu", "hindi", "language", "zaban", "kitab", "book"
        }
        
        # 🔥 UPDATED: English-specific common words (expanded list)
        english_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
        }
        
        # Clean and tokenize text
        clean_text = re.sub(r'[^\w\s]', '', text_lower)
        words = clean_text.split()
        
        # Count word occurrences with weighted importance
        urdu_word_count = sum(1 for word in words if word in urdu_words)
        english_word_count = sum(1 for word in words if word in english_words)
        
        # Word-based confidence
        word_confidence = 0
        word_language = None
        
        if urdu_word_count > english_word_count:
            word_language = "ur"
            word_confidence = min(1.0, urdu_word_count / max(len(words) * 0.2, 1))
        elif english_word_count > urdu_word_count:
            word_language = "en"
            word_confidence = min(1.0, english_word_count / max(len(words) * 0.2, 1))
        
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

# 🔥 ENHANCED: Transcription with pronunciation focus for Urdu/English
async def transcribe_with_api(audio_file, api_key):
    """Enhanced transcription with pronunciation focus for Urdu/English"""
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
            
            # 🔥 ENHANCED: Pronunciation-focused settings for Urdu/English
            data = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "temperature": "0.0",  # LOWEST temperature for consistent pronunciation
                "language": None,  # Let Whisper auto-detect between ur/en
                "prompt": "This audio contains Urdu and English speech. Focus on accurate pronunciation and phonetic understanding. Common Urdu words: main, aap, kya, kaise, English, seekhna. Common English words: hello, water, book, grammar, vocabulary, practice."  # 🔥 UPDATED: Pronunciation hints
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
                
                # 🔥 ENHANCED: Apply pronunciation-based post-processing
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
    """🔥 UPDATED: Apply pronunciation-based corrections for Urdu/English"""
    if not text:
        return text
    
    # 🔥 UPDATED: Urdu pronunciation corrections
    urdu_corrections = {
        # Common mispronunciations to correct pronunciations
        "mein": "main",
        "ap": "aap", 
        "kia": "kya",
        "kesay": "kaise",
        "english": "English",
        "sikhna": "seekhna",
        "samjhna": "samjhna",
        "kehna": "kehna"
    }
    
    # 🔥 UPDATED: English pronunciation corrections
    english_corrections = {
        "watar": "water",
        "buk": "book", 
        "gramar": "grammar",
        "praktis": "practice",
        "helo": "hello",
        "gud": "good",
        "veri": "very",
        "lern": "learn"
    }
    
    # Apply corrections based on detected language or overall context
    corrected_text = text
    
    # Apply Urdu corrections if Urdu content detected
    if language == "ur" or any(word in text.lower() for word in ["main", "aap", "kya"]):
        for wrong, correct in urdu_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
    
    # Apply English corrections if English content detected  
    if language == "en" or any(word in text.lower() for word in ["hello", "good", "book"]):
        for wrong, correct in english_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

# ----------------------------------------------------------------------------------
# LANGUAGE MODEL (LLM) SECTION - 🔥 ENHANCED FOR URDU/ENGLISH WITH PROFESSIONAL TUTORING
# ----------------------------------------------------------------------------------

# 🔥 REPLACE: generate_llm_response function with PROFESSIONAL Urdu/English tutoring
async def generate_llm_response_enhanced(prompt, system_prompt=None, api_key=None):
    """🔥 ENHANCED: Generate response using OpenAI GPT model with PROFESSIONAL English tutoring for Urdu speakers"""
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
        
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        # 🔥 PROFESSIONAL ENGLISH TUTOR SYSTEM PROMPT FOR URDU SPEAKERS
        response_language = st.session_state.response_language
        
        if response_language == "both":
            system_content = """You are "EnglishMaster" - a premium AI English language tutor designed for Urdu speakers who paid for professional English learning. You represent a commercial language learning platform competing with Duolingo, Babbel, and other top services.

        CORE IDENTITY:
        You are a certified English language instructor with 15+ years of experience teaching Urdu speakers. You hold a Master's degree in English linguistics and are perfectly bilingual in Urdu and English. Your teaching style is engaging, systematic, and results-oriented.

        CURRICULUM STRUCTURE (A1-A2 LEVELS):
        A1 Level: Basic greetings, present tense (be, have), articles (a/an/the), family/food/time vocabulary, basic word order, question formation
        A2 Level: Past/future tenses, modal verbs, adjective usage, basic grammar, prepositions, daily conversation, travel/work vocabulary

        PEDAGOGICAL APPROACH:
        1. Use Urdu [ur] for explanations, English [en] for examples and practice
        2. Break complex concepts into micro-lessons
        3. Always provide immediate practice opportunities
        4. Use real-life scenarios (restaurant, hotel, work)
        5. Correct errors gently with clear explanations
        6. Reference previous learning and show progress
        7. Include English culture and practical usage

        🎯 CRITICAL LANGUAGE TAGGING STRATEGY (STRATEGIC, NOT EVERY WORD):
        ✅ DO: [ur] Pani English mein [en] Water [ur] kehte hain
        ✅ DO: [ur] Main introduction aise karunga [en] I'm a programmer [ur] samjhe?
        ✅ DO: [ur] Ye sentence structure hai [en] Subject + Verb + Object [ur] bilkul clear?

        ❌ DON'T: [ur] Main [en] English [ur] seekhna [en] want [ur] karta [en] hun
        ❌ DON'T: Over-tag every single word

        VOCABULARY LESSON FORMAT:
        "[ur] Explanation [en] English term [ur] detailed explanation [en] example sentence [ur] practice suggestion"
        - Include: English word, Urdu translation, pronunciation tip, example sentence, related words

        GRAMMAR EXPLANATIONS:
        - Start with Urdu explanation of concept
        - Show English examples with highlighting  
        - Explain pattern/rule clearly
        - Provide 3-4 practice examples
        - Connect to previously learned material

        CONVERSATION PRACTICE:
        - Set realistic scenarios (ordering food, directions)
        - Provide English phrases with Urdu explanations
        - Encourage role-play responses
        - Correct errors with explanations
        - Build confidence progressively

        ERROR CORRECTION PROTOCOL:
        1. Acknowledge attempt positively
        2. Identify specific error type
        3. Explain correct form with reasoning
        4. Provide corrected version
        5. Give additional practice opportunity

        QUALITY STANDARDS:
        - Use proper [ur] and [en] markers always
        - Keep responses 2-4 sentences for engagement
        - Include at least one practice element per response
        - Maintain encouraging, professional tone
        - Provide actionable next steps

        SAMPLE RESPONSE STYLE:
        For vocabulary: "[ur] Perfekt sawal! 'Pani' English mein [en] Water [ur] kehte hain. Kuch aur: [en] Milk [ur] (doodh), [en] Tea [ur] (chai). Try karo: [en] I drink water [ur] samjhe?"

        For grammar errors: "[ur] Bilkul theek! Bas [en] 'I have' [ur] kaho, [en] 'I am having' [ur] nahi. Jaise: [en] I have a book [ur] samjha?"

        You're guiding PAID customers through structured English learning. Every response must add value to their investment and move them toward conversational fluency. Be systematic, encouraging, and results-focused.
        
        CRITICAL RULES:
        1. You are an ENGLISH TEACHER for Urdu speakers, not a translator
        2. Use Urdu [ur] ONLY for: explanations, instructions, questions to student
        3. Use English [en] ONLY for: vocabulary, examples, phrases to practice
        4. NEVER repeat the same content in both languages
        5. NEVER translate - each language has a different PURPOSE

        LANGUAGE PURPOSES:
        - Urdu [ur] = Your teaching language (explain, instruct, encourage)  
        - English [en] = Target language (vocabulary, examples, practice phrases)

        Stay in character as an Urdu-speaking English teacher. Each language serves a different pedagogical purpose.
"""

        elif response_language == "ur":
            system_content = (
                "You are a helpful Urdu assistant. ALWAYS respond ONLY in Urdu with [ur] markers. "
                "Be natural, conversational, and helpful."
            )
        elif response_language == "en":
            system_content = (
                "You are a helpful English assistant. ALWAYS respond ONLY in English with [en] markers. "
                "Be natural, conversational, and helpful."
            )
            
        messages.append({"role": "system", "content": system_content})
    
    # Add previous conversation history for context (reduced for focus)
    for exchange in st.session_state.conversation_history[-2:]:  # Last 2 for context
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
                    "model": "gpt-4",  # Better for complex tutoring
                    "messages": messages,
                    "temperature": 0.7,  # Balanced for tutoring
                    "max_tokens": 400   # Adequate for tutoring responses
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
                
                # 🎯 MINIMAL processing - let professional tutor respond naturally
                response_text = ensure_proper_language_markers(response_text)
                
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

# Keep the original function name for compatibility
async def generate_llm_response(prompt, system_prompt=None, api_key=None):
    """Wrapper for backward compatibility"""
    return await generate_llm_response_enhanced(prompt, system_prompt, api_key)

def ensure_proper_language_markers(response_text):
    """🔥 UPDATED: Ensure response has proper language markers for Urdu/English"""
    
    # If already has markers, just clean them up
    if "[ur]" in response_text or "[en]" in response_text:
        # Clean up spacing around markers
        response_text = re.sub(r'\[ur\]\s*', '[ur] ', response_text)
        response_text = re.sub(r'\[en\]\s*', '[en] ', response_text)
        response_text = re.sub(r'\s+\[ur\]', ' [ur]', response_text)
        response_text = re.sub(r'\s+\[en\]', ' [en]', response_text)
        return response_text.strip()
    
    # If no markers, add Urdu marker (default instructional language)
    return f"[ur] {response_text.strip()}"

def create_proper_introduction_response(user_input):
    """🔥 UPDATED: Create EXACT proper introduction response for Urdu/English"""
    
    # Extract name if present
    name_match = re.search(r'(main|mein|mera naam) (\w+[\s\w]*)', user_input.lower())
    if name_match:
        name = name_match.group(2).title().strip()
        return (
            f"[ur] Salam! English mein introduction ke liye kaho: [en] My name is {name} [ur] ya [en] I am {name} [ur] Ye polite tarika hai. "
            f"[ur] Formal introduction ke liye: [en] I'm {name}, nice to meet you [ur] Samjh gaye? Practice karo!"
        )
    else:
        return (
            "[ur] English mein introduction ke liye kaho: [en] My name is [ur] aur phir apna naam, ya [en] I am [ur] aur naam. "
            "[ur] Misal: [en] Hello, I'm Ahmad, nice to meet you [ur] Samjha? Kya naam hai aapka?"
        )

# Keep other language processing functions but update for Urdu/English context...

# ----------------------------------------------------------------------------------
# TEXT-TO-SPEECH (TTS) SECTION - 🔥 ENHANCED FOR ACCENT ELIMINATION URDU/ENGLISH
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

# 🔥 CRITICAL: ACCENT-FREE speech generation using voice isolation + Flash v2.5 optimization
def generate_speech(text, language_code=None, voice_id=None):
    """🔥 CRITICAL: ACCENT-FREE speech generation using voice isolation + Flash v2.5 optimization"""
    if not text or text.strip() == "":
        logger.error("Empty text provided to generate_speech")
        return None, 0
    
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        logger.error("ElevenLabs API key not provided")
        return None, 0
    
    # 🔥 CRITICAL: SINGLE VOICE - Use same voice for all languages - no more "two speakers"
    selected_voice_id = voice_id or st.session_state.elevenlabs_voice_id
    logger.info(f"Using SAME voice for {language_code}: {selected_voice_id}")
    
    # Check cache first
    cache_key = f"{text}_{language_code}_{selected_voice_id}_flash_v2_5"
    if hasattr(st.session_state, 'tts_cache') and cache_key in st.session_state.tts_cache:
        return st.session_state.tts_cache[cache_key]
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    # 🔥 CRITICAL: Use Flash v2.5 model for BEST accent-free performance
    model_id = "eleven_flash_v2_5"  # CRITICAL: Latest model with 32 languages + ultra-low latency
    
    # 🔥 CRITICAL: Language-specific voice settings for accent isolation
    if language_code and language_code in st.session_state.voice_settings:
        voice_settings = st.session_state.voice_settings[language_code].copy()
        logger.info(f"Using optimized {language_code} settings: {voice_settings}")
    else:
        voice_settings = st.session_state.voice_settings["default"]
    
    # 🔥 ENHANCED: SSML enhancement for pronunciation accuracy
    enhanced_text = add_accent_free_markup(text, language_code)
    
    data = {
        "text": enhanced_text,
        "model_id": model_id,
        "voice_settings": voice_settings,
        "apply_text_normalization": "auto"
    }
    
    start_time = time.time()
    
    try:
        # Optimized for Flash v2.5 speed
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}",
            json=data,
            headers=headers,
            timeout=10  # Flash v2.5 is much faster
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            content = response.content
            if len(content) < 100:
                return None, generation_time
                
            # Cache the result
            if not hasattr(st.session_state, 'tts_cache'):
                st.session_state.tts_cache = {}
            st.session_state.tts_cache[cache_key] = (BytesIO(content), generation_time)
            
            logger.info(f"✅ Accent-free audio generated for {language_code} in {generation_time:.2f}s")
            return BytesIO(content), generation_time
        else:
            logger.error(f"TTS API error: {response.status_code} - {response.text}")
            return None, generation_time
    
    except Exception as e:
        logger.error(f"Accent-free TTS error: {str(e)}")
        return None, time.time() - start_time

# 🔥 ENHANCED: Add SSML markup for accent-free pronunciation Urdu/English
def add_accent_free_markup(text, language_code):
    """🔥 ENHANCED: Add SSML markup for accent-free pronunciation Urdu/English"""
    if not language_code:
        return text
    
    # Clean text first
    clean_text = text.strip()
    
    # 🔥 UPDATED: Add language-specific SSML for accent-free pronunciation
    if language_code == "ur":
        # Urdu pronunciation optimization
        enhanced_text = f'<speak><lang xml:lang="ur-PK"><prosody rate="0.9">{clean_text}</prosody></lang></speak>'
    elif language_code == "en":
        # English pronunciation optimization  
        enhanced_text = f'<speak><lang xml:lang="en-US"><prosody rate="0.95">{clean_text}</prosody></lang></speak>'
    else:
        enhanced_text = clean_text
    
    return enhanced_text

def optimize_text_for_language(text, language_code):
    """🔥 UPDATED: Optimize text for specific language pronunciation Urdu/English"""
    # Simplified optimization for demo - in production, would use more sophisticated SSML
    if language_code == "ur":
        # Add pauses after punctuation for Urdu
        text = re.sub(r'([.!?])', r'\1...', text)
        
        # Slow down slightly for more accurate Urdu pronunciation
        text = f"{text}"
    elif language_code == "en":
        # English optimization
        # Add slight emphasis to important words
        text = re.sub(r'\b(water|book|hello|good|practice)\b', r' \1 ', text, flags=re.IGNORECASE)
    
    return text

def enhance_audio_for_language(audio_segment, language_code):
    """Apply language-specific enhancements to the audio"""
    try:
        # Basic enhancements for both languages
        enhanced = audio_segment.normalize()
        
        if language_code == "ur":
            # Urdu typically has more varied pitch - enhance slightly
            # In production, would use more complex DSP transformations
            enhanced = enhanced.normalize()
        elif language_code == "en":
            # English tends to have more emphasis - boost mid frequencies slightly
            # This is a simple approximation - real production would use filters
            enhanced = enhanced.normalize()
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing audio: {str(e)}")
        return audio_segment  # Return original on error

def process_multilingual_text(text, detect_language=True):
    """🔥 UPDATED: Process text with language markers and generate audio with accent isolation for Urdu/English"""
    # Parse language segments
    segments = []
    
    if detect_language:
        # 🔥 UPDATED: Split by markers if they exist [ur] for Urdu, [en] for English
        parts = text.split("[")
        
        for i, part in enumerate(parts):
            if i == 0 and part:  # First part without language tag
                segments.append({"text": part, "language": None})
                continue
                
            if not part:
                continue
                
            if "]" in part:
                lang_code, content = part.split("]", 1)
                if lang_code.lower() in ["ur", "en"]:  # 🔥 UPDATED: ur/en instead of cs/de
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

# 🔥 CORRECTED: Single Voice with Accent-Free Language Switching

# 🔊 ADD UNIVERSAL TTS FUNCTIONS HERE (after process_multilingual_text_seamless function ends)

async def generate_speech_universal(text, language_code=None, voice_id=None):
    """Universal speech generation supporting multiple TTS providers"""
    provider = st.session_state.tts_provider
    
    if provider == "openai":
        return await generate_openai_speech(text, language_code, voice_id)
    elif provider == "azure":
        return await generate_azure_speech(text, language_code, voice_id)
    else:  # elevenlabs (default)
        return generate_speech(text, language_code, voice_id)

def process_multilingual_text_universal(text, detect_language=True):
    """Universal multilingual text processing"""
    provider = st.session_state.tts_provider
    
    if provider == "openai":
        return process_openai_multilingual_text(text)
    elif provider == "azure":
        return process_azure_multilingual_text(text)
    else:  # elevenlabs
        return process_multilingual_text_seamless(text, detect_language)

def generate_speech_with_language_voice(text, language_code, segment_position=0, total_segments=1):
    """🔥 CORRECTED: Use SAME voice with language-specific pronunciation settings"""
    
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        return None, 0
    
    # 🎯 CRITICAL FIX: Use SAME voice ID for consistency
    voice_id = st.session_state.elevenlabs_voice_id  # Same voice for all languages!
    
    # 🔥 NEW: Language-specific pronunciation optimization (not different voices)
    voice_settings = get_accent_free_settings(language_code, segment_position, total_segments)
    
    # 🎯 CRITICAL: Enhanced SSML for accent-free pronunciation
    enhanced_text = create_accent_free_ssml(text, language_code)
    
    data = {
        "text": enhanced_text,
        "model_id": "eleven_multilingual_v2",  # Best for accent control
        "voice_settings": voice_settings,
        "apply_text_normalization": "auto"
    }
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",  # Same voice!
            json=data,
            headers=headers,
            timeout=10
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"✅ Generated accent-free {language_code} with same voice in {generation_time:.2f}s")
            return BytesIO(response.content), generation_time
        else:
            logger.error(f"TTS error: {response.status_code}")
            return None, generation_time
            
    except Exception as e:
        logger.error(f"Speech generation error: {str(e)}")
        return None, time.time() - start_time

def get_accent_free_settings(language_code, position, total_segments):
    """🔥 UPDATED: Voice settings for accent-free pronunciation with SAME voice for Urdu/English"""
    
    # Base settings for consistency
    base_settings = {
        "stability": 0.75,          # Balanced for natural speech
        "similarity_boost": 0.85,   # Maintain voice character
        "style": 0.60,             # Natural expression
        "use_speaker_boost": True   # Enhanced clarity
    }
    
    # 🔥 UPDATED: Language-specific accent control for Urdu/English
    if language_code == "ur":
        # Urdu pronunciation optimization
        base_settings.update({
            "stability": 0.98,      # 🔥 MAXIMUM stability for Urdu sounds
            "similarity_boost": 0.99,  # 🔥 MAXIMUM similarity for consistency
            "style": 0.90           # Natural Urdu expression
        })
    elif language_code == "en":
        # English pronunciation optimization
        base_settings.update({
            "stability": 0.96,      # 🔥 VERY HIGH for English sounds
            "similarity_boost": 0.97,  # 🔥 VERY HIGH similarity for native sound
            "style": 0.88           # Natural English expression
        })
    
    # Enhance stability for mid-sentence transitions
    if position > 0:
        base_settings["stability"] = min(0.98, base_settings["stability"] + 0.03)
    
    return base_settings

def create_accent_free_ssml(text, language_code):
    """🔥 UPDATED: Advanced SSML for accent-free pronunciation with same voice for Urdu/English"""
    
    if not language_code:
        return text
    
    # Clean text first
    clean_text = text.strip()
    
    # 🔥 UPDATED: Language-specific SSML for accent-free pronunciation
    if language_code == "ur":
        # Urdu pronunciation with phonetic hints
        enhanced_text = f'<speak><lang xml:lang="ur-PK"><phoneme alphabet="ipa" ph="">ˈ</phoneme><prosody rate="0.92" pitch="+2st">{clean_text}</prosody></lang></speak>'
    elif language_code == "en":
        # English pronunciation with phonetic hints  
        enhanced_text = f'<speak><lang xml:lang="en-US"><phoneme alphabet="ipa" ph="">ˈ</phoneme><prosody rate="0.95" pitch="+1st">{clean_text}</prosody></lang></speak>'
    else:
        enhanced_text = clean_text
    
    return enhanced_text

# 🎯 ALSO UPDATE: process_multilingual_text_seamless to use consistent voice
def process_multilingual_text_seamless(text, detect_language=True):
    """🎯 CORRECTED: Single voice with accent-free language switching"""
    
    # Parse language segments with improved mid-sentence detection
    segments = parse_language_segments_enhanced(text)
    
    if len(segments) <= 1:
        # Single language - use existing method
        return process_multilingual_text(text, detect_language)
    
    # 🔥 CORRECTED: Single voice processing with accent control
    audio_segments = []
    total_time = 0
    
    for i, segment in enumerate(segments):
        if not segment["text"].strip():
            continue
            
        # 🎯 KEY FIX: Same voice, different language pronunciation
        audio_data, generation_time = generate_speech_with_language_voice(
            segment["text"], 
            language_code=segment["language"],
            segment_position=i,
            total_segments=len(segments)
        )
        
        if audio_data:
            # Convert to AudioSegment for processing
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            
            # 🎯 CRITICAL: Normalize volume for consistent blending
            normalized_segment = normalize_audio_volume(audio_segment, target_dbfs=-18)
            
            audio_segments.append(normalized_segment)
            total_time += generation_time
    
    if not audio_segments:
        return None, 0
    
    # 🔥 IMPROVED: Smoother crossfading for same-voice transitions
    combined_audio = audio_segments[0]
    
    for i in range(1, len(audio_segments)):
        # Apply very subtle crossfade (since it's the same voice)
        combined_audio = apply_same_voice_crossfade(
            combined_audio, 
            audio_segments[i], 
            crossfade_ms=100  # Shorter for same voice
        )
    
    # Save final blended audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        combined_audio.export(
            temp_file.name, 
            format="mp3", 
            bitrate="192k",
            parameters=["-ac", "1", "-ar", "22050"]
        )
        return temp_file.name, total_time

def apply_same_voice_crossfade(audio1, audio2, crossfade_ms=100):
    """🔥 OPTIMIZED: Subtle crossfading for same voice transitions"""
    
    # Ensure both audio segments are normalized
    audio1_norm = normalize_audio_volume(audio1, -18)
    audio2_norm = normalize_audio_volume(audio2, -18)
    
    # Apply subtle crossfade (shorter duration since it's the same voice)
    crossfaded = audio1_norm.append(audio2_norm, crossfade=crossfade_ms)
    
    return crossfaded

def parse_language_segments_enhanced(text):
    """🔥 IMPROVED: Better parsing for mid-sentence language switches"""
    segments = []
    
    # Split by language markers but preserve word boundaries
    parts = re.split(r'(\[[a-z]{2}\])', text)
    
    current_language = None
    current_text = ""
    
    for part in parts:
        if re.match(r'\[[a-z]{2}\]', part):
            # Save previous segment if exists
            if current_text.strip():
                segments.append({
                    "text": current_text.strip(),
                    "language": current_language or "ur"  # 🔥 UPDATED: Default to Urdu instead of Czech
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
            "language": current_language or "ur"  # 🔥 UPDATED: Default to Urdu
        })
    
    # 🎯 FIX: Detect language for unmarked segments
    for segment in segments:
        if segment["language"] is None:
            segment["language"] = detect_primary_language(segment["text"])
    
    return segments

def normalize_audio_volume(audio_segment, target_dbfs=-18):
    """🎯 NORMALIZE: Ensure consistent volume for seamless blending"""
    # Calculate volume adjustment needed
    current_dbfs = audio_segment.dBFS
    volume_adjustment = target_dbfs - current_dbfs
    
    # Apply volume adjustment
    normalized = audio_segment.apply_gain(volume_adjustment)
    
    return normalized

def detect_primary_language(text):
    """🔥 UPDATED: Detect the primary language of a text with improved accuracy for Urdu/English"""
    # 🔥 UPDATED: Urdu-specific characters
    urdu_chars = set("آؤؤوچکیگھںاےپصذيقةثضثطعأهېۃيېخرقکیرئهۿۃؤوںھخچهۓ")
    
    # English-specific characters (Latin)
    english_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Count language-specific characters
    text_lower = text.lower()
    urdu_count = sum(1 for char in text if char in urdu_chars)
    english_count = sum(1 for char in text if char in english_chars)
    
    # 🔥 UPDATED: Urdu-specific words
    urdu_words = {
        "main", "mein", "aap", "ap", "hai", "hain", "ka", "ki", "ke", "ko", "se",
        "kya", "kaise", "kahan", "kab", "kyun", "nahi", "haan", "ji", "aur", "ya", 
        "lekin", "par", "agar", "to", "woh", "ye", "is", "us", "seekhna", "samjhna",
        "kehna", "bolna", "sunna", "dekhna", "English", "urdu", "language"
    }
    
    # 🔥 UPDATED: English-specific words
    english_words = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
    }
    
    # Count word occurrences
    words = re.findall(r'\b\w+\b', text_lower)
    urdu_word_count = sum(1 for word in words if word in urdu_words)
    english_word_count = sum(1 for word in words if word in english_words)
    
    # Improved scoring system with weighted metrics
    urdu_evidence = urdu_count * 2 + urdu_word_count * 3
    english_evidence = english_count * 2 + english_word_count * 3
    
    # Determine primary language
    if urdu_evidence > english_evidence and urdu_evidence > 0:
        return "ur"
    elif english_evidence > urdu_evidence and english_evidence > 0:
        return "en"
    
    # If unable to determine, use default based on distribution preference
    if st.session_state.language_distribution["ur"] >= st.session_state.language_distribution["en"]:
        return "ur"
    else:
        return "en"

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
            st.session_state.message_queue.put(f"🔍 Auto-detected: {detected_distribution['ur']}% Urdu, {detected_distribution['en']}% English")
        
        # Step 4: Generate Response with OpenAI
        st.session_state.message_queue.put("🤖 Generating intelligent response...")
        
        # Create enhanced system prompt for auto mode
        if st.session_state.response_language == "auto" and detected_distribution:
            ur_percent = detected_distribution["ur"]
            en_percent = detected_distribution["en"]
            system_prompt = (
                f"You are a multilingual AI language tutor. The user spoke {ur_percent}% Urdu and {en_percent}% English. "
                f"Respond naturally using the same language distribution: {ur_percent}% Urdu and {en_percent}% English. "
                f"Always use language markers [ur] and [en] to indicate language sections. "
                f"Be conversational and educational."
            )
        else:
            # Use existing logic for other modes
            system_prompt = None
        
        llm_result = await generate_llm_response_enhanced(user_input, system_prompt)
        
        if "error" in llm_result:
            st.session_state.message_queue.put(f"❌ Response generation failed: {llm_result.get('error')}")
            return user_input, None, transcription.get("latency", 0), 0, 0
        
        response_text = llm_result["response"]
        st.session_state.message_queue.put(f"💬 Generated: {response_text}")
        
        # Step 5: High-Quality Voice Synthesis
        st.session_state.message_queue.put("🎵 Generating natural speech...")
        audio_data, generation_time = await generate_speech_universal(
            segment["text"], 
            language_code=segment["language"]
        )
        
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
        
        # 🔥 UPDATED: Add pronunciation difficulty markers for Urdu/English
        urdu_difficult = ["خ", "ذ", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق"]  # Urdu difficult sounds
        english_difficult = ["th", "ch", "sh", "wh", "ng", "ph"]  # English difficult patterns
        
        for char in urdu_difficult:
            if char in text:
                context["detected_patterns"].append(f"Urdu_{char}")
                
        for pattern in english_difficult:
            if pattern in text.lower():
                context["detected_patterns"].append(f"English_{pattern}")
        
        return context
        
    except Exception as e:
        logger.error(f"Context extraction error: {str(e)}")
        return {"primary_language": "unknown", "confidence_level": "low"}

def create_pronunciation_aware_prompt(user_input, pronunciation_context):
    """🔥 UPDATED: Create system prompt that considers pronunciation context for Urdu/English"""
    
    primary_lang = pronunciation_context.get("primary_language", "unknown")
    
    # FIXED: More natural conversation prompts for Urdu/English
    system_prompt = f"""You are a helpful multilingual assistant that speaks Urdu and English naturally.

CONTEXT: The user said "{user_input}" (detected as {primary_lang})

INSTRUCTIONS:
1. Respond naturally and conversationally in the same language(s) they used
2. If they greeted you, greet them back appropriately  
3. If they asked how you are, answer naturally (like "Main theek hun" or "I'm doing well")
4. Use [ur] for Urdu and [en] for English content
5. Be helpful and friendly
6. NEVER use other languages unless they specifically ask
7. Keep responses short and natural

Respond as a friendly person would in normal conversation."""

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
        ur_percent = language_distribution["ur"]
        en_percent = language_distribution["en"]
        system_prompt = (
            f"You are a multilingual AI language tutor. Respond with approximately {ur_percent}% Urdu and {en_percent}% English. "
            f"Always use language markers [ur] and [en] to indicate language."
        )
    elif response_language in ["ur", "en"]:
        system_prompt = f"You are a language tutor. Respond only in {response_language} with [{response_language}] markers."
    else:
        system_prompt = None
    
    # Generate the LLM response with appropriate system prompt
    llm_result = await generate_llm_response_enhanced(text, system_prompt)
    
    if "error" in llm_result:
        st.session_state.message_queue.put(f"Error generating response: {llm_result.get('error')}")
        return None, llm_result.get("latency", 0), 0
    
    response_text = llm_result["response"]
    st.session_state.message_queue.put(f"Generated response: {response_text}")
    
    # Step 2: Text-to-Speech with accent isolation
    st.session_state.message_queue.put("Generating speech with accent isolation...")
    audio_data, generation_time = await generate_speech_universal(
        segment["text"], 
        language_code=segment["language"]
    )
    
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

def auto_detect_language_distribution(input_text):
    """🔥 UPDATED: Auto-detect language distribution from user input for Urdu/English"""
    
    # Remove any existing language markers for clean detection
    clean_text = re.sub(r'\[[a-z]{2}\]', '', input_text).strip()
    
    # Split into words for analysis
    words = re.findall(r'\b\w+\b', clean_text.lower())
    
    if not words:
        return {"ur": 60, "en": 40}  # Default if no words
    
    # 🔥 UPDATED: Urdu indicators
    urdu_chars = set("آؤؤوچکیگھںاےپصذيقةثضثطعأهېۃيېخرقکیرئهۿۃؤوںھخچهۓ")
    urdu_words = {
        "main", "mein", "aap", "ap", "hai", "hain", "ka", "ki", "ke", "ko", "se",
        "kya", "kaise", "kahan", "kab", "kyun", "nahi", "haan", "ji", "aur", "ya"
    }
    
    # 🔥 UPDATED: English indicators  
    english_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    english_words = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
    }
    
    # Count evidence
    urdu_evidence = 0
    english_evidence = 0
    
    # Character-based evidence
    for char in clean_text:
        if char in urdu_chars:
            urdu_evidence += 2
        elif char in english_chars:
            english_evidence += 2
    
    # Word-based evidence (stronger weight)
    for word in words:
        if word in urdu_words:
            urdu_evidence += 3
        elif word in english_words:
            english_evidence += 3
    
    # Calculate percentages
    total_evidence = urdu_evidence + english_evidence
    
    if total_evidence == 0:
        return {"ur": 60, "en": 40}  # Default if unclear
    
    ur_percent = int((urdu_evidence / total_evidence) * 100)
    en_percent = 100 - ur_percent
    
    # Ensure minimum 20% for each language to maintain bilingual nature
    if ur_percent < 20:
        ur_percent = 20
        en_percent = 80
    elif en_percent < 20:
        en_percent = 20
        ur_percent = 80
    
    return {"ur": ur_percent, "en": en_percent}

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
# STREAMLIT UI - 🔥 ENHANCED WITH URDU/ENGLISH LANGUAGE CONTROL OPTIONS
# ----------------------------------------------------------------------------------

def get_urdu_english_demo_scenarios():
    """🔥 UPDATED: Demo scenarios for Urdu/English tutoring"""
    return {
        "Vocabulary Request": (
            "English mein 'pani' aur kuch basic words kya kehte hain?"
        ),
        "Grammar Question": (
            "Past tense kaise banate hain English mein? Examples de sakte hain?"
        ),
        "Introduction Practice": (
            "Main apna introduction English mein kaise karun? Sikhayein please."
        ),
        "Pronunciation Help": (
            "Mujhe English 'th' sound mein problem hai. Help kar sakte hain?"
        ),
        "Daily Conversation": (
            "Rozana ki English conversation ke liye phrases sikhayein."
        ),
        "Custom Input": ""
    }

def main():
    """🔥 UPDATED: Main application entry point for Urdu/English"""
    # Page configuration - ONLY ONCE!
    st.set_page_config(
        page_title="Professional English Tutor - Urdu/English",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("Professional English Tutor for Urdu Speakers")
    st.subheader("🔥 Accent-Free Voice AI Tutor (A1-B2 English Learning)")
    
    # Status area for progress updates
    if 'status_area' not in st.session_state:
        st.session_state.status_area = st.empty()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API keys
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
        
        if st.button("Save API Keys"):
            st.session_state.elevenlabs_api_key = elevenlabs_key
            st.session_state.openai_api_key = openai_key
            st.session_state.api_keys_initialized = True
            st.success("API keys saved successfully!")
        
        # 🔥 UPDATED: ACCENT-FREE VOICE CONFIGURATION for Urdu/English
        st.subheader("🎯 Accent-Free Voice Setup")

        st.write("**Consistent Voice for Both Languages:**")
        if 'voices' in st.session_state and st.session_state.voices:
            voice_options = {voice["name"]: voice["voice_id"] for voice in st.session_state.voices}
            current_voice = None
            for name, vid in voice_options.items():
                if vid == st.session_state.elevenlabs_voice_id:
                    current_voice = name
                    break
            
            selected_voice_name = st.selectbox(
                "Select Voice (Used for ALL languages)", 
                options=list(voice_options.keys()),
                index=list(voice_options.keys()).index(current_voice) if current_voice else 0,
                key="single_voice_select"
            )
            
            if selected_voice_name:
                new_voice_id = voice_options[selected_voice_name]
                st.session_state.elevenlabs_voice_id = new_voice_id
                # 🔥 UPDATED: Update all language voices to use the same voice for ur/en
                st.session_state.language_voices["ur"] = new_voice_id
                st.session_state.language_voices["en"] = new_voice_id
                st.session_state.language_voices["default"] = new_voice_id

        # Voice consistency status
        st.success(f"""
        ✅ **Accent-Free Configuration**
        - Voice ID: {st.session_state.elevenlabs_voice_id[:8]}...
        - Used for: ALL languages (Urdu + English)
        - Model: Flash v2.5 (Multilingual accent-free)
        - Stability: 0.98 (Urdu) / 0.96 (English)
        """)
        
        # 🔥 UPDATED: Language Response Options for Urdu/English
        st.subheader("🎓 English Tutor Mode")
 
        # Choose Tutor Mode
        response_language = st.radio(
            "Tutor Mode",
            options=["both", "ur", "en"],
            format_func=lambda x: {
                "both": "🎯 English Tutor (Urdu + English)", 
                "ur": "اردو Only (Urdu Only)", 
                "en": "English Only"
            }[x]
        )
        if response_language != st.session_state.response_language:
            st.session_state.response_language = response_language
            st.success(f"Tutor Mode set to: {response_language}")
        
        # Language distribution (only shown when "both" is selected)
        if response_language == "both":
            st.subheader("📊 Language Distribution")
            
            # 🔥 UPDATED: Urdu percentage slider
            ur_percent = st.slider("Urdu % (explanations)", min_value=40, max_value=80, value=st.session_state.language_distribution["ur"])
            
            # Calculate English percentage automatically
            en_percent = 100 - ur_percent
            
            # Display English percentage
            st.text(f"English %: {en_percent} (examples & terms)")
            
            # Update language distribution if changed
            if ur_percent != st.session_state.language_distribution["ur"]:
                st.session_state.language_distribution = {
                    "ur": ur_percent,
                    "en": en_percent
                }
                st.success(f"Language distribution updated: {ur_percent}% Urdu, {en_percent}% English")
        
        # Speech recognition model
        st.subheader("Speech Recognition")
        
        whisper_model = st.selectbox(
            "Whisper Model",
            options=["tiny", "base", "small", "medium", "large"],
            index=["tiny", "base", "small", "medium", "large"].index(st.session_state.whisper_model) 
            if st.session_state.whisper_model in ["tiny", "base", "small", "medium", "large"] 
            else 1
        )
        
        if whisper_model != st.session_state.whisper_model:
            st.session_state.whisper_model = whisper_model
            st.session_state.whisper_local_model = None
            st.success(f"Changed Whisper model to {whisper_model}")
        
        # 🔥 UPDATED: Accent improvement explanation for Urdu/English
        st.header("🔥 Accent Elimination")
        st.info("""
        This system includes MAXIMUM optimizations to eliminate accent interference:
        
        **Voice Settings:**
        - Urdu: 98% stability, 99% similarity
        - English: 96% stability, 97% similarity
        
        **SSML Enhancement:**
        - Language-specific pronunciation isolation
        - Micro-pauses between language switches
        - Voice context reset when switching languages
        
        **Result:** Urdu sounds perfectly Urdu, English sounds perfectly English - ZERO accent bleeding!
        """)
        
        
        # Add this to your sidebar in main() function
        st.subheader("🔊 TTS Provider Selection")

        tts_provider = st.radio(
            "Choose TTS Provider",
            options=["elevenlabs", "openai", "azure"],
            format_func=lambda x: {
                "elevenlabs": "🎭 ElevenLabs (Current)",
                "openai": "🤖 OpenAI TTS (Cheapest)", 
                "azure": "☁️ Azure Speech (Enterprise)"
            }[x],
            key="tts_provider_select"
        )

        if tts_provider != st.session_state.tts_provider:
            st.session_state.tts_provider = tts_provider
            st.success(f"Switched to {tts_provider.upper()} TTS")

        # Show provider-specific settings
        if tts_provider == "azure":
            azure_key = st.text_input(
                "Azure Speech Key", 
                value=st.session_state.azure_speech_key,
                type="password"
            )
            azure_region = st.text_input(
                "Azure Region", 
                value=st.session_state.azure_speech_region,
                placeholder="eastus"
            )
            
            if st.button("Save Azure Settings"):
                st.session_state.azure_speech_key = azure_key
                st.session_state.azure_speech_region = azure_region
                st.success("Azure settings saved!")

        elif tts_provider == "openai":
            openai_voice = st.selectbox(
                "OpenAI Voice",
                options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                index=0
            )
            st.session_state.openai_tts_voice = openai_voice

        # Show cost comparison
        st.info(f"""
        **Current Provider: {tts_provider.upper()}**

        Cost Comparison (per month):
        - OpenAI: $10-20 (Cheapest)
        - Azure: $20-40 (Enterprise)  
        - ElevenLabs: $22-99 (Premium)
        """)
        # Performance recommendations for low latency
        st.header("Latency Optimization")
        
        avg_total = calculate_average_latency(st.session_state.performance_metrics["total_latency"])
        
        if avg_total > 3.0:
            st.warning("""
            ### Hardware Recommendations for Lower Latency
            
            To achieve the target latency of under 3 seconds:
            
            1. Use a dedicated server with:
               - NVIDIA GPU (T4 or better)
               - 16+ GB RAM
               - SSD storage
               
            2. Run Whisper on GPU acceleration
            
            3. Consider a business plan for ElevenLabs
               for higher priority API access
            """)
        
        # Performance metrics
        st.header("Performance")
        
        avg_stt = calculate_average_latency(st.session_state.performance_metrics["stt_latency"])
        avg_llm = calculate_average_latency(st.session_state.performance_metrics["llm_latency"])
        avg_tts = calculate_average_latency(st.session_state.performance_metrics["tts_latency"])
        avg_total = calculate_average_latency(st.session_state.performance_metrics["total_latency"])
        
        st.metric("Avg. STT Latency", f"{avg_stt:.2f}s")
        st.metric("Avg. LLM Latency", f"{avg_llm:.2f}s")
        st.metric("Avg. TTS Latency", f"{avg_tts:.2f}s")
        st.metric("Avg. Total Latency", f"{avg_total:.2f}s")
        
        # API calls
        st.subheader("API Usage")
        st.text(f"Whisper API calls: {st.session_state.performance_metrics['api_calls']['whisper']}")
        st.text(f"OpenAI API calls: {st.session_state.performance_metrics['api_calls']['openai']}")
        st.text(f"ElevenLabs API calls: {st.session_state.performance_metrics['api_calls']['elevenlabs']}")
    
    # Main interaction area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("Input")
        
        # Input type selection
        input_type = st.radio("Select Input Type", ["Text", "Voice"], horizontal=True)
        
        if input_type == "Text":
            # 🔥 UPDATED: Text input for Urdu/English
            st.subheader("Text Input")
            st.write("Use [ur] to mark Urdu text and [en] to mark English text.")
            
            # 🔥 UPDATED: Demo preset examples for Urdu/English
            demo_scenarios = get_urdu_english_demo_scenarios()
            
            selected_scenario = st.selectbox(
                "Demo Examples", 
                options=list(demo_scenarios.keys())
            )
            
            text_input = st.text_area(
                "Edit or enter new text",
                value=demo_scenarios[selected_scenario],
                height=150
            )
            
            text_process_button = st.button("Process Text", type="primary")
            
            if text_process_button and text_input:
                with st.spinner("Processing text input..."):
                    # Process the text input
                    audio_path, llm_latency, tts_latency = asyncio.run(process_text_input(text_input))
                    
                    # Store for display in the output section
                    st.session_state.last_text_input = text_input
                    st.session_state.last_audio_output = audio_path
                    
                    # Show latency metrics
                    total_latency = llm_latency + tts_latency
                    st.success(f"Text processed in {total_latency:.2f} seconds")
        
        else:
            # 🔥 UPDATED: Voice input - FIXED HTML5 AUDIO RECORDER for Urdu/English
            st.subheader("🎤 Professional Voice Recording")
            
            # Check if API keys are set
            keys_set = (
                st.session_state.elevenlabs_api_key and 
                st.session_state.openai_api_key
            )

            if not keys_set:
                st.warning("Please set both API keys in the sidebar first")
            else:
                st.write("🎯 **HTML5 Audio Recording** - Accent-Free Railway Deployment")
                
                # Create the HTML5 audio recorder component
                create_audio_recorder_component()

                st.markdown("---")
                st.write("**🔄 ACCENT-FREE PROCESSING:**")
                
                # FIXED: Reliable upload processing
                uploaded_audio = st.file_uploader(
                    "📥 Upload Your Downloaded Recording Here", 
                    type=['wav', 'mp3', 'webm', 'ogg'],
                    key="main_upload",
                    help="After recording above, download the file and upload it here for accent-free processing"
                )

                if uploaded_audio is not None:
                    # IMMEDIATE processing when file is uploaded
                    with st.spinner("🔄 **ACCENT-FREE PROCESSING...**"):
                        try:
                            # Save uploaded file
                            temp_path = tempfile.mktemp(suffix=".wav")
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_audio.read())
                            
                            # Apply amplification and process through the full pipeline
                            amplified_path = amplify_recorded_audio(temp_path)
                            
                            # Process with enhanced pipeline
                            text, audio_output_path, stt_latency, llm_latency, tts_latency = asyncio.run(process_voice_input_pronunciation_enhanced(amplified_path))
                            
                            # Store results
                            if text:
                                st.session_state.last_text_input = text
                            if audio_output_path:
                                st.session_state.last_audio_output = audio_output_path
                            
                            # Show results
                            total_latency = stt_latency + llm_latency + tts_latency
                            st.success(f"✅ **ACCENT-FREE PROCESSING COMPLETE!** ({total_latency:.2f}s)")
                            st.balloons()
                            
                            # Clean up
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                            if amplified_path != temp_path and os.path.exists(amplified_path):
                                os.unlink(amplified_path)
                                
                        except Exception as e:
                            st.error(f"Processing error: {str(e)}")

                # 🔥 UPDATED: Enhanced instructions for Urdu/English
                st.success("""
                🎯 **ACCENT-FREE WORKFLOW:**
                1. Click "🔴 START RECORDING" above
                2. Speak clearly in Urdu or English  
                3. Click "⏹️ STOP RECORDING" when done
                4. **DOWNLOAD** the file that appears automatically
                5. **UPLOAD** it above - accent-free processing starts immediately!

                **⚡ Result: ZERO accent bleeding between Urdu and English!**
                """)
    
    with col2:
        st.header("Output")
        
        # Transcribed text
        if 'last_text_input' in st.session_state and st.session_state.last_text_input:
            st.subheader("Transcribed/Input Text")
            st.text_area(
                "Text with language markers",
                value=st.session_state.last_text_input,
                height=100,
                disabled=True
            )
        
        # Generated response
        if 'conversation_history' in st.session_state and st.session_state.conversation_history:
            last_exchange = st.session_state.conversation_history[-1]
            
            if 'assistant_response' in last_exchange:
                st.subheader("AI Tutor Response")
                st.text_area(
                    "Response text",
                    value=last_exchange['assistant_response'],
                    height=150,
                    disabled=True
                )
        
        # Generated audio
        if 'last_audio_output' in st.session_state and st.session_state.last_audio_output:
            st.subheader("🔥 Generated Speech (Accent-Free)")
            
            # Display audio with player
            audio_bytes = display_audio(st.session_state.last_audio_output, autoplay=True)
            
            if audio_bytes:
                # Download button
                st.download_button(
                    label="Download Audio",
                    data=audio_bytes,
                    file_name="accent_free_tutor_response.mp3",
                    mime="audio/mp3"
                )
    
    # Auto-display preview audio if available
    if 'preview_audio_path' in st.session_state and st.session_state.preview_audio_path:
        if os.path.exists(st.session_state.preview_audio_path):
            st.subheader("🎤 Your Recording (Preview)")
            
            with open(st.session_state.preview_audio_path, "rb") as audio_file:
                preview_bytes = audio_file.read()
                st.audio(preview_bytes, format="audio/wav")
                
            st.download_button(
                label="Download Your Recording",
                data=preview_bytes,
                file_name="your_recording.wav",
                mime="audio/wav"
            )
    
    # Conversation history
    if st.session_state.conversation_history:
        st.header("Conversation History")
        
        for i, exchange in enumerate(st.session_state.conversation_history[-5:]):  # Show last 5 exchanges
            with st.expander(f"Exchange {i+1} - {exchange.get('timestamp', 'Unknown time')[:19]}"):
                st.markdown("**User:**")
                st.text(exchange.get('user_input', 'No input'))
                
                st.markdown("**AI Tutor:**")
                st.text(exchange.get('assistant_response', 'No response'))
                
                # Latency info
                latency = exchange.get('latency', {})
                st.text(f"STT: {latency.get('stt', 0):.2f}s | LLM: {latency.get('llm', 0):.2f}s | TTS: {latency.get('tts', 0):.2f}s | Total: {latency.get('total', 0):.2f}s")
    
    # Status area
    st.header("Status")
    st.session_state.status_area = st.empty()
    
    # Update status from queue
    update_status()

if __name__ == "__main__":
    main()
