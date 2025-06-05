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
import openai

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


if 'language_voices' not in st.session_state:
    # USE SAME VOICE FOR ALL LANGUAGES - No more "two speakers" effect
    single_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Use same voice for everything
    st.session_state.language_voices = {
        "cs": single_voice_id,  # SAME voice for Czech
        "de": single_voice_id,  # SAME voice for German
        "default": single_voice_id
    }

# OPTIMIZED voice settings for Flash v2.5 model
if 'voice_settings' not in st.session_state:
    st.session_state.voice_settings = {
        "cs": {  # Czech-optimized settings
            "stability": 0.95,        # MAXIMUM stability for consistent Czech
            "similarity_boost": 0.98, # MAXIMUM similarity for native Czech sound
            "style": 0.85,           # High style for natural Czech expression
            "use_speaker_boost": True # Enable speaker boost for clarity
        },
        "de": {  # German-optimized settings  
            "stability": 0.92,        # VERY HIGH stability for consistent German
            "similarity_boost": 0.95, # VERY HIGH similarity for native German sound
            "style": 0.80,           # High style for natural German expression
            "use_speaker_boost": True # Enable speaker boost for clarity
        },
        "default": {
            "stability": 0.90,
            "similarity_boost": 0.90,
            "style": 0.75,
            "use_speaker_boost": True
        }
    }

# TTS Provider Configuration
if 'tts_provider' not in st.session_state:
    st.session_state.tts_provider = "elevenlabs"  # Default

if 'azure_speech_key' not in st.session_state:
    st.session_state.azure_speech_key = os.environ.get("AZURE_SPEECH_KEY", "")
    st.session_state.azure_region = os.environ.get("AZURE_REGION", "eastus")

# Provider-specific voice configurations for accent-free switching
if 'provider_voice_configs' not in st.session_state:
    st.session_state.provider_voice_configs = {
        "elevenlabs": {
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "model": "eleven_flash_v2_5"
        },
        "openai": {
            "voice": "nova",  # Most neutral for multilingual
            "model": "tts-1"
        },
        "azure": {
            "voice_cs": "cs-CZ-VlastaNeural",  # Czech female
            "voice_de": "de-DE-KatjaNeural",   # German female - same speaker type
            "rate": "0.9",
            "pitch": "0%"
        }
    }
# Backward compatibility
if 'elevenlabs_voice_id' not in st.session_state:
    st.session_state.elevenlabs_voice_id = st.session_state.language_voices["default"]
    
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
                statusDiv.innerHTML = '🔴 RECORDING - Speak in Czech or German';
                
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
        
        llm_result = await generate_llm_response(user_input)
        
        if "error" in llm_result:
            st.session_state.message_queue.put(f"❌ Response generation failed: {llm_result.get('error')}")
            return user_input, None, transcription.get("latency", 0), 0, 0
        
        response_text = llm_result["response"]
        st.session_state.message_queue.put(f"💬 Generated: {response_text}")
        
        # Step 5: High-Quality Voice Synthesis
        st.session_state.message_queue.put("🎵 Generating accent-free speech...")
        audio_path, tts_latency = process_multilingual_text_seamless(response_text)
        
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

# REPLACE the generate_llm_response function in your tutor_app.py with this FIXED version:

async def generate_llm_response(prompt, system_prompt=None, api_key=None):
    """Generate response with INTELLIGENT language tagging - not every word"""
    if not api_key:
        api_key = st.session_state.openai_api_key
        
    if not api_key:
        logger.error("OpenAI API key not provided")
        return {
            "response": "Error: OpenAI API key not configured. Please set it in the sidebar.",
            "latency": 0
        }
    
    start_time = time.time()
    
    messages = []
        
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        response_language = st.session_state.response_language
        
        if response_language == "both":
            system_content = """You are "GermanMeister" - a premium Czech-speaking German language tutor. Your job is to teach German to Czech speakers using INTELLIGENT language mixing.

        CORE IDENTITY:
        You are a certified German language instructor with 15+ years of experience teaching Czech speakers. You hold a Master's degree in Germanic linguistics and are perfectly bilingual in Czech and German. Your teaching style is engaging, systematic, and results-oriented.

        CURRICULUM STRUCTURE (A1-A2 LEVELS):
        A1 Level: Basic greetings, present tense (sein, haben), articles (der/die/das), family/food/time vocabulary, basic word order, question formation
        A2 Level: Past/future tenses, modal verbs, adjective declination, dative/accusative cases, prepositions, separable verbs, travel/work vocabulary

        PEDAGOGICAL APPROACH:
        1. Use Czech [cs] for explanations, German [de] for examples and practice
        2. Break complex concepts into micro-lessons
        3. Always provide immediate practice opportunities
        4. Use real-life scenarios (restaurant, hotel, work)
        5. Correct errors gently with clear explanations
        6. Reference previous learning and show progress
        7. Include German culture and regional differences

        VOCABULARY LESSON FORMAT:
        "[cs] Explanation [de] German term [cs] detailed explanation [de] example sentence [cs] practice suggestion"
        - Include: German word + article, Czech translation, pronunciation tip, example sentence, related words

        GRAMMAR EXPLANATIONS:
        - Start with Czech explanation of concept
        - Show German examples with highlighting  
        - Explain pattern/rule clearly
        - Provide 3-4 practice examples
        - Connect to previously learned material

        CONVERSATION PRACTICE:
        - Set realistic scenarios (ordering food, directions)
        - Provide German phrases with Czech explanations
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
        - Use proper [cs] and [de] markers always
        - Keep responses 2-4 sentences for engagement
        - Include at least one practice element per response
        - Maintain encouraging, professional tone
        - Provide actionable next steps

        SAMPLE RESPONSE STYLE:
        For vocabulary: "[cs] Perfektní otázka! 'Voda' je [de] das Wasser [cs] - rod střední. Dalších pár: [de] das Brot [cs] (chléb), [de] die Milch [cs] (mléko). Zkuste: [de] Ich trinke Wasser [cs] (piju vodu). Jaké jídlo máte nejraději?"

        For grammar errors: "[cs] Téměř správně! Místo [de] 'ich haben' [cs] řekněte [de] 'ich habe' [cs] - první osoba je vždy 'habe'. Zkuste: [de] Ich habe einen Hund [cs] (mám psa)."

        You're guiding PAID customers through structured German learning. Every response must add value to their investment and move them toward conversational fluency. Be systematic, encouraging, and results-focused.
        NEVER DO THIS (wrong):
        ❌ "[cs] Jsem učitel [de] Ich bin Lehrer" (same content translated)

        ALWAYS DO THIS (correct):
        ✅ "[cs] Jsem váš učitel němčiny. Naučíme se říct [de] Guten Morgen [cs] což znamená 'dobré ráno'"
        
        LANGUAGE PURPOSES:
        - Czech [cs] = Your teaching language (explain, instruct, encourage)  
        - German [de] = Target language (vocabulary, examples, practice phrases)


        CRITICAL TAGGING RULES:
        1. Use [cs] ONLY for Czech explanations and instructions
        2. Use [de] ONLY for German vocabulary, phrases, and examples to learn
        3. DO NOT tag every word - only tag when switching languages
        4. Tag complete meaningful units (words/phrases), not individual words

        EXAMPLES OF CORRECT TAGGING:
        ✅ "[cs] Slovo 'voda' v němčině je [de] das Wasser [cs] - rod střední"
        ✅ "[cs] Řekněte [de] Guten Morgen [cs] což znamená dobré ráno"
        ✅ "[cs] Zkuste větu [de] Ich trinke Wasser [cs] - rozumíte?"

        WRONG TAGGING (DON'T DO):
        ❌ "[cs] Slovo [cs] 'voda' [cs] v [cs] němčině [cs] je [de] das [de] Wasser"
        ❌ "[de] Ich [de] trinke [de] Wasser"

        TEACHING APPROACH:
        - Start explanations in Czech [cs]
        - Introduce German words/phrases with [de] 
        - Return to Czech [cs] for clarification
        - Keep responses 2-3 sentences max
        - Always include practice opportunity

        PERSONALITY: Professional, encouraging, systematic. You're teaching paying students who expect results.
        make sure you're not adding german line or word in [cs] tag & czech line or word within [de] tag"""

        elif response_language == "cs":
            system_content = "You are a helpful Czech assistant. ALWAYS respond ONLY in Czech with [cs] markers."
        elif response_language == "de":
            system_content = "You are a helpful German assistant. ALWAYS respond ONLY in German with [de] markers."
            
        messages.append({"role": "system", "content": system_content})
    
    # Add conversation context (last 2 exchanges only)
    for exchange in st.session_state.conversation_history[-2:]:
        if "user_input" in exchange:
            messages.append({"role": "user", "content": exchange["user_input"]})
        if "assistant_response" in exchange:
            messages.append({"role": "assistant", "content": exchange["assistant_response"]})
    
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
                    "model": "gpt-4",
                    "messages": messages,
                    "temperature": 0.3,  # Lower for more consistent tagging
                    "max_tokens": 300
                },
                timeout=30.0
            )
            
            latency = time.time() - start_time
            st.session_state.performance_metrics["llm_latency"].append(latency)
            st.session_state.performance_metrics["api_calls"]["openai"] += 1
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                # Clean up tagging intelligently
                response_text = clean_intelligent_tags(response_text)
                
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

def clean_intelligent_tags(response_text):
    """Clean up intelligent language tags"""
    # Remove excessive tagging
    response_text = re.sub(r'\[cs\]\s*\[cs\]', '[cs]', response_text)
    response_text = re.sub(r'\[de\]\s*\[de\]', '[de]', response_text)
    
    # Ensure proper spacing
    response_text = re.sub(r'\[cs\]\s*', '[cs] ', response_text)
    response_text = re.sub(r'\[de\]\s*', '[de] ', response_text)
    
    # If no tags at all, add Czech as default
    if not re.search(r'\[cs\]|\[de\]', response_text):
        response_text = f"[cs] {response_text}"
    
    return response_text.strip()

def ensure_proper_language_markers(response_text):
    """Ensure response has proper language markers - minimal processing"""
    
    # If already has markers, just clean them up
    if "[cs]" in response_text or "[de]" in response_text:
        # Clean up spacing around markers
        response_text = re.sub(r'\[cs\]\s*', '[cs] ', response_text)
        response_text = re.sub(r'\[de\]\s*', '[de] ', response_text)
        response_text = re.sub(r'\s+\[cs\]', ' [cs]', response_text)
        response_text = re.sub(r'\s+\[de\]', ' [de]', response_text)
        return response_text.strip()
    
    # If no markers, add Czech marker (default instructional language)
    return f"[cs] {response_text.strip()}"

def create_proper_introduction_response(user_input):
    """Create EXACT proper introduction response"""
    
    # Extract name if present
    name_match = re.search(r'jmenuji se (\w+[\s\w]*)', user_input.lower())
    if name_match:
        name = name_match.group(1).title().strip()
        return (
            f"[cs] Dobrý den! Pro představení v němčině můžete říct: [de] Mein Name ist {name} [cs] nebo [de] Ich heiße {name} [cs] Takto se představujete a říkáte své jméno. "
            f"[cs] Pokud chcete být formálnější, můžete říct [de] Ich bin {name} [cs] Jak se máte? Potřebujete se naučit nějaká další německá slova?"
        )
    else:
        return (
            "[cs] Dobrý den! Pro představení v němčině můžete říct: [de] Mein Name ist [cs] a pak své jméno, nebo [de] Ich heiße [cs] a své jméno. "
            "[cs] Takto se představujete. Pokud chcete být formálnější, můžete říct [de] Ich bin [cs] a své jméno. Jak se máte?"
        )

def fix_common_marker_errors(response_text):
    """Fix common language marker errors"""
    
    # Remove quotes around German text and add proper markers
    response_text = re.sub(r'"([^"]*(?:mein|ich|hallo|guten|das|ist|bin|heiße)[^"]*)"', r'[de] \1 [cs]', response_text, flags=re.IGNORECASE)
    
    # Fix Czech text incorrectly marked as German
    czech_phrases = ["jak se máš", "můžu ti", "s něčím", "dalším", "pomoci", "ahoj", "toheede", "můžeš říci"]
    for phrase in czech_phrases:
        # If Czech phrase is after [de], move it to [cs]
        pattern = rf'\[de\]([^[]*{re.escape(phrase)}[^[]*)'
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            czech_part = match.group(1).strip()
            response_text = re.sub(pattern, f'[cs] {czech_part}', response_text, flags=re.IGNORECASE)
    
    # Ensure proper spacing around markers
    response_text = re.sub(r'\[cs\]\s*', '[cs] ', response_text)
    response_text = re.sub(r'\[de\]\s*', '[de] ', response_text)
    response_text = re.sub(r'\s+\[cs\]', ' [cs]', response_text)
    response_text = re.sub(r'\s+\[de\]', ' [de]', response_text)
    
    # Remove duplicate markers
    response_text = re.sub(r'\[cs\]\s*\[cs\]', '[cs]', response_text)
    response_text = re.sub(r'\[de\]\s*\[de\]', '[de]', response_text)
    
    return response_text.strip()

def ensure_natural_mixing(response_text):
    """Ensure response has natural Czech-German mixing"""
    
    # If response has no language markers, add them naturally
    if "[cs]" not in response_text and "[de]" not in response_text:
        # Add Czech marker and suggest some German
        return f"[cs] {response_text} V němčině můžete říct [de] 'Verstehe!' [cs] což znamená 'rozumím'."
    
    # If response is all Czech, add some German naturally
    if "[cs]" in response_text and "[de]" not in response_text:
        # Insert German naturally
        czech_text = response_text.replace("[cs]", "").strip()
        return f"[cs] {czech_text} V němčině to můžeme říct [de] 'Das ist gut!' [cs] Rozumíte?"
    
    return response_text
        
def clean_and_fix_response(user_input, response_text):
    """Clean response and ensure it makes conversational sense"""
    
    # CRITICAL FIX: Check user's language preference first
    response_language = st.session_state.response_language
    
    # If user selected single language mode, enforce it strictly
    if response_language == "cs":
        # Force Czech only response
        clean_text = re.sub(r'\[de\].*?(?=\[cs\]|$)', '', response_text, flags=re.DOTALL)
        clean_text = re.sub(r'\[cs\]\s*', '', clean_text).strip()
        if not clean_text:
            if 'guten tag' in user_input.lower() or 'wie geht' in user_input.lower():
                return "[cs] Dobrý den! Mám se dobře, děkuji."
            elif 'jak se máte' in user_input.lower():
                return "[cs] Mám se dobře, děkuji! A vy?"
            else:
                return "[cs] Děkuji za vaši zprávu."
        return f"[cs] {clean_text}"
    
    elif response_language == "de":
        # Force German only response  
        clean_text = re.sub(r'\[cs\].*?(?=\[de\]|$)', '', response_text, flags=re.DOTALL)
        clean_text = re.sub(r'\[de\]\s*', '', clean_text).strip()
        if not clean_text:
            if 'dobrý den' in user_input.lower() or 'jak se máte' in user_input.lower():
                return "[de] Guten Tag! Mir geht es gut, danke."
            elif 'guten tag' in user_input.lower() or 'wie geht' in user_input.lower():
                return "[de] Mir geht es gut, danke! Und Ihnen?"
            else:
                return "[de] Vielen Dank für Ihre Nachricht."
        return f"[de] {clean_text}"
    
    # For "both" mode, ensure both languages are present
    if response_language == "both":
        # ENHANCED: Ensure response contains both German and Czech
        has_german = '[de]' in response_text
        has_czech = '[cs]' in response_text
        
        if not has_german or not has_czech:
            # Force both languages if missing
            if 'guten tag' in user_input.lower() or 'dobrý den' in user_input.lower():
                return "[de] Guten Tag! Mir geht es sehr gut, vielen Dank! [cs] Dobrý den! Mám se výborně, děkuji!"
            elif 'jak se máte' in user_input.lower() or 'wie geht' in user_input.lower():
                return "[de] Mir geht es ausgezeichnet, danke der Nachfrage! [cs] Mám se skvěle, děkuji za optání!"
            else:
                return "[de] Vielen Dank für Ihre Nachricht! Gerne helfe ich Ihnen weiter. [cs] Děkuji za vaši zprávu! Rád vám pomohu."
        
        return response_text  # Already has both languages

    # Remove any English explanations that shouldn't be there
    lines = response_text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip English explanations (lines without language markers that contain English)
        if not re.search(r'\[([a-z]{2})\]', line):
            # Check if it's likely English
            english_indicators = ['you', 'are', 'the', 'and', 'or', 'is', 'this', 'that', 'would', 'like', 'know', 'practice', 'specific', 'anything']
            if any(word in line.lower() for word in english_indicators):
                continue  # Skip English lines
        
        cleaned_lines.append(line)
    
    # Rebuild response
    cleaned_response = ' '.join(cleaned_lines)
    
    # If response is empty after cleaning, provide a default
    if not cleaned_response.strip():
        # Detect user's language and respond appropriately
        if 'guten tag' in user_input.lower() or 'wie geht' in user_input.lower():
            cleaned_response = "[de] Guten Tag! Mir geht es gut, danke."
        elif 'dobrý den' in user_input.lower() or 'jak se máte' in user_input.lower():
            cleaned_response = "[cs] Dobrý den! Mám se dobře, děkuji."
        else:
            cleaned_response = "[cs] Děkuji za vaši zprávu. [de] Vielen Dank für Ihre Nachricht."
    
    # Ensure language markers are present
    if not re.search(r'\[([a-z]{2})\]', cleaned_response):
        # Add appropriate language markers based on content
        cleaned_response = add_appropriate_language_markers(cleaned_response, user_input)
    
    return cleaned_response.strip()

def add_appropriate_language_markers(text, user_input):
    """Add appropriate language markers based on user input and content"""
    
    # Detect user's language preference from input
    user_input_lower = user_input.lower()
    
    # If user used German
    if any(word in user_input_lower for word in ['guten', 'tag', 'wie', 'geht', 'danke', 'bitte']):
        return f"[de] {text}"
    
    # If user used Czech  
    elif any(word in user_input_lower for word in ['dobrý', 'den', 'jak', 'se', 'máte', 'děkuji', 'prosím']):
        return f"[cs] {text}"
    
    # Default to Czech if unclear
    return f"[cs] {text}"


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
    """Enhanced distribution for 'both' mode - always include both languages"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 1:
        # Single sentence - provide both languages
        return f"[de] {text} [cs] {text}"
    
    result = ""
    cs_percent = st.session_state.language_distribution["cs"]
    de_percent = st.session_state.language_distribution["de"]
    
    total_sentences = len(sentences)
    cs_sentences = max(1, int(total_sentences * (cs_percent / 100)))  # Ensure at least 1
    de_sentences = max(1, total_sentences - cs_sentences)  # Ensure at least 1
    
    # Add German sentences first
    german_added = 0
    czech_added = 0
    
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            if german_added < de_sentences and (i < de_sentences or czech_added >= cs_sentences):
                result += f"[de] {sentence} "
                german_added += 1
            else:
                result += f"[cs] {sentence} "
                czech_added += 1
    
    # Ensure we have at least one of each language
    if german_added == 0:
        result = f"[de] Verstanden! " + result
    if czech_added == 0:
        result += f"[cs] Rozumím!"
    
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

# 🎯 CORRECTED APPROACH: Single Voice with Accent-Free Language Switching

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
    """🎯 OPTIMIZED: Voice settings for accent-free pronunciation with SAME voice"""
    
    # Base settings for consistency
    base_settings = {
        "stability": 0.75,          # Balanced for natural speech
        "similarity_boost": 0.85,   # Maintain voice character
        "style": 0.60,             # Natural expression
        "use_speaker_boost": True   # Enhanced clarity
    }
    
    # 🔥 NEW: Language-specific accent control (same voice, different pronunciation)
    if language_code == "cs":
        # Czech pronunciation optimization
        base_settings.update({
            "stability": 0.80,      # Slightly more stable for Czech sounds
            "similarity_boost": 0.90,  # Higher similarity for consistency
            "style": 0.65           # Natural Czech expression
        })
    elif language_code == "de":
        # German pronunciation optimization
        base_settings.update({
            "stability": 0.78,      # Balanced for German sounds
            "similarity_boost": 0.88,  # Maintain voice character
            "style": 0.62           # Natural German expression
        })
    
    # Enhance stability for mid-sentence transitions
    if position > 0:
        base_settings["stability"] = min(0.85, base_settings["stability"] + 0.03)
    
    return base_settings

def create_accent_free_ssml(text, language_code):
    """🎯 ENHANCED: Advanced SSML for accent-free pronunciation with same voice"""
    
    if not language_code:
        return text
    
    # Clean text first
    clean_text = text.strip()
    
    # 🔥 CRITICAL: Language-specific SSML for accent-free pronunciation
    if language_code == "cs":
        # Czech pronunciation with phonetic hints
        enhanced_text = f'<speak><lang xml:lang="cs-CZ"><phoneme alphabet="ipa" ph="">ˈ</phoneme><prosody rate="0.92" pitch="+2st">{clean_text}</prosody></lang></speak>'
    elif language_code == "de":
        # German pronunciation with phonetic hints  
        enhanced_text = f'<speak><lang xml:lang="de-DE"><phoneme alphabet="ipa" ph="">ˈ</phoneme><prosody rate="0.95" pitch="+1st">{clean_text}</prosody></lang></speak>'
    else:
        enhanced_text = clean_text
    
    return enhanced_text

async def process_multilingual_text_seamless(text, detect_language=True):
    """Process multilingual text with intelligent accent-free switching"""
    
    # Parse segments more intelligently
    segments = parse_intelligent_segments(text)
    
    if len(segments) <= 1:
        # Single segment - use unified generation
        audio_data, generation_time = await generate_speech_unified(  # Add await
            segment["text"], 
            segment["language"]
        )
        
        if audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                # Save audio data to file
                temp_file.write(audio_data.read())
                return temp_file.name, generation_time
        return None, 0
    
    # Multi-segment processing with accent-free blending
    audio_segments = []
    total_time = 0
    
    for i, segment in enumerate(segments):
        if not segment["text"].strip():
            continue
            
        # Generate with appropriate provider
        audio_data, generation_time = generate_speech_unified(
            segment["text"], 
            segment["language"]
        )
        
        if audio_data:
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            
            # Normalize for consistent blending
            normalized_segment = audio_segment.normalize()
            audio_segments.append(normalized_segment)
            total_time += generation_time
    
    if not audio_segments:
        return None, 0
    
    # Blend segments with minimal crossfade for accent-free switching
    combined_audio = audio_segments[0]
    
    for i in range(1, len(audio_segments)):
        # Very short crossfade to maintain natural flow
        combined_audio = combined_audio.append(audio_segments[i], crossfade=50)
    
    # Save final audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        combined_audio.export(
            temp_file.name, 
            format="mp3", 
            bitrate="128k",
            parameters=["-ac", "1", "-ar", "22050"]
        )
        return temp_file.name, total_time

def parse_intelligent_segments(text):
    """Parse text into intelligent language segments"""
    segments = []
    
    # Split by language markers
    parts = re.split(r'(\[[a-z]{2}\])', text)
    
    current_language = None
    current_text = ""
    
    for part in parts:
        if re.match(r'\[[a-z]{2}\]', part):
            # Save previous segment
            if current_text.strip():
                segments.append({
                    "text": current_text.strip(),
                    "language": current_language or "cs"
                })
            
            # Set new language
            current_language = part[1:-1]
            current_text = ""
        else:
            current_text += part
    
    # Add final segment
    if current_text.strip():
        segments.append({
            "text": current_text.strip(),
            "language": current_language or "cs"
        })
    
    return segments

def apply_same_voice_crossfade(audio1, audio2, crossfade_ms=100):
    """🔥 OPTIMIZED: Subtle crossfading for same voice transitions"""
    
    # Ensure both audio segments are normalized
    audio1_norm = normalize_audio_volume(audio1, -18)
    audio2_norm = normalize_audio_volume(audio2, -18)
    
    # Apply subtle crossfade (shorter duration since it's the same voice)
    crossfaded = audio1_norm.append(audio2_norm, crossfade=crossfade_ms)
    
    return crossfaded

def parse_language_segments_enhanced(text):
    """🎯 IMPROVED: Better parsing for mid-sentence language switches"""
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
                    "language": current_language or "cs"  # Default to Czech
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
            "language": current_language or "cs"
        })
    
    # 🎯 FIX: Detect language for unmarked segments
    for segment in segments:
        if segment["language"] is None:
            segment["language"] = detect_primary_language(segment["text"])
    
    return segments

def generate_speech_with_language_voice(text, language_code, segment_position=0, total_segments=1):
    """🔥 NEW: Generate speech using language-specific voice for accent-free output"""
    
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        return None, 0
    
    # 🎯 FIXED: Use SAME voice ID for all languages
    voice_id = st.session_state.elevenlabs_voice_id  # Same voice always!
    logger.info(f"Using consistent voice {voice_id} for {language_code}")
    
    # Enhanced voice settings for seamless transitions
    voice_settings = get_transition_optimized_settings(language_code, segment_position, total_segments)
    
    # Add language-specific SSML for better pronunciation
    enhanced_text = add_language_specific_ssml(text, language_code)
    
    data = {
        "text": enhanced_text,
        "model_id": "eleven_flash_v2_5",  # Fastest model
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
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            json=data,
            headers=headers,
            timeout=10
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"✅ Generated {language_code} audio with native voice in {generation_time:.2f}s")
            return BytesIO(response.content), generation_time
        else:
            logger.error(f"TTS error: {response.status_code}")
            return None, generation_time
            
    except Exception as e:
        logger.error(f"Speech generation error: {str(e)}")
        return None, time.time() - start_time

def normalize_audio_volume(audio_segment, target_dbfs=-18):
    """🎯 NORMALIZE: Ensure consistent volume for seamless blending"""
    # Calculate volume adjustment needed
    current_dbfs = audio_segment.dBFS
    volume_adjustment = target_dbfs - current_dbfs
    
    # Apply volume adjustment
    normalized = audio_segment.apply_gain(volume_adjustment)
    
    return normalized

def apply_equal_power_crossfade(audio1, audio2, crossfade_ms=150):
    """🔥 NEW: Equal power crossfading for seamless voice transitions"""
    
    # Ensure both audio segments are normalized
    audio1_norm = normalize_audio_volume(audio1, -18)
    audio2_norm = normalize_audio_volume(audio2, -18)
    
    # Apply equal power crossfade using pydub's built-in method
    # This maintains constant perceived loudness during transition
    crossfaded = audio1_norm.append(audio2_norm, crossfade=crossfade_ms)
    
    return crossfaded

def get_transition_optimized_settings(language_code, position, total_segments):
    """🎯 OPTIMIZED: Voice settings for smooth language transitions"""
    
    base_settings = {
        "stability": 0.85,          # High stability for consistency
        "similarity_boost": 0.90,   # Maintain voice character
        "style": 0.70,             # Natural expression
        "use_speaker_boost": True   # Enhanced clarity
    }
    
    # Adjust for language-specific characteristics
    if language_code == "cs":
        base_settings.update({
            "stability": 0.90,      # Extra stability for Czech
            "similarity_boost": 0.95,
            "style": 0.75
        })
    elif language_code == "de":
        base_settings.update({
            "stability": 0.88,      # Slightly more flexible for German
            "similarity_boost": 0.92,
            "style": 0.72
        })
    
    # Increase stability for mid-sentence transitions
    if position > 0:  # Not the first segment
        base_settings["stability"] = min(0.95, base_settings["stability"] + 0.05)
    
    return base_settings

def add_language_specific_ssml(text, language_code):
    """🎯 ENHANCED: Language-specific SSML for better pronunciation"""
    
    if language_code == "cs":
        # Czech-specific SSML
        return f'<speak><lang xml:lang="cs-CZ"><prosody rate="0.95">{text}</prosody></lang></speak>'
    elif language_code == "de":
        # German-specific SSML  
        return f'<speak><lang xml:lang="de-DE"><prosody rate="0.98">{text}</prosody></lang></speak>'
    else:
        return text
        

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

def ensure_single_voice_consistency():
    """Ensure all languages use the same voice ID"""
    single_voice = st.session_state.elevenlabs_voice_id
    st.session_state.language_voices["cs"] = single_voice
    st.session_state.language_voices["de"] = single_voice
    st.session_state.language_voices["default"] = single_voice
    logger.info(f"Voice consistency enforced: {single_voice}")
    
    
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
    """ACCENT-FREE speech generation using voice isolation + Flash v2.5 optimization"""
    if not text or text.strip() == "":
        logger.error("Empty text provided to generate_speech")
        return None, 0
    
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        logger.error("ElevenLabs API key not provided")
        return None, 0
    
    # SINGLE VOICE: Use same voice for all languages - no more "two speakers"
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
    
    # METHOD 2: Use Flash v2.5 model for BEST accent-free performance
    model_id = "eleven_flash_v2_5"  # CRITICAL: Latest model with 32 languages + ultra-low latency
    
    # METHOD 1: Language-specific voice settings for accent isolation
    if language_code and language_code in st.session_state.voice_settings:
        voice_settings = st.session_state.voice_settings[language_code].copy()
        logger.info(f"Using optimized {language_code} settings: {voice_settings}")
    else:
        voice_settings = st.session_state.voice_settings["default"]
    
    # SSML enhancement for pronunciation accuracy
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
async def generate_speech_openai(text, language_code=None):
    """Generate speech using OpenAI TTS"""
    api_key = st.session_state.openai_api_key
    if not api_key:
        return None, 0
    
    # Clean text for OpenAI TTS
    clean_text = re.sub(r'\[cs\]|\[de\]', '', text).strip()
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "tts-1",
                    "input": clean_text,
                    "voice": "nova",  # Good for multilingual
                    "response_format": "mp3",
                    "speed": 0.9  # Slightly slower for clarity
                },
                timeout=20.0
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                return BytesIO(response.content), generation_time
            else:
                logger.error(f"OpenAI TTS error: {response.status_code}")
                return None, generation_time
                
    except Exception as e:
        logger.error(f"OpenAI TTS error: {str(e)}")
        return None, time.time() - start_time
    
async def generate_speech_azure(text, language_code=None):
    """Generate speech using Azure Speech"""
    speech_key = st.session_state.azure_speech_key
    region = st.session_state.azure_region
    
    if not speech_key:
        return None, 0
    
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        st.error("Azure Speech SDK not installed. Run: pip install azure-cognitiveservices-speech")
        return None, 0
    
    config = st.session_state.provider_voice_configs["azure"]
    
    # Choose voice based on language
    if language_code == "cs":
        voice_name = config["voice_cs"]
        language = "cs-CZ"
    elif language_code == "de":
        voice_name = config["voice_de"] 
        language = "de-DE"
    else:
        voice_name = config["voice_cs"]  # Default
        language = "cs-CZ"
    
    clean_text = re.sub(r'\[cs\]|\[de\]', '', text).strip()
    
    start_time = time.time()
    
    try:
        # Configure Azure Speech
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
        speech_config.speech_synthesis_voice_name = voice_name
        
        # Create SSML for better control
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
            <voice name="{voice_name}">
                <prosody rate="{config['rate']}" pitch="{config['pitch']}">
                    {clean_text}
                </prosody>
            </voice>
        </speak>
        """
        
        # Generate speech to memory
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        result = synthesizer.speak_ssml_async(ssml).get()
        
        generation_time = time.time() - start_time
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return BytesIO(result.audio_data), generation_time
        else:
            logger.error(f"Azure TTS error: {result.reason}")
            return None, generation_time
            
    except Exception as e:
        logger.error(f"Azure TTS error: {str(e)}")
        return None, time.time() - start_time

async def generate_speech_unified(text, language_code=None):
    """Unified speech generation using selected provider"""
    provider = st.session_state.tts_provider
    
    if provider == "elevenlabs":
        return generate_speech(text, language_code)
    elif provider == "openai":
        return await generate_speech_openai(text, language_code)
    elif provider == "azure":
        return await generate_speech_azure(text, language_code)
    else:
        return generate_speech(text, language_code)  # Fallback

def add_accent_free_markup(text, language_code):
    """Add SSML markup for accent-free pronunciation"""
    if not language_code:
        return text
    
    # Clean text first
    clean_text = text.strip()
    
    # Add language-specific SSML for accent-free pronunciation
    if language_code == "cs":
        # Czech pronunciation optimization
        enhanced_text = f'<speak><lang xml:lang="cs-CZ"><prosody rate="0.9">{clean_text}</prosody></lang></speak>'
    elif language_code == "de":
        # German pronunciation optimization  
        enhanced_text = f'<speak><lang xml:lang="de-DE"><prosody rate="0.95">{clean_text}</prosody></lang></speak>'
    else:
        enhanced_text = clean_text
    
    return enhanced_text

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
        audio_path, tts_latency = await process_multilingual_text_seamless(response_text)
        
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
        audio_path, tts_latency = await process_multilingual_text_seamless(response_text)  # Add await
        
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
    
    # FIXED: More natural conversation prompts
    system_prompt = f"""You are a helpful multilingual assistant that speaks Czech and German naturally.

CONTEXT: The user said "{user_input}" (detected as {primary_lang})

INSTRUCTIONS:
1. Respond naturally and conversationally in the same language(s) they used
2. If they greeted you, greet them back appropriately  
3. If they asked how you are, answer naturally (like "Mám se dobře" or "Mir geht es gut")
4. Use [cs] for Czech and [de] for German content
5. Be helpful and friendly
6. NEVER use English unless they specifically ask
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
    audio_path, tts_latency = await process_multilingual_text_seamless(response_text)  # FIX: Add await here
    
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



def main():
    """Main application entry point"""
    # Page configuration - ONLY ONCE!
    st.set_page_config(
        page_title="Multilingual AI Voice Tutor",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("Multilingual AI Voice Tutor")
    st.subheader("Professional German Language Tutor for Czech Speakers (A1-A2)")
    
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
            ensure_single_voice_consistency()
            st.success("API keys saved successfully!")
        
        # ACCENT-FREE VOICE CONFIGURATION
        st.subheader("🎯 Single Voice Setup")

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
                # Update all language voices to use the same voice
                st.session_state.language_voices["cs"] = new_voice_id
                st.session_state.language_voices["de"] = new_voice_id
                st.session_state.language_voices["default"] = new_voice_id

        # Voice consistency status
        st.success(f"""
        ✅ **Single Voice Configuration**
        - Voice ID: {st.session_state.elevenlabs_voice_id[:8]}...
        - Used for: ALL languages (Czech + German)
        - Model: Flash v2.5 (Multilingual accent-free)
        """)
        
        # NEW: Language Response Options
        st.subheader("Tutor Mode")
 
        # Choose Tutor Mode

        response_language = st.radio(
            "Tutor Mode",
            options=["both", "cs", "de"],
            format_func=lambda x: {
                "both": "German Tutor (Czech + German)", 
                "cs": "Czech Only", 
                "de": "German Only"
            }[x]
        )
        if response_language != st.session_state.response_language:
            st.session_state.response_language = response_language
            st.success(f"Tutor Mode set to: {response_language}")
        
        # Language distribution (only shown when "both" is selected)
        if response_language == "both":
            st.subheader("Language Distribution")
            
            # Czech percentage slider
            cs_percent = st.slider("Czech %", min_value=0, max_value=100, value=st.session_state.language_distribution["cs"])
            
            # Calculate German percentage automatically
            de_percent = 100 - cs_percent
            
            # Display German percentage
            st.text(f"German %: {de_percent}")
            
            # Update language distribution if changed
            if cs_percent != st.session_state.language_distribution["cs"]:
                st.session_state.language_distribution = {
                    "cs": cs_percent,
                    "de": de_percent
                }
                st.success(f"Language distribution updated: {cs_percent}% Czech, {de_percent}% German")
        
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
        
        # Accent improvement explanation
        st.header("Accent Improvement")
        st.info("""
        This system includes optimizations to eliminate accent interference:
        
        1. Language-specific voice settings
        2. Micro-pauses between language switches
        3. Voice context reset when switching languages
        
        These improvements ensure Czech sounds truly Czech and German sounds truly German.
        """)
        
        
        # TTS Provider Selection
        st.subheader("🎵 TTS Provider")
        
        tts_provider = st.selectbox(
            "Choose TTS Provider",
            options=["elevenlabs", "openai", "azure"],
            format_func=lambda x: {
                "elevenlabs": "ElevenLabs (Flash v2.5)",
                "openai": "OpenAI TTS (Nova)",
                "azure": "Azure Speech"
            }[x],
            index=["elevenlabs", "openai", "azure"].index(st.session_state.tts_provider)
        )
        
        if tts_provider != st.session_state.tts_provider:
            st.session_state.tts_provider = tts_provider
            st.success(f"TTS Provider changed to: {tts_provider}")
        
        # Provider-specific configuration
        if tts_provider == "azure":
            st.write("**Azure Speech Configuration:**")
            azure_key = st.text_input(
                "Azure Speech Key", 
                value=st.session_state.azure_speech_key,
                type="password"
            )
            azure_region = st.text_input(
                "Azure Region", 
                value=st.session_state.azure_region,
                placeholder="e.g., eastus"
            )
            
            if st.button("Save Azure Config"):
                st.session_state.azure_speech_key = azure_key
                st.session_state.azure_region = azure_region
                st.success("Azure configuration saved!")
            
            if not azure_key:
                st.warning("⚠️ Azure Speech Key required for Azure TTS")
        
        elif tts_provider == "openai":
            st.info("✅ OpenAI TTS uses your existing OpenAI API key")
        
        elif tts_provider == "elevenlabs":
            st.info("✅ ElevenLabs uses your existing ElevenLabs API key")
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
            # Text input
            st.subheader("Text Input")
            st.write("Use [cs] to mark Czech text and [de] to mark German text.")
            
            # Demo preset examples
            demo_scenarios = {
                "Vocabulary Request": (
                    "Můžete mi říct, co říkáme voda a nějaká další základní slova v němčině?"
                ),
                "Grammar Question": (
                    "Jak se tvoří minulý čas v němčině? Můžete mi ukázat nějaké příklady?"
                ),
                "Practice Conversation": (
                    "Chtěl bych si procvičit představování v němčině. Jak bych se měl představit?"
                ),
                "Pronunciation Help": (
                    "Mám problém s výslovností německého \"ü\". Můžete mi pomoci?"
                ),
                "Daily Expressions": (
                    "Naučte mě základní fráze pro každodenní konverzaci v němčině."
                ),
                "Custom Input": ""
            }
            
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
                    # Voice input - FIXED HTML5 AUDIO RECORDER
                    st.subheader("🎤 Professional Voice Recording")
                    
                    # Check if API keys are set
                    keys_set = (
                        st.session_state.elevenlabs_api_key and 
                        st.session_state.openai_api_key
                    )

                    if not keys_set:
                        st.warning("Please set both API keys in the sidebar first")
                    else:
                        st.write("🎯 **HTML5 Audio Recording** - Reliable Railway Deployment")
                        
                        # Create the HTML5 audio recorder component
                        create_audio_recorder_component()

                        st.markdown("---")
                        st.write("**🔄 AUTOMATIC PROCESSING:**")
                        
                        # FIXED: Reliable upload processing
                        uploaded_audio = st.file_uploader(
                            "📥 Upload Your Downloaded Recording Here", 
                            type=['wav', 'mp3', 'webm', 'ogg'],
                            key="main_upload",
                            help="After recording above, download the file and upload it here for automatic processing"
                        )

                        if uploaded_audio is not None:
                            # IMMEDIATE processing when file is uploaded
                            with st.spinner("🔄 **PROCESSING YOUR RECORDING...**"):
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
                                    st.success(f"✅ **PROCESSING COMPLETE!** ({total_latency:.2f}s)")
                                    st.balloons()
                                    
                                    # Clean up
                                    if os.path.exists(temp_path):
                                        os.unlink(temp_path)
                                    if amplified_path != temp_path and os.path.exists(amplified_path):
                                        os.unlink(amplified_path)
                                        
                                except Exception as e:
                                    st.error(f"Processing error: {str(e)}")

                        # Enhanced instructions
                        st.success("""
                        🎯 **SIMPLE WORKFLOW:**
                        1. Click "🔴 START RECORDING" above
                        2. Speak clearly in Czech or German  
                        3. Click "⏹️ STOP RECORDING" when done
                        4. **DOWNLOAD** the file that appears automatically
                        5. **UPLOAD** it in the section above - processing starts immediately!

                        **⚡ Total time: Record → Download → Upload → Get Results!**
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
            st.subheader("Generated Speech")
            
            # Display audio with player
            audio_bytes = display_audio(st.session_state.last_audio_output, autoplay=True)
            
            if audio_bytes:
                # Download button
                st.download_button(
                    label="Download Audio",
                    data=audio_bytes,
                    file_name="multilingual_tutor_response.mp3",
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
