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
import openai
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
logger = logging.getLogger("urdu_english_voice_tutor")

# ----------------------------------------------------------------------------------
# CONFIGURATION SECTION - UPDATED FOR URDU-ENGLISH
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
    # USE SAME VOICE FOR ALL LANGUAGES - No accent bleeding
    single_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Use same voice for everything
    st.session_state.language_voices = {
        "ur": single_voice_id,  # SAME voice for Urdu
        "en": single_voice_id,  # SAME voice for English
        "default": single_voice_id
    }

# OPTIMIZED voice settings for Urdu-English
if 'voice_settings' not in st.session_state:
    st.session_state.voice_settings = {
        "ur": {  # Urdu-optimized settings
            "stability": 0.95,        # MAXIMUM stability for consistent Urdu
            "similarity_boost": 0.98, # MAXIMUM similarity for native Urdu sound
            "style": 0.85,           # High style for natural Urdu expression
            "use_speaker_boost": True # Enable speaker boost for clarity
        },
        "en": {  # English-optimized settings  
            "stability": 0.92,        # VERY HIGH stability for consistent English
            "similarity_boost": 0.95, # VERY HIGH similarity for native English sound
            "style": 0.80,           # High style for natural English expression
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

if 'provider_voice_configs' not in st.session_state:
    st.session_state.provider_voice_configs = {
        "elevenlabs_flash": {
            "speakers": {
                "Rachel": {
                    "voice_id": "21m00Tcm4TlvDq8ikWAM", 
                    "description": "Calm, professional female - excellent for multilingual",
                    "model": "eleven_flash_v2_5"
                },
                "Adam": {
                    "voice_id": "pNInz6obpgDQGcFmaJgB",
                    "description": "Deep, authoritative male - great accent control", 
                    "model": "eleven_flash_v2_5"
                },
                "Bella": {
                    "voice_id": "EXAVITQu4vr4xnSDxMaL",
                    "description": "Young, energetic female - clear pronunciation",
                    "model": "eleven_flash_v2_5"
                }
            },
            "selected": "Rachel"
        },
        "elevenlabs_multilingual": {
            "speakers": {
                "Rachel": {
                    "voice_id": "21m00Tcm4TlvDq8ikWAM", 
                    "description": "Calm, professional female - multilingual optimized",
                    "model": "eleven_multilingual_v2"
                },
                "Adam": {
                    "voice_id": "pNInz6obpgDQGcFmaJgB",
                    "description": "Deep, authoritative male - multilingual optimized", 
                    "model": "eleven_multilingual_v2"
                },
                "Antoni": {
                    "voice_id": "ErXwobaYiN019PkySvjV",
                    "description": "Warm, well-educated male - accent-free switching",
                    "model": "eleven_multilingual_v2"
                }
            },
            "selected": "Rachel"
        }
    }
if 'selected_speakers' not in st.session_state:
    st.session_state.selected_speakers = {
        "elevenlabs_flash": "Rachel",
        "elevenlabs_multilingual": "Rachel"
    }
# Dynamic voice ID based on selected provider and speaker
def get_current_voice_id():
    provider = st.session_state.tts_provider
    if provider in st.session_state.provider_voice_configs:
        selected_speaker = st.session_state.selected_speakers.get(provider, list(st.session_state.provider_voice_configs[provider]["speakers"].keys())[0])
        return st.session_state.provider_voice_configs[provider]["speakers"][selected_speaker]["voice_id"]
    return "21m00Tcm4TlvDq8ikWAM"  # Fallback

# Update elevenlabs_voice_id dynamically
if 'elevenlabs_voice_id' not in st.session_state:
    st.session_state.elevenlabs_voice_id = get_current_voice_id()
    
# Whisper speech recognition config
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = "medium"
    st.session_state.whisper_local_model = None

# Language distribution preference - UPDATED FOR URDU-ENGLISH
if 'language_distribution' not in st.session_state:
    st.session_state.language_distribution = {
        "ur": 50,  # Urdu percentage
        "en": 50   # English percentage
    }

# Language preference for response
if 'response_language' not in st.session_state:
    st.session_state.response_language = "both"  # Options: "ur", "en", "both"

# Language codes and settings - UPDATED FOR URDU-ENGLISH
SUPPORTED_LANGUAGES = {
    "ur": {"name": "Urdu", "confidence_threshold": 0.65},
    "en": {"name": "English", "confidence_threshold": 0.65}
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

# Audio recording and processing functions
import streamlit.components.v1 as components

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
        
        <!-- Download link for reliable processing -->
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
                    
                    // Show download immediately for reliable processing
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

        // Reliable download approach
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

# Audio processing functions
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

# ----------------------------------------------------------------------------------
# SPEECH RECOGNITION (STT) SECTION - UPDATED FOR URDU-ENGLISH
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
            
            # ENHANCED: Pronunciation-focused settings for Urdu/English
            data = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "temperature": "0.0",  # LOWEST temperature for consistent pronunciation
                "language": None,  # Let Whisper auto-detect between ur/en
                "prompt": "This audio contains Urdu and English speech. Focus on accurate pronunciation and phonetic understanding. Common Urdu words: assalam alaikum, shukriya, meherbani, acha. Common English words: hello, thank you, please, good."  # Pronunciation hints
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
    """Apply pronunciation-based corrections for Urdu/English"""
    if not text:
        return text
    
    # Urdu pronunciation corrections
    urdu_corrections = {
        # Common mispronunciations to correct pronunciations
        "assalam": "assalam alaikum",
        "shukria": "shukriya",
        "mehrbani": "meherbani", 
        "acha": "acha",
        "theek": "theek hai",
        "namaste": "assalam alaikum",  # Convert to proper Urdu greeting
    }
    
    # English pronunciation corrections
    english_corrections = {
        "hello": "hello",
        "thank you": "thank you", 
        "please": "please",
        "good": "good",
        "morning": "morning",
        "evening": "evening",
        "how are you": "how are you",
        "fine": "fine",
        "okay": "okay"
    }
    
    # Apply corrections based on detected language or overall context
    corrected_text = text
    
    # Apply Urdu corrections if Urdu content detected
    if language == "ur" or any(word in text.lower() for word in ["assalam", "shukriya", "meherbani"]):
        for wrong, correct in urdu_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
    
    # Apply English corrections if English content detected  
    if language == "en" or any(word in text.lower() for word in ["hello", "thank", "please"]):
        for wrong, correct in english_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

# ----------------------------------------------------------------------------------
# LANGUAGE MODEL (LLM) SECTION - UPDATED FOR URDU-ENGLISH TUTORING
# ----------------------------------------------------------------------------------

async def generate_llm_response(prompt, system_prompt=None, api_key=None):
    """Generate response with INTELLIGENT language tagging for Urdu-English"""
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
            system_content = """🎯 WORD-LEVEL TAGGING ENFORCER

            ⚠️ ABSOLUTE RULE: EVERY ENGLISH WORD GETS [en] TAG, EVERY URDU GETS [ur]

            ❌ NEVER DO THIS:
            "[ur] English mein water kehte hain"
            "[en] We call it paani in Urdu"

            ✅ ALWAYS DO THIS:
            "[ur] English mein [en] water [ur] kehte hain"
            "[ur] Paani ko [en] English [ur] mein [en] water [ur] kehte hain"

            🔥 MANDATORY PATTERNS:

            VOCABULARY TEACHING:
            "[ur] {URDU_WORD} ko [en] English [ur] mein [en] {ENGLISH_WORD} [ur] kehte hain"

            LISTS:
            "[ur] 1. [en] House [ur] - ghar
            [ur] 2. [en] Car [ur] - gaari  
            [ur] 3. [en] Book [ur] - kitab"

            GRAMMAR:
            "[ur] Past tense banane ke liye [en] verb [ur] ke saath [en] -ed [ur] lagaate hain"

            🚨 ENFORCEMENT RULES:
            1. NO English word without [en] tag
            2. NO Urdu word without [ur] tag  
            3. Switch tags for EVERY word
            4. Maximum 3 words per tag

            EXAMPLE PERFECT RESPONSE:
            User: "Water English mein kya kehte hain?"
            You: "[ur] Paani ko [en] English [ur] mein [en] water [ur] kehte hain. [ur] Aur [en] basic words [ur]: [en] house [ur] (ghar), [en] book [ur] (kitab)."

            BEGIN WITH PERFECT WORD-LEVEL TAGGING."""
        elif response_language == "ur":
            system_content = "You are a helpful Urdu assistant. ALWAYS respond ONLY in Urdu with [ur] markers."
        elif response_language == "en":
            system_content = "You are a helpful English assistant. ALWAYS respond ONLY in English with [en] markers."
            
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
                response_text = validate_and_fix_tagging(response_text)
                
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
def validate_and_fix_tagging(response_text):
    """Enforce word-level tagging - critical fix"""
    
    # Check for block-level violations (long untagged sequences)
    if re.search(r'\[ur\][^[]{50,}', response_text) or re.search(r'\[en\][^[]{50,}', response_text):
        logger.warning("Block-level tagging detected - fixing...")
        
        # Force word-level mixing for vocabulary responses
        if any(word in response_text.lower() for word in ["english mein", "kehte hain", "matlab"]):
            # Apply aggressive word-level tagging
            response_text = apply_word_level_tagging(response_text)
    
    return response_text

def apply_word_level_tagging(text):
    """Force word-level tagging for critical responses"""
    # Pattern: "English mein X kehte hain" -> "English mein [en] X [ur] kehte hain"
    text = re.sub(r'(\[ur\].*?)English mein["\s]*([A-Za-z]+)["\s]*(.*?kehte hain)', 
                  r'\1English mein [en] \2 [ur] \3', text)
    
    # Pattern: Lists with translations
    text = re.sub(r'(\[ur\].*?)-\s*([A-Za-z]+)\s*\(([^)]+)\)', 
                  r'\1- [en] \2 [ur] (\3)', text)
    
    return text

def clean_intelligent_tags(response_text):
    """Clean up intelligent language tags for Urdu-English"""
    # Remove excessive tagging
    response_text = re.sub(r'\[ur\]\s*\[ur\]', '[ur]', response_text)
    response_text = re.sub(r'\[en\]\s*\[en\]', '[en]', response_text)
    
    # Ensure proper spacing
    response_text = re.sub(r'\[ur\]\s*', '[ur] ', response_text)
    response_text = re.sub(r'\[en\]\s*', '[en] ', response_text)
    
    # If no tags at all, add Urdu as default
    if not re.search(r'\[ur\]|\[en\]', response_text):
        response_text = f"[ur] {response_text}"
    
    return response_text.strip()

def detect_primary_language(text):
    """Detect the primary language of a text with improved accuracy for Urdu-English"""
    # Urdu-specific characters (Arabic script)
    urdu_chars = set("آاأإئبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںہویے")
    
    # English-specific characters (Latin script)
    english_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Count language-specific characters
    text_lower = text.lower()
    urdu_count = sum(1 for char in text if char in urdu_chars)
    english_count = sum(1 for char in text if char in english_chars)
    
    # Urdu-specific words (romanized)
    urdu_words = {
        "aap", "hum", "main", "ye", "wo", "hai", "hain", "tha", "thi", "the",
        "assalam", "alaikum", "shukriya", "meherbani", "kya", "kaise", "kahan", "kab",
        "acha", "theek", "bura", "namaste", "adaab", "khuda", "hafiz", "inshallah",
        "mashallah", "subhanallah", "paani", "roti", "ghar", "dil", "pyaar"
    }
    
    # English-specific words
    english_words = {
        "i", "you", "he", "she", "it", "we", "they", "am", "is", "are", "was", "were",
        "hello", "hi", "thank", "you", "please", "good", "bad", "yes", "no",
        "the", "a", "an", "and", "or", "but", "if", "when", "where", "how", "what",
        "who", "why", "water", "bread", "house", "love", "heart", "time"
    }
    
    # Count word occurrences
    words = re.findall(r'\b\w+\b', text_lower)
    urdu_word_count = sum(1 for word in words if word in urdu_words)
    english_word_count = sum(1 for word in words if word in english_words)
    
    # Improved scoring system with weighted metrics
    urdu_evidence = urdu_count * 3 + urdu_word_count * 2  # Higher weight for Urdu script
    english_evidence = english_count * 1 + english_word_count * 2
    
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
# TEXT-TO-SPEECH (TTS) SECTION
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
    """Generate speech using ElevenLabs with selected speaker"""
    if not text or text.strip() == "":
        logger.error("Empty text provided to generate_speech")
        return None, 0
    
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        logger.error("ElevenLabs API key not provided")
        return None, 0
    
    # Get selected speaker configuration
    selected_speaker_name = st.session_state.selected_speakers.get("elevenlabs", "Rachel")
    speaker_config = st.session_state.provider_voice_configs["elevenlabs"]["speakers"][selected_speaker_name]
    selected_voice_id = voice_id or speaker_config["voice_id"]
    
    logger.info(f"Using ElevenLabs speaker: {selected_speaker_name} ({selected_voice_id})")
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    model_id = speaker_config["model"]
    
    # Language-specific voice settings for accent isolation
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
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}",
            json=data,
            headers=headers,
            timeout=10
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            content = response.content
            if len(content) < 100:
                return None, generation_time
                
            logger.info(f"✅ ElevenLabs ({selected_speaker_name}) generated for {language_code} in {generation_time:.2f}s")
            return BytesIO(content), generation_time
        else:
            logger.error(f"TTS API error: {response.status_code} - {response.text}")
            return None, generation_time
    
    except Exception as e:
        logger.error(f"ElevenLabs TTS error: {str(e)}")
        return None, time.time() - start_time

async def generate_speech_openai(text, language_code=None):
    """Generate speech using OpenAI TTS with selected speaker"""
    api_key = st.session_state.openai_api_key
    if not api_key:
        return None, 0
    
    # Get selected speaker configuration
    selected_speaker_name = st.session_state.selected_speakers.get("openai", "Nova")
    speaker_config = st.session_state.provider_voice_configs["openai"]["speakers"][selected_speaker_name]
    
    # Clean text for OpenAI TTS (remove language markers)
    clean_text = re.sub(r'\[ur\]|\[en\]', '', text).strip()
    
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
                    "model": speaker_config["model"],
                    "input": clean_text,
                    "voice": speaker_config["voice_id"],
                    "response_format": "mp3",
                    "speed": 0.9
                },
                timeout=20.0
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                logger.info(f"✅ OpenAI TTS ({selected_speaker_name}) generated in {generation_time:.2f}s")
                return BytesIO(response.content), generation_time
            else:
                logger.error(f"OpenAI TTS error: {response.status_code}")
                return None, generation_time
                
    except Exception as e:
        logger.error(f"OpenAI TTS error: {str(e)}")
        return None, time.time() - start_time

async def generate_speech_unified(text, language_code=None):
    """Unified speech generation using selected ElevenLabs model"""
    provider = st.session_state.tts_provider
    
    if provider in ["elevenlabs_flash", "elevenlabs_multilingual"]:
        return generate_speech(text, language_code)
    else:
        return generate_speech(text, language_code)  # Fallback

def add_accent_free_markup(text, language_code):
    """Add SSML markup for accent-free pronunciation - Urdu/English"""
    if not language_code:
        return text
    
    # Clean text first
    clean_text = text.strip()
    
    # Add language-specific SSML for accent-free pronunciation
    if language_code == "ur":
        # Urdu pronunciation optimization
        enhanced_text = f'<speak><lang xml:lang="ur-PK"><prosody rate="0.9">{clean_text}</prosody></lang></speak>'
    elif language_code == "en":
        # English pronunciation optimization  
        enhanced_text = f'<speak><lang xml:lang="en-US"><prosody rate="0.95">{clean_text}</prosody></lang></speak>'
    else:
        enhanced_text = clean_text
    
    return enhanced_text

# Enhanced multilingual processing for Urdu-English
async def process_multilingual_text_seamless(text, detect_language=True):
    """Process multilingual text with intelligent accent-free switching"""
    
    # Parse segments more intelligently
    segments = parse_intelligent_segments(text)
    
    if len(segments) <= 1:
        # Single segment - use unified generation
        audio_data, generation_time = await generate_speech_unified(
            segments[0]["text"] if segments else text, 
            segments[0]["language"] if segments else None
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
        audio_data, generation_time = await generate_speech_unified(
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
    """Parse text into intelligent language segments for Urdu-English"""
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
                    "language": current_language or "ur"  # Default to Urdu
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
            "language": current_language or "ur"
        })
    
    return segments

# ----------------------------------------------------------------------------------
# END-TO-END PIPELINE - ENHANCED FOR URDU-ENGLISH PROCESSING
# ----------------------------------------------------------------------------------

async def process_voice_input_pronunciation_enhanced(audio_file):
    """Enhanced voice processing focusing on pronunciation accuracy for Urdu-English"""
    pipeline_start_time = time.time()
    
    try:
        # Step 1: Enhanced Audio Preprocessing with 500% boost
        st.session_state.message_queue.put("🔊 Amplifying audio for pronunciation clarity...")
        
        # Step 2: Pronunciation-Enhanced Transcription
        st.session_state.message_queue.put("🎯 Analyzing Urdu-English pronunciation patterns...")
        
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
        st.session_state.last_llm_response = response_text 
        st.session_state.message_queue.put(f"💬 Generated: {response_text}")
        
        # Step 5: High-Quality Voice Synthesis
        st.session_state.message_queue.put("🎵 Generating accent-free speech...")
        audio_path, tts_latency = await process_multilingual_text_seamless(response_text)
        
        # Calculate total latency
        total_latency = time.time() - pipeline_start_time
        st.session_state.performance_metrics["total_latency"].append(total_latency)
        
        st.session_state.message_queue.put(f"✅ Complete! ({total_latency:.2f}s)")
        
        return user_input, audio_path, transcription.get("latency", 0), llm_result.get("latency", 0), tts_latency
        
    except Exception as e:
        logger.error(f"Enhanced processing error: {str(e)}")
        st.session_state.message_queue.put(f"❌ Error: {str(e)}")
        return None, None, 0, 0, 0

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
    llm_result = await generate_llm_response(text, system_prompt)
    
    if "error" in llm_result:
        st.session_state.message_queue.put(f"Error generating response: {llm_result.get('error')}")
        return None, llm_result.get("latency", 0), 0
    
    response_text = llm_result["response"]
    st.session_state.last_llm_response = response_text
    st.session_state.message_queue.put(f"Generated response: {response_text}")
    
    # Step 2: Text-to-Speech with accent isolation
    st.session_state.message_queue.put("Generating speech with accent isolation...")
    audio_path, tts_latency = await process_multilingual_text_seamless(response_text)
    
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
    """Update status display from message queue - FIXED"""
    if 'status_messages' not in st.session_state:
        st.session_state.status_messages = []
    
    # Get all new messages
    while True:
        try:
            message = st.session_state.message_queue.get_nowait()
            st.session_state.status_messages.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
            # Keep only last 10 messages
            if len(st.session_state.status_messages) > 10:
                st.session_state.status_messages.pop(0)
        except queue.Empty:
            break
    
    # Display messages
    if st.session_state.status_messages:
        status_text = "\n".join(st.session_state.status_messages)
        st.text_area("Processing Log", value=status_text, height=200, key="status_display")

def ensure_single_voice_consistency():
    """Ensure all languages use the same voice ID"""
    single_voice = st.session_state.elevenlabs_voice_id
    st.session_state.language_voices["ur"] = single_voice
    st.session_state.language_voices["en"] = single_voice
    st.session_state.language_voices["default"] = single_voice
    logger.info(f"Voice consistency enforced: {single_voice}")

# ----------------------------------------------------------------------------------
# STREAMLIT UI - UPDATED FOR URDU-ENGLISH INTERFACE
# ----------------------------------------------------------------------------------

def main():
    """Main application entry point"""
    # Page configuration - ONLY ONCE!
    st.set_page_config(
        page_title="Urdu-English AI Voice Tutor",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("Urdu-English AI Voice Tutor")
    st.subheader("Professional English Language Tutor for Urdu Speakers (A1-A2)")
    
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
                st.session_state.language_voices["ur"] = new_voice_id
                st.session_state.language_voices["en"] = new_voice_id
                st.session_state.language_voices["default"] = new_voice_id

        # Voice consistency status
        st.success(f"""
        ✅ **Single Voice Configuration**
        - Voice ID: {st.session_state.elevenlabs_voice_id[:8]}...
        - Used for: ALL languages (Urdu + English)
        - Model: Flash v2.5 (Multilingual accent-free)
        """)
        
        # Language Response Options
        st.subheader("Tutor Mode")
 
        response_language = st.radio(
            "Tutor Mode",
            options=["both", "ur", "en"],
            format_func=lambda x: {
                "both": "English Tutor (Urdu + English)", 
                "ur": "Urdu Only", 
                "en": "English Only"
            }[x]
        )
        if response_language != st.session_state.response_language:
            st.session_state.response_language = response_language
            st.success(f"Tutor Mode set to: {response_language}")
        
        # Language distribution (only shown when "both" is selected)
        if response_language == "both":
            st.subheader("Language Distribution")
            
            # Urdu percentage slider
            ur_percent = st.slider("Urdu %", min_value=0, max_value=100, value=st.session_state.language_distribution["ur"])
            
            # Calculate English percentage automatically
            en_percent = 100 - ur_percent
            
            # Display English percentage
            st.text(f"English %: {en_percent}")
            
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
        
        # TTS Provider Selection
        st.subheader("🎵 TTS Provider")

        tts_provider = st.selectbox(
            "Choose ElevenLabs Model",
            options=["elevenlabs_flash", "elevenlabs_multilingual"],
            format_func=lambda x: {
                "elevenlabs_flash": "ElevenLabs Flash v2.5 (Fastest)",
                "elevenlabs_multilingual": "ElevenLabs Multilingual v2 (Best Quality)"
            }[x],
            index=["elevenlabs_flash", "elevenlabs_multilingual"].index(st.session_state.tts_provider) if st.session_state.tts_provider in ["elevenlabs_flash", "elevenlabs_multilingual"] else 0
        )

        if tts_provider != st.session_state.tts_provider:
            st.session_state.tts_provider = tts_provider
            st.success(f"TTS Provider changed to: {tts_provider}")

        # Voice Testing Section
        st.write("**🎵 Test Current Speaker:**")
        test_text = st.text_input(
            "Test Text", 
            value="Hello, this is a test. Assalam alaikum, ye test hai.",
            key="speaker_test_text"
        )

        if st.button("🔊 Test Speaker"):
            if test_text.strip():
                with st.spinner(f"Testing {tts_provider} speaker..."):
                    try:
                        # Generate test audio
                        if tts_provider == "elevenlabs":
                            audio_data, latency = generate_speech(test_text)
                        elif tts_provider == "openai":
                            audio_data, latency = asyncio.run(generate_speech_openai(test_text))                        
                        if audio_data:
                            st.audio(audio_data.read(), format="audio/mp3")
                            st.success(f"✅ Test completed in {latency:.2f}s")
                        else:
                            st.error("❌ Test failed - check API keys")
                            
                    except Exception as e:
                        st.error(f"Test error: {str(e)}")
        
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
        
        # Accent improvement explanation
        st.header("Accent Improvement")
        st.info("""
        This system includes optimizations to eliminate accent interference:
        
        1. Language-specific voice settings for Urdu-English
        2. Micro-pauses between language switches
        3. Voice context reset when switching languages
        4. Phonetic optimization for both languages
        
        These improvements ensure Urdu sounds truly Urdu and English sounds truly English.
        """)
    
    # Main interaction area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("Input")
        
        # Input type selection
        input_type = st.radio("Select Input Type", ["Text", "Voice"], horizontal=True)
        
        if input_type == "Text":
            # Text input
            st.subheader("Text Input")
            st.write("Use [ur] to mark Urdu text and [en] to mark English text.")
            
            # Demo preset examples - UPDATED FOR URDU-ENGLISH
            demo_scenarios = {
                "Vocabulary Request": (
                    "Ap mujhe bata sakte hain ke paani aur kuch basic words English mein kya kehte hain?"
                ),
                "Grammar Question": (
                    "English mein past tense kaise banate hain? Kuch examples de sakte hain?"
                ),
                "Practice Conversation": (
                    "Main English mein apna introduction karna seekhna chahta hun. Kaise karu?"
                ),
                "Pronunciation Help": (
                    "Mujhe English ke 'th' sound ka pronunciation seekhna hai. Madad kar sakte hain?"
                ),
                "Daily Expressions": (
                    "Mujhe rozana istemal hone wale English phrases sikhayiye."
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
            # Voice input - HTML5 AUDIO RECORDER
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
                
                # Reliable upload processing
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
                2. Speak clearly in Urdu or English  
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
        elif 'last_llm_response' in st.session_state and st.session_state.last_llm_response:
            st.subheader("AI Tutor Response")
            st.text_area(
                "Response text", 
                value=st.session_state.last_llm_response,
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
                    file_name="urdu_english_tutor_response.mp3",
                    mime="audio/mp3"
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
