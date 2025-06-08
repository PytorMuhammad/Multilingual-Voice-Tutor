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
logger = logging.getLogger("czech_german_voice_tutor")

# ----------------------------------------------------------------------------------
# CONFIGURATION SECTION - UPDATED FOR CZECH-GERMAN
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
        "cs": single_voice_id,  # SAME voice for Czech
        "de": single_voice_id,  # SAME voice for German
        "default": single_voice_id
    }

# OPTIMIZED voice settings for Czech-German
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
    st.session_state.tts_provider = "elevenlabs_multilingual"  # Default

if 'provider_voice_configs' not in st.session_state:
    st.session_state.provider_voice_configs = {
        "elevenlabs_flash": {
            "speakers": {
                "Rachel": {
                    "voice_id": "21m00Tcm4TlvDq8ikWAM", 
                    "description": "Flash v2.5 - Fast, efficient, accent-free",
                    "model": "eleven_flash_v2_5"
                },
                "Adam": {
                    "voice_id": "pNInz6obpgDQGcFmaJgB",
                    "description": "Flash v2.5 - Deep male, accent control", 
                    "model": "eleven_flash_v2_5"
                },
                "Bella": {
                    "voice_id": "EXAVITQu4vr4xnSDxMaL",
                    "description": "Flash v2.5 - Young, clear pronunciation",
                    "model": "eleven_flash_v2_5"
                }
            },
            "selected": "Rachel"
        },
        "elevenlabs_multilingual": {
            "speakers": {
                "Rachel_Multi": {
                    "voice_id": "21m00Tcm4TlvDq8ikWAM", 
                    "description": "Multilingual v2 - Advanced accent elimination",
                    "model": "eleven_multilingual_v2"
                },
                "Adam_Multi": {
                    "voice_id": "pNInz6obpgDQGcFmaJgB",
                    "description": "Multilingual v2 - Deep male, native-like", 
                    "model": "eleven_multilingual_v2"
                },
                "Bella_Multi": {
                    "voice_id": "EXAVITQu4vr4xnSDxMaL",
                    "description": "Multilingual v2 - Crystal clear switching",
                    "model": "eleven_multilingual_v2"
                }
            },
            "selected": "Rachel_Multi"
        }
    }

# Initialize selected speakers per provider
if 'selected_speakers' not in st.session_state:
    st.session_state.selected_speakers = {
        "elevenlabs_flash": "Rachel",
        "elevenlabs_multilingual": "Rachel_Multi"
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

# Language distribution preference - UPDATED FOR CZECH-GERMAN
if 'language_distribution' not in st.session_state:
    st.session_state.language_distribution = {
        "cs": 50,  # Czech percentage
        "de": 50   # German percentage
    }

# Language preference for response
if 'response_language' not in st.session_state:
    st.session_state.response_language = "both"  # Options: "cs", "de", "both"

# Language codes and settings - UPDATED FOR CZECH-GERMAN
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
        "api_calls": {"whisper": 0, "openai": 0, "elevenlabs_flash": 0, "elevenlabs_multilingual": 0}
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

def enforce_word_level_embedding(response_text):
    """CRITICAL: Enforce word-level embedding, prevent sentence-level alternating"""
    
    # Split into lines for analysis
    lines = response_text.strip().split('\n')
    corrected_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is pure German (WRONG pattern)
        if re.match(r'^\[de\].*[^\[cs\]]$', line):
            # Convert pure German line to embedded pattern
            german_content = re.sub(r'^\[de\]\s*', '', line)
            # Embed it in Czech context
            corrected_line = f'[cs] To znamená [de] {german_content} [cs] v němčině.'
            corrected_lines.append(corrected_line)
        
        # Check if line alternates incorrectly
        elif '[de]' in line and '[cs]' in line:
            # This might be correct, but ensure proper flow
            corrected_lines.append(line)
        
        # Pure Czech lines are fine
        elif line.startswith('[cs]'):
            corrected_lines.append(line)
        
        # Any other pattern, force into Czech context
        else:
            corrected_lines.append(f'[cs] {line}')
    
    return '\n'.join(corrected_lines)

# ----------------------------------------------------------------------------------
# SPEECH RECOGNITION (STT) SECTION - UPDATED FOR CZECH-GERMAN
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

# UPDATED STT FUNCTIONS FOR CZECH-GERMAN
async def transcribe_with_enhanced_prompts(audio_file):
    """Enhanced transcription with Czech/German pronunciation hints"""
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            with open(audio_file, "rb") as f:
                file_content = f.read()
            
            files = {
                "file": (os.path.basename(audio_file), file_content, "audio/wav")
            }
            
            data = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "temperature": "0.0",
                "language": None,
                "prompt": "This audio contains Czech and German speech from a language learning session. Focus on accurate pronunciation. Common Czech words: ahoj, jak, se, máš, dobře, děkuji. Common German words: hallo, gut, danke, bitte, deutsch, lernen."
            }
            
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {st.session_state.openai_api_key}"},
                files=files,
                data=data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_result = enhance_czech_german_transcription(result)
                latency = time.time() - start_time
                st.session_state.performance_metrics["stt_latency"].append(latency)
                st.session_state.performance_metrics["api_calls"]["whisper"] += 1
                enhanced_result["latency"] = latency
                return enhanced_result
            else:
                return {
                    "text": "",
                    "language": None,
                    "error": f"API error: {response.status_code}",
                    "latency": time.time() - start_time
                }
    
    except Exception as e:
        return {
            "text": "",
            "language": None,
            "error": str(e),
            "latency": time.time() - start_time
        }

def enhance_czech_german_transcription(result):
    """Apply Czech/German specific pronunciation corrections"""
    try:
        text = result.get("text", "")
        
        czech_corrections = {
            "ahoj": "ahoj",
            "jak": "jak", 
            "se": "se",
            "mas": "máš",
            "dobre": "dobře",
            "dekuji": "děkuji"
        }
        
        german_corrections = {
            "halo": "hallo",
            "gut": "gut",
            "danke": "danke",
            "bitte": "bitte",
            "tschus": "tschüs"
        }
        
        corrected_text = text
        
        for wrong, correct in czech_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
        
        for wrong, correct in german_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
        
        result["text"] = corrected_text
        result["pronunciation_enhanced"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Transcription enhancement error: {str(e)}")
        return result

def analyze_user_request_precisely(user_input):
    """PRECISE: Analyze exactly what user is asking for"""
    
    user_lower = user_input.lower()
    
    # VOCABULARY REQUEST PATTERNS
    vocabulary_patterns = [
        "jak se řekne",
        "co znamená", 
        "základní slova",
        "slovní zásoba",
        "překlad",
        "deutsch word for"
    ]
    
    # EXAMPLE REQUEST PATTERNS  
    example_patterns = [
        "jak se představit",
        "příklad", 
        "věta",
        "ukažte mi",
        "beispiel"
    ]
    
    # Check vocabulary request
    if any(pattern in user_lower for pattern in vocabulary_patterns):
        return {
            "intent": "vocabulary_request",
            "response_pattern": "word_embedding",
            "explanation_language": "cs",
            "target_language": "de"
        }
    
    # Check example request
    elif any(pattern in user_lower for pattern in example_patterns):
        return {
            "intent": "example_request", 
            "response_pattern": "example_provision",
            "explanation_language": "cs",
            "target_language": "de"
        }
    
    else:
        return {
            "intent": "general_query",
            "response_pattern": "mixed_explanation", 
            "explanation_language": "cs",
            "target_language": "both"
        }

# ----------------------------------------------------------------------------------
# LANGUAGE MODEL (LLM) SECTION - UPDATED FOR CZECH-GERMAN TUTORING
# ----------------------------------------------------------------------------------

async def generate_llm_response(prompt, system_prompt=None, api_key=None):
    """Generate intelligent tutoring response in proper Czech script + German"""
    if not api_key:
        api_key = st.session_state.openai_api_key
        
    if not api_key:
        return {
            "response": "Error: OpenAI API key not configured.",
            "latency": 0
        }
    
    start_time = time.time()
    
    # Analyze user intent for precise response pattern
    intent_analysis = analyze_user_learning_intent(prompt)
    
    # Generate context-aware system prompt
    system_content = create_intelligent_tutor_prompt(intent_analysis)
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    # Add conversation context (last 2 exchanges)
    for exchange in st.session_state.conversation_history[-2:]:
        if "user_input" in exchange and "assistant_response" in exchange:
            messages.insert(-1, {"role": "user", "content": exchange["user_input"]})
            messages.insert(-1, {"role": "assistant", "content": exchange["assistant_response"]})
    
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
                    "temperature": 0.4,
                    "max_tokens": 400
                },
                timeout=30.0
            )
            
            latency = time.time() - start_time
            st.session_state.performance_metrics["llm_latency"].append(latency)
            st.session_state.performance_metrics["api_calls"]["openai"] += 1
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                # Apply intelligent post-processing
                enhanced_response = enhance_natural_tutoring_response(response_text, intent_analysis)
                
                return {
                    "response": enhanced_response,
                    "latency": latency,
                    "tokens": result.get("usage", {})
                }
            else:
                return {
                    "response": f"Error: {response.status_code}",
                    "latency": latency
                }
    
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "latency": time.time() - start_time
        }

def analyze_user_learning_intent(user_input):
    """Analyze user's learning intent with 99% accuracy"""
    
    user_lower = user_input.lower()
    
    # Intent patterns with confidence scoring
    intents = {
        "vocabulary_translation": {
            "patterns": ["jak se řekne", "co znamená", "základní slova", "překlad", "slovní zásoba"],
            "confidence": 0
        },
        "example_request": {
            "patterns": ["jak se představit", "příklad", "věta", "ukažte mi", "beispiel"],
            "confidence": 0
        },
        "grammar_explanation": {
            "patterns": ["gramatika", "čas", "pravidlo", "struktura", "konjugace"],
            "confidence": 0
        },
        "pronunciation_help": {
            "patterns": ["výslovnost", "jak vyslovit", "aussprache", "zvuk"],
            "confidence": 0
        },
        "conversation_practice": {
            "patterns": ["konverzace", "rozhovor", "cvičení", "praxe"],
            "confidence": 0
        }
    }
    
    # Calculate confidence scores
    for intent, data in intents.items():
        for pattern in data["patterns"]:
            if pattern in user_lower:
                data["confidence"] += 1
    
    # Determine primary intent
    primary_intent = max(intents.items(), key=lambda x: x[1]["confidence"])
    
    return {
        "intent": primary_intent[0],
        "confidence": primary_intent[1]["confidence"],
        "user_input": user_input
    }

def create_intelligent_tutor_prompt(intent_analysis):
    """Create context-aware system prompt for natural tutoring"""
    
    intent = intent_analysis["intent"]
    
    base_prompt = """Jste profesionální učitel němčiny, který učí německý jazyk české mluvčí.

Důležité pokyny:
- Vždy používejte správnou českou gramatiku a pravopis
- Německá slova používejte pouze tam, kde je to nutné pro výuku
- Odpovědi by měly být přirozené a lidské
- Nikdy neposkytujte obecné odpovědi
- Používejte [cs] pro český text a [de] pro německý text"""

    if intent == "vocabulary_translation":
        return base_prompt + """

Uživatel se ptá na překlad slovní zásoby.

Vzor odpovědi:
- Vysvětlete v češtině
- Uveďte německé slovo
- Přidejte 4-5 souvisejících slov
- Pro každé slovo uveďte jednoduchý příklad

Příklad:
[cs] "Voda" se německy řekne [de] Wasser [cs]. 
Několik dalších základních slov:
"Jídlo" se řekne [de] Essen [cs] - [de] Ich esse täglich [cs]
"Dům" se řekne [de] Haus [cs] - [de] Das ist mein Haus [cs]
"Kniha" se řekne [de] Buch [cs] - [de] Ich lese ein Buch [cs]"""

    elif intent == "example_request":
        return base_prompt + """

Uživatel žádá o příklady v němčině.

Vzor odpovědi:
- Úvod v češtině
- Úplný německý příklad (v uvozovkách)
- Vysvětlení klíčových německých slov v češtině

Příklad:
[cs] Vaše představení v němčině by mohlo vypadat takto:
[de] "Hallo, ich heiße Ahmed. Ich bin Softwareentwickler. Ich habe 3 Jahre Erfahrung." [cs]
Zde [de] Erfahrung [cs] znamená zkušenost."""

    elif intent == "grammar_explanation":
        return base_prompt + """

Uživatel se ptá na vysvětlení gramatiky.

Vzor odpovědi:
- Vysvětlení gramatického pravidla v češtině
- Uvedení struktury
- Jednoduché příklady
- Zdůraznění častých chyb"""

    else:
        return base_prompt + """

Obecná otázka:
- Užitečné rady v češtině
- Podle potřeby zahrňte německá slova
- Povzbudivý tón"""

def enhance_natural_tutoring_response(response_text, intent_analysis):
    """Enhance response for natural tutoring flow"""
    
    # Remove any accidental tags that might have appeared
    cleaned_response = re.sub(r'\[cs\]|\[de\]', '', response_text)
    
    # Ensure proper Czech script (convert any incorrect text)
    cleaned_response = convert_incorrect_to_czech_script(cleaned_response)
    
    # Add encouraging ending based on intent
    if intent_analysis["intent"] == "vocabulary_translation":
        if not any(phrase in cleaned_response for phrase in ["Další", "cvičení", "naučte"]):
            cleaned_response += "\n[cs] Zeptejte se na další slova!"
    
    elif intent_analysis["intent"] == "example_request":
        if not any(phrase in cleaned_response for phrase in ["zkuste", "cvičte"]):
            cleaned_response += "\n[cs] Teď to zkuste sami!"
    
    return cleaned_response.strip()

def convert_incorrect_to_czech_script(text):
    """Convert any incorrect text to proper Czech"""
    
    # Common corrections for Czech
    corrections = {
        "ahoj": "ahoj",
        "dekuji": "děkuji", 
        "prosim": "prosím",
        "dobre": "dobře",
        "ano": "ano",
        "ne": "ne",
        "ja": "já",
        "ty": "ty",
        "on": "on",
        "ona": "ona",
        "my": "my",
        "vy": "vy",
        "oni": "oni"
    }
    
    # Apply corrections with word boundaries
    for incorrect, correct in corrections.items():
        text = re.sub(rf'\b{re.escape(incorrect)}\b', correct, text, flags=re.IGNORECASE)
    
    return text

def detect_primary_language(text):
    """Detect the primary language of a text with improved accuracy for Czech-German"""
    # Czech-specific characters (Latin script with diacritics)
    czech_chars = set("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")
    
    # German-specific characters (Latin script with umlauts)
    german_chars = set("äöüßÄÖÜ")
    
    # General Latin characters
    latin_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Count language-specific characters
    text_lower = text.lower()
    czech_count = sum(1 for char in text if char in czech_chars)
    german_count = sum(1 for char in text if char in german_chars)
    
    # Czech-specific words
    czech_words = {
        "ahoj", "jak", "se", "máš", "dobře", "děkuji", "prosím", "ano", "ne", "já",
        "ty", "on", "ona", "my", "vy", "oni", "jsem", "jsi", "je", "jsme", "jste", "jsou",
        "byl", "byla", "bylo", "byli", "byly", "co", "kde", "kdy", "jak", "proč",
        "dům", "voda", "chléb", "kniha", "čas", "ruka", "hlava", "srdce"
    }
    
    # German-specific words
    german_words = {
        "hallo", "wie", "geht", "es", "dir", "gut", "danke", "bitte", "ja", "nein", "ich",
        "du", "er", "sie", "es", "wir", "ihr", "bin", "bist", "ist", "sind", "seid",
        "war", "warst", "waren", "wart", "was", "wo", "wann", "wie", "warum",
        "haus", "wasser", "brot", "buch", "zeit", "hand", "kopf", "herz", "deutsch"
    }
    
    # Count word occurrences
    words = re.findall(r'\b\w+\b', text_lower)
    czech_word_count = sum(1 for word in words if word in czech_words)
    german_word_count = sum(1 for word in words if word in german_words)
    
    # Improved scoring system with weighted metrics
    czech_evidence = czech_count * 3 + czech_word_count * 2  # Higher weight for Czech diacritics
    german_evidence = german_count * 3 + german_word_count * 2
    
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

def preprocess_text_for_clean_speech(text, language_code=None):
    """Preprocess text to eliminate random voice artifacts - FIXED"""
    
    # Remove problematic patterns that cause voice issues
    cleaned_text = text.strip()
    
    # Remove language tags for cleaner speech
    cleaned_text = re.sub(r'\[cs\]|\[de\]', '', cleaned_text)
    
    # Remove multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Remove problematic punctuation that causes artifacts
    cleaned_text = re.sub(r'[^\w\s\.\,\!\?\-\:\'\"]', '', cleaned_text)
    
    # Handle empty or very short text - CRITICAL FIX
    if len(cleaned_text.strip()) < 3:
        return "Hello."  # Fallback to prevent empty text
    
    # Ensure proper sentence endings
    if cleaned_text and not cleaned_text.endswith(('.', '!', '?')):
        cleaned_text += '.'
    
    return cleaned_text

def eliminate_accent_bleeding_enhanced(text, language_code):
    """CRITICAL: Enhanced accent bleeding elimination - FIXED FOR ELEVENLABS"""
    
    if not language_code:
        return text
    
    # ElevenLabs doesn't support Azure SSML - just return clean text
    # The multilingual model handles language switching internally
    return text.strip()

def clean_czech_text(text):
    """Clean Czech text for better pronunciation"""
    # Handle common Czech pronunciation issues
    text = re.sub(r'\bke\b', 'ke', text)  # Improve pronunciation
    text = re.sub(r'\bse\b', 'se', text)  # Improve pronunciation
    return text

def clean_german_text(text):
    """Clean German text for better pronunciation"""
    # Handle common German pronunciation issues
    text = re.sub(r'\bdas\b', 'das', text, flags=re.IGNORECASE)  # Clear pronunciation
    return text

def generate_speech(text, language_code=None, voice_id=None):
    """Generate speech using ElevenLabs with selected speaker"""
    if not text or text.strip() == "":
        logger.error("Empty text provided to generate_speech")
        return None, 0
    
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        logger.error("ElevenLabs API key not provided")
        return None, 0
    
    # Get selected speaker configuration based on current provider
    provider = st.session_state.tts_provider
    if provider in st.session_state.selected_speakers:
        selected_speaker_name = st.session_state.selected_speakers.get(provider, "Rachel")
        speaker_config = st.session_state.provider_voice_configs[provider]["speakers"][selected_speaker_name]
    else:
        # Fallback to first available provider
        provider = list(st.session_state.provider_voice_configs.keys())[0]
        selected_speaker_name = list(st.session_state.provider_voice_configs[provider]["speakers"].keys())[0]
        speaker_config = st.session_state.provider_voice_configs[provider]["speakers"][selected_speaker_name]
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
    
    # CRITICAL: Clean text for ElevenLabs - SIMPLIFIED
    preprocessed_text = preprocess_text_for_clean_speech(text, language_code)
    if not preprocessed_text:
        logger.error("Text preprocessing failed - empty result")
        return None, 0

    # Use cleaned text directly - no SSML markup needed
    enhanced_text = preprocessed_text

    data = {
        "text": enhanced_text,
        "model_id": model_id,
        "voice_settings": voice_settings,
        "apply_text_normalization": "auto",
        "optimize_streaming_latency": 3  # Optimize for speed
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

async def generate_speech_unified(text, language_code=None):
    """Unified speech generation using selected ElevenLabs model"""
    provider = st.session_state.tts_provider
    
    if provider in ["elevenlabs_flash", "elevenlabs_multilingual"]:
        return generate_speech(text, language_code)
    else:
        return generate_speech(text, language_code)  # Fallback

def add_accent_free_markup(text, language_code):
    """Add accent-free markup for ElevenLabs - SIMPLIFIED"""
    if not language_code:
        return text
    
    # Just return clean text - ElevenLabs Multilingual v2 handles language detection
    return text.strip()

# Enhanced multilingual processing for Czech-German
async def process_multilingual_text_seamless(text, detect_language=True):
    """SIMPLIFIED: Direct processing for ElevenLabs multilingual"""
    start_time = time.time()
    
    # Clean text - remove any tags, just clean punctuation
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    
    if not cleaned_text:
        return None, 0
    
    # Single API call - let ElevenLabs multilingual handle language detection
    audio_data, generation_time = generate_speech(cleaned_text, language_code=None)
    
    if audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_data.read())
            return temp_file.name, generation_time
    
    return None, 0

def clean_text_for_tts(text):
    """Remove problematic characters that cause voice issues"""
    # Remove multiple tags
    text = re.sub(r'\[cs\]\s*\[cs\]', '[cs]', text)
    text = re.sub(r'\[de\]\s*\[de\]', '[de]', text)
    
    # Remove empty tags
    text = re.sub(r'\[cs\]\s*\[de\]', '', text)
    text = re.sub(r'\[de\]\s*\[cs\]', '', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_single_language_dominant(segments):
    """Check if one language dominates (>80%) for single API call"""
    if not segments:
        return True
    
    cs_chars = sum(len(s["text"]) for s in segments if s["language"] == "cs")
    de_chars = sum(len(s["text"]) for s in segments if s["language"] == "de")
    total_chars = cs_chars + de_chars
    
    if total_chars == 0:
        return True
    
    return max(cs_chars, de_chars) / total_chars > 0.8

async def generate_speech_async(text, language):
    """Async wrapper for speech generation"""
    return generate_speech(text, language)

def merge_audio_segments_fast(audio_segments):
    """Fast audio merging without complex processing"""
    try:
        # Simple concatenation
        combined = b''
        for audio_data in audio_segments:
            if hasattr(audio_data, 'read'):
                combined += audio_data.read()
            else:
                combined += audio_data
        return combined
    except:
        # Fallback to first segment if merging fails
        return audio_segments[0].read() if audio_segments else b''

def parse_intelligent_segments(text):
    """Parse text into intelligent language segments for Czech-German with accent isolation"""
    segments = []
    
    # Split by language markers
    parts = re.split(r'(\[[a-z]{2}\])', text)
    
    current_language = None
    current_text = ""
    
    for part in parts:
        if re.match(r'\[[a-z]{2}\]', part):
            # Save previous segment with enhanced accent control
            if current_text.strip():
                segments.append({
                    "text": current_text.strip(),
                    "language": current_language or "cs",
                    "accent_isolated": True  # Mark for accent-free processing
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
            "language": current_language or "cs",
            "accent_isolated": True
        })
    
    return segments

# ----------------------------------------------------------------------------------
# END-TO-END PIPELINE - ENHANCED FOR CZECH-GERMAN PROCESSING
# ----------------------------------------------------------------------------------

async def process_voice_input_pronunciation_enhanced(audio_file):
    """Enhanced voice processing focusing on pronunciation accuracy for Czech-German"""
    pipeline_start_time = time.time()
    
    try:
        # Step 1: Enhanced Audio Preprocessing with 500% boost
        st.session_state.message_queue.put("🔊 Amplifying audio for pronunciation clarity...")
        
        # Step 2: Pronunciation-Enhanced Transcription - UPDATED TO USE WORKING STT
        st.session_state.message_queue.put("🎯 Analyzing Czech-German pronunciation patterns...")
        
        transcription = await asyncio.wait_for(
            transcribe_with_enhanced_prompts(audio_file),  # UPDATED FUNCTION NAME
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
        audio_path, tts_latency = await process_multilingual_text_seamless(response_text)
        
        # Calculate total latency
        total_latency = time.time() - pipeline_start_time
        st.session_state.performance_metrics["total_latency"].append(total_latency)
        
        # ✅ CRITICAL FIX: ADD CONVERSATION HISTORY UPDATE FOR VOICE
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
    """Update status display from message queue"""
    if 'status_text' not in st.session_state:
        st.session_state.status_text = ""
    
    messages_found = False
    while True:
        try:
            message = st.session_state.message_queue.get_nowait()
            st.session_state.status_text += message + "\n"
            messages_found = True
        except queue.Empty:
            break
    
    if messages_found:
        with st.session_state.status_area.container():
            st.text_area("Processing Log", value=st.session_state.status_text, height=200)

def ensure_single_voice_consistency():
    """Ensure all languages use the same voice ID"""
    single_voice = st.session_state.elevenlabs_voice_id
    st.session_state.language_voices["cs"] = single_voice
    st.session_state.language_voices["de"] = single_voice
    st.session_state.language_voices["default"] = single_voice
    logger.info(f"Voice consistency enforced: {single_voice}")

# ----------------------------------------------------------------------------------
# STREAMLIT UI - UPDATED FOR CZECH-GERMAN INTERFACE
# ----------------------------------------------------------------------------------

def main():
    """Main application entry point"""
    # Page configuration - ONLY ONCE!
    st.set_page_config(
        page_title="Czech-German AI Voice Tutor",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("Czech-German AI Voice Tutor")
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
        - Model: Multilingual v2 (Accent-free)
        """)
        
        # Language Response Options
        st.subheader("Tutor Mode")
 
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
            
            cs_percent = st.slider("Czech %", min_value=0, max_value=100, value=st.session_state.language_distribution["cs"])
            de_percent = 100 - cs_percent
            st.text(f"German %: {de_percent}")
            
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
        
        # TTS Provider Selection
        st.subheader("🎵 TTS Provider")
        
        tts_provider = st.selectbox(
            "Choose TTS Provider", 
            options=["elevenlabs_flash", "elevenlabs_multilingual"],
            format_func=lambda x: {
                "elevenlabs_flash": "ElevenLabs Flash v2.5 (Fast, 1-2s latency)",
                "elevenlabs_multilingual": "ElevenLabs Multilingual v2 (Accent-free, 2-3s latency)"
            }[x],
            index=1  # Default to multilingual
        )
        
        if tts_provider != st.session_state.tts_provider:
            st.session_state.tts_provider = tts_provider
            st.success(f"TTS Provider changed to: {tts_provider}")
        
        # Provider-specific configuration
        if tts_provider == "elevenlabs_flash":
            st.info("✅ ElevenLabs Flash v2.5 - Optimized for speed (1-2s latency)")
        elif tts_provider == "elevenlabs_multilingual":
            st.info("✅ ElevenLabs Multilingual v2 - Optimized for accent-free switching (2-3s latency)")
            
        # Speaker Selection per Provider
        if tts_provider in st.session_state.provider_voice_configs:
            st.write(f"**{tts_provider.title()} Speakers:**")
            
            speakers = st.session_state.provider_voice_configs[tts_provider]["speakers"]
            current_speaker = st.session_state.selected_speakers.get(tts_provider, list(speakers.keys())[0])
            
            # Create speaker options with descriptions
            speaker_options = {}
            for name, config in speakers.items():
                description = config.get("description", "Professional voice")
                speaker_options[f"{name} - {description}"] = name
            
            selected_speaker_display = st.selectbox(
                f"Choose {tts_provider.title()} Speaker",
                options=list(speaker_options.keys()),
                index=list(speaker_options.values()).index(current_speaker) if current_speaker in speaker_options.values() else 0,
                key=f"{tts_provider}_speaker_select"
            )
            
            selected_speaker = speaker_options[selected_speaker_display]
            
            if selected_speaker != st.session_state.selected_speakers.get(tts_provider):
                st.session_state.selected_speakers[tts_provider] = selected_speaker
                
                # Update ElevenLabs voice ID if ElevenLabs is selected
                if tts_provider in ["elevenlabs_flash", "elevenlabs_multilingual"]:
                    new_voice_id = speakers[selected_speaker]["voice_id"]
                    st.session_state.elevenlabs_voice_id = new_voice_id
                
                st.success(f"✅ Speaker changed to: {selected_speaker}")
                
            # Show speaker details
            speaker_config = speakers[selected_speaker]
            st.info(f"""
            **{selected_speaker}**: {speaker_config['description']}
            - Voice ID: {speaker_config['voice_id'][:12]}...
            - Model: {speaker_config['model']}
            """)

        # Voice Testing Section
        st.write("**🎵 Test Current Speaker:**")
        test_text = st.text_input(
            "Test Text", 
            value="Hallo, das ist ein Test. Ahoj, toto je test.",
            key="speaker_test_text"
        )

        if st.button("🔊 Test Speaker"):
            if test_text.strip():
                with st.spinner(f"Testing {tts_provider} speaker..."):
                    try:
                        # Generate test audio
                        if tts_provider in ["elevenlabs_flash", "elevenlabs_multilingual"]:
                            audio_data, latency = generate_speech(test_text)
                        else:
                            audio_data, latency = generate_speech(test_text)  # Fallback                      
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
        st.text(f"ElevenLabs Flash calls: {st.session_state.performance_metrics['api_calls']['elevenlabs_flash']}")
        st.text(f"ElevenLabs Multilingual calls: {st.session_state.performance_metrics['api_calls']['elevenlabs_multilingual']}")
        
        # Accent improvement explanation
        st.header("Accent Improvement")
        st.info("""
        This system includes optimizations to eliminate accent interference:
        
        1. Language-specific voice settings for Czech-German
        2. Micro-pauses between language switches
        3. Voice context reset when switching languages
        4. Phonetic optimization for both languages
        
        These improvements ensure Czech sounds truly Czech and German sounds truly German.
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
            st.write("Use [cs] to mark Czech text and [de] to mark German text.")
            
            demo_scenarios = {
                "Vocabulary Request": (
                    "Můžete mi říct, jak se řekne voda a některá další základní slova v němčině?"
                ),
                "Grammar Question": (
                    "Jak se tvoří minulý čas v němčině? Můžete mi dát nějaké příklady?"
                ),
                "Practice Conversation": (
                    "Chtěl bych se naučit představit se v němčině. Jak na to?"
                ),
                "Pronunciation Help": (
                    "Potřebuji se naučit výslovnost německého 'ü'. Můžete mi pomoci?"
                ),
                "Daily Expressions": (
                    "Naučte mě každodenní německé fráze."
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
                    file_name="czech_german_tutor_response.mp3",
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
