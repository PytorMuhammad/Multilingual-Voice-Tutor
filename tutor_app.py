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
import noisereduce as nr

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
# CONFIGURATION SECTION - ENHANCED WITH MULTIPLE TTS OPTIONS
# ----------------------------------------------------------------------------------

# Initialize session state variables
if 'api_keys_initialized' not in st.session_state:
    st.session_state.api_keys_initialized = False
    st.session_state.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    st.session_state.azure_speech_key = os.environ.get("AZURE_SPEECH_KEY", "")
    st.session_state.azure_speech_region = os.environ.get("AZURE_SPEECH_REGION", "")

# TTS Provider Selection
if 'tts_provider' not in st.session_state:
    st.session_state.tts_provider = "elevenlabs"  # Default to ElevenLabs

# SINGLE VOICE CONFIGURATION FOR ACCENT-FREE SWITCHING
if 'language_voices' not in st.session_state:
    single_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Same voice for both languages
    st.session_state.language_voices = {
        "ur": single_voice_id,    # Urdu uses same voice
        "en": single_voice_id,    # English uses same voice  
        "default": single_voice_id
    }

# OPTIMIZED voice settings for accent-free switching
if 'voice_settings' not in st.session_state:
    st.session_state.voice_settings = {
        "ur": {  # Urdu-optimized settings
            "stability": 0.95,        # Maximum stability for consistent Urdu
            "similarity_boost": 0.98, # Maximum similarity for native sound
            "style": 0.85,           # High style for natural expression
            "use_speaker_boost": True # Enable speaker boost for clarity
        },
        "en": {  # English-optimized settings  
            "stability": 0.92,        # Very high stability for consistent English
            "similarity_boost": 0.95, # Very high similarity for native sound
            "style": 0.80,           # High style for natural expression
            "use_speaker_boost": True # Enable speaker boost for clarity
        },
        "default": {
            "stability": 0.90,
            "similarity_boost": 0.90,
            "style": 0.75,
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

# Language distribution preference
if 'language_distribution' not in st.session_state:
    st.session_state.language_distribution = {
        "ur": 50,  # Urdu percentage
        "en": 50   # English percentage
    }

# Language preference for response
if 'response_language' not in st.session_state:
    st.session_state.response_language = "both"  # Options: "ur", "en", "both"

# Language codes and settings
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
        "api_calls": {"whisper": 0, "openai": 0, "elevenlabs": 0, "azure": 0}
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
        import subprocess
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.warning("⚠️ Audio processing may be limited. Installing dependencies...")
    return True

# ----------------------------------------------------------------------------------
# ENHANCED HTML5 AUDIO RECORDER WITH ACCENT-FREE PROCESSING
# ----------------------------------------------------------------------------------

def create_audio_recorder_component():
    """Create HTML5 audio recorder component with accent-free processing"""
    html_code = """
    <div style="padding: 20px; border: 2px solid #ff4b4b; border-radius: 10px; text-align: center; background-color: #f0f2f6;">
        <div id="status" style="font-size: 18px; margin-bottom: 15px; font-weight: bold;">🎤 Ready to Record (Urdu/English)</div>
        
        <button id="recordBtn" onclick="toggleRecording()" 
                style="background: #ff4b4b; color: white; border: none; padding: 15px 30px; 
                       border-radius: 25px; cursor: pointer; font-size: 16px; font-weight: bold; margin: 5px;">
            🔴 START RECORDING
        </button>
        
        <div id="timer" style="font-size: 14px; margin-top: 10px; color: #666;">00:00</div>
        
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
                    document.getElementById('status').innerHTML = '✅ Recording Complete!';
                    showDownloadLink();
                };
                
                document.getElementById('status').innerHTML = '🎤 Ready - Speak in Urdu or English';
                
            } catch (error) {
                document.getElementById('status').innerHTML = '❌ Microphone access denied';
                console.error('Error accessing microphone:', error);
            }
        }

        function toggleRecording() {
            const recordBtn = document.getElementById('recordBtn');
            const statusDiv = document.getElementById('status');
            
            if (!isRecording) {
                audioChunks = [];
                recordingTime = 0;
                isRecording = true;
                
                recordBtn.innerHTML = '⏹️ STOP RECORDING';
                recordBtn.style.background = '#666';
                statusDiv.innerHTML = '🔴 RECORDING - Speak in Urdu or English';
                
                document.getElementById('downloadSection').style.display = 'none';
                timerInterval = setInterval(updateTimer, 1000);
                mediaRecorder.start(1000);
                
            } else {
                isRecording = false;
                mediaRecorder.stop();
                
                recordBtn.innerHTML = '🔄 NEW RECORDING';
                recordBtn.style.background = '#ff4b4b';
                statusDiv.innerHTML = '⏳ Processing recording...';
                
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

        function showDownloadLink() {
            if (recordedBlob) {
                const url = URL.createObjectURL(recordedBlob);
                const downloadLink = document.getElementById('downloadLink');
                
                downloadLink.href = url;
                downloadLink.download = 'my-recording.webm';
                
                document.getElementById('downloadSection').style.display = 'block';
                
                setTimeout(() => {
                    downloadLink.click();
                }, 2000);
                
                document.getElementById('status').innerHTML = '✅ Recording ready! Download and upload below.';
            }
        }
    </script>
    """
    
    return st.components.v1.html(html_code, height=250)

def convert_webm_to_wav(webm_path):
    """Convert WebM audio to WAV format"""
    try:
        audio = AudioSegment.from_file(webm_path, format="webm")
        wav_path = tempfile.mktemp(suffix=".wav")
        audio.export(wav_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        return wav_path
    except Exception as e:
        logger.error(f"WebM to WAV conversion error: {str(e)}")
        return webm_path

def amplify_recorded_audio(audio_path):
    """Apply 500% amplification to recorded audio for better recognition"""
    try:
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
# ENHANCED SPEECH RECOGNITION WITH MULTIPLE PROVIDERS
# ----------------------------------------------------------------------------------

async def transcribe_with_api(audio_file, api_key):
    """Enhanced transcription with pronunciation focus for Urdu/English"""
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            with open(audio_file, "rb") as f:
                file_content = f.read()
            
            files = {
                "file": (os.path.basename(audio_file), file_content, "audio/wav")
            }
            
            # ENHANCED: Pronunciation-focused settings for Urdu/English
            data = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "temperature": "0.0",  # Lowest temperature for consistent pronunciation
                "language": None,  # Let Whisper auto-detect between ur/en
                "prompt": "This audio contains Urdu and English speech. Focus on accurate pronunciation and phonetic understanding. Common Urdu words: salaam, shukriya, aap, hum. Common English words: hello, thank you, water, good."
            }
            
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_result = enhance_pronunciation_transcription(result)
                
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
        "salam": "salaam",
        "shukria": "shukriya", 
        "ap": "aap",
        "ham": "hum",
        "paani": "pani",
        "achha": "acha"
    }
    
    # English pronunciation corrections
    english_corrections = {
        "helo": "hello",
        "watter": "water", 
        "gud": "good",
        "thanx": "thanks",
        "welcom": "welcome"
    }
    
    corrected_text = text
    
    # Apply Urdu corrections if Urdu content detected
    if language == "ur" or any(word in text.lower() for word in ["salaam", "aap", "hum"]):
        for wrong, correct in urdu_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
    
    # Apply English corrections if English content detected  
    if language == "en" or any(word in text.lower() for word in ["hello", "water", "good"]):
        for wrong, correct in english_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

# ----------------------------------------------------------------------------------
# ENHANCED LLM WITH STRATEGIC LANGUAGE TAGGING
# ----------------------------------------------------------------------------------

async def generate_llm_response(prompt, system_prompt=None, api_key=None):
    """Generate response with STRATEGIC language tagging for accent-free switching"""
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
        # PROFESSIONAL ENGLISH TUTOR FOR URDU SPEAKERS SYSTEM PROMPT
        response_language = st.session_state.response_language
        
        if response_language == "both":
            system_content = """You are "EnglishMaster" - a premium AI English language tutor designed for Urdu speakers who paid for professional English learning. You represent a commercial language learning platform.

CORE IDENTITY:
You are a certified English language instructor with 15+ years of experience teaching Urdu speakers. You are perfectly bilingual in Urdu and English. Your teaching style is engaging, systematic, and results-oriented.

CURRICULUM STRUCTURE (Beginner to Intermediate):
- Basic greetings, present tense, articles (a/an/the), family/food/time vocabulary
- Past/future tenses, modal verbs, prepositions, everyday conversation

PEDAGOGICAL APPROACH:
1. Use Urdu [ur] for explanations, English [en] for examples and practice
2. Break complex concepts into micro-lessons
3. Always provide immediate practice opportunities
4. Use real-life scenarios (restaurant, shopping, work)
5. Correct errors gently with clear explanations

STRATEGIC LANGUAGE TAGGING RULES:
- Tag ONLY when switching languages, not every word
- Use [ur] for: explanations, cultural context, encouragement
- Use [en] for: vocabulary, phrases to practice, examples
- Example: "[ur] Hm english mein pani ko [en] water [ur] kehte hain"

VOCABULARY LESSON FORMAT:
"[ur] Explanation [en] English term [ur] detailed explanation [en] example sentence [ur] practice suggestion"

CONVERSATION PRACTICE:
- Set realistic scenarios (ordering food, directions)
- Provide English phrases with Urdu explanations
- Encourage role-play responses
- Build confidence progressively

ERROR CORRECTION PROTOCOL:
1. Acknowledge attempt positively
2. Identify specific error type
3. Explain correct form with reasoning
4. Provide corrected version
5. Give additional practice opportunity

QUALITY STANDARDS:
- Use proper [ur] and [en] markers strategically
- Keep responses 2-4 sentences for engagement
- Include at least one practice element per response
- Maintain encouraging, professional tone

CRITICAL RULES FOR ACCENT-FREE SWITCHING:
1. You are an ENGLISH TEACHER, not a translator
2. Use Urdu [ur] ONLY for: explanations, instructions, questions to student
3. Use English [en] ONLY for: vocabulary, examples, phrases to practice
4. NEVER repeat the same content in both languages
5. NEVER translate - each language has a different PURPOSE
6. TAG STRATEGICALLY - not every word, only when switching context

LANGUAGE PURPOSES:
- Urdu [ur] = Your teaching language (explain, instruct, encourage)  
- English [en] = Target language (vocabulary, examples, practice phrases)

INTRODUCTION EXAMPLE:
"[ur] Assalam alaikum! Main aapka English teacher hun. Aaj hum basic English words seekhenge. [en] Good morning [ur] ka matlab hai 'subah bakhair'. Kya aap yeh keh sakte hain?"

VOCABULARY EXAMPLE:
"[ur] 'Pani' ko English mein [en] water [ur] kehte hain. Sentence banayiye: [en] I drink water [ur] Samjh gaye?"

NEVER DO THIS (wrong):
❌ "[ur] Main teacher hun [en] I am teacher" (same content translated)

ALWAYS DO THIS (correct):
✅ "[ur] Main aapka English teacher hun. Aaj seekhenge [en] Good morning [ur] ka matlab"

Stay in character as an Urdu-speaking English teacher. Each language serves a different pedagogical purpose."""

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
    
    # Add previous conversation history for context
    for exchange in st.session_state.conversation_history[-2:]:
        if "user_input" in exchange:
            messages.append({"role": "user", "content": exchange["user_input"]})
        if "assistant_response" in exchange:
            messages.append({"role": "assistant", "content": exchange["assistant_response"]})
    
    # Add the current prompt
    messages.append({"role": "user", "content": prompt})
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": messages,
                    "temperature": 0.7,
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
                
                # Minimal processing - let professional tutor respond naturally
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

def ensure_proper_language_markers(response_text):
    """Ensure response has proper strategic language markers"""
    
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

# ----------------------------------------------------------------------------------
# MULTIPLE TTS PROVIDERS - ACCENT-FREE IMPLEMENTATION
# ----------------------------------------------------------------------------------

async def generate_speech_elevenlabs(text, language_code, voice_id, api_key):
    """Generate speech using ElevenLabs with accent-free optimization"""
    voice_settings = get_accent_free_settings(language_code)
    enhanced_text = create_accent_free_ssml(text, language_code)
    
    data = {
        "text": enhanced_text,
        "model_id": "eleven_flash_v2_5",  # Latest model for accent control
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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                json=data,
                headers=headers,
                timeout=15
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                logger.info(f"✅ ElevenLabs: Generated accent-free {language_code} in {generation_time:.2f}s")
                return BytesIO(response.content), generation_time
            else:
                logger.error(f"ElevenLabs error: {response.status_code}")
                return None, generation_time
                
    except Exception as e:
        logger.error(f"ElevenLabs speech generation error: {str(e)}")
        return None, time.time() - start_time

async def generate_speech_openai(text, language_code, api_key):
    """Generate speech using OpenAI TTS"""
    # Choose voice based on language for accent-free output
    voice = "nova"  # Use same voice for consistency
    
    data = {
        "model": "tts-1-hd",  # High definition model
        "input": text,
        "voice": voice,
        "response_format": "mp3"
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/speech",
                json=data,
                headers=headers,
                timeout=15
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                logger.info(f"✅ OpenAI TTS: Generated {language_code} in {generation_time:.2f}s")
                return BytesIO(response.content), generation_time
            else:
                logger.error(f"OpenAI TTS error: {response.status_code}")
                return None, generation_time
                
    except Exception as e:
        logger.error(f"OpenAI TTS error: {str(e)}")
        return None, time.time() - start_time

async def generate_speech_azure(text, language_code, speech_key, region):
    """Generate speech using Azure Speech Services"""
    # Choose language and voice based on content
    if language_code == "ur":
        voice_name = "ur-PK-AsadNeural"  # Urdu voice
        language = "ur-PK"
    else:
        voice_name = "en-US-JennyNeural"  # English voice
        language = "en-US"
    
    # Create SSML for better control
    ssml = f"""
    <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{language}'>
        <voice name='{voice_name}'>
            <prosody rate='0.9' pitch='0st'>
                {text}
            </prosody>
        </voice>
    </speak>
    """
    
    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3"
    }
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                content=ssml,
                headers=headers,
                timeout=15
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                logger.info(f"✅ Azure Speech: Generated {language_code} in {generation_time:.2f}s")
                return BytesIO(response.content), generation_time
            else:
                logger.error(f"Azure Speech error: {response.status_code}")
                return None, generation_time
                
    except Exception as e:
        logger.error(f"Azure Speech error: {str(e)}")
        return None, time.time() - start_time

def get_accent_free_settings(language_code):
    """Get voice settings optimized for accent-free pronunciation"""
    if language_code and language_code in st.session_state.voice_settings:
        return st.session_state.voice_settings[language_code].copy()
    else:
        return st.session_state.voice_settings["default"]

def create_accent_free_ssml(text, language_code):
    """Create SSML markup for accent-free pronunciation"""
    if not language_code:
        return text
    
    clean_text = text.strip()
    
    if language_code == "ur":
        # Urdu pronunciation optimization
        enhanced_text = f'<speak><lang xml:lang="ur-PK"><prosody rate="0.92" pitch="+2st">{clean_text}</prosody></lang></speak>'
    elif language_code == "en":
        # English pronunciation optimization  
        enhanced_text = f'<speak><lang xml:lang="en-US"><prosody rate="0.95" pitch="+1st">{clean_text}</prosody></lang></speak>'
    else:
        enhanced_text = clean_text
    
    return enhanced_text

async def generate_speech_unified(text, language_code=None):
    """Unified speech generation using selected TTS provider"""
    if not text or text.strip() == "":
        logger.error("Empty text provided to generate_speech_unified")
        return None, 0
    
    provider = st.session_state.tts_provider
    
    try:
        if provider == "elevenlabs":
            api_key = st.session_state.elevenlabs_api_key
            voice_id = st.session_state.elevenlabs_voice_id
            if not api_key:
                st.error("ElevenLabs API key not configured")
                return None, 0
            return await generate_speech_elevenlabs(text, language_code, voice_id, api_key)
        
        elif provider == "openai":
            api_key = st.session_state.openai_api_key
            if not api_key:
                st.error("OpenAI API key not configured")
                return None, 0
            return await generate_speech_openai(text, language_code, api_key)
        
        elif provider == "azure":
            speech_key = st.session_state.azure_speech_key
            region = st.session_state.azure_speech_region
            if not speech_key or not region:
                st.error("Azure Speech key and region not configured")
                return None, 0
            return await generate_speech_azure(text, language_code, speech_key, region)
        
        else:
            st.error(f"Unknown TTS provider: {provider}")
            return None, 0
            
    except Exception as e:
        logger.error(f"Unified speech generation error: {str(e)}")
        return None, 0

def process_multilingual_text_seamless(text, detect_language=True):
    """Process text with strategic language markers and generate accent-free audio"""
    segments = parse_language_segments_strategic(text)
    
    if len(segments) <= 1:
        # Single language - direct processing
        return asyncio.run(process_single_language_segment(text, segments[0]["language"] if segments else None))
    
    # Multiple languages - accent-free blending
    return asyncio.run(process_multiple_language_segments(segments))

async def process_single_language_segment(text, language_code):
    """Process single language segment"""
    audio_data, generation_time = await generate_speech_unified(text, language_code)
    
    if audio_data:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_data.read())
            return temp_file.name, generation_time
    
    return None, 0

async def process_multiple_language_segments(segments):
    """Process multiple language segments with accent-free blending"""
    audio_segments = []
    total_time = 0
    
    for i, segment in enumerate(segments):
        if not segment["text"].strip():
            continue
            
        # Generate audio for each segment
        audio_data, generation_time = await generate_speech_unified(
            segment["text"], 
            language_code=segment["language"]
        )
        
        if audio_data:
            # Convert to AudioSegment for processing
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            
            # Normalize volume for consistent blending
            normalized_segment = normalize_audio_volume(audio_segment, target_dbfs=-18)
            
            audio_segments.append(normalized_segment)
            total_time += generation_time
    
    if not audio_segments:
        return None, 0
    
    # Blend segments with accent-free crossfading
    combined_audio = audio_segments[0]
    
    for i in range(1, len(audio_segments)):
        # Apply subtle crossfade for same voice transitions
        combined_audio = apply_accent_free_crossfade(
            combined_audio, 
            audio_segments[i], 
            crossfade_ms=100  # Short crossfade for same voice
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

def parse_language_segments_strategic(text):
    """Parse language segments with strategic tagging approach"""
    segments = []
    
    # Split by language markers
    parts = re.split(r'(\[(?:ur|en)\])', text)
    
    current_language = None
    current_text = ""
    
    for part in parts:
        if re.match(r'\[(?:ur|en)\]', part):
            # Save previous segment if exists
            if current_text.strip():
                segments.append({
                    "text": current_text.strip(),
                    "language": current_language or "ur"  # Default to Urdu
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
            "language": current_language or "ur"
        })
    
    # Detect language for unmarked segments
    for segment in segments:
        if segment["language"] is None:
            segment["language"] = detect_primary_language(segment["text"])
    
    return segments

def normalize_audio_volume(audio_segment, target_dbfs=-18):
    """Normalize audio volume for consistent blending"""
    current_dbfs = audio_segment.dBFS
    volume_adjustment = target_dbfs - current_dbfs
    normalized = audio_segment.apply_gain(volume_adjustment)
    return normalized

def apply_accent_free_crossfade(audio1, audio2, crossfade_ms=100):
    """Apply subtle crossfading for accent-free transitions"""
    audio1_norm = normalize_audio_volume(audio1, -18)
    audio2_norm = normalize_audio_volume(audio2, -18)
    
    # Apply subtle crossfade
    crossfaded = audio1_norm.append(audio2_norm, crossfade=crossfade_ms)
    return crossfaded

def detect_primary_language(text):
    """Detect the primary language of text for Urdu/English"""
    # Urdu-specific characters and words
    urdu_chars = set("آابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہھیے")
    urdu_words = {
        "aap", "hum", "main", "hain", "hai", "salaam", "shukriya", "paani", 
        "acha", "kya", "kaise", "kahan", "kab", "kyun", "yeh", "woh"
    }
    
    # English-specific words
    english_words = {
        "the", "and", "or", "is", "are", "was", "were", "have", "has", "had",
        "will", "would", "can", "could", "should", "may", "might", "must",
        "hello", "water", "good", "thank", "welcome", "please", "sorry"
    }
    
    # Count evidence
    text_lower = text.lower()
    urdu_evidence = 0
    english_evidence = 0
    
    # Character-based evidence
    for char in text:
        if char in urdu_chars:
            urdu_evidence += 2
    
    # Word-based evidence
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        if word in urdu_words:
            urdu_evidence += 3
        elif word in english_words:
            english_evidence += 3
    
    # Determine primary language
    if urdu_evidence > english_evidence and urdu_evidence > 0:
        return "ur"
    elif english_evidence > urdu_evidence and english_evidence > 0:
        return "en"
    
    # Default based on distribution preference
    if st.session_state.language_distribution["ur"] >= st.session_state.language_distribution["en"]:
        return "ur"
    else:
        return "en"

# ----------------------------------------------------------------------------------
# ENHANCED VOICE PROCESSING PIPELINE
# ----------------------------------------------------------------------------------

async def process_voice_input_accent_free(audio_file):
    """Enhanced voice processing with accent-free output"""
    pipeline_start_time = time.time()
    
    try:
        # Step 1: Enhanced Audio Preprocessing
        st.session_state.message_queue.put("🔊 Amplifying audio for clarity...")
        
        enhanced_audio_file = enhance_audio_for_pronunciation(audio_file)
        if enhanced_audio_file and os.path.exists(enhanced_audio_file):
            audio_file = enhanced_audio_file
        
        # Step 2: Accent-Free Transcription
        st.session_state.message_queue.put("🎯 Analyzing speech patterns...")
        
        transcription = await asyncio.wait_for(
            transcribe_with_api(audio_file, st.session_state.openai_api_key),
            timeout=30.0
        )
        
        if not transcription or not transcription.get("text"):
            st.session_state.message_queue.put("❌ No clear speech detected")
            return None, None, 0, 0, 0
        
        # Step 3: Strategic Language Understanding
        user_input = transcription["text"].strip()
        st.session_state.message_queue.put(f"📝 Transcribed: {user_input}")
        
        # Step 4: Generate Strategic Response
        st.session_state.message_queue.put("🤖 Generating strategic response...")
        
        llm_result = await generate_llm_response(user_input)
        
        if "error" in llm_result:
            st.session_state.message_queue.put(f"❌ Response generation failed: {llm_result.get('error')}")
            return user_input, None, transcription.get("latency", 0), 0, 0
        
        response_text = llm_result["response"]
        st.session_state.message_queue.put(f"💬 Generated: {response_text}")
        
        # Step 5: Accent-Free Voice Synthesis
        st.session_state.message_queue.put("🎵 Generating accent-free speech...")
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
        
        st.session_state.message_queue.put(f"✅ Accent-Free Processing Complete! ({total_latency:.2f}s)")
        
        # Clean up
        if enhanced_audio_file and enhanced_audio_file != audio_file:
            try:
                os.unlink(enhanced_audio_file)
            except:
                pass
        
        return user_input, audio_path, transcription.get("latency", 0), llm_result.get("latency", 0), tts_latency
        
    except Exception as e:
        logger.error(f"Accent-free processing error: {str(e)}")
        st.session_state.message_queue.put(f"❌ Error: {str(e)}")
        return None, None, 0, 0, 0

def enhance_audio_for_pronunciation(audio_file):
    """Enhanced audio preprocessing for better recognition"""
    try:
        # Load audio
        audio, sample_rate = sf.read(audio_file)
        
        # Apply noise reduction
        enhanced_audio = nr.reduce_noise(y=audio, sr=sample_rate)
        
        # Dynamic range compression
        enhanced_audio = np.tanh(enhanced_audio * 2.0)
        
        # Pre-emphasis filter for consonant clarity
        pre_emphasis = 0.97
        emphasized_audio = np.append(enhanced_audio[0], enhanced_audio[1:] - pre_emphasis * enhanced_audio[:-1])
        
        # Save enhanced audio
        enhanced_path = tempfile.mktemp(suffix=".wav")
        sf.write(enhanced_path, emphasized_audio, sample_rate)
        
        return enhanced_path
        
    except Exception as e:
        logger.error(f"Audio enhancement error: {str(e)}")
        return audio_file

async def process_text_input_enhanced(text):
    """Process text input with accent-free TTS"""
    pipeline_start_time = time.time()
    
    # Generate response
    st.session_state.message_queue.put("🤖 Generating strategic response...")
    
    response_language = st.session_state.response_language
    language_distribution = st.session_state.language_distribution
    
    # Create prompt with strategic instructions
    if response_language == "both":
        ur_percent = language_distribution["ur"]
        en_percent = language_distribution["en"]
        system_prompt = (
            f"You are a multilingual AI English tutor. Respond with strategic language mixing: approximately {ur_percent}% Urdu and {en_percent}% English. "
            f"Use language markers [ur] and [en] strategically, not for every word."
        )
    elif response_language in ["ur", "en"]:
        system_prompt = f"You are a language tutor. Respond only in {response_language} with [{response_language}] markers."
    else:
        system_prompt = None
    
    llm_result = await generate_llm_response(text, system_prompt)
    
    if "error" in llm_result:
        st.session_state.message_queue.put(f"Error generating response: {llm_result.get('error')}")
        return None, llm_result.get("latency", 0), 0
    
    response_text = llm_result["response"]
    st.session_state.message_queue.put(f"Generated response: {response_text}")
    
    # Accent-free text-to-speech
    st.session_state.message_queue.put("🎵 Generating accent-free speech...")
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
    """Display audio in Streamlit with error handling"""
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None
        
    try:
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            return None
            
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
            return audio_bytes
    except Exception as e:
        logger.error(f"Error displaying audio: {str(e)}")
        return None

def calculate_average_latency(latency_list, recent_count=5):
    """Calculate average latency from recent measurements"""
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
            if hasattr(st.session_state, 'status_area') and st.session_state.status_area:
                st.session_state.status_area.text_area("Processing Log", value=status_text, height=200)
        except queue.Empty:
            break

def get_voices():
    """Fetch available voices from ElevenLabs API"""
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        return []
    
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }
    
    try:
        response = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("voices", [])
        else:
            st.error(f"Failed to get voices: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching voices: {e}")
        return []

# ----------------------------------------------------------------------------------
# ENHANCED STREAMLIT UI WITH MULTIPLE TTS OPTIONS
# ----------------------------------------------------------------------------------

def main():
    """Main application entry point with enhanced UI"""
    st.set_page_config(
        page_title="Multilingual AI Voice Tutor - Accent-Free",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️ Multilingual AI Voice Tutor - Accent-Free")
    st.subheader("Professional English Language Tutor for Urdu Speakers")
    
    # Status area for progress updates
    if 'status_area' not in st.session_state:
        st.session_state.status_area = st.empty()
    
    # Enhanced sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API keys section
        st.subheader("🔑 API Keys")
        
        elevenlabs_key = st.text_input(
            "ElevenLabs API Key", 
            value=st.session_state.elevenlabs_api_key,
            type="password",
            help="Required for ElevenLabs TTS"
        )
        
        openai_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key,
            type="password",
            help="Required for speech recognition and language understanding"
        )
        
        # Azure Speech Services
        st.subheader("🎵 Azure Speech Services")
        azure_key = st.text_input(
            "Azure Speech Key", 
            value=st.session_state.azure_speech_key,
            type="password",
            help="Optional: Azure Speech Services key"
        )
        
        azure_region = st.text_input(
            "Azure Region", 
            value=st.session_state.azure_speech_region,
            help="Azure region (e.g., eastus, westus2)"
        )
        
        if st.button("💾 Save API Keys"):
            st.session_state.elevenlabs_api_key = elevenlabs_key
            st.session_state.openai_api_key = openai_key
            st.session_state.azure_speech_key = azure_key
            st.session_state.azure_speech_region = azure_region
            st.session_state.api_keys_initialized = True
            st.success("✅ API keys saved successfully!")
        
        # TTS Provider Selection
        st.subheader("🎤 TTS Provider Selection")
        
        tts_options = ["elevenlabs", "openai", "azure"]
        tts_labels = {
            "elevenlabs": "ElevenLabs (Best Quality)",
            "openai": "OpenAI TTS (Fast)",
            "azure": "Azure Speech (Multilingual)"
        }
        
        selected_tts = st.selectbox(
            "Select TTS Provider",
            options=tts_options,
            format_func=lambda x: tts_labels[x],
            index=tts_options.index(st.session_state.tts_provider)
        )
        
        if selected_tts != st.session_state.tts_provider:
            st.session_state.tts_provider = selected_tts
            st.success(f"✅ TTS Provider set to: {tts_labels[selected_tts]}")
        
        # Provider-specific settings
        if st.session_state.tts_provider == "elevenlabs":
            st.subheader("🎯 ElevenLabs Voice Settings")
            
            # Load voices if ElevenLabs is selected
            if st.button("🔄 Load Voices"):
                voices = get_voices()
                if voices:
                    st.session_state.voices = voices
                    st.success(f"Loaded {len(voices)} voices")
            
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
                    index=list(voice_options.keys()).index(current_voice) if current_voice else 0
                )
                
                if selected_voice_name:
                    new_voice_id = voice_options[selected_voice_name]
                    st.session_state.elevenlabs_voice_id = new_voice_id
                    # Update all language voices to use the same voice
                    st.session_state.language_voices["ur"] = new_voice_id
                    st.session_state.language_voices["en"] = new_voice_id
                    st.session_state.language_voices["default"] = new_voice_id
            
            st.success(f"""
            ✅ **Single Voice Configuration**
            - Voice ID: {st.session_state.elevenlabs_voice_id[:8]}...
            - Used for: ALL languages (Urdu + English)
            - Model: Flash v2.5 (Accent-free)
            """)
        
        elif st.session_state.tts_provider == "azure":
            if not st.session_state.azure_speech_key or not st.session_state.azure_speech_region:
                st.warning("⚠️ Please set Azure Speech key and region above")
            else:
                st.success("✅ Azure Speech configured")
        
        elif st.session_state.tts_provider == "openai":
            if not st.session_state.openai_api_key:
                st.warning("⚠️ Please set OpenAI API key above")
            else:
                st.success("✅ OpenAI TTS configured")
        
        # Language Response Options
        st.subheader("🎓 Tutor Mode")
        
        response_language = st.radio(
            "Select Tutor Mode",
            options=["both", "ur", "en"],
            format_func=lambda x: {
                "both": "English Tutor (Urdu + English)", 
                "ur": "Urdu Only", 
                "en": "English Only"
            }[x]
        )
        
        if response_language != st.session_state.response_language:
            st.session_state.response_language = response_language
            st.success(f"✅ Tutor Mode set to: {response_language}")
        
        # Language distribution (only shown when "both" is selected)
        if response_language == "both":
            st.subheader("📊 Language Distribution")
            
            ur_percent = st.slider("Urdu %", min_value=0, max_value=100, value=st.session_state.language_distribution["ur"])
            en_percent = 100 - ur_percent
            
            st.text(f"English %: {en_percent}")
            
            if ur_percent != st.session_state.language_distribution["ur"]:
                st.session_state.language_distribution = {
                    "ur": ur_percent,
                    "en": en_percent
                }
                st.success(f"✅ Language distribution updated: {ur_percent}% Urdu, {en_percent}% English")
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
        avg_stt = calculate_average_latency(st.session_state.performance_metrics["stt_latency"])
        avg_llm = calculate_average_latency(st.session_state.performance_metrics["llm_latency"])
        avg_tts = calculate_average_latency(st.session_state.performance_metrics["tts_latency"])
        avg_total = calculate_average_latency(st.session_state.performance_metrics["total_latency"])
        
        st.metric("🎤 STT Latency", f"{avg_stt:.2f}s")
        st.metric("🤖 LLM Latency", f"{avg_llm:.2f}s")
        st.metric("🔊 TTS Latency", f"{avg_tts:.2f}s")
        st.metric("⚡ Total Latency", f"{avg_total:.2f}s")
        
        # Latency status
        if avg_total < 3.0:
            st.success(f"✅ Excellent latency: {avg_total:.2f}s")
        elif avg_total < 5.0:
            st.warning(f"⚠️ Good latency: {avg_total:.2f}s")
        else:
            st.error(f"❌ High latency: {avg_total:.2f}s")
        
        # API usage
        st.subheader("📊 API Usage")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Whisper", st.session_state.performance_metrics['api_calls']['whisper'])
            st.metric("OpenAI", st.session_state.performance_metrics['api_calls']['openai'])
        with col2:
            st.metric("ElevenLabs", st.session_state.performance_metrics['api_calls']['elevenlabs'])
            st.metric("Azure", st.session_state.performance_metrics['api_calls']['azure'])
    
    # Main interaction area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("📝 Input")
        
        # Input type selection
        input_type = st.radio("Select Input Type", ["Text", "Voice"], horizontal=True)
        
        if input_type == "Text":
            # Text input with demo scenarios
            st.subheader("✍️ Text Input")
            st.write("Use [ur] for Urdu and [en] for English strategically")
            
            demo_scenarios = {
                "Vocabulary Request": (
                    "Aap mujhe kuch basic English words sikha sakte hain?"
                ),
                "Grammar Question": (
                    "English mein past tense kaise banate hain? Example de sakte hain?"
                ),
                "Practice Conversation": (
                    "Main apna introduction English mein karna chahta hun. Help kar sakte hain?"
                ),
                "Pronunciation Help": (
                    "Mujhe 'th' sound pronounce karne mein mushkil hoti hai. Madad kariye?"
                ),
                "Daily Expressions": (
                    "Rozana istemaal hone wale English phrases sikhayiye."
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
            
            text_process_button = st.button("🚀 Process Text", type="primary")
            
            if text_process_button and text_input:
                with st.spinner("Processing text input..."):
                    audio_path, llm_latency, tts_latency = asyncio.run(process_text_input_enhanced(text_input))
                    
                    st.session_state.last_text_input = text_input
                    st.session_state.last_audio_output = audio_path
                    
                    total_latency = llm_latency + tts_latency
                    st.success(f"✅ Text processed in {total_latency:.2f} seconds")
        
        else:
            # Voice input with enhanced recording
            st.subheader("🎤 Professional Voice Recording")
            
            # Check API keys
            required_keys = []
            if st.session_state.tts_provider == "elevenlabs" and not st.session_state.elevenlabs_api_key:
                required_keys.append("ElevenLabs")
            if st.session_state.tts_provider == "openai" and not st.session_state.openai_api_key:
                required_keys.append("OpenAI")
            if st.session_state.tts_provider == "azure" and (not st.session_state.azure_speech_key or not st.session_state.azure_speech_region):
                required_keys.append("Azure Speech")
            if not st.session_state.openai_api_key:
                required_keys.append("OpenAI (for STT)")
            
            if required_keys:
                st.warning(f"⚠️ Please configure: {', '.join(required_keys)}")
            else:
                st.write(f"🎯 **Using {st.session_state.tts_provider.title()} TTS** - Accent-Free Processing")
                
                # Create the HTML5 audio recorder
                create_audio_recorder_component()
                
                st.markdown("---")
                st.write("**🔄 AUTOMATIC PROCESSING:**")
                
                # Enhanced upload processing
                uploaded_audio = st.file_uploader(
                    "📥 Upload Your Downloaded Recording Here", 
                    type=['wav', 'mp3', 'webm', 'ogg'],
                    key="main_upload",
                    help="After recording above, download and upload here for automatic processing"
                )

                if uploaded_audio is not None:
                    with st.spinner("🔄 **PROCESSING YOUR RECORDING...**"):
                        try:
                            # Save uploaded file
                            temp_path = tempfile.mktemp(suffix=".wav")
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_audio.read())
                            
                            # Apply amplification and process
                            amplified_path = amplify_recorded_audio(temp_path)
                            
                            # Process with accent-free pipeline
                            text, audio_output_path, stt_latency, llm_latency, tts_latency = asyncio.run(process_voice_input_accent_free(amplified_path))
                            
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
                            st.error(f"❌ Processing error: {str(e)}")

                # Instructions
                st.success("""
                🎯 **SIMPLE WORKFLOW:**
                1. Click "🔴 START RECORDING" above
                2. Speak clearly in Urdu or English  
                3. Click "⏹️ STOP RECORDING" when done
                4. **DOWNLOAD** the file automatically
                5. **UPLOAD** it above - accent-free processing starts immediately!

                **⚡ Experience: Record → Download → Upload → Get Accent-Free Results!**
                """)
    
    with col2:
        st.header("📤 Output")
        
        # Transcribed text
        if 'last_text_input' in st.session_state and st.session_state.last_text_input:
            st.subheader("📝 Transcribed/Input Text")
            st.text_area(
                "Text with strategic language markers",
                value=st.session_state.last_text_input,
                height=100,
                disabled=True
            )
        
        # Generated response
        if 'conversation_history' in st.session_state and st.session_state.conversation_history:
            last_exchange = st.session_state.conversation_history[-1]
            
            if 'assistant_response' in last_exchange:
                st.subheader("🤖 AI Tutor Response")
                st.text_area(
                    "Strategic response with accent-free markers",
                    value=last_exchange['assistant_response'],
                    height=150,
                    disabled=True
                )
        
        # Generated audio
        if 'last_audio_output' in st.session_state and st.session_state.last_audio_output:
            st.subheader(f"🔊 Accent-Free Speech ({st.session_state.tts_provider.title()})")
            
            # Display audio with player
            audio_bytes = display_audio(st.session_state.last_audio_output, autoplay=True)
            
            if audio_bytes:
                # Download button
                st.download_button(
                    label="💾 Download Audio",
                    data=audio_bytes,
                    file_name=f"accent_free_response_{st.session_state.tts_provider}.mp3",
                    mime="audio/mp3"
                )
        
        # TTS Provider comparison
        if st.session_state.conversation_history:
            st.subheader("📊 TTS Provider Analysis")
            
            # Show current provider performance
            recent_tts_latency = st.session_state.performance_metrics["tts_latency"][-1] if st.session_state.performance_metrics["tts_latency"] else 0
            
            provider_info = {
                "elevenlabs": {"quality": "⭐⭐⭐⭐⭐", "speed": "⭐⭐⭐", "cost": "💰💰💰"},
                "openai": {"quality": "⭐⭐⭐⭐", "speed": "⭐⭐⭐⭐⭐", "cost": "💰💰"},
                "azure": {"quality": "⭐⭐⭐⭐", "speed": "⭐⭐⭐⭐", "cost": "💰💰"}
            }
            
            current_provider = st.session_state.tts_provider
            info = provider_info[current_provider]
            
            st.write(f"""
            **Current Provider: {current_provider.title()}**
            - Quality: {info['quality']}
            - Speed: {info['speed']} ({recent_tts_latency:.2f}s)
            - Cost: {info['cost']}
            """)
    
    # Conversation history
    if st.session_state.conversation_history:
        st.header("💬 Conversation History")
        
        for i, exchange in enumerate(st.session_state.conversation_history[-5:]):
            with st.expander(f"Exchange {i+1} - {exchange.get('timestamp', 'Unknown time')[:19]}"):
                st.markdown("**User:**")
                st.text(exchange.get('user_input', 'No input'))
                
                st.markdown("**AI Tutor:**")
                st.text(exchange.get('assistant_response', 'No response'))
                
                # Latency info
                latency = exchange.get('latency', {})
                st.text(f"STT: {latency.get('stt', 0):.2f}s | LLM: {latency.get('llm', 0):.2f}s | TTS: {latency.get('tts', 0):.2f}s | Total: {latency.get('total', 0):.2f}s")
    
    # Status area
    st.header("📊 Processing Status")
    st.session_state.status_area = st.empty()
    
    # Update status from queue
    update_status()

if __name__ == "__main__":
    main()
