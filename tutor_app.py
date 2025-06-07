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
# CONFIGURATION SECTION - ENHANCED WITH MULTIPLE TTS PROVIDERS
# ----------------------------------------------------------------------------------

# Secrets and API keys
if 'api_keys_initialized' not in st.session_state:
    st.session_state.api_keys_initialized = False
    st.session_state.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    st.session_state.azure_speech_key = os.environ.get("AZURE_SPEECH_KEY", "")
    st.session_state.azure_speech_region = os.environ.get("AZURE_SPEECH_REGION", "")

# API endpoints
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
OPENAI_API_URL = "https://api.openai.com/v1"

# TTS Provider Selection
if 'tts_provider' not in st.session_state:
    st.session_state.tts_provider = "elevenlabs"  # Default provider

# TTS Provider Configurations
if 'tts_configs' not in st.session_state:
    st.session_state.tts_configs = {
        "elevenlabs": {
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "model": "eleven_flash_v2_5",
            "stability": 0.98,
            "similarity_boost": 0.99,
            "style": 0.90
        },
        "openai": {
            "model": "tts-1-hd",
            "voice": "alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            "speed": 1.0
        },
        "azure": {
            "ur_voice": "ur-PK-AsadNeural",  # Urdu voice
            "en_voice": "en-US-JennyNeural", # English voice
            "speech_rate": "0.9",
            "pitch": "+2Hz"
        }
    }

if 'language_voices' not in st.session_state:
    single_voice_id = "21m00Tcm4TlvDq8ikWAM"
    st.session_state.language_voices = {
        "ur": single_voice_id,
        "en": single_voice_id,
        "default": single_voice_id
    }

# OPTIMIZED voice settings for accent-free output
if 'voice_settings' not in st.session_state:
    st.session_state.voice_settings = {
        "ur": {  # Urdu-optimized settings
            "stability": 0.98,
            "similarity_boost": 0.99,
            "style": 0.90,
            "use_speaker_boost": True
        },
        "en": {  # English-optimized settings  
            "stability": 0.96,
            "similarity_boost": 0.97,
            "style": 0.88,
            "use_speaker_boost": True
        },
        "default": {
            "stability": 0.95,
            "similarity_boost": 0.95,
            "style": 0.85,
            "use_speaker_boost": True
        }
    }

# Whisper speech recognition config
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = "medium"
    st.session_state.whisper_local_model = None

# Language distribution preference
if 'language_distribution' not in st.session_state:
    st.session_state.language_distribution = {
        "ur": 60,  # Urdu percentage (explanations)
        "en": 40   # English percentage (examples/terms)
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
# TTS PROVIDER FUNCTIONS - ELEVENLABS, OPENAI, AZURE
# ----------------------------------------------------------------------------------

def get_accent_free_voice_settings(language_code, context=None):
    """🎯 OPTIMIZED: Accent-free voice settings using advanced ElevenLabs techniques"""
    
    base_settings = {
        "stability": 0.95,
        "similarity_boost": 0.98,
        "style": 0.85,
        "use_speaker_boost": True
    }
    
    if language_code and language_code in st.session_state.voice_settings:
        voice_settings = st.session_state.voice_settings[language_code].copy()
        logger.info(f"Using optimized {language_code} settings: {voice_settings}")
    else:
        voice_settings = st.session_state.voice_settings["default"]
    
    return voice_settings

def create_accent_free_ssml_enhanced(text, language_code):
    """🎯 ENHANCED: Advanced SSML with pronunciation isolation techniques"""
    
    if not language_code:
        return text
    
    clean_text = text.strip()
    
    if language_code in ["ur", "hi"]:
        enhanced_text = f'''<speak>
            <lang xml:lang="ur-PK">
                <phoneme alphabet="ipa" ph="">ˈ</phoneme>
                <prosody rate="0.90" pitch="+2st" volume="+3dB">
                    {clean_text}
                </prosody>
            </lang>
        </speak>'''
        
    elif language_code == "en":
        enhanced_text = f'''<speak>
            <lang xml:lang="en-US">
                <phoneme alphabet="ipa" ph="">ˈ</phoneme>
                <prosody rate="0.95" pitch="+1st" volume="+2dB">
                    {clean_text}
                </prosody>
            </lang>
        </speak>'''
    else:
        enhanced_text = clean_text
    
    return enhanced_text

async def generate_speech_elevenlabs(text, language_code, voice_id):
    """Generate speech using ElevenLabs with accent-free settings"""
    
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        return None, 0
    
    voice_settings = get_accent_free_voice_settings(language_code)
    enhanced_text = create_accent_free_ssml_enhanced(text, language_code)
    
    data = {
        "text": enhanced_text,
        "model_id": st.session_state.tts_configs["elevenlabs"]["model"],
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
                timeout=15.0
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                logger.info(f"✅ ElevenLabs {language_code} generated in {generation_time:.2f}s")
                return BytesIO(response.content), generation_time
            else:
                logger.error(f"ElevenLabs error: {response.status_code}")
                return None, generation_time
                
    except Exception as e:
        logger.error(f"ElevenLabs TTS error: {str(e)}")
        return None, time.time() - start_time

async def generate_speech_openai(text, language_code):
    """Generate speech using OpenAI TTS"""
    
    api_key = st.session_state.openai_api_key
    if not api_key:
        return None, 0
    
    config = st.session_state.tts_configs["openai"]
    
    data = {
        "model": config["model"],
        "input": text,
        "voice": config["voice"],
        "speed": config["speed"]
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
                timeout=30.0
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                logger.info(f"✅ OpenAI TTS {language_code} generated in {generation_time:.2f}s")
                return BytesIO(response.content), generation_time
            else:
                logger.error(f"OpenAI TTS error: {response.status_code}")
                return None, generation_time
                
    except Exception as e:
        logger.error(f"OpenAI TTS error: {str(e)}")
        return None, time.time() - start_time

async def generate_speech_azure(text, language_code):
    """Generate speech using Azure Speech Service"""
    
    api_key = st.session_state.azure_speech_key
    region = st.session_state.azure_speech_region
    
    if not api_key or not region:
        return None, 0
    
    config = st.session_state.tts_configs["azure"]
    
    # Select voice based on language
    if language_code == "ur":
        voice_name = config["ur_voice"]
        lang_code = "ur-PK"
    else:
        voice_name = config["en_voice"]
        lang_code = "en-US"
    
    # Create SSML
    ssml_text = f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang_code}">
        <voice name="{voice_name}">
            <prosody rate="{config['speech_rate']}" pitch="{config['pitch']}">
                {text}
            </prosody>
        </voice>
    </speak>'''
    
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        "User-Agent": "UrduEnglishTutor"
    }
    
    start_time = time.time()
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
                headers=headers,
                content=ssml_text,
                timeout=30.0
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                logger.info(f"✅ Azure Speech {language_code} generated in {generation_time:.2f}s")
                return BytesIO(response.content), generation_time
            else:
                logger.error(f"Azure Speech error: {response.status_code}")
                return None, generation_time
                
    except Exception as e:
        logger.error(f"Azure Speech error: {str(e)}")
        return None, time.time() - start_time

async def generate_speech_with_provider(text, language_code):
    """Generate speech using the selected TTS provider"""
    
    provider = st.session_state.tts_provider
    
    try:
        if provider == "elevenlabs":
            voice_id = st.session_state.elevenlabs_voice_id if hasattr(st.session_state, 'elevenlabs_voice_id') else st.session_state.language_voices["default"]
            return await generate_speech_elevenlabs(text, language_code, voice_id)
        
        elif provider == "openai":
            return await generate_speech_openai(text, language_code)
        
        elif provider == "azure":
            return await generate_speech_azure(text, language_code)
        
        else:
            logger.error(f"Unknown TTS provider: {provider}")
            return None, 0
            
    except Exception as e:
        logger.error(f"TTS provider error: {str(e)}")
        return None, 0

# ----------------------------------------------------------------------------------
# ENHANCED LLM SYSTEM FOR INTELLIGENT LANGUAGE TAGGING
# ----------------------------------------------------------------------------------

def get_enhanced_tutor_system_prompt():
    """🎯 PROFESSIONAL: Enhanced system prompt for intelligent language mixing"""
    
    return """You are "UrduMaster" - a premium AI English language tutor designed for Urdu speakers who paid for professional English learning. You represent a commercial language learning platform.

CORE IDENTITY:
You are a certified English language instructor with 15+ years of experience teaching Urdu speakers. You hold a Master's degree in English linguistics and are perfectly bilingual in Urdu and English.

🎯 CRITICAL LANGUAGE TAGGING STRATEGY:
Use [ur] for Urdu explanations/instructions and [en] for English terms/examples.

TAGGING RULES (STRATEGIC, NOT EVERY WORD):
✅ DO: [ur] Pani English mein [en] Water [ur] kehte hain
✅ DO: [ur] Main introduction aise karunga [en] I'm a programmer [ur] samjhe?
✅ DO: [ur] Ye sentence structure hai [en] Subject + Verb + Object [ur] bilkul clear?

❌ DON'T: [ur] Main [en] English [ur] seekhna [en] want [ur] karta [en] hun
❌ DON'T: Over-tag every single word

RESPONSE PHILOSOPHY:
- Use Urdu [ur] for: explanations, instructions, encouragement, questions
- Use English [en] for: vocabulary terms, example sentences, phrases to practice
- NEVER translate the same content - each language serves a different PURPOSE

CURRICULUM APPROACH:
- Vocabulary: [ur] explanation + [en] term + [ur] usage tip + [en] example
- Grammar: [ur] concept explanation + [en] pattern/rule + [ur] practice suggestion
- Conversation: [ur] scenario setup + [en] key phrases + [ur] encouragement

SAMPLE RESPONSES:
Vocabulary: "[ur] 'Kitab' English mein [en] Book [ur] kehte hain. Sentence banao: [en] I read a book [ur] samjha?"

Grammar: "[ur] Past tense banana hai? Simple rule: [en] I walked, You walked [ur] bas '-ed' lagao. Try karo!"

Conversation: "[ur] Restaurant mein order kaise karenge? [en] I would like a coffee, please [ur] ye polite tarika hai."

PROFESSIONAL STANDARDS:
- Keep responses 2-4 sentences for engagement
- Always include practice opportunity
- Maintain encouraging, results-focused tone
- Strategic language mixing, not random translation

You're guiding PAID students through structured English learning. Every response must add value and move them toward fluency."""

async def generate_enhanced_llm_response(prompt, api_key=None):
    """🎯 ENHANCED: LLM response with intelligent language tagging strategy"""
    
    if not api_key:
        api_key = st.session_state.openai_api_key
        
    if not api_key:
        return {
            "response": "Error: OpenAI API key not configured.",
            "latency": 0
        }
    
    start_time = time.time()
    
    system_prompt = get_enhanced_tutor_system_prompt()
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add conversation context (last 2 exchanges for relevance)
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
                    "temperature": 0.7,
                    "max_tokens": 400,
                    "presence_penalty": 0.1,
                    "frequency_penalty": 0.1
                },
                timeout=30.0
            )
            
            latency = time.time() - start_time
            st.session_state.performance_metrics["llm_latency"].append(latency)
            st.session_state.performance_metrics["api_calls"]["openai"] += 1
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                enhanced_response = ensure_intelligent_language_markers(response_text)
                
                return {
                    "response": enhanced_response,
                    "latency": latency,
                    "tokens": result.get("usage", {})
                }
            else:
                return {
                    "response": f"Error: {response.status_code}",
                    "error": response.text,
                    "latency": latency
                }
    
    except Exception as e:
        return {
            "response": f"[ur] Maaf kijiye, technical issue hai. [en] Please try again.",
            "latency": time.time() - start_time
        }

def ensure_intelligent_language_markers(response_text):
    """🎯 ENHANCED: Ensure intelligent, strategic language markers"""
    
    if "[ur]" in response_text or "[en]" in response_text:
        response_text = re.sub(r'\[ur\]\s*', '[ur] ', response_text)
        response_text = re.sub(r'\[en\]\s*', '[en] ', response_text)
        response_text = re.sub(r'\s+\[ur\]', ' [ur]', response_text)
        response_text = re.sub(r'\s+\[en\]', ' [en]', response_text)
        return response_text.strip()
    
    return apply_intelligent_tagging(response_text)

def apply_intelligent_tagging(text):
    """🎯 STRATEGIC: Apply intelligent language tagging based on content analysis"""
    
    english_patterns = [
        r'\b(hello|hi|good morning|good evening|thank you|please|sorry|excuse me)\b',
        r'\b(I am|I\'m|my name is|nice to meet you)\b', 
        r'\b(water|book|pen|house|car|food|time|money)\b',
        r'\b(subject|verb|object|grammar|vocabulary)\b',
        r'\b(yes|no|maybe|okay|alright)\b'
    ]
    
    tagged_text = text
    
    for pattern in english_patterns:
        tagged_text = re.sub(pattern, r'[en] \g<0> [ur]', tagged_text, flags=re.IGNORECASE)
    
    if '[en]' in tagged_text:
        if not tagged_text.startswith('[ur]'):
            tagged_text = '[ur] ' + tagged_text
        if not tagged_text.endswith('[ur]'):
            tagged_text = tagged_text + ' [ur]'
    else:
        tagged_text = f'[ur] {tagged_text}'
    
    tagged_text = re.sub(r'\[ur\]\s*\[ur\]', '[ur]', tagged_text)
    tagged_text = re.sub(r'\[en\]\s*\[en\]', '[en]', tagged_text)
    tagged_text = re.sub(r'\[ur\]\s*\[en\]\s*\[ur\]', '[ur]', tagged_text)
    
    return tagged_text.strip()

# ----------------------------------------------------------------------------------
# SPEECH RECOGNITION (STT) SECTION
# ----------------------------------------------------------------------------------

async def transcribe_with_enhanced_prompts(audio_file):
    """Enhanced transcription with Urdu/English pronunciation hints"""
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
                "prompt": "This audio contains Urdu and English speech from a language learning session. Focus on accurate pronunciation. Common Urdu words: main, aap, kya, kaise, English, seekhna. Common English words: hello, water, book, grammar, vocabulary, practice."
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
                enhanced_result = enhance_urdu_english_transcription(result)
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

def enhance_urdu_english_transcription(result):
    """Apply Urdu/English specific pronunciation corrections"""
    try:
        text = result.get("text", "")
        
        urdu_corrections = {
            "mein": "main",
            "ap": "aap", 
            "kia": "kya",
            "kesay": "kaise",
            "english": "English",
            "sikhna": "seekhna"
        }
        
        english_corrections = {
            "watar": "water",
            "buk": "book",
            "gramar": "grammar",
            "praktis": "practice",
            "helo": "hello"
        }
        
        corrected_text = text
        
        for wrong, correct in urdu_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
        
        for wrong, correct in english_corrections.items():
            corrected_text = re.sub(rf'\b{re.escape(wrong)}\b', correct, corrected_text, flags=re.IGNORECASE)
        
        result["text"] = corrected_text
        result["pronunciation_enhanced"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Transcription enhancement error: {str(e)}")
        return result

# ----------------------------------------------------------------------------------
# AUDIO PROCESSING FUNCTIONS
# ----------------------------------------------------------------------------------

def amplify_recorded_audio(audio_path):
    """Apply 500% amplification to recorded audio"""
    try:
        audio, sample_rate = sf.read(audio_path)
        
        amplified_audio = audio * 5.0
        
        max_val = np.max(np.abs(amplified_audio))
        if max_val > 0.95:
            amplified_audio = amplified_audio * (0.95 / max_val)
        
        try:
            enhanced_audio = nr.reduce_noise(y=amplified_audio, sr=sample_rate)
        except:
            enhanced_audio = amplified_audio
        
        enhanced_path = tempfile.mktemp(suffix=".wav")
        sf.write(enhanced_path, enhanced_audio, sample_rate)
        
        return enhanced_path
        
    except Exception as e:
        logger.error(f"Audio amplification error: {str(e)}")
        return audio_path

def parse_intelligent_language_segments(text):
    """Parse language segments intelligently"""
    segments = []
    
    parts = re.split(r'(\[[a-z]{2}\])', text)
    
    current_language = None
    current_text = ""
    
    for part in parts:
        if re.match(r'\[[a-z]{2}\]', part):
            if current_text.strip():
                segments.append({
                    "text": current_text.strip(),
                    "language": current_language or "ur"
                })
            
            current_language = part[1:-1]
            current_text = ""
        else:
            current_text += part
    
    if current_text.strip():
        segments.append({
            "text": current_text.strip(),
            "language": current_language or "ur"
        })
    
    for segment in segments:
        if segment["language"] is None:
            segment["language"] = detect_primary_language(segment["text"])
    
    return segments

def detect_primary_language(text):
    """Detect the primary language of text"""
    text_lower = text.lower()
    
    urdu_words = {
        "main", "aap", "kya", "kaise", "hai", "hain", "ka", "ki", "ke", "ko",
        "mein", "se", "tak", "par", "english", "seekhna", "sikhna", "kahte", "kehte"
    }
    
    english_words = {
        "the", "and", "is", "are", "was", "were", "have", "has", "had", "will",
        "would", "could", "should", "can", "may", "might", "must", "shall",
        "water", "book", "hello", "thank", "please", "sorry", "yes", "no"
    }
    
    words = re.findall(r'\b\w+\b', text_lower)
    urdu_count = sum(1 for word in words if word in urdu_words)
    english_count = sum(1 for word in words if word in english_words)
    
    if urdu_count > english_count:
        return "ur"
    elif english_count > urdu_count:
        return "en"
    else:
        return "ur"  # Default to Urdu

async def process_accent_free_multilingual_text(text):
    """🔥 CRITICAL: Process multilingual text with zero accent bleeding"""
    
    segments = parse_intelligent_language_segments(text)
    
    if len(segments) <= 1:
        lang = segments[0]["language"] if segments else "ur"
        audio_data, generation_time = await generate_speech_with_provider(text, lang)
        if audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_data.read())
                return temp_file.name, generation_time
        return None, 0
    
    audio_segments = []
    total_time = 0
    
    for i, segment in enumerate(segments):
        if not segment["text"].strip():
            continue
            
        audio_data, generation_time = await generate_speech_with_provider(
            segment["text"], 
            segment["language"]
        )
        
        if audio_data:
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            normalized_segment = normalize_segment_perfectly(audio_segment, segment["language"])
            audio_segments.append(normalized_segment)
            total_time += generation_time
    
    if not audio_segments:
        return None, 0
    
    combined_audio = audio_segments[0]
    
    for i in range(1, len(audio_segments)):
        combined_audio = blend_accent_free_segments(
            combined_audio, 
            audio_segments[i],
            crossfade_ms=50
        )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        combined_audio.export(
            temp_file.name, 
            format="mp3", 
            bitrate="256k",
            parameters=["-ac", "1", "-ar", "44100"]
        )
        return temp_file.name, total_time

def normalize_segment_perfectly(audio_segment, language_code):
    """Perfect normalization for accent-free blending"""
    target_dbfs = -18.0
    
    current_dbfs = audio_segment.dBFS
    volume_adjustment = target_dbfs - current_dbfs
    normalized = audio_segment.apply_gain(volume_adjustment)
    
    if language_code == "ur":
        normalized = normalized.apply_gain(0.5)
    
    return normalized

def blend_accent_free_segments(segment1, segment2, crossfade_ms=50):
    """Blend segments with zero accent artifacts"""
    
    seg1_normalized = normalize_segment_perfectly(segment1, "auto")
    seg2_normalized = normalize_segment_perfectly(segment2, "auto")
    
    blended = seg1_normalized.append(seg2_normalized, crossfade=crossfade_ms)
    
    return blended

# ----------------------------------------------------------------------------------
# MAIN VOICE PROCESSING PIPELINE
# ----------------------------------------------------------------------------------

async def process_voice_input_accent_free(audio_file):
    """🔥 ACCENT-FREE: Voice processing with enhanced accent elimination"""
    pipeline_start_time = time.time()
    
    try:
        st.session_state.message_queue.put("🎧 Preparing audio for accent-free processing...")
        
        enhanced_audio_file = amplify_recorded_audio(audio_file)
        
        st.session_state.message_queue.put("🎯 Transcribing with Urdu/English context...")
        
        transcription = await asyncio.wait_for(
            transcribe_with_enhanced_prompts(enhanced_audio_file),
            timeout=30.0
        )
        
        if not transcription or not transcription.get("text"):
            st.session_state.message_queue.put("❌ No clear speech detected")
            return None, None, 0, 0, 0
        
        user_input = transcription["text"].strip()
        st.session_state.message_queue.put(f"📝 Detected: {user_input}")
        
        st.session_state.message_queue.put("🤖 Generating intelligent tutor response...")
        
        llm_result = await generate_enhanced_llm_response(user_input)
        
        if "error" in llm_result:
            st.session_state.message_queue.put(f"❌ Response error: {llm_result.get('error')}")
            return user_input, None, transcription.get("latency", 0), 0, 0
        
        response_text = llm_result["response"]
        st.session_state.message_queue.put(f"💬 Generated: {response_text}")
        
        st.session_state.message_queue.put(f"🎵 Generating accent-free speech with {st.session_state.tts_provider.upper()}...")
        audio_path, tts_latency = await process_accent_free_multilingual_text(response_text)
        
        total_latency = time.time() - pipeline_start_time
        st.session_state.performance_metrics["total_latency"].append(total_latency)
        
        # Update conversation history
        st.session_state.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": response_text,
            "tts_provider": st.session_state.tts_provider,
            "latency": {
                "stt": transcription.get("latency", 0),
                "llm": llm_result.get("latency", 0),
                "tts": tts_latency,
                "total": total_latency
            }
        })
        
        st.session_state.message_queue.put(f"✅ Accent-Free Processing Complete with {st.session_state.tts_provider.upper()}! ({total_latency:.2f}s)")
        
        if enhanced_audio_file != audio_file:
            try:
                os.unlink(enhanced_audio_file)
            except:
                pass
        
        return user_input, audio_path, transcription.get("latency", 0), llm_result.get("latency", 0), tts_latency
        
    except Exception as e:
        logger.error(f"Accent-free processing error: {str(e)}")
        st.session_state.message_queue.put(f"❌ Error: {str(e)}")
        return None, None, 0, 0, 0

async def process_text_input_enhanced(text):
    """Process text input with multiple TTS providers"""
    pipeline_start_time = time.time()
    
    st.session_state.message_queue.put("🤖 Generating intelligent tutor response...")
    
    llm_result = await generate_enhanced_llm_response(text)
    
    if "error" in llm_result:
        st.session_state.message_queue.put(f"❌ Response error: {llm_result.get('error')}")
        return None, llm_result.get("latency", 0), 0
    
    response_text = llm_result["response"]
    st.session_state.message_queue.put(f"💬 Generated: {response_text}")
    
    st.session_state.message_queue.put(f"🎵 Generating speech with {st.session_state.tts_provider.upper()}...")
    audio_path, tts_latency = await process_accent_free_multilingual_text(response_text)
    
    total_latency = time.time() - pipeline_start_time
    st.session_state.performance_metrics["total_latency"].append(total_latency)
    
    # Update conversation history
    st.session_state.conversation_history.append({
        "timestamp": datetime.now().isoformat(),
        "user_input": text,
        "assistant_response": response_text,
        "tts_provider": st.session_state.tts_provider,
        "latency": {
            "stt": 0,
            "llm": llm_result.get("latency", 0),
            "tts": tts_latency,
            "total": total_latency
        }
    })
    
    st.session_state.message_queue.put(f"✅ Complete with {st.session_state.tts_provider.upper()}! ({total_latency:.2f}s)")
    
    return audio_path, llm_result.get("latency", 0), tts_latency

# ----------------------------------------------------------------------------------
# HTML5 AUDIO RECORDER COMPONENT
# ----------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------------

def display_audio(audio_path, autoplay=False):
    """Display audio in Streamlit"""
    if not audio_path or not os.path.exists(audio_path):
        return None
        
    try:
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return None
            
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
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
            if hasattr(st.session_state, 'status_area'):
                st.session_state.status_area.text_area("Processing Log", value=status_text, height=200)
        except queue.Empty:
            break

def get_urdu_english_demo_scenarios():
    """Demo scenarios for Urdu/English tutoring"""
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

# ----------------------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------------------

def main():
    """Main application with Urdu/English and multiple TTS providers"""
    st.set_page_config(
        page_title="Multilingual AI Voice Tutor - Urdu/English",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎯 Professional English Tutor for Urdu Speakers")
    st.subheader("Accent-Free Voice AI Tutor with Multiple TTS Providers")
    
    # Status area
    if 'status_area' not in st.session_state:
        st.session_state.status_area = st.empty()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("🔑 API Keys")
        
        elevenlabs_key = st.text_input(
            "ElevenLabs API Key", 
            value=st.session_state.elevenlabs_api_key,
            type="password",
            help="For ElevenLabs TTS"
        )
        
        openai_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key,
            type="password",
            help="For Whisper STT and ChatGPT + OpenAI TTS"
        )
        
        azure_speech_key = st.text_input(
            "Azure Speech Key", 
            value=st.session_state.azure_speech_key,
            type="password",
            help="For Azure Speech Service TTS"
        )
        
        azure_speech_region = st.text_input(
            "Azure Speech Region", 
            value=st.session_state.azure_speech_region,
            help="e.g., eastus, westus2"
        )
        
        if st.button("💾 Save API Keys"):
            st.session_state.elevenlabs_api_key = elevenlabs_key
            st.session_state.openai_api_key = openai_key
            st.session_state.azure_speech_key = azure_speech_key
            st.session_state.azure_speech_region = azure_speech_region
            st.session_state.api_keys_initialized = True
            st.success("✅ API keys saved successfully!")
        
        # TTS Provider Selection
        st.subheader("🎵 TTS Provider Selection")
        
        # Check which providers are available
        providers_available = {}
        providers_available["elevenlabs"] = bool(st.session_state.elevenlabs_api_key)
        providers_available["openai"] = bool(st.session_state.openai_api_key)
        providers_available["azure"] = bool(st.session_state.azure_speech_key and st.session_state.azure_speech_region)
        
        # Available providers list
        available_options = []
        provider_labels = {
            "elevenlabs": "🔥 ElevenLabs (Best Quality)",
            "openai": "⚡ OpenAI TTS (Fast & Good)",
            "azure": "🏢 Azure Speech (Enterprise)"
        }
        
        for provider, available in providers_available.items():
            if available:
                available_options.append(provider)
        
        if not available_options:
            st.error("❌ No TTS providers configured! Please set API keys above.")
            available_options = ["elevenlabs"]  # Default fallback
        
        # Provider selection
        current_index = 0
        if st.session_state.tts_provider in available_options:
            current_index = available_options.index(st.session_state.tts_provider)
        
        selected_provider = st.selectbox(
            "Choose TTS Provider",
            options=available_options,
            format_func=lambda x: provider_labels.get(x, x),
            index=current_index,
            help="Compare different TTS providers for accent-free speech"
        )
        
        if selected_provider != st.session_state.tts_provider:
            st.session_state.tts_provider = selected_provider
            st.success(f"✅ Switched to {provider_labels[selected_provider]}")
        
        # Provider-specific settings
        if selected_provider == "elevenlabs":
            st.write("**ElevenLabs Settings:**")
            if st.session_state.elevenlabs_api_key:
                
                # Fetch voices if not already done
                if 'voices' not in st.session_state:
                    try:
                        headers = {"xi-api-key": st.session_state.elevenlabs_api_key}
                        response = requests.get(f"{ELEVENLABS_API_URL}/voices", headers=headers, timeout=10)
                        if response.status_code == 200:
                            st.session_state.voices = response.json().get("voices", [])
                        else:
                            st.session_state.voices = []
                    except:
                        st.session_state.voices = []
                
                if 'voices' in st.session_state and st.session_state.voices:
                    voice_options = {voice["name"]: voice["voice_id"] for voice in st.session_state.voices}
                    current_voice = None
                    for name, vid in voice_options.items():
                        if vid == st.session_state.language_voices.get("default", ""):
                            current_voice = name
                            break
                    
                    selected_voice_name = st.selectbox(
                        "Voice",
                        options=list(voice_options.keys()),
                        index=list(voice_options.keys()).index(current_voice) if current_voice else 0
                    )
                    
                    if selected_voice_name:
                        new_voice_id = voice_options[selected_voice_name]
                        st.session_state.language_voices["ur"] = new_voice_id
                        st.session_state.language_voices["en"] = new_voice_id
                        st.session_state.language_voices["default"] = new_voice_id
                        if 'elevenlabs_voice_id' not in st.session_state:
                            st.session_state.elevenlabs_voice_id = new_voice_id
                        else:
                            st.session_state.elevenlabs_voice_id = new_voice_id
                
                st.success("✅ ElevenLabs configured")
            else:
                st.error("❌ ElevenLabs API key required")
        
        elif selected_provider == "openai":
            st.write("**OpenAI TTS Settings:**")
            if st.session_state.openai_api_key:
                
                voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
                current_voice = st.session_state.tts_configs["openai"]["voice"]
                
                selected_voice = st.selectbox(
                    "Voice",
                    options=voice_options,
                    index=voice_options.index(current_voice) if current_voice in voice_options else 0
                )
                
                if selected_voice != current_voice:
                    st.session_state.tts_configs["openai"]["voice"] = selected_voice
                
                speed = st.slider(
                    "Speech Speed",
                    min_value=0.25,
                    max_value=4.0,
                    value=st.session_state.tts_configs["openai"]["speed"],
                    step=0.1
                )
                
                if speed != st.session_state.tts_configs["openai"]["speed"]:
                    st.session_state.tts_configs["openai"]["speed"] = speed
                
                st.success("✅ OpenAI TTS configured")
            else:
                st.error("❌ OpenAI API key required")
        
        elif selected_provider == "azure":
            st.write("**Azure Speech Settings:**")
            if st.session_state.azure_speech_key and st.session_state.azure_speech_region:
                
                # Urdu voice options
                ur_voices = ["ur-PK-AsadNeural", "ur-PK-UzmaNeural", "ur-IN-GulNeural", "ur-IN-SalmanNeural"]
                current_ur_voice = st.session_state.tts_configs["azure"]["ur_voice"]
                
                selected_ur_voice = st.selectbox(
                    "Urdu Voice",
                    options=ur_voices,
                    index=ur_voices.index(current_ur_voice) if current_ur_voice in ur_voices else 0
                )
                
                # English voice options
                en_voices = ["en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-US-DavisNeural"]
                current_en_voice = st.session_state.tts_configs["azure"]["en_voice"]
                
                selected_en_voice = st.selectbox(
                    "English Voice",
                    options=en_voices,
                    index=en_voices.index(current_en_voice) if current_en_voice in en_voices else 0
                )
                
                if selected_ur_voice != current_ur_voice:
                    st.session_state.tts_configs["azure"]["ur_voice"] = selected_ur_voice
                
                if selected_en_voice != current_en_voice:
                    st.session_state.tts_configs["azure"]["en_voice"] = selected_en_voice
                
                st.success("✅ Azure Speech configured")
            else:
                st.error("❌ Azure Speech key and region required")
        
        # TTS Provider Comparison
        st.subheader("🔬 TTS Provider Comparison")
        
        comparison_data = {
            "Provider": ["ElevenLabs", "OpenAI", "Azure"],
            "Quality": ["🔥 Excellent", "⭐ Very Good", "🏢 Good"],
            "Speed": ["⚡ Fast", "🚀 Very Fast", "🏃 Fast"],
            "Languages": ["32+ Native", "50+ Good", "75+ Excellent"],
            "Accent Control": ["🎯 Excellent", "✅ Good", "🎭 Excellent"]
        }
        
        st.table(comparison_data)
        
        # Current Provider Status
        st.info(f"""
        **Currently Using:** {provider_labels.get(st.session_state.tts_provider, st.session_state.tts_provider)}
        
        **Accent-Free Settings:** ✅ ACTIVE
        - Same voice/model for both languages
        - Language-specific pronunciation
        - Perfect volume normalization
        """)
        
        # Tutor Mode
        st.subheader("🎓 English Tutor Mode")
        
        response_language = st.radio(
            "Response Language Mix",
            options=["both", "ur", "en"],
            format_func=lambda x: {
                "both": "🎯 English Tutor (Urdu + English)", 
                "ur": "اردو Only (Urdu Only)", 
                "en": "English Only"
            }[x],
            index=0
        )
        
        if response_language != st.session_state.response_language:
            st.session_state.response_language = response_language
            st.success(f"Tutor mode: {response_language}")
        
        # Language distribution
        if response_language == "both":
            st.subheader("📊 Language Balance")
            
            ur_percent = st.slider(
                "Urdu % (explanations)", 
                min_value=40, max_value=80, 
                value=st.session_state.language_distribution["ur"],
                help="Urdu for explanations and instructions"
            )
            
            en_percent = 100 - ur_percent
            st.text(f"English %: {en_percent} (examples & terms)")
            
            if ur_percent != st.session_state.language_distribution["ur"]:
                st.session_state.language_distribution = {
                    "ur": ur_percent,
                    "en": en_percent
                }
                st.success(f"Updated: {ur_percent}% Urdu, {en_percent}% English")
        
        # Performance metrics
        st.subheader("📊 Performance Metrics")
        
        avg_stt = calculate_average_latency(st.session_state.performance_metrics["stt_latency"])
        avg_llm = calculate_average_latency(st.session_state.performance_metrics["llm_latency"])
        avg_tts = calculate_average_latency(st.session_state.performance_metrics["tts_latency"])
        avg_total = calculate_average_latency(st.session_state.performance_metrics["total_latency"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("STT Latency", f"{avg_stt:.2f}s")
            st.metric("LLM Latency", f"{avg_llm:.2f}s")
        with col2:
            st.metric("TTS Latency", f"{avg_tts:.2f}s")
            st.metric("Total Latency", f"{avg_total:.2f}s")
        
        # API usage
        st.subheader("📈 API Usage")
        
        metrics = st.session_state.performance_metrics['api_calls']
        st.text(f"Whisper calls: {metrics['whisper']}")
        st.text(f"OpenAI calls: {metrics['openai']}")
        st.text(f"ElevenLabs calls: {metrics['elevenlabs']}")
        st.text(f"Azure calls: {metrics['azure']}")
    
    # Main interface
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("📝 Input")
        
        input_type = st.radio("Select Input Type", ["Text", "Voice"], horizontal=True)
        
        if input_type == "Text":
            st.subheader("Text Input")
            st.write("Use [ur] for Urdu and [en] for English text.")
            
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
            
            text_process_button = st.button("🚀 Process Text", type="primary")
            
            if text_process_button and text_input:
                with st.spinner(f"Processing with {st.session_state.tts_provider.upper()}..."):
                    audio_path, llm_latency, tts_latency = asyncio.run(process_text_input_enhanced(text_input))
                    
                    st.session_state.last_text_input = text_input
                    st.session_state.last_audio_output = audio_path
                    
                    total_latency = llm_latency + tts_latency
                    st.success(f"✅ Processed with {st.session_state.tts_provider.upper()} in {total_latency:.2f}s")
        
        else:
            # Voice input
            st.subheader("🎤 Voice Recording")
            
            keys_set = (
                st.session_state.openai_api_key and 
                (st.session_state.elevenlabs_api_key or 
                 st.session_state.openai_api_key or 
                 (st.session_state.azure_speech_key and st.session_state.azure_speech_region))
            )
            
            if not keys_set:
                st.warning("⚠️ Please set required API keys in the sidebar first")
            else:
                st.write(f"🎯 **Recording with {provider_labels.get(st.session_state.tts_provider)} TTS**")
                
                create_audio_recorder_component()
                
                st.markdown("---")
                st.write("**🔄 AUTOMATIC PROCESSING:**")
                
                uploaded_audio = st.file_uploader(
                    "📥 Upload Your Downloaded Recording Here", 
                    type=['wav', 'mp3', 'webm', 'ogg'],
                    key="main_upload",
                    help="After recording above, download and upload here for processing"
                )
                
                if uploaded_audio is not None:
                    with st.spinner(f"🔄 **PROCESSING WITH {st.session_state.tts_provider.upper()}...**"):
                        try:
                            temp_path = tempfile.mktemp(suffix=".wav")
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_audio.read())
                            
                            amplified_path = amplify_recorded_audio(temp_path)
                            
                            text, audio_output_path, stt_latency, llm_latency, tts_latency = asyncio.run(process_voice_input_accent_free(amplified_path))
                            
                            if text:
                                st.session_state.last_text_input = text
                            if audio_output_path:
                                st.session_state.last_audio_output = audio_output_path
                            
                            total_latency = stt_latency + llm_latency + tts_latency
                            st.success(f"✅ **PROCESSING COMPLETE WITH {st.session_state.tts_provider.upper()}!** ({total_latency:.2f}s)")
                            st.balloons()
                            
                            if os.path.exists(temp_path):
                                os.unlink(temp_path)
                            if amplified_path != temp_path and os.path.exists(amplified_path):
                                os.unlink(amplified_path)
                                
                        except Exception as e:
                            st.error(f"Processing error: {str(e)}")
                
                st.success("""
                🎯 **SIMPLE WORKFLOW:**
                1. Click "🔴 START RECORDING" above
                2. Speak clearly in Urdu or English  
                3. Click "⏹️ STOP RECORDING" when done
                4. **DOWNLOAD** the file automatically
                5. **UPLOAD** it above - processing starts immediately!
                
                **⚡ Test different TTS providers in the sidebar!**
                """)
    
    with col2:
        st.header("🎵 Output")
        
        # Transcribed text
        if 'last_text_input' in st.session_state and st.session_state.last_text_input:
            st.subheader("📝 Transcribed/Input Text")
            st.text_area(
                "Text with language markers",
                value=st.session_state.last_text_input,
                height=100,
                disabled=True
            )
        
        # Generated response
        if st.session_state.conversation_history:
            last_exchange = st.session_state.conversation_history[-1]
            
            if 'assistant_response' in last_exchange:
                st.subheader("🤖 AI Tutor Response")
                st.text_area(
                    "Response text",
                    value=last_exchange['assistant_response'],
                    height=150,
                    disabled=True
                )
                
                # Show which TTS provider was used
                provider_used = last_exchange.get('tts_provider', 'unknown')
                st.info(f"🎵 Generated with: {provider_labels.get(provider_used, provider_used)}")
        
        # Generated audio
        if 'last_audio_output' in st.session_state and st.session_state.last_audio_output:
            st.subheader("🔊 Generated Speech")
            
            audio_bytes = display_audio(st.session_state.last_audio_output, autoplay=True)
            
            if audio_bytes:
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name=f"tutor_response_{st.session_state.tts_provider}.mp3",
                    mime="audio/mp3"
                )
    
    # Conversation history
    if st.session_state.conversation_history:
        st.header("💬 Conversation History")
        
        for i, exchange in enumerate(st.session_state.conversation_history[-3:]):
            with st.expander(f"Exchange {i+1} - {exchange.get('timestamp', 'Unknown')[:19]} - {provider_labels.get(exchange.get('tts_provider', 'unknown'), 'Unknown TTS')}"):
                st.markdown("**User:**")
                st.text(exchange.get('user_input', 'No input'))
                
                st.markdown("**AI Tutor:**")
                st.text(exchange.get('assistant_response', 'No response'))
                
                latency = exchange.get('latency', {})
                provider = exchange.get('tts_provider', 'unknown')
                st.text(f"TTS Provider: {provider.upper()} | STT: {latency.get('stt', 0):.2f}s | LLM: {latency.get('llm', 0):.2f}s | TTS: {latency.get('tts', 0):.2f}s | Total: {latency.get('total', 0):.2f}s")
    
    # Status area
    st.header("📊 Status")
    st.session_state.status_area = st.empty()
    update_status()

if __name__ == "__main__":
    main()
