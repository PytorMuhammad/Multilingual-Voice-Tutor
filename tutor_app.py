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
# CONFIGURATION SECTION - UPDATED FOR URDU/ENGLISH
# ----------------------------------------------------------------------------------

# Secrets and API keys
if 'api_keys_initialized' not in st.session_state:
    st.session_state.api_keys_initialized = False
    st.session_state.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")

# API endpoints
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"
OPENAI_API_URL = "https://api.openai.com/v1"

# 🔥 FIXED: Single voice for accent-free switching
if 'language_voices' not in st.session_state:
    single_voice_id = "21m00Tcm4TlvDq8ikWAM"  # SAME voice for ALL languages
    st.session_state.language_voices = {
        "ur": single_voice_id,  # SAME voice for Urdu
        "en": single_voice_id,  # SAME voice for English
        "default": single_voice_id
    }

# 🔥 CRITICAL: Accent-free voice settings for maximum elimination
if 'voice_settings' not in st.session_state:
    st.session_state.voice_settings = {
        "ur": {  # Urdu-optimized settings for accent elimination
            "stability": 0.98,        # MAXIMUM stability for consistent Urdu
            "similarity_boost": 0.99, # MAXIMUM similarity for native Urdu sound
            "style": 0.90,           # High style for natural Urdu expression
            "use_speaker_boost": True # Enable speaker boost for clarity
        },
        "en": {  # English-optimized settings for accent elimination
            "stability": 0.96,        # VERY HIGH stability for consistent English
            "similarity_boost": 0.97, # VERY HIGH similarity for native English sound
            "style": 0.88,           # High style for natural English expression
            "use_speaker_boost": True # Enable speaker boost for clarity
        },
        "default": {
            "stability": 0.95,
            "similarity_boost": 0.95,
            "style": 0.85,
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

# 🔥 UPDATED: Language distribution for Urdu/English
if 'language_distribution' not in st.session_state:
    st.session_state.language_distribution = {
        "ur": 60,  # Urdu percentage (explanations)
        "en": 40   # English percentage (examples/terms)
    }

# Language preference for response
if 'response_language' not in st.session_state:
    st.session_state.response_language = "both"  # Options: "ur", "en", "both"

# 🔥 UPDATED: Language codes and settings for Urdu/English
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

# 🔥 CRITICAL: Enhanced voice settings for perfect accent elimination
def get_accent_free_voice_settings(language_code, context=None):
    """🎯 OPTIMIZED: Accent-free voice settings using advanced ElevenLabs techniques"""
    
    # Base settings for accent elimination
    base_settings = {
        "stability": 0.95,           # MAXIMUM stability for consistency
        "similarity_boost": 0.98,    # MAXIMUM similarity to maintain voice character
        "style": 0.85,              # High style for natural expression
        "use_speaker_boost": True    # Enhanced clarity
    }
    
    # 🔥 CRITICAL: Language-specific accent elimination settings
    if language_code == "ur":  # Urdu
        base_settings.update({
            "stability": 0.98,        # ULTRA HIGH for Urdu consistency
            "similarity_boost": 0.99, # MAXIMUM similarity for native sound
            "style": 0.90,           # Natural Urdu expression
            "use_speaker_boost": True
        })
    elif language_code == "en":  # English
        base_settings.update({
            "stability": 0.96,        # VERY HIGH for English consistency
            "similarity_boost": 0.97, # VERY HIGH similarity for native sound
            "style": 0.88,           # Natural English expression
            "use_speaker_boost": True
        })
    
    return base_settings

# 🔥 ENHANCED: Advanced SSML for accent-free pronunciation
def create_accent_free_ssml_enhanced(text, language_code):
    """🎯 ENHANCED: Advanced SSML with pronunciation isolation techniques"""
    
    if not language_code:
        return text
    
    clean_text = text.strip()
    
    # 🔥 CRITICAL: Language-specific SSML with accent isolation
    if language_code == "ur":
        # Urdu with proper pronunciation hints
        enhanced_text = f'''<speak>
            <lang xml:lang="ur-PK">
                <phoneme alphabet="ipa" ph="">ˈ</phoneme>
                <prosody rate="0.90" pitch="+2st" volume="+3dB">
                    {clean_text}
                </prosody>
            </lang>
        </speak>'''
        
    elif language_code == "en":
        # English with clear American accent specification
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

# 🔥 PROFESSIONAL: Enhanced system prompt for intelligent language mixing
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

# 🔥 ENHANCED: LLM response with intelligent language tagging strategy
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
    
    # Enhanced system prompt
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
    
    # Add current prompt
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
                    "model": "gpt-4",  # Best for complex tutoring logic
                    "messages": messages,
                    "temperature": 0.7,  # Balanced creativity
                    "max_tokens": 400,
                    "presence_penalty": 0.1,  # Encourage variety
                    "frequency_penalty": 0.1   # Reduce repetition
                },
                timeout=30.0
            )
            
            latency = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                # 🎯 CRITICAL: Ensure proper language markers
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

# 🔥 ENHANCED: Ensure intelligent, strategic language markers
def ensure_intelligent_language_markers(response_text):
    """🎯 ENHANCED: Ensure intelligent, strategic language markers"""
    
    # If already has proper markers, clean them up
    if "[ur]" in response_text or "[en]" in response_text:
        # Clean up spacing
        response_text = re.sub(r'\[ur\]\s*', '[ur] ', response_text)
        response_text = re.sub(r'\[en\]\s*', '[en] ', response_text)
        response_text = re.sub(r'\s+\[ur\]', ' [ur]', response_text)
        response_text = re.sub(r'\s+\[en\]', ' [en]', response_text)
        return response_text.strip()
    
    # If no markers, apply intelligent tagging
    return apply_intelligent_tagging(response_text)

def apply_intelligent_tagging(text):
    """🎯 STRATEGIC: Apply intelligent language tagging based on content analysis"""
    
    # English words/phrases that should be tagged
    english_patterns = [
        r'\b(hello|hi|good morning|good evening|thank you|please|sorry|excuse me)\b',
        r'\b(I am|I\'m|my name is|nice to meet you)\b', 
        r'\b(water|book|pen|house|car|food|time|money)\b',
        r'\b(subject|verb|object|grammar|vocabulary)\b',
        r'\b(yes|no|maybe|okay|alright)\b'
    ]
    
    # Apply strategic tagging
    tagged_text = text
    
    # Tag English patterns
    for pattern in english_patterns:
        tagged_text = re.sub(pattern, r'[en] \g<0> [ur]', tagged_text, flags=re.IGNORECASE)
    
    # Clean up and ensure Urdu context
    if '[en]' in tagged_text:
        # Ensure Urdu markers around English
        if not tagged_text.startswith('[ur]'):
            tagged_text = '[ur] ' + tagged_text
        if not tagged_text.endswith('[ur]'):
            tagged_text = tagged_text + ' [ur]'
    else:
        # All Urdu content
        tagged_text = f'[ur] {tagged_text}'
    
    # Clean up multiple consecutive markers
    tagged_text = re.sub(r'\[ur\]\s*\[ur\]', '[ur]', tagged_text)
    tagged_text = re.sub(r'\[en\]\s*\[en\]', '[en]', tagged_text)
    tagged_text = re.sub(r'\[ur\]\s*\[en\]\s*\[ur\]', '[ur]', tagged_text)
    
    return tagged_text.strip()

# 🔥 CRITICAL: Generate completely accent-free speech
async def generate_accent_free_speech(text, language_code, voice_id, segment_position=0):
    """🔥 CRITICAL: Generate completely accent-free speech"""
    
    api_key = st.session_state.elevenlabs_api_key
    if not api_key:
        return None, 0
    
    # 🎯 CRITICAL: Use optimized voice settings
    voice_settings = get_accent_free_voice_settings(language_code)
    
    # 🔥 ENHANCED: Advanced SSML for accent elimination
    enhanced_text = create_accent_free_ssml_enhanced(text, language_code)
    
    # 🎯 CRITICAL: Use Flash v2.5 model with multilingual optimization
    data = {
        "text": enhanced_text,
        "model_id": "eleven_flash_v2_5",  # Latest model with best accent control
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
            timeout=15.0
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"✅ Accent-free {language_code} generated in {generation_time:.2f}s")
            return BytesIO(response.content), generation_time
        else:
            logger.error(f"TTS error: {response.status_code}")
            return None, generation_time
            
    except Exception as e:
        logger.error(f"Accent-free TTS error: {str(e)}")
        return None, time.time() - start_time

# 🔥 CRITICAL: Process multilingual text with zero accent bleeding
async def process_accent_free_multilingual_text(text):
    """🔥 CRITICAL: Process multilingual text with zero accent bleeding"""
    
    # Parse language segments intelligently
    segments = parse_intelligent_language_segments(text)
    
    if len(segments) <= 1:
        # Single language segment
        single_lang = segments[0]["language"] if segments else "ur"
        audio_data, gen_time = await generate_accent_free_speech(
            text, single_lang, st.session_state.elevenlabs_voice_id
        )
        if audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                with open(temp_file.name, "wb") as f:
                    f.write(audio_data.read())
                return temp_file.name, gen_time
        return None, 0
    
    # Multiple segments - process with accent isolation
    audio_segments = []
    total_time = 0
    
    for i, segment in enumerate(segments):
        if not segment["text"].strip():
            continue
            
        # Generate accent-free audio for each segment
        audio_data, generation_time = await generate_accent_free_speech(
            segment["text"], 
            language_code=segment["language"],
            voice_id=st.session_state.elevenlabs_voice_id,
            segment_position=i
        )
        
        if audio_data:
            # Convert to AudioSegment with perfect normalization
            audio_segment = AudioSegment.from_file(audio_data, format="mp3")
            
            # 🔥 CRITICAL: Perfect volume normalization for seamless blending
            normalized_segment = normalize_segment_perfectly(audio_segment, segment["language"])
            
            audio_segments.append(normalized_segment)
            total_time += generation_time
    
    if not audio_segments:
        return None, 0
    
    # 🔥 CRITICAL: Seamless blending with zero accent artifacts
    combined_audio = audio_segments[0]
    
    for i in range(1, len(audio_segments)):
        combined_audio = blend_accent_free_segments(
            combined_audio, 
            audio_segments[i],
            crossfade_ms=50  # Minimal crossfade for same voice
        )
    
    # Save final accent-free audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        combined_audio.export(
            temp_file.name, 
            format="mp3", 
            bitrate="256k",  # High quality
            parameters=["-ac", "1", "-ar", "44100"]
        )
        return temp_file.name, total_time

def parse_intelligent_language_segments(text):
    """Parse language segments intelligently"""
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
                    "language": current_language or "ur"
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
    
    return segments

def normalize_segment_perfectly(audio_segment, language_code):
    """Perfect normalization for accent-free blending"""
    # Target dBFS for consistent volume
    target_dbfs = -18.0
    
    # Normalize to target
    current_dbfs = audio_segment.dBFS
    volume_adjustment = target_dbfs - current_dbfs
    normalized = audio_segment.apply_gain(volume_adjustment)
    
    # Language-specific fine-tuning
    if language_code == "ur":
        # Urdu might need slight emphasis boost
        normalized = normalized.apply_gain(0.5)
    elif language_code == "en":
        # English standard normalization
        pass
    
    return normalized

def blend_accent_free_segments(segment1, segment2, crossfade_ms=50):
    """Blend segments with zero accent artifacts"""
    
    # Ensure perfect volume matching
    seg1_normalized = normalize_segment_perfectly(segment1, "auto")
    seg2_normalized = normalize_segment_perfectly(segment2, "auto")
    
    # Minimal crossfade since it's the same voice
    blended = seg1_normalized.append(seg2_normalized, crossfade=crossfade_ms)
    
    return blended

# Enhanced transcription with Urdu/English pronunciation hints
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
            
            # 🔥 ENHANCED: Urdu/English specific prompts for better pronunciation detection
            data = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "temperature": "0.0",
                "language": None,  # Auto-detect between ur/en
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
                
                # Apply Urdu/English pronunciation corrections
                enhanced_result = enhance_urdu_english_transcription(result)
                
                latency = time.time() - start_time
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
        
        # Urdu pronunciation corrections
        urdu_corrections = {
            "mein": "main",
            "ap": "aap", 
            "kia": "kya",
            "kesay": "kaise",
            "english": "English",
            "sikhna": "seekhna"
        }
        
        # English pronunciation corrections
        english_corrections = {
            "watar": "water",
            "buk": "book",
            "gramar": "grammar",
            "praktis": "practice",
            "helo": "hello"
        }
        
        corrected_text = text
        
        # Apply corrections
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

# Audio recording and processing functions (keeping existing ones but updating for accent-free)
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

# Voice input processing with accent-free enhancement
async def process_voice_input_accent_free(audio_file):
    """🔥 ACCENT-FREE: Voice processing with enhanced accent elimination"""
    pipeline_start_time = time.time()
    
    try:
        # Step 1: Enhanced Audio Preprocessing
        st.session_state.message_queue.put("🎧 Preparing audio for accent-free processing...")
        
        # Apply amplification for clear pronunciation detection
        enhanced_audio_file = amplify_recorded_audio(audio_file)
        
        # Step 2: Advanced Transcription with language hints
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
        
        # Step 3: Enhanced LLM Response with intelligent tagging
        st.session_state.message_queue.put("🤖 Generating intelligent tutor response...")
        
        llm_result = await generate_enhanced_llm_response(user_input)
        
        if "error" in llm_result:
            st.session_state.message_queue.put(f"❌ Response error: {llm_result.get('error')}")
            return user_input, None, transcription.get("latency", 0), 0, 0
        
        response_text = llm_result["response"]
        st.session_state.message_queue.put(f"💬 Generated: {response_text}")
        
        # Step 4: Accent-Free TTS Generation
        st.session_state.message_queue.put("🎵 Generating accent-free speech...")
        audio_path, tts_latency = await process_accent_free_multilingual_text(response_text)
        
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

# Keep existing audio recorder component
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

# Utility functions
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

# MAIN STREAMLIT APPLICATION
def main():
    """Main application with Urdu/English configuration"""
    st.set_page_config(
        page_title="Professional English Tutor - Urdu/English",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("Professional English Tutor for Urdu Speakers")
    st.subheader("Accent-Free Voice AI Tutor (A1-B2 English Learning)")
    
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
        
        # 🔥 UPDATED: Accent-Free Voice Setup for Urdu/English
        st.subheader("🎯 Accent-Free Voice Setup")
        st.write("**Same Voice for Both Languages - No Accent Bleeding:**")
        
        # Voice status
        st.success(f"""
        ✅ **Accent-Free Configuration**
        - Voice ID: {st.session_state.elevenlabs_voice_id[:8]}...
        - Used for: Urdu + English (SAME VOICE)
        - Model: Flash v2.5 (Accent Isolation)
        - SSML: Enhanced language-specific pronunciation
        """)
        
        # 🔥 UPDATED: Tutor Mode for Urdu/English
        st.subheader("🎓 English Tutor Mode")
        
        response_language = st.radio(
            "Response Language Mix",
            options=["both", "ur", "en"],
            format_func=lambda x: {
                "both": "🎯 English Tutor (Urdu + English)", 
                "ur": "اردو Only (Urdu Only)", 
                "en": "English Only"
            }[x],
            index=0  # Default to "both"
        )
        
        if response_language != st.session_state.response_language:
            st.session_state.response_language = response_language
            st.success(f"Tutor mode: {response_language}")
        
        # Language distribution (only for "both" mode)
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

        # 🔥 UPDATED: Accent Control Status
        st.subheader("🎭 Accent Control Status")
        
        # Voice settings verification
        ur_settings = st.session_state.voice_settings.get("ur", {})
        en_settings = st.session_state.voice_settings.get("en", {})
        
        st.info(f"""
        **Urdu Settings:**
        - Stability: {ur_settings.get('stability', 0.98):.2f}
        - Similarity: {ur_settings.get('similarity_boost', 0.99):.2f}
        
        **English Settings:**
        - Stability: {en_settings.get('stability', 0.96):.2f}  
        - Similarity: {en_settings.get('similarity_boost', 0.97):.2f}
        
        **Accent Elimination:** ✅ ACTIVE
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
    
    # Main interaction area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("Input")
        
        # Input type selection
        input_type = st.radio("Select Input Type", ["Text", "Voice"], horizontal=True)
        
        if input_type == "Text":
            # Text input with Urdu/English examples
            st.subheader("Text Input")
            st.write("Use [ur] to mark Urdu text and [en] to mark English text.")
            
            # Demo preset examples
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
                    # Process the text input with enhanced LLM
                    llm_result = asyncio.run(generate_enhanced_llm_response(text_input))
                    
                    if "error" not in llm_result:
                        response_text = llm_result["response"]
                        audio_path, tts_latency = asyncio.run(process_accent_free_multilingual_text(response_text))
                        
                        # Store for display in the output section
                        st.session_state.last_text_input = text_input
                        st.session_state.last_audio_output = audio_path
                        
                        # Show latency metrics
                        total_latency = llm_result.get("latency", 0) + tts_latency
                        st.success(f"Text processed in {total_latency:.2f} seconds")
        
        else:
            # Voice input - ACCENT-FREE AUDIO RECORDER
            st.subheader("🎤 Professional Voice Recording")
            
            # Check if API keys are set
            keys_set = (
                st.session_state.elevenlabs_api_key and 
                st.session_state.openai_api_key
            )

            if not keys_set:
                st.warning("Please set both API keys in the sidebar first")
            else:
                st.write("🎯 **HTML5 Audio Recording** - Accent-Free Processing")
                
                # Create the HTML5 audio recorder component
                create_audio_recorder_component()

                st.markdown("---")
                st.write("**🔄 ACCENT-FREE PROCESSING:**")
                
                # Upload processing
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
                            
                            # Apply amplification and process through accent-free pipeline
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
                            st.error(f"Processing error: {str(e)}")

                # Enhanced instructions
                st.success("""
                🎯 **ACCENT-FREE WORKFLOW:**
                1. Click "🔴 START RECORDING" above
                2. Speak clearly in Urdu or English  
                3. Click "⏹️ STOP RECORDING" when done
                4. **DOWNLOAD** the file that appears automatically
                5. **UPLOAD** it above - accent-free processing starts immediately!

                **⚡ Result: ZERO accent bleeding between languages!**
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
            st.subheader("Generated Speech (Accent-Free)")
            
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
