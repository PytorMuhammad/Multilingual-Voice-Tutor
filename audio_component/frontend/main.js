let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordingTime = 0;
let timerInterval;

// Initialize Streamlit communication
const { Streamlit } = window.parent;

// Initialize when loaded
window.addEventListener('load', function() {
    initializeRecorder();
    
    // Set initial height
    Streamlit.setFrameHeight(150);
});

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
            processAudioDirectly();
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
        recordBtn.className = 'recording';
        statusDiv.innerHTML = '🔴 RECORDING - Speak in Czech or German';
        
        // Start timer
        timerInterval = setInterval(updateTimer, 1000);
        
        // Start recording
        mediaRecorder.start(1000);
        
    } else {
        // Stop recording
        isRecording = false;
        mediaRecorder.stop();
        
        recordBtn.innerHTML = '🔄 NEW RECORDING';
        recordBtn.className = '';
        statusDiv.innerHTML = '⚡ Processing automatically...';
        
        // Stop timer
        clearInterval(timerInterval);
        
        // Show processing indicator
        document.getElementById('processing').style.display = 'block';
    }
}

function updateTimer() {
    recordingTime++;
    const minutes = Math.floor(recordingTime / 60);
    const seconds = recordingTime % 60;
    document.getElementById('timer').innerHTML = 
        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

// CRITICAL: Direct processing - no downloads!
function processAudioDirectly() {
    if (audioChunks.length === 0) {
        console.error('No audio data recorded');
        return;
    }
    
    const recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
    
    // Convert to base64 and send directly to Streamlit
    const reader = new FileReader();
    reader.onloadend = function() {
        const base64Data = reader.result.split(',')[1];
        
        // CRITICAL: Send directly to Python - no file handling!
        const audioResult = {
            audio_data: base64Data,
            sample_rate: 16000,
            format: 'webm',
            timestamp: Date.now()
        };
        
        // Send to Streamlit Python backend
        Streamlit.setComponentValue(audioResult);
        
        // Update UI
        document.getElementById('status').innerHTML = '✅ Audio sent to processing!';
        document.getElementById('processing').style.display = 'none';
        
        console.log('Audio data sent directly to Python:', base64Data.substring(0, 50) + '...');
    };
    
    reader.readAsDataURL(recordedBlob);
}

// Handle Streamlit events
function onRender(event) {
    // Component ready
}

// Register with Streamlit
window.addEventListener("message", function(event) {
    if (event.data.type === "streamlit:render") {
        onRender(event.data);
    }
});
