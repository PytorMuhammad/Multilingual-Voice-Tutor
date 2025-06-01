import React, { useState, useRef, useEffect } from "react"
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib"

interface AudioRecorderState {
  isRecording: boolean
  recordingTime: number
  status: string
  hasPermission: boolean
}

const AudioRecorder: React.FC = () => {
  const [state, setState] = useState<AudioRecorderState>({
    isRecording: false,
    recordingTime: 0,
    status: "🎤 Ready to Record",
    hasPermission: false
  })

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    // Initialize microphone access
    initializeMicrophone()
    
    // Cleanup on unmount
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }, [])

  const initializeMicrophone = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000 // Optimal for Whisper
        } 
      })
      
      streamRef.current = stream
      
      setState(prev => ({
        ...prev,
        hasPermission: true,
        status: "🎤 Ready - Click START to Record"
      }))

      // Setup MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })
      
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        processRecording()
      }

    } catch (error) {
      setState(prev => ({
        ...prev,
        hasPermission: false,
        status: "❌ Microphone access denied"
      }))
    }
  }

  const startRecording = () => {
    if (!mediaRecorderRef.current || !state.hasPermission) return

    // Reset chunks
    audioChunksRef.current = []
    
    // Start recording
    mediaRecorderRef.current.start(1000) // Collect data every second
    
    setState(prev => ({
      ...prev,
      isRecording: true,
      recordingTime: 0,
      status: "🔴 RECORDING - Speak in Czech or German"
    }))

    // Start timer
    timerRef.current = setInterval(() => {
      setState(prev => ({
        ...prev,
        recordingTime: prev.recordingTime + 1
      }))
    }, 1000)
  }

  const stopRecording = () => {
    if (!mediaRecorderRef.current || !state.isRecording) return

    // Stop recording
    mediaRecorderRef.current.stop()
    
    setState(prev => ({
      ...prev,
      isRecording: false,
      status: "⏳ Processing automatically..."
    }))

    // Clear timer
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  const processRecording = async () => {
    if (audioChunksRef.current.length === 0) {
      setState(prev => ({
        ...prev,
        status: "❌ No audio recorded"
      }))
      return
    }

    try {
      // Create blob from chunks
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
      
      // Convert to base64
      const reader = new FileReader()
      reader.onloadend = () => {
        const base64String = reader.result as string
        const base64Data = base64String.split(',')[1] // Remove data:audio/webm;base64, prefix
        
        // Send data back to Streamlit
        Streamlit.setComponentValue({
          audioData: base64Data,
          timestamp: Date.now(),
          duration: state.recordingTime,
          status: "complete"
        })
        
        setState(prev => ({
          ...prev,
          status: "✅ Audio sent for processing!",
          recordingTime: 0
        }))

        // Reset for next recording after 2 seconds
        setTimeout(() => {
          setState(prev => ({
            ...prev,
            status: "🎤 Ready for next recording"
          }))
        }, 2000)
      }
      
      reader.readAsDataURL(audioBlob)
      
    } catch (error) {
      setState(prev => ({
        ...prev,
        status: "❌ Processing failed"
      }))
    }
  }

  const toggleRecording = () => {
    if (state.isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const buttonStyle: React.CSSProperties = {
    background: state.isRecording ? '#666' : '#ff4b4b',
    color: 'white',
    border: 'none',
    padding: '15px 30px',
    borderRadius: '25px',
    cursor: state.hasPermission ? 'pointer' : 'not-allowed',
    fontSize: '16px',
    fontWeight: 'bold',
    margin: '5px',
    opacity: state.hasPermission ? 1 : 0.5
  }

  const containerStyle: React.CSSProperties = {
    padding: '20px',
    border: '2px solid #ff4b4b',
    borderRadius: '10px',
    textAlign: 'center',
    backgroundColor: '#f0f2f6',
    fontFamily: 'Arial, sans-serif'
  }

  return (
    <div style={containerStyle}>
      <div style={{ fontSize: '18px', marginBottom: '15px', fontWeight: 'bold' }}>
        {state.status}
      </div>
      
      <button 
        onClick={toggleRecording}
        disabled={!state.hasPermission}
        style={buttonStyle}
      >
        {state.isRecording ? '⏹️ STOP RECORDING' : '🔴 START RECORDING'}
      </button>
      
      <div style={{ fontSize: '14px', marginTop: '10px', color: '#666' }}>
        {formatTime(state.recordingTime)}
      </div>

      {!state.hasPermission && (
        <div style={{ marginTop: '10px', color: '#ff4b4b', fontSize: '14px' }}>
          Please allow microphone access to use the recorder
        </div>
      )}
    </div>
  )
}

export default withStreamlitConnection(AudioRecorder)
