"""
Enhanced Voice Recorder with Start/Stop/Send functionality
"""

import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import io
import threading
import queue
from datetime import datetime
from typing import Optional

class EnhancedVoiceRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.audio_data = []
        self.sample_rate = 16000
        
    def render_voice_controls_compact(self):
        """Render compact voice controls for inline use"""
        
        # Single button interface that changes based on state
        if not self.is_recording:
            if st.button("🎤 Voice", key="voice_compact", help="Click to start recording"):
                self.start_recording()
                st.rerun()
        else:
            if st.button("⏹️ Stop", key="voice_stop_compact", type="secondary", help="Click to stop recording"):
                audio_data = self.stop_recording()
                if audio_data:
                    st.session_state['pending_audio'] = audio_data
                st.rerun()
        
        # Handle pending audio
        if st.session_state.get('pending_audio'):
            st.success("🎵 Audio recorded! Type a message or it will be sent automatically.")
            # Auto-send after a short delay or on next interaction
            return st.session_state.pop('pending_audio')
        
        # Show simple recording status
        if self.is_recording:
            st.warning("🔴 Recording... Click Stop when done.")
        
        return None
    
    def render_voice_controls(self):
        """Render voice recording controls with start/stop/send"""
        # Use a single button layout to avoid nesting issues when inside expanders
        
        if not self.is_recording:
            if st.button("🎤 Start Recording", key="start_rec", 
                       use_container_width=True, type="primary"):
                self.start_recording()
                st.rerun()
        else:
            if st.button("⏹️ Stop Recording", key="stop_rec", 
                       use_container_width=True, type="secondary"):
                audio_data = self.stop_recording()
                if audio_data:
                    st.session_state['pending_audio'] = audio_data
                st.rerun()
        
        # Show pending audio and send button
        if st.session_state.get('pending_audio'):
            st.audio(st.session_state['pending_audio'], format='audio/wav')
            if st.button("📤 Send Audio", key="send_audio", type="primary", use_container_width=True):
                return st.session_state.pop('pending_audio')
        
        # Show recording status
        if self.is_recording:
            self._show_recording_status()
        
        return None
    
    def _show_recording_status(self):
        """Show live recording status with waveform"""
        status_container = st.container()
        with status_container:
            cols = st.columns([1, 3, 1])
            with cols[1]:
                st.markdown("""
                <div style="text-align: center; padding: 20px;">
                    <div class="recording-indicator">
                        <span class="pulse"></span>
                        <span style="color: #ff4444; font-weight: bold;">Recording...</span>
                    </div>
                    <div class="waveform-container">
                        <canvas id="waveform"></canvas>
                    </div>
                </div>
                <style>
                    .recording-indicator { margin-bottom: 20px; }
                    .pulse {
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: #ff4444;
                        animation: pulse 1.5s infinite;
                        margin-right: 10px;
                    }
                    @keyframes pulse {
                        0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.7); }
                        70% { box-shadow: 0 0 0 10px rgba(255, 68, 68, 0); }
                        100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
                    }
                </style>
                """, unsafe_allow_html=True)
    
    def start_recording(self):
        """Start audio recording"""
        self.is_recording = True
        self.audio_data = []
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        if self.audio_data:
            return self._create_wav_file(self.audio_data)
        return None
    
    def _record_audio(self):
        """Record audio in background thread"""
        def callback(indata, frames, time, status):
            if self.is_recording:
                self.audio_queue.put(indata.copy())
        
        try:
            with sd.InputStream(callback=callback, channels=1, 
                              samplerate=self.sample_rate):
                while self.is_recording:
                    if not self.audio_queue.empty():
                        self.audio_data.append(self.audio_queue.get())
        except Exception as e:
            st.error(f"Recording error: {str(e)}")
            self.is_recording = False
    
    def _create_wav_file(self, audio_chunks):
        """Create WAV file from audio chunks"""
        if not audio_chunks:
            return None
            
        audio_data = np.concatenate(audio_chunks)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.getvalue()
    
    def render_advanced_controls(self):
        """Render advanced recording controls without nested expanders"""
        # Don't create expander since this is called from within an expander
        st.markdown("**🎛️ Advanced Audio Settings**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.sample_rate = st.selectbox(
                "Sample Rate", 
                [8000, 16000, 22050, 44100], 
                index=1,
                help="Higher sample rates provide better quality but larger files"
            )
        
        with col2:
            noise_reduction = st.checkbox(
                "Noise Reduction", 
                value=True,
                help="Apply basic noise reduction to recordings"
            )
        
        # Voice Activity Detection
        vad_enabled = st.checkbox(
            "Voice Activity Detection",
            value=False,
            help="Automatically detect speech vs silence"
        )
        
        if vad_enabled:
            vad_sensitivity = st.slider(
                "VAD Sensitivity",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                help="Higher values are more sensitive to sound"
            )
    
    def render_advanced_controls_standalone(self):
        """Render advanced recording controls with expander (for standalone use)"""
        with st.expander("🎛️ Advanced Audio Settings"):
            self.render_advanced_controls()
    
    def render_recording_history(self):
        """Show history of recordings in current session without expander"""
        if 'recording_history' not in st.session_state:
            st.session_state.recording_history = []
        
        if st.session_state.recording_history:
            st.markdown("**📼 Recording History**")
            for i, recording in enumerate(st.session_state.recording_history):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"Recording {i+1} - {recording['timestamp']}")
                
                with col2:
                    st.audio(recording['data'], format='audio/wav')
                
                with col3:
                    if st.button("🗑️", key=f"delete_rec_{i}"):
                        st.session_state.recording_history.pop(i)
                        st.rerun()
    
    def render_recording_history_standalone(self):
        """Show history of recordings with expander (for standalone use)"""
        if 'recording_history' not in st.session_state:
            st.session_state.recording_history = []
        
        if st.session_state.recording_history:
            with st.expander("📼 Recording History"):
                for i, recording in enumerate(st.session_state.recording_history):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"Recording {i+1} - {recording['timestamp']}")
                    
                    with col2:
                        st.audio(recording['data'], format='audio/wav')
                    
                    with col3:
                        if st.button("🗑️", key=f"delete_rec_{i}"):
                            st.session_state.recording_history.pop(i)
                            st.rerun() 