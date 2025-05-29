import whisper
import tempfile
import os
from typing import Optional
import streamlit as st

class WhisperModel:
    def __init__(self, config):
        self.config = config
    
    def load_model(self, model_size: str):
        """Load Whisper model"""
        try:
            with st.spinner(f"Loading Whisper {model_size} model..."):
                return whisper.load_model(model_size)
        except Exception as e:
            st.error(f"Error loading Whisper model: {str(e)}")
            return None
    
    def transcribe(self, audio_data: bytes, model) -> Optional[str]:
        """Transcribe audio to text"""
        if model is None:
            return None
        
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            # Transcribe
            result = model.transcribe(tmp_file_path)
            
            # Clean up
            os.unlink(tmp_file_path)
            
            return result["text"]
        except Exception as e:
            st.error(f"Error in speech-to-text: {str(e)}")
            return None 