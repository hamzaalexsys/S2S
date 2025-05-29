import streamlit as st
from typing import Dict, Any

def render_sidebar(assistant, session_manager) -> Dict[str, Any]:
    """Render sidebar with settings and configuration"""
    settings = {}
    
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Settings")
        
        # Ollama Status
        _render_ollama_status(assistant, session_manager)
        settings['selected_model'] = session_manager.get('selected_model')
        
        # Model Installation
        _render_model_installation(assistant, session_manager)
        
        # Whisper Settings
        settings['whisper_model_size'] = _render_whisper_settings(session_manager)
        
        # Voice Settings
        settings.update(_render_voice_settings(session_manager))
        
        # PDF Upload
        _render_pdf_upload(assistant, session_manager)
        
        # Display processed PDFs
        _render_processed_pdfs(session_manager)
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            session_manager.clear_conversation()
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return settings

def _render_ollama_status(assistant, session_manager):
    """Render Ollama status section"""
    st.subheader("🦙 Ollama Status")
    
    if st.button("🔄 Refresh Ollama Status"):
        assistant.check_ollama_status()
    
    status = session_manager.get('ollama_status', {})
    
    if status.get('status') == 'running':
        st.markdown(
            '<div class="status-box status-success">✅ Ollama is running</div>', 
            unsafe_allow_html=True
        )
        if status.get('models'):
            st.write("Available models:")
            selected = st.selectbox("Select Model:", status['models'])
            session_manager.set('selected_model', selected)
        else:
            st.warning("No models installed. Install a model below.")
    elif status.get('status') == 'not_running':
        st.markdown(
            '<div class="status-box status-warning">⚠️ Ollama is installed but not running</div>', 
            unsafe_allow_html=True
        )
        st.info("Start Ollama by running: `ollama serve` in terminal")
    else:
        st.markdown(
            '<div class="status-box status-error">❌ Ollama not installed</div>', 
            unsafe_allow_html=True
        )
        st.info("Install Ollama from: https://ollama.ai")

def _render_model_installation(assistant, session_manager):
    """Render model installation section"""
    st.subheader("📦 Install Models")
    
    from config.settings import OllamaConfig
    config = OllamaConfig()
    
    new_model = st.selectbox(
        "Install a new model:",
        [""] + config.available_models
    )
    
    if new_model and st.button(f"Install {new_model}"):
        with st.spinner(f"Installing {new_model}... This may take several minutes."):
            if assistant.install_ollama_model(new_model):
                st.success(f"Successfully installed {new_model}!")
                assistant.check_ollama_status()
                st.rerun()
            else:
                st.error(f"Failed to install {new_model}")

def _render_whisper_settings(session_manager):
    """Render Whisper settings"""
    st.subheader("🎤 Whisper Settings")
    
    from config.settings import WhisperConfig
    config = WhisperConfig()
    
    model_size = st.selectbox(
        "Whisper Model Size:",
        config.available_sizes,
        index=config.available_sizes.index(session_manager.get('whisper_model_size', 'base')),
        help="Larger models are more accurate but slower"
    )
    
    session_manager.set('whisper_model_size', model_size)
    return model_size

def _render_voice_settings(session_manager):
    """Render voice settings"""
    st.subheader("🔊 Voice Settings")
    
    enable_speech_input = st.checkbox(
        "Enable Speech Input", 
        value=session_manager.get('enable_speech_input', True)
    )
    enable_speech_output = st.checkbox(
        "Enable Speech Output", 
        value=session_manager.get('enable_speech_output', False)
    )
    
    session_manager.set('enable_speech_input', enable_speech_input)
    session_manager.set('enable_speech_output', enable_speech_output)
    
    # Add TTS test section
    if enable_speech_output:
        st.markdown("---")
        st.write("**🧪 Test Text-to-Speech:**")
        
        # Browser audio test first
        st.write("**🔊 Browser Audio Test:**")
        if st.button("🎵 Test Browser Audio", key="test_browser_audio"):
            # Create a simple beep sound for testing
            import numpy as np
            import io
            import wave
            
            # Generate a simple sine wave beep
            sample_rate = 44100
            duration = 1.0  # 1 second
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave_data = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Convert to 16-bit integers
            wave_data = (wave_data * 32767).astype(np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave_data.tobytes())
            
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()
            
            st.success("✅ Browser audio test - You should hear a beep!")
            st.audio(audio_bytes, format='audio/wav', autoplay=True)
            
            # Also provide HTML5 audio
            import base64
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <audio controls autoplay style="width: 100%;">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown("**HTML5 Audio Player:**", unsafe_allow_html=True)
            st.markdown(audio_html, unsafe_allow_html=True)
            
            st.info("If you don't hear the beep, check your browser's audio settings, volume, and ensure audio is not muted.")
        
        st.markdown("---")
        
        test_text = st.text_input(
            "Test text:", 
            value="Hello, this is a test of the text to speech system.",
            key="tts_test_input"
        )
        
        if st.button("🎵 Test TTS", key="test_tts_button"):
            if test_text.strip():
                tts_engine = session_manager.get('tts_engine')
                if tts_engine:
                    with st.spinner("Generating test audio..."):
                        # Import here to avoid circular imports
                        from models.tts_model import TTSModel
                        from config.settings import AppConfig
                        
                        config = AppConfig()
                        tts_model = TTSModel(config.tts)
                        
                        audio_file = tts_model.generate_audio(test_text, tts_engine)
                        if audio_file:
                            try:
                                with open(audio_file, 'rb') as f:
                                    audio_bytes = f.read()
                                
                                st.success("✅ Test audio generated!")
                                st.audio(audio_bytes, format='audio/wav', autoplay=True)
                                
                                # Also provide HTML5 audio player
                                import base64
                                audio_base64 = base64.b64encode(audio_bytes).decode()
                                audio_html = f"""
                                <audio controls autoplay style="width: 100%;">
                                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                                    Your browser does not support the audio element.
                                </audio>
                                """
                                st.markdown(audio_html, unsafe_allow_html=True)
                                
                                # Clean up test file
                                import os
                                try:
                                    os.unlink(audio_file)
                                except:
                                    pass
                                    
                            except Exception as e:
                                st.error(f"Error playing test audio: {str(e)}")
                        else:
                            st.error("Failed to generate test audio")
                else:
                    st.error("TTS engine not initialized")
            else:
                st.warning("Please enter some text to test")
    
    return {
        'enable_speech_input': enable_speech_input,
        'enable_speech_output': enable_speech_output
    }

def _render_pdf_upload(assistant, session_manager):
    """Render PDF upload section"""
    st.subheader("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload PDF files to create a knowledge base"
    )
    
    if uploaded_files:
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                if assistant.process_pdfs(uploaded_files):
                    st.success(f"Successfully processed {len(uploaded_files)} PDF files!")
                else:
                    st.error("Failed to process PDFs")

def _render_processed_pdfs(session_manager):
    """Display processed PDFs"""
    pdf_texts = session_manager.get('pdf_texts', [])
    if pdf_texts:
        st.subheader("📚 Processed Documents")
        for pdf in pdf_texts:
            st.text(f"• {pdf['filename']}") 