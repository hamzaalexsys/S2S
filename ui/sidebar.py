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