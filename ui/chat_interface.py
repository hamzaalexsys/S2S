import streamlit as st
import os
from audio_recorder_streamlit import audio_recorder
from ui.components import render_chat_message, render_statistics

def render_chat_interface(assistant, session_manager, settings):
    """Render the main chat interface"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        _render_conversation(session_manager)
        _handle_user_input(assistant, session_manager, settings)
    
    with col2:
        render_statistics(session_manager, settings)

def _render_conversation(session_manager):
    """Render conversation history"""
    st.subheader("💬 Conversation")
    chat_container = st.container()
    
    with chat_container:
        for message in session_manager.get('chat_history', []):
            render_chat_message(message)

def _handle_user_input(assistant, session_manager, settings):
    """Handle user input (text or voice)"""
    input_method = st.radio(
        "Choose input method:",
        ["Text", "Voice"] if settings.get('enable_speech_input') else ["Text"],
        horizontal=True
    )
    
    user_input = None
    
    if input_method == "Text":
        user_input = st.chat_input("Type your message here...")
    
    elif input_method == "Voice" and settings.get('enable_speech_input'):
        st.write("🎤 Click to record your voice:")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="2x"
        )
        
        if audio_bytes:
            with st.spinner("Converting speech to text..."):
                user_input = assistant.speech_to_text(audio_bytes)
                if user_input:
                    st.success(f"Recognized: {user_input}")
    
    # Process input if available
    if user_input:
        _process_user_input(user_input, assistant, session_manager, settings)

def _process_user_input(user_input, assistant, session_manager, settings):
    """Process user input and generate response"""
    status = session_manager.get('ollama_status', {})
    selected_model = settings.get('selected_model')
    
    if status.get('status') == 'running' and selected_model:
        # Add user message
        session_manager.append_message("user", user_input)
        
        # Get AI response
        with st.spinner("Thinking..."):
            ai_response = assistant.get_ai_response(user_input, selected_model)
        
        # Add AI response
        session_manager.append_message("assistant", ai_response)
        
        # Handle text-to-speech
        if settings.get('enable_speech_output') and ai_response:
            with st.spinner("Generating speech..."):
                audio_file = assistant.text_to_speech(ai_response)
                if audio_file:
                    st.audio(audio_file)
                    try:
                        os.unlink(audio_file)
                    except:
                        pass
        
        st.rerun()
    
    elif user_input and status.get('status') != 'running':
        st.warning("Please make sure Ollama is running and a model is selected.") 