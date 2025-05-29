import streamlit as st
import os
import threading
import time
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
    
    # Display pending audio if available
    _display_pending_audio(session_manager)

def _display_pending_audio(session_manager):
    """Display any pending audio that was generated"""
    current_audio_key = session_manager.get('current_audio_key')
    if current_audio_key:
        audio_data = session_manager.get(current_audio_key)
        if audio_data and isinstance(audio_data, dict):
            print(f"🔊 Displaying pending audio: {current_audio_key}")
            
            audio_bytes = audio_data.get('audio_data')
            audio_text = audio_data.get('text', 'Audio response')
            
            if audio_bytes:
                st.markdown("---")
                st.markdown("🔊 **Audio Response:**")
                
                # Display multiple audio players for better compatibility
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Manual play button
                    if st.button("🔊 Play Audio", key=f"manual_play_{current_audio_key}"):
                        st.audio(audio_bytes, format='audio/wav', autoplay=True)
                        st.success("Playing audio!")
                
                with col2:
                    st.info(f"Audio for: {audio_text}")
                
                # Primary audio player with autoplay
                st.audio(audio_bytes, format='audio/wav', autoplay=True)
                
                # HTML5 audio player as backup
                import base64
                audio_base64 = base64.b64encode(audio_bytes).decode()
                
                # Calculate approximate duration
                duration_seconds = len(audio_bytes) / (44100 * 2)
                
               
                
                # Add dismiss button
                if st.button("✅ Audio Played", key=f"dismiss_{current_audio_key}"):
                    session_manager.set('current_audio_key', None)
                    session_manager.set(current_audio_key, None)  # Clear the audio data
                    st.rerun()
                
                print(f"✅ Audio displayed successfully: {len(audio_bytes)} bytes")
            else:
                print("❌ No audio data found in pending audio")
                session_manager.set('current_audio_key', None)

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
    print(f"\n🎯 Processing user input...")
    print(f"👤 User input: {user_input}")
    
    status = session_manager.get('ollama_status', {})
    selected_model = settings.get('selected_model')
    
    print(f"🔍 Ollama status: {status.get('status', 'unknown')}")
    print(f"🤖 Selected model: {selected_model}")
    
    if status.get('status') == 'running' and selected_model:
        print("✅ Ollama is running and model is selected - proceeding...")
        
        # Add user message
        print("💾 Adding user message to chat history...")
        session_manager.append_message("user", user_input)
        
        # Get AI response
        print("🤖 Requesting AI response...")
        with st.spinner("Thinking..."):
            ai_response = assistant.get_ai_response(user_input, selected_model)
        
        print(f"📨 Received AI response: {ai_response[:100]}{'...' if len(ai_response) > 100 else ''}")
        
        # Add AI response
        print("💾 Adding AI response to chat history...")
        session_manager.append_message("assistant", ai_response)
        
        # Handle text-to-speech BEFORE rerun
        audio_generated = False
        if settings.get('enable_speech_output') and ai_response:
            print("🔊 Text-to-speech is enabled, generating audio...")
            audio_file = assistant.text_to_speech(ai_response)
            if audio_file:
                print(f"🎵 Audio file generated: {audio_file}")
                
                try:
                    # Read and store audio in session state immediately
                    with open(audio_file, 'rb') as f:
                        audio_bytes = f.read()
                    
                    print(f"📊 Audio file size: {len(audio_bytes)} bytes")
                    
                    # Store audio with a unique key
                    audio_key = f"pending_audio_{int(time.time() * 1000)}"
                    session_manager.set(audio_key, {
                        'audio_data': audio_bytes,
                        'text': ai_response[:50] + "..." if len(ai_response) > 50 else ai_response,
                        'timestamp': time.time()
                    })
                    session_manager.set('current_audio_key', audio_key)
                    audio_generated = True
                    print(f"💾 Audio stored in session with key: {audio_key}")
                    
                except Exception as e:
                    print(f"❌ Error processing audio: {str(e)}")
                
                # Schedule file cleanup
                def cleanup_audio_file(filepath):
                    time.sleep(30)
                    try:
                        if os.path.exists(filepath):
                            os.unlink(filepath)
                            print(f"🗑️ Temporary audio file cleaned up: {filepath}")
                    except Exception as e:
                        print(f"⚠️ Could not clean up temporary audio file: {e}")
                
                cleanup_thread = threading.Thread(target=cleanup_audio_file, args=(audio_file,))
                cleanup_thread.daemon = True
                cleanup_thread.start()
            else:
                print("❌ Failed to generate audio file")
        else:
            print("🔇 Text-to-speech disabled or no response to convert")
        
        # Now rerun to show the conversation
        print("🔄 Refreshing UI...")
        st.rerun()
    
    elif user_input and status.get('status') != 'running':
        print("❌ Cannot process input - Ollama not running or no model selected")
        st.warning("Please make sure Ollama is running and a model is selected.") 