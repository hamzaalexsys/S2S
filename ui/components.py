import streamlit as st

def render_chat_message(message):
    """Render a single chat message"""
    if message["role"] == "user":
        st.markdown(
            f'<div class="chat-message user-message">'
            f'<strong>You:</strong> {message["content"]}'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message assistant-message">'
            f'<strong>Assistant:</strong> {message["content"]}'
            f'</div>',
            unsafe_allow_html=True
        )

def render_statistics(session_manager, settings):
    """Render session statistics and system status"""
    st.subheader("📊 Session Info")
    
    # Session metrics
    info_container = st.container()
    with info_container:
        st.metric("Messages", len(session_manager.get('chat_history', [])))
        st.metric("PDFs Processed", len(session_manager.get('pdf_texts', [])))
        
        if session_manager.get('pdf_knowledge_base'):
            st.success("✅ Knowledge Base Active")
        else:
            st.info("📄 Upload PDFs to enable document QA")
        
        if settings.get('enable_speech_input'):
            st.success("🎤 Speech Input Enabled")
        
        if settings.get('enable_speech_output'):
            st.success("🔊 Speech Output Enabled")
    
    # System Status
    st.subheader("🔧 System Status")
    status_container = st.container()
    with status_container:
        status = session_manager.get('ollama_status', {})
        
        # Ollama status
        if status.get('status') == 'running':
            st.success("🦙 Ollama: Running")
        else:
            st.error("🦙 Ollama: Not Running")
        
        # Whisper status
        if session_manager.get('whisper_model'):
            st.success("🎤 Whisper: Loaded")
        else:
            st.info("🎤 Whisper: Not Loaded")
    
    # Help section
    render_help_section()

def render_help_section():
    """Render help section"""
    st.subheader("❓ Help")
    with st.expander("How to use"):
        st.markdown("""
        **Getting Started:**
        1. Install Ollama from https://ollama.ai
        2. Run `ollama serve` in terminal
        3. Install a model (e.g., llama3.2)
        4. Upload PDF files to create knowledge base
        5. Start chatting!
        
        **Features:**
        - 🗣️ Local voice input (Whisper)
        - 🔊 Voice output (pyttsx3)
        - 📄 PDF document analysis
        - 💭 Conversation memory
        - 🧠 Context-aware responses
        - 🔒 100% local - no API keys needed!
        
        **Models:**
        - **llama3.2:1b** - Fast, lightweight
        - **llama3.2** - Balanced performance
        - **mistral** - Good for coding
        - **codellama** - Specialized for code
        """) 