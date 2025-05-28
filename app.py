import streamlit as st

# Fix for PyTorch compatibility with Streamlit
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress PyTorch warnings that can interfere with Streamlit
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from core.session_manager import SessionManager
from core.assistant import LocalAIAssistant
from ui.sidebar import render_sidebar
from ui.chat_interface import render_chat_interface
from ui.styles import apply_custom_styles

def main():
    # Page configuration
    st.set_page_config(
        page_title="Local AI Assistant with Speech & PDF",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styles
    apply_custom_styles()
    
    # Initialize session manager
    session_manager = SessionManager()
    session_manager.initialize()
    
    # Initialize assistant
    assistant = LocalAIAssistant(session_manager)
    
    # Header
    st.markdown('<div class="main-header">🤖 Local AI Assistant with Speech & PDF</div>', 
                unsafe_allow_html=True)
    st.markdown("**No API keys required! Uses local Ollama + Whisper + HuggingFace embeddings**")
    
    # Render sidebar
    settings = render_sidebar(assistant, session_manager)
    
    # Render main chat interface
    render_chat_interface(assistant, session_manager, settings)

if __name__ == "__main__":
    main() 