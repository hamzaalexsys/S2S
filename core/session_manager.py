import streamlit as st
from langchain.memory import ConversationBufferMemory
from typing import Dict, List, Any

class SessionManager:
    def __init__(self):
        self.state = st.session_state
    
    def initialize(self):
        """Initialize all session state variables"""
        defaults = {
            'chat_history': [],
            'conversation_memory': None,
            'pdf_knowledge_base': None,
            'pdf_texts': [],
            'tts_engine': None,
            'whisper_model': None,
            'ollama_status': {'status': 'not_checked', 'models': []},
            'selected_model': None,
            'enable_speech_input': True,
            'enable_speech_output': False,
            'whisper_model_size': 'base'
        }
        
        for key, value in defaults.items():
            if key not in self.state:
                if key == 'conversation_memory':
                    self.state[key] = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                else:
                    self.state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from session state"""
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a value in session state"""
        self.state[key] = value
    
    def append_message(self, role: str, content: str):
        """Append a message to chat history"""
        self.state.chat_history.append({
            "role": role,
            "content": content
        })
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.state.chat_history = []
        self.state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def add_pdf_text(self, filename: str, text: str):
        """Add processed PDF text"""
        self.state.pdf_texts.append({
            "filename": filename,
            "text": text
        }) 