# Refactored Local AI Assistant Project Structure

## Project Directory Structure
```
local_ai_assistant/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Project dependencies
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration settings
├── core/
│   ├── __init__.py
│   ├── session_manager.py    # Session state management
│   └── assistant.py           # Main assistant orchestrator
├── models/
│   ├── __init__.py
│   ├── ollama_model.py       # Ollama LLM interface
│   ├── whisper_model.py      # Whisper speech recognition
│   └── tts_model.py          # Text-to-speech engine
├── services/
│   ├── __init__.py
│   ├── pdf_processor.py      # PDF text extraction
│   ├── knowledge_base.py     # Vector database management
│   └── embeddings.py         # Embedding service
├── ui/
│   ├── __init__.py
│   ├── components.py         # Reusable UI components
│   ├── sidebar.py            # Sidebar configuration
│   ├── chat_interface.py     # Main chat interface
│   └── styles.py             # CSS styles
└── utils/
    ├── __init__.py
    ├── audio_utils.py        # Audio processing utilities
    └── helpers.py            # General helper functions
```

## File Contents

### 1. `app.py` - Main Application Entry Point
```python
import streamlit as st
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
```

### 2. `config/settings.py` - Configuration Settings
```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class WhisperConfig:
    model_size: str = "base"
    available_sizes: List[str] = None
    
    def __post_init__(self):
        if self.available_sizes is None:
            self.available_sizes = ["tiny", "base", "small", "medium", "large"]

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    available_models: List[str] = None
    timeout: int = 5
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = [
                "llama3.2", "llama3.2:1b", "llama3.1", 
                "mistral", "codellama", "qwen2.5"
            ]

@dataclass
class TTSConfig:
    rate: int = 150
    volume: float = 0.9

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class AppConfig:
    whisper: WhisperConfig = None
    ollama: OllamaConfig = None
    tts: TTSConfig = None
    embedding: EmbeddingConfig = None
    
    def __post_init__(self):
        if self.whisper is None:
            self.whisper = WhisperConfig()
        if self.ollama is None:
            self.ollama = OllamaConfig()
        if self.tts is None:
            self.tts = TTSConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
```

### 3. `core/session_manager.py` - Session State Management
```python
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
```

### 4. `core/assistant.py` - Main Assistant Orchestrator
```python
from typing import Optional, List
from models.ollama_model import OllamaModel
from models.whisper_model import WhisperModel
from models.tts_model import TTSModel
from services.pdf_processor import PDFProcessor
from services.knowledge_base import KnowledgeBase
from config.settings import AppConfig

class LocalAIAssistant:
    def __init__(self, session_manager):
        self.session_manager = session_manager
        self.config = AppConfig()
        
        # Initialize models
        self.ollama = OllamaModel(self.config.ollama)
        self.whisper = WhisperModel(self.config.whisper)
        self.tts = TTSModel(self.config.tts)
        
        # Initialize services
        self.pdf_processor = PDFProcessor()
        self.knowledge_base = KnowledgeBase(self.config.embedding)
        
        # Initialize TTS engine on startup
        self._init_tts_engine()
    
    def _init_tts_engine(self):
        """Initialize TTS engine"""
        engine = self.tts.init_engine()
        if engine:
            self.session_manager.set('tts_engine', engine)
    
    def check_ollama_status(self):
        """Check Ollama status and available models"""
        status = self.ollama.check_status()
        self.session_manager.set('ollama_status', status)
        return status
    
    def install_ollama_model(self, model_name: str) -> bool:
        """Install an Ollama model"""
        return self.ollama.install_model(model_name)
    
    def load_whisper_model(self, model_size: str = "base"):
        """Load Whisper model"""
        current_model = self.session_manager.get('whisper_model')
        if current_model is None:
            model = self.whisper.load_model(model_size)
            if model:
                self.session_manager.set('whisper_model', model)
            return model
        return current_model
    
    def process_pdfs(self, uploaded_files) -> bool:
        """Process uploaded PDF files"""
        texts = []
        for file in uploaded_files:
            text = self.pdf_processor.extract_text(file)
            if text:
                texts.append(text)
                self.session_manager.add_pdf_text(file.name, text)
        
        if texts:
            knowledge_base = self.knowledge_base.create_from_texts(texts)
            if knowledge_base:
                self.session_manager.set('pdf_knowledge_base', knowledge_base)
                return True
        return False
    
    def speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """Convert speech to text"""
        model = self.session_manager.get('whisper_model')
        if model is None:
            model = self.load_whisper_model(
                self.session_manager.get('whisper_model_size', 'base')
            )
        return self.whisper.transcribe(audio_data, model)
    
    def text_to_speech(self, text: str) -> Optional[str]:
        """Convert text to speech"""
        engine = self.session_manager.get('tts_engine')
        if engine:
            return self.tts.generate_audio(text, engine)
        return None
    
    def get_ai_response(self, user_input: str, model_name: str) -> str:
        """Get AI response with context from knowledge base"""
        knowledge_base = self.session_manager.get('pdf_knowledge_base')
        chat_history = self.session_manager.get('chat_history', [])
        
        return self.ollama.generate_response(
            user_input=user_input,
            model_name=model_name,
            knowledge_base=knowledge_base,
            chat_history=chat_history
        )
```

### 5. `models/ollama_model.py` - Ollama Model Interface
```python
import requests
import subprocess
from typing import Dict, List, Optional, Any
from langchain.llms import Ollama

class OllamaModel:
    def __init__(self, config):
        self.config = config
        self.base_url = config.base_url
    
    def check_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and get available models"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", 
                timeout=self.config.timeout
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    'status': 'running',
                    'models': [model['name'] for model in models]
                }
            else:
                return {'status': 'not_running', 'models': []}
        except requests.exceptions.RequestException:
            return {'status': 'not_installed', 'models': []}
    
    def install_model(self, model_name: str) -> bool:
        """Install an Ollama model"""
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=300
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def generate_response(
        self, 
        user_input: str, 
        model_name: str,
        knowledge_base: Optional[Any] = None,
        chat_history: List[Dict] = None
    ) -> str:
        """Generate AI response using Ollama"""
        try:
            llm = Ollama(model=model_name, base_url=self.base_url)
            
            # Prepare context from knowledge base
            context = ""
            if knowledge_base:
                docs = knowledge_base.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in docs])
            
            # Build prompt
            system_prompt = self._build_system_prompt(context)
            conversation_context = self._build_conversation_context(chat_history)
            
            full_prompt = (
                f"{system_prompt}\n\n"
                f"Conversation History:\n{conversation_context}\n\n"
                f"Human: {user_input}\nAssistant:"
            )
            
            return llm(full_prompt)
            
        except Exception as e:
            return (
                "I'm sorry, I encountered an error while processing your request. "
                "Please make sure Ollama is running with a model installed."
            )
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with context"""
        return f"""You are a helpful AI assistant. 
        You have access to the following context from uploaded documents:
        {context}
        
        Use this context to answer questions when relevant, but also provide 
        general assistance when needed. Be conversational and helpful. 
        Keep responses concise but informative."""
    
    def _build_conversation_context(self, chat_history: List[Dict]) -> str:
        """Build conversation context from history"""
        if not chat_history:
            return ""
        
        context = ""
        for message in chat_history[-6:]:  # Last 6 messages
            role = "Human" if message["role"] == "user" else "Assistant"
            context += f"{role}: {message['content']}\n"
        
        return context
```

### 6. `models/whisper_model.py` - Whisper Speech Recognition
```python
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
```

### 7. `models/tts_model.py` - Text-to-Speech Model
```python
import pyttsx3
import tempfile
from typing import Optional
import streamlit as st

class TTSModel:
    def __init__(self, config):
        self.config = config
    
    def init_engine(self):
        """Initialize TTS engine"""
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
            engine.setProperty('rate', self.config.rate)
            engine.setProperty('volume', self.config.volume)
            return engine
        except:
            return None
    
    def generate_audio(self, text: str, engine) -> Optional[str]:
        """Generate audio from text"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                engine.save_to_file(text, tmp_file.name)
                engine.runAndWait()
                return tmp_file.name
        except Exception as e:
            st.error(f"Error in text-to-speech: {str(e)}")
            return None
```

### 8. `services/pdf_processor.py` - PDF Processing Service
```python
import PyPDF2
import io
from typing import Optional
import streamlit as st

class PDFProcessor:
    def extract_text(self, pdf_file) -> Optional[str]:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
```

### 9. `services/knowledge_base.py` - Knowledge Base Service
```python
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st

class KnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.model_name
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )
    
    def create_from_texts(self, texts: List[str]) -> Optional[FAISS]:
        """Create FAISS knowledge base from texts"""
        try:
            chunks = []
            for text in texts:
                chunks.extend(self.text_splitter.split_text(text))
            
            knowledge_base = FAISS.from_texts(chunks, self.embeddings)
            return knowledge_base
        except Exception as e:
            st.error(f"Error creating knowledge base: {str(e)}")
            return None
```

### 10. `ui/styles.py` - CSS Styles
```python
import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to the app"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
        }
        .assistant-message {
            background-color: #f3e5f5;
            border-left: 5px solid #9c27b0;
        }
        .sidebar-section {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .status-box {
            padding: 0.5rem;
            border-radius: 0.3rem;
            margin: 0.5rem 0;
        }
        .status-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .status-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
    </style>
    """, unsafe_allow_html=True)
```

### 11. `ui/sidebar.py` - Sidebar Component
```python
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
```

### 12. `ui/chat_interface.py` - Chat Interface
```python
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
```

### 13. `ui/components.py` - Reusable UI Components
```python
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
```

### 14. `requirements.txt` - Project Dependencies
```txt
streamlit>=1.28.0
whisper>=1.0.0
openai-whisper
pyttsx3>=2.90
PyPDF2>=3.0.0
SpeechRecognition>=3.10.0
audio-recorder-streamlit>=0.0.8
numpy>=1.24.0
langchain>=0.1.0
langchain-community>=0.0.10
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
requests>=2.31.0
```

### 15. `__init__.py` files
Create empty `__init__.py` files in each directory to make them Python packages:

```python
# config/__init__.py
# core/__init__.py
# models/__init__.py
# services/__init__.py
# ui/__init__.py
# utils/__init__.py
```

## Key Benefits of This Refactored Structure

### 1. **Separation of Concerns**
- Each module has a single, well-defined responsibility
- Business logic is separated from UI components
- Configuration is centralized

### 2. **Modularity & Scalability**
- Easy to add new models or services
- Components can be tested independently
- Clear interfaces between modules

### 3. **Maintainability**
- Easier to debug and update specific features
- Consistent coding patterns across modules
- Clear dependency structure

### 4. **Reusability**
- UI components can be reused across different views
- Services can be used by multiple controllers
- Models are abstracted and can be swapped easily

### 5. **Configuration Management**
- All settings in one place
- Easy to add new configuration options
- Type-safe configuration with dataclasses

## Usage Instructions

1. Create the directory structure as shown above
2. Copy each file content to its respective location
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app.py`

## Future Enhancements

1. **Add Unit Tests**
   - Create a `tests/` directory
   - Add unit tests for each service and model
   - Use pytest for testing framework

2. **Add Logging**
   - Create a `utils/logger.py` module
   - Add structured logging throughout the application
   - Configure log levels per module

3. **Add Database Support**
   - Create a `database/` directory
   - Add models for persistent storage
   - Implement conversation history storage

4. **Add API Layer**
   - Create an `api/` directory
   - Add FastAPI endpoints for programmatic access
   - Enable REST API for the assistant

5. **Add Authentication**
   - Create an `auth/` directory
   - Implement user authentication
   - Add user-specific knowledge bases