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