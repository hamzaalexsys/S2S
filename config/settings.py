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