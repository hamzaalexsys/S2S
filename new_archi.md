# Enhanced RAG Implementation Guide

## 1. Advanced RAG Architecture

### A. Hybrid Search Implementation
```python
# services/enhanced_knowledge_base.py
from typing import List, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

class HybridKnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.documents = []
        self.metadata = []
        self.bm25 = None
        self.dense_index = None
        self.reranker = SentenceTransformer('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Tuple[str, dict, float]]:
        """Combine BM25 and dense retrieval with reranking"""
        # Get BM25 results
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_k = np.argsort(bm25_scores)[-k:][::-1]
        
        # Get dense retrieval results
        dense_results = self.dense_index.similarity_search_with_score(query, k=k)
        
        # Combine and deduplicate
        candidates = self._merge_results(bm25_top_k, dense_results)
        
        # Rerank with cross-encoder
        reranked = self._rerank_results(query, candidates)
        
        return reranked[:k]
```

### B. Contextual Chunking with Overlap
```python
# services/document_chunker.py
class SmartDocumentChunker:
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_with_context(self, text: str, metadata: dict) -> List[dict]:
        """Create chunks with contextual information"""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_size += len(sentence)
            
            if current_size >= self.chunk_size:
                # Add context from surrounding chunks
                chunk_text = ' '.join(current_chunk)
                
                chunk_data = {
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_index': len(chunks),
                        'start_sentence': max(0, i - len(current_chunk) + 1),
                        'end_sentence': i,
                        'context_before': sentences[max(0, i - len(current_chunk) - 2):max(0, i - len(current_chunk) + 1)],
                        'context_after': sentences[i+1:i+3] if i+1 < len(sentences) else []
                    }
                }
                chunks.append(chunk_data)
                
                # Overlap handling
                overlap_sentences = int(self.overlap / (current_size / len(current_chunk)))
                current_chunk = current_chunk[-overlap_sentences:]
                current_size = sum(len(s) for s in current_chunk)
        
        return chunks
```

### C. Query Expansion and Understanding
```python
# services/query_processor.py
class QueryProcessor:
    def __init__(self):
        self.llm = None  # Initialize with Ollama
    
    def expand_query(self, query: str) -> List[str]:
        """Generate query variations for better retrieval"""
        prompt = f"""Generate 3 alternative phrasings for this question that preserve the intent:
        Original: {query}
        
        Alternatives:"""
        
        response = self.llm(prompt)
        alternatives = [query] + self._parse_alternatives(response)
        return alternatives
    
    def extract_intent(self, query: str) -> dict:
        """Extract query intent and entities"""
        prompt = f"""Analyze this query and extract:
        1. Intent (question/command/clarification)
        2. Key entities
        3. Subject area
        
        Query: {query}"""
        
        response = self.llm(prompt)
        return self._parse_intent(response)
```

## 2. Fine-tuning for Education

### A. Educational Response Templates
```python
# models/educational_ollama.py
class EducationalOllama(OllamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.educational_prompts = {
            'explain': """You are an expert educator. Explain this concept in a clear, 
                         structured way with examples. Use the Feynman technique.""",
            'quiz': """Create an interactive quiz question about this topic with 
                      multiple choice answers and explanations.""",
            'summarize': """Summarize this content for a student, highlighting key 
                           concepts and their relationships.""",
            'elaborate': """Provide a detailed explanation with real-world applications 
                           and connections to other concepts."""
        }
    
    def generate_educational_response(self, query: str, mode: str = 'explain'):
        prompt = self.educational_prompts.get(mode, self.educational_prompts['explain'])
        return self.generate_response(query, system_prompt=prompt)
```

### B. Adaptive Learning System
```python
# services/learning_tracker.py
class LearningTracker:
    def __init__(self):
        self.user_profile = {
            'knowledge_level': {},
            'learning_style': 'visual',
            'pace': 'moderate',
            'interests': []
        }
    
    def adapt_response(self, response: str, topic: str) -> str:
        """Adapt response based on user's learning profile"""
        level = self.user_profile['knowledge_level'].get(topic, 'beginner')
        
        if level == 'beginner':
            return self._simplify_response(response)
        elif level == 'advanced':
            return self._add_technical_details(response)
        
        return response
    
    def track_understanding(self, query: str, response: str, feedback: dict):
        """Track user's understanding level"""
        # Update knowledge profile based on interaction
        pass
```

## 3. Enhanced Voice Interface

### A. Advanced Speech Recognition
```python
# models/enhanced_whisper.py
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class EnhancedWhisperModel:
    def __init__(self, config):
        self.config = config
        self.processor = None
        self.model = None
        self.vad_model = None  # Voice Activity Detection
    
    def load_model(self, model_size: str = "base"):
        """Load Whisper with additional models"""
        # Load Whisper
        self.model = WhisperForConditionalGeneration.from_pretrained(
            f"openai/whisper-{model_size}"
        )
        self.processor = WhisperProcessor.from_pretrained(
            f"openai/whisper-{model_size}"
        )
        
        # Load VAD for better speech detection
        self.vad_model = self._load_vad_model()
    
    def transcribe_with_confidence(self, audio_data: bytes) -> dict:
        """Transcribe with confidence scores and timestamps"""
        # Preprocess audio
        audio = self._preprocess_audio(audio_data)
        
        # Detect speech segments
        speech_segments = self.vad_model.get_speech_segments(audio)
        
        # Transcribe with detailed output
        inputs = self.processor(audio, return_tensors="pt")
        
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs.input_features,
                return_timestamps=True,
                return_token_timestamps=True
            )
        
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True,
            output_word_offsets=True
        )
        
        return {
            'text': transcription[0],
            'confidence': self._calculate_confidence(predicted_ids),
            'segments': speech_segments,
            'language': self._detect_language(audio)
        }
```

### B. Real-time Voice Interaction
```python
# ui/voice_interface.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import queue
import threading

class RealTimeVoiceInterface:
    def __init__(self, assistant):
        self.assistant = assistant
        self.audio_queue = queue.Queue()
        self.is_recording = False
    
    def render_voice_interface(self):
        """Render push-to-talk interface"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Push-to-talk button
            if st.button("🎤 Hold to Speak", key="ptt", 
                        use_container_width=True,
                        on_click=self.toggle_recording):
                self.is_recording = not self.is_recording
            
            if self.is_recording:
                st.info("🔴 Recording... Release to send")
                self._start_recording()
            else:
                st.info("🎤 Press and hold to speak")
        
        # Voice activity indicator
        if self.is_recording:
            self._render_voice_activity()
    
    def _render_voice_activity(self):
        """Show real-time voice activity"""
        placeholder = st.empty()
        
        # Real-time waveform visualization
        with placeholder.container():
            st.markdown("""
                <div class="voice-activity">
                    <div class="waveform">
                        <span></span><span></span><span></span>
                        <span></span><span></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
```

## 4. Modern Chat Interface

### A. Enhanced UI Components
```python
# ui/modern_chat.py
def render_modern_chat_interface(assistant, session_manager):
    """Render modern chat interface with animations"""
    
    # Custom CSS for modern look
    st.markdown("""
    <style>
        .chat-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        
        .message-bubble {
            animation: slideIn 0.3s ease-out;
            backdrop-filter: blur(10px);
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(2) { animation-delay: -0.32s; }
        .typing-dot:nth-child(3) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% {
                transform: scale(0.8);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat messages with avatars
    for message in session_manager.get('chat_history', []):
        render_message_with_avatar(message)
```

### B. Interactive Message Components
```python
# ui/message_components.py
def render_interactive_message(message: dict):
    """Render message with interactive elements"""
    
    col1, col2 = st.columns([1, 10])
    
    with col1:
        # Avatar
        if message['role'] == 'user':
            st.markdown("👤")
        else:
            st.markdown("🤖")
    
    with col2:
        # Message content with actions
        message_container = st.container()
        
        with message_container:
            st.markdown(message['content'])
            
            # Action buttons
            col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 7])
            
            with col_a:
                if st.button("📋", key=f"copy_{message['id']}", 
                           help="Copy message"):
                    st.toast("Copied to clipboard!")
            
            with col_b:
                if st.button("🔊", key=f"speak_{message['id']}", 
                           help="Read aloud"):
                    assistant.text_to_speech(message['content'])
            
            with col_c:
                if st.button("📝", key=f"edit_{message['id']}", 
                           help="Edit message"):
                    st.session_state[f"editing_{message['id']}"] = True
```

## 5. Smart Model Management

### A. Automatic Model Recommendations
```python
# services/model_manager.py
class SmartModelManager:
    def __init__(self):
        self.model_specs = {
            'llama3.2:1b': {
                'size': '1.3GB',
                'ram': '2GB',
                'speed': 'fast',
                'quality': 'good',
                'use_cases': ['chat', 'simple_qa']
            },
            'llama3.2': {
                'size': '3.8GB',
                'ram': '4GB',
                'speed': 'moderate',
                'quality': 'better',
                'use_cases': ['chat', 'qa', 'analysis']
            },
            'mixtral': {
                'size': '26GB',
                'ram': '32GB',
                'speed': 'slow',
                'quality': 'excellent',
                'use_cases': ['complex_reasoning', 'coding', 'analysis']
            }
        }
    
    def recommend_model(self, use_case: str, system_specs: dict) -> str:
        """Recommend best model based on use case and system"""
        available_ram = system_specs.get('ram', 8)
        
        suitable_models = []
        for model, specs in self.model_specs.items():
            if use_case in specs['use_cases']:
                required_ram = int(specs['ram'].replace('GB', ''))
                if required_ram <= available_ram:
                    suitable_models.append((model, specs))
        
        # Sort by quality
        suitable_models.sort(key=lambda x: x[1]['quality'], reverse=True)
        
        return suitable_models[0][0] if suitable_models else 'llama3.2:1b'
```

### B. One-Click Model Setup
```python
# ui/model_setup.py
def render_quick_setup(assistant, session_manager):
    """One-click setup for common configurations"""
    
    st.subheader("🚀 Quick Setup")
    
    presets = {
        "Educator": {
            "model": "llama3.2",
            "whisper": "small",
            "features": ["voice_input", "voice_output", "educational_mode"]
        },
        "Researcher": {
            "model": "mixtral",
            "whisper": "medium",
            "features": ["pdf_analysis", "advanced_rag"]
        },
        "Casual Chat": {
            "model": "llama3.2:1b",
            "whisper": "tiny",
            "features": ["voice_input"]
        }
    }
    
    selected_preset = st.selectbox("Choose a preset:", list(presets.keys()))
    
    if st.button("Apply Preset"):
        preset = presets[selected_preset]
        
        with st.spinner(f"Setting up {selected_preset} configuration..."):
            # Install model if needed
            if not assistant.check_model_installed(preset['model']):
                assistant.install_ollama_model(preset['model'])
            
            # Configure settings
            session_manager.set('selected_model', preset['model'])
            session_manager.set('whisper_model_size', preset['whisper'])
            
            for feature in preset['features']:
                session_manager.set(f'enable_{feature}', True)
            
            st.success(f"✅ {selected_preset} setup complete!")
```

## 6. Additional Urgent Features

### A. Conversation Export and Import
```python
# services/conversation_manager.py
class ConversationManager:
    def export_conversation(self, chat_history: List[dict], format: str = 'json') -> bytes:
        """Export conversation in various formats"""
        if format == 'json':
            return json.dumps(chat_history, indent=2).encode()
        elif format == 'markdown':
            md_content = self._convert_to_markdown(chat_history)
            return md_content.encode()
        elif format == 'pdf':
            return self._generate_pdf(chat_history)
    
    def import_conversation(self, file_data: bytes, format: str) -> List[dict]:
        """Import conversation from file"""
        # Implementation for importing conversations
        pass
```

### B. Multi-modal Support
```python
# services/multimodal.py
class MultiModalProcessor:
    def __init__(self):
        self.image_model = None
        self.ocr_engine = None
    
    def process_image(self, image_data: bytes) -> dict:
        """Process image for context"""
        # Extract text via OCR
        text = self.ocr_engine.extract_text(image_data)
        
        # Generate image description
        description = self.image_model.describe(image_data)
        
        return {
            'text': text,
            'description': description,
            'type': 'image'
        }
```

### C. Collaborative Features
```python
# services/collaboration.py
class CollaborationService:
    def share_session(self, session_id: str) -> str:
        """Generate shareable link for session"""
        # Create temporary share link
        share_token = self._generate_share_token(session_id)
        return f"http://localhost:8501/?share={share_token}"
    
    def join_session(self, share_token: str) -> dict:
        """Join shared session"""
        # Load shared session state
        pass
```
# Immediate Implementation Guide: Top Priority Features

## 1. Enhanced Voice Recording with Start/Stop/Send
# ui/enhanced_voice_recorder.py

import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import io
import threading
import queue
from datetime import datetime

class EnhancedVoiceRecorder:
    def __init__(self):
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.audio_data = []
        self.sample_rate = 16000
        
    def render_voice_controls(self):
        """Render voice recording controls with start/stop/send"""
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if not self.is_recording:
                if st.button("🎤 Start Recording", key="start_rec", 
                           use_container_width=True, type="primary"):
                    self.start_recording()
                    st.rerun()
            else:
                if st.button("⏹️ Stop Recording", key="stop_rec", 
                           use_container_width=True, type="secondary"):
                    audio_data = self.stop_recording()
                    if audio_data:
                        st.session_state['pending_audio'] = audio_data
                    st.rerun()
        
        with col2:
            if st.session_state.get('pending_audio'):
                st.audio(st.session_state['pending_audio'], format='audio/wav')
        
        with col3:
            if st.session_state.get('pending_audio'):
                if st.button("📤 Send", key="send_audio", type="primary"):
                    return st.session_state.pop('pending_audio')
        
        # Show recording status
        if self.is_recording:
            self._show_recording_status()
        
        return None
    
    def _show_recording_status(self):
        """Show live recording status with waveform"""
        status_container = st.container()
        with status_container:
            cols = st.columns([1, 3, 1])
            with cols[1]:
                st.markdown("""
                <div style="text-align: center; padding: 20px;">
                    <div class="recording-indicator">
                        <span class="pulse"></span>
                        <span style="color: #ff4444; font-weight: bold;">Recording...</span>
                    </div>
                    <div class="waveform-container">
                        <canvas id="waveform"></canvas>
                    </div>
                </div>
                <style>
                    .recording-indicator { margin-bottom: 20px; }
                    .pulse {
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: #ff4444;
                        animation: pulse 1.5s infinite;
                        margin-right: 10px;
                    }
                    @keyframes pulse {
                        0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.7); }
                        70% { box-shadow: 0 0 0 10px rgba(255, 68, 68, 0); }
                        100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
                    }
                </style>
                """, unsafe_allow_html=True)
    
    def start_recording(self):
        """Start audio recording"""
        self.is_recording = True
        self.audio_data = []
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        if self.audio_data:
            return self._create_wav_file(self.audio_data)
        return None
    
    def _record_audio(self):
        """Record audio in background thread"""
        def callback(indata, frames, time, status):
            if self.is_recording:
                self.audio_queue.put(indata.copy())
        
        with sd.InputStream(callback=callback, channels=1, 
                          samplerate=self.sample_rate):
            while self.is_recording:
                if not self.audio_queue.empty():
                    self.audio_data.append(self.audio_queue.get())
    
    def _create_wav_file(self, audio_chunks):
        """Create WAV file from audio chunks"""
        audio_data = np.concatenate(audio_chunks)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.getvalue()


## 2. Smart Model Management with System Detection
# services/system_analyzer.py

import psutil
import platform
import subprocess
import json

class SystemAnalyzer:
    def get_system_info(self):
        """Analyze system capabilities"""
        info = {
            'os': platform.system(),
            'cpu': {
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'model': platform.processor()
            },
            'memory': {
                'total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
                'available_gb': round(psutil.virtual_memory().available / (1024**3), 1)
            },
            'gpu': self._detect_gpu()
        }
        return info
    
    def _detect_gpu(self):
        """Detect GPU availability"""
        gpu_info = {'available': False, 'type': None, 'memory_gb': 0}
        
        try:
            # Try NVIDIA
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                name, memory = result.stdout.strip().split(', ')
                gpu_info = {
                    'available': True,
                    'type': 'NVIDIA',
                    'name': name,
                    'memory_gb': int(memory.replace(' MiB', '')) / 1024
                }
        except:
            pass
        
        return gpu_info
    
    def recommend_models(self):
        """Recommend models based on system capabilities"""
        info = self.get_system_info()
        recommendations = []
        
        available_ram = info['memory']['available_gb']
        
        model_requirements = {
            'llama3.2:1b': {'ram': 2, 'description': 'Fastest, good for chat'},
            'llama3.2': {'ram': 4, 'description': 'Balanced performance'},
            'mistral:7b': {'ram': 8, 'description': 'High quality responses'},
            'mixtral:8x7b': {'ram': 48, 'description': 'Best quality, slow'},
            'phi3:mini': {'ram': 2, 'description': 'Efficient, good reasoning'},
            'gemma2:2b': {'ram': 3, 'description': 'Google model, fast'},
            'qwen2.5:3b': {'ram': 4, 'description': 'Good multilingual support'}
        }
        
        for model, reqs in model_requirements.items():
            if reqs['ram'] <= available_ram * 0.7:  # Use 70% of available RAM
                recommendations.append({
                    'model': model,
                    'description': reqs['description'],
                    'ram_required': reqs['ram'],
                    'suitable': True
                })
        
        return {
            'system_info': info,
            'recommendations': recommendations
        }


## 3. Enhanced RAG with Source Citations
# models/enhanced_ollama.py

class EnhancedOllamaModel(OllamaModel):
    def generate_with_sources(self, query: str, context_docs: list, model_name: str):
        """Generate response with source citations"""
        
        # Format context with source tracking
        formatted_context = self._format_context_with_sources(context_docs)
        
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.
Always cite your sources using [Source N] format when using information from the context.

Context:
{formatted_context}

Question: {query}

Instructions:
1. Answer based on the context provided
2. Cite sources using [Source N] format
3. If the context doesn't contain relevant information, say so
4. Be concise but thorough

Answer:"""
        
        response = self.generate_response(prompt, model_name)
        
        # Extract citations and create source mapping
        sources_used = self._extract_citations(response)
        source_details = [context_docs[i-1] for i in sources_used if i <= len(context_docs)]
        
        return {
            'response': response,
            'sources': source_details,
            'citations': sources_used
        }
    
    def _format_context_with_sources(self, docs):
        """Format documents with source numbers"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"[Source {i}] {doc['metadata']['filename']}: {doc['text']}")
        return "\n\n".join(formatted)
    
    def _extract_citations(self, text):
        """Extract source citations from response"""
        import re
        citations = re.findall(r'\[Source (\d+)\]', text)
        return list(set(int(c) for c in citations))


## 4. Modern Chat UI with Markdown Support
# ui/modern_chat_components.py

def render_modern_message(message: dict, index: int):
    """Render a modern chat message with animations and features"""
    
    # Message wrapper with animation
    st.markdown(f'''
    <div class="message-wrapper animate-in" style="animation-delay: {index * 0.1}s">
        <div class="message-bubble {message['role']}-message">
            <div class="message-header">
                <span class="message-icon">{'👤' if message['role'] == 'user' else '🤖'}</span>
                <span class="message-time">{message.get('timestamp', '')}</span>
            </div>
            <div class="message-content">{message['content']}</div>
            <div class="message-actions">
                <button class="action-btn" onclick="copyMessage('{index}')">📋</button>
                <button class="action-btn" onclick="speakMessage('{index}')">🔊</button>
                <button class="action-btn" onclick="editMessage('{index}')">✏️</button>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Add CSS for modern styling
    st.markdown("""
    <style>
        .message-wrapper {
            margin: 20px 0;
            opacity: 0;
            animation: fadeInUp 0.5s ease forwards;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message-bubble {
            position: relative;
            padding: 15px 20px;
            border-radius: 18px;
            max-width: 70%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .message-bubble:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
        }
        
        .assistant-message {
            background: #f7f7f8;
            color: #333;
            border: 1px solid #e1e4e8;
        }
        
        .message-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.85em;
            opacity: 0.8;
        }
        
        .message-content {
            line-height: 1.6;
            word-wrap: break-word;
        }
        
        .message-actions {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .message-bubble:hover .message-actions {
            opacity: 1;
        }
        
        .action-btn {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 0.9em;
            padding: 5px;
            border-radius: 5px;
            transition: background 0.2s ease;
        }
        
        .action-btn:hover {
            background: rgba(0,0,0,0.1);
        }
    </style>
    
    <script>
        function copyMessage(index) {
            // Copy message content to clipboard
            const content = document.querySelector(`#message-${index} .message-content`).innerText;
            navigator.clipboard.writeText(content);
        }
        
        function speakMessage(index) {
            // Trigger TTS for message
            window.parent.postMessage({action: 'speak', index: index}, '*');
        }
        
        function editMessage(index) {
            // Enable message editing
            window.parent.postMessage({action: 'edit', index: index}, '*');
        }
    </script>
    """, unsafe_allow_html=True)


## 5. Educational Mode with Interactive Learning
# services/educational_assistant.py

class EducationalAssistant:
    def __init__(self, ollama_model):
        self.ollama = ollama_model
        self.learning_modes = {
            'explain': self._explain_mode,
            'quiz': self._quiz_mode,
            'practice': self._practice_mode,
            'summarize': self._summarize_mode
        }
        
    def process_educational_query(self, query: str, mode: str = 'explain', 
                                 context: dict = None):
        """Process query in educational mode"""
        
        handler = self.learning_modes.get(mode, self._explain_mode)
        return handler(query, context)
    
    def _explain_mode(self, query: str, context: dict):
        """Explain concept step by step"""
        prompt = f"""You are an expert educator using the Feynman Technique.
        
Task: Explain {query}

Follow this structure:
1. **Simple Explanation**: Explain as if to a beginner
2. **Core Concepts**: Break down the key ideas
3. **Analogy**: Provide a relatable comparison
4. **Example**: Give a practical example
5. **Check Understanding**: Ask a question to verify comprehension

Make it engaging and clear."""
        
        response = self.ollama.generate_response(prompt)
        
        # Add interactive elements
        return self._add_interactive_elements(response, 'explanation')
    
    def _quiz_mode(self, topic: str, context: dict):
        """Generate interactive quiz"""
        prompt = f"""Create an interactive quiz about {topic}.

Format:
**Question**: [Clear question]

**Options**:
A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]

**Hint**: [Helpful hint without giving away the answer]

After user selects, provide:
- Correct answer
- Explanation why it's correct
- Why other options are incorrect"""
        
        quiz_data = self.ollama.generate_response(prompt)
        return self._format_quiz(quiz_data)
    
    def _practice_mode(self, skill: str, context: dict):
        """Generate practice problems"""
        prompt = f"""Create a practice problem for {skill}.

Include:
1. Problem statement
2. Space for student work
3. Hints (hidden by default)
4. Step-by-step solution
5. Similar practice problems

Make it progressively challenging."""
        
        return self.ollama.generate_response(prompt)
    
    def _add_interactive_elements(self, content: str, content_type: str):
        """Add interactive UI elements to educational content"""
        
        # Parse content and add interactive components
        interactive_content = {
            'content': content,
            'type': content_type,
            'interactive_elements': []
        }
        
        # Add comprehension check buttons
        if content_type == 'explanation':
            interactive_content['interactive_elements'].append({
                'type': 'comprehension_check',
                'questions': self._extract_comprehension_questions(content)
            })
        
        return interactive_content


## 6. Streamlined PDF Processing with Better UI
# ui/enhanced_pdf_interface.py

def render_pdf_upload_interface(assistant, session_manager):
    """Enhanced PDF upload interface with drag-and-drop"""
    
    st.markdown("""
    <style>
        .pdf-upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            transition: all 0.3s ease;
        }
        
        .pdf-upload-area:hover {
            border-color: #764ba2;
            background: #f0f1f3;
        }
        
        .pdf-list {
            margin-top: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .pdf-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .pdf-processing {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .processing-spinner {
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #2196f3;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Upload area
    st.markdown('<div class="pdf-upload-area">', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here or click to browse",
        type=['pdf'],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process uploaded files
    if uploaded_files:
        # Show file list
        st.markdown("### 📄 Files to Process")
        
        total_size = sum(file.size for file in uploaded_files)
        st.info(f"Total: {len(uploaded_files)} files, {format_file_size(total_size)}")
        
        # Process button
        if st.button("🚀 Process All PDFs", type="primary", use_container_width=True):
            process_pdfs_with_progress(assistant, uploaded_files, session_manager)
    
    # Show processed documents
    render_processed_documents(session_manager)


def process_pdfs_with_progress(assistant, files, session_manager):
    """Process PDFs with progress indication"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        # Update progress
        progress = (i + 1) / len(files)
        progress_bar.progress(progress)
        status_text.text(f"Processing {file.name}... ({i+1}/{len(files)})")
        
        # Process file
        try:
            text = assistant.pdf_processor.extract_text(file)
            if text:
                session_manager.add_pdf_text(file.name, text)
                st.success(f"✅ {file.name} processed successfully")
        except Exception as e:
            st.error(f"❌ Error processing {file.name}: {str(e)}")
    
    # Create knowledge base
    status_text.text("Creating knowledge base...")
    texts = [pdf['text'] for pdf in session_manager.get('pdf_texts', [])]
    
    if texts:
        knowledge_base = assistant.knowledge_base.create_from_texts(texts)
        session_manager.set('pdf_knowledge_base', knowledge_base)
        st.success("🎉 Knowledge base created successfully!")
    
    progress_bar.empty()
    status_text.empty()


## 7. Settings Persistence and Profiles
# services/settings_manager.py

import json
from pathlib import Path

class SettingsManager:
    def __init__(self, settings_dir: str = ".streamlit_assistant"):
        self.settings_dir = Path.home() / settings_dir
        self.settings_dir.mkdir(exist_ok=True)
        self.settings_file = self.settings_dir / "settings.json"
        self.profiles_dir = self.settings_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
    
    def save_settings(self, settings: dict, profile_name: str = "default"):
        """Save current settings to profile"""
        profile_file = self.profiles_dir / f"{profile_name}.json"
        
        with open(profile_file, 'w') as f:
            json.dump(settings, f, indent=2)
        
        return True
    
    def load_settings(self, profile_name: str = "default"):
        """Load settings from profile"""
        profile_file = self.profiles_dir / f"{profile_name}.json"
        
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def list_profiles(self):
        """List available profiles"""
        return [f.stem for f in self.profiles_dir.glob("*.json")]
    
    def export_profile(self, profile_name: str):
        """Export profile for sharing"""
        profile_file = self.profiles_dir / f"{profile_name}.json"
        
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                return f.read()
        
        return None


# Integration in sidebar
def render_profile_manager(session_manager):
    """Render profile management UI"""
    settings_manager = SettingsManager()
    
    st.subheader("👤 Profiles")
    
    # Profile selector
    profiles = ["default"] + settings_manager.list_profiles()
    selected_profile = st.selectbox("Select Profile:", profiles)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Current"):
            current_settings = {
                'model': session_manager.get('selected_model'),
                'whisper_size': session_manager.get('whisper_model_size'),
                'voice_input': session_manager.get('enable_speech_input'),
                'voice_output': session_manager.get('enable_speech_output'),
            }
            settings_manager.save_settings(current_settings, selected_profile)
            st.success("Profile saved!")
    
    with col2:
        if st.button("📂 Load Profile"):
            settings = settings_manager.load_settings(selected_profile)
            if settings:
                for key, value in settings.items():
                    session_manager.set(key, value)
                st.success("Profile loaded!")
                st.rerun()

## Helper function
def format_file_size(size_bytes: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

    # Updated requirements.txt for Enhanced Local AI Assistant

# Core Streamlit and UI
streamlit>=1.28.0
streamlit-webrtc>=0.45.0  # For real-time audio streaming
audio-recorder-streamlit>=0.0.8
streamlit-javascript>=1.0.0  # For custom JS interactions

# Speech Processing
openai-whisper>=20230918  # Latest Whisper
whisper>=1.0.0
SpeechRecognition>=3.10.0
pyttsx3>=2.90
sounddevice>=0.4.6  # For better audio recording
pyaudio>=0.2.11
webrtcvad>=2.0.10  # Voice Activity Detection

# PDF and Document Processing
PyPDF2>=3.0.0
pdfplumber>=0.9.0  # Better PDF text extraction
pytesseract>=0.3.10  # OCR support
Pillow>=10.0.0  # Image processing
python-docx>=0.8.11  # DOCX support
python-pptx>=0.6.21  # PowerPoint support

# AI Models and NLP
langchain>=0.1.0
langchain-community>=0.0.10
sentence-transformers>=2.2.0
transformers>=4.35.0  # For advanced models
torch>=2.0.0  # PyTorch for models
accelerate>=0.24.0  # Model acceleration

# Vector Stores and Search
faiss-cpu>=1.7.4
chromadb>=0.4.0  # Alternative vector store
rank-bm25>=0.2.2  # BM25 search

# Enhanced RAG
llama-index>=0.9.0  # Advanced RAG capabilities
unstructured>=0.10.0  # Better document parsing

# System and Performance
psutil>=5.9.0  # System monitoring
nvidia-ml-py3>=7.352.0  # NVIDIA GPU monitoring
gputil>=1.4.0  # GPU utilities

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0

# API and Networking
requests>=2.31.0
aiohttp>=3.8.0  # Async HTTP
websockets>=11.0  # WebSocket support

# Utilities
python-dotenv>=1.0.0  # Environment variables
pydantic>=2.0.0  # Data validation
click>=8.1.0  # CLI tools
rich>=13.0.0  # Better terminal output

# Visualization
plotly>=5.0.0  # Interactive plots
matplotlib>=3.7.0
seaborn>=0.12.0

# Development and Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0  # Code formatting
flake8>=6.0.0  # Linting

# Optional: Advanced Features
# streamlit-aggrid>=0.3.4  # Advanced tables
# streamlit-chat>=0.1.1  # Chat UI components
# streamlit-elements>=0.1.0  # Custom components