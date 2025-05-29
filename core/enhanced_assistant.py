"""
Enhanced AI Assistant with Advanced Features
"""

from typing import Optional, List, Dict, Any
from models.enhanced_ollama import EnhancedOllamaModel
from models.whisper_model import WhisperModel
from models.tts_model import TTSModel
from services.enhanced_knowledge_base import HybridKnowledgeBase
from services.system_analyzer import SystemAnalyzer
from services.settings_manager import SettingsManager
from ui.enhanced_voice_recorder import EnhancedVoiceRecorder
from ui.enhanced_pdf_interface import EnhancedPDFProcessor
from config.settings import AppConfig
import streamlit as st
from datetime import datetime

class EnhancedLocalAIAssistant:
    def __init__(self, session_manager):
        self.session_manager = session_manager
        self.config = AppConfig()
        
        # Initialize enhanced models
        self.ollama = EnhancedOllamaModel(self.config.ollama)
        self.whisper = WhisperModel(self.config.whisper)
        self.tts = TTSModel(self.config.tts)
        
        # Initialize enhanced services
        self.system_analyzer = SystemAnalyzer()
        self.settings_manager = SettingsManager()
        self.voice_recorder = EnhancedVoiceRecorder()
        self.pdf_processor = EnhancedPDFProcessor()
        
        # Initialize knowledge bases - restore from session if available
        self.enhanced_knowledge_base = self._restore_enhanced_knowledge_base()
        self.regular_knowledge_base = session_manager.get('pdf_knowledge_base')
        
        print(f"🔄 Assistant initialized:")
        print(f"   🧠 Enhanced KB: {'✅ Restored' if self.enhanced_knowledge_base else '❌ None'}")
        print(f"   📖 Regular KB: {'✅ Available' if self.regular_knowledge_base else '❌ None'}")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        # Load settings
        current_profile = self.session_manager.get('current_profile', 'default')
        settings = self.settings_manager.load_settings(current_profile)
        
        # Apply settings to session
        for key, value in settings.items():
            if not self.session_manager.get(key):
                self.session_manager.set(key, value)
        
        # Initialize TTS engine
        self._init_tts_engine()
        
        # Check system capabilities
        self._analyze_system()
    
    def _init_tts_engine(self):
        """Initialize TTS engine with enhanced settings"""
        try:
            engine = self.tts.init_engine()
            if engine:
                # Apply voice settings
                voice_speed = self.session_manager.get('voice_speed', 1.0)
                voice_pitch = self.session_manager.get('voice_pitch', 1.0)
                
                # Configure engine with user preferences
                self.tts.configure_voice(engine, speed=voice_speed, pitch=voice_pitch)
                
                self.session_manager.set('tts_engine', engine)
                return True
            return False
        except Exception as e:
            st.error(f"TTS initialization error: {str(e)}")
            return False
    
    def _analyze_system(self):
        """Analyze system capabilities and recommend settings"""
        try:
            analysis = self.system_analyzer.recommend_models()
            self.session_manager.set('system_analysis', analysis)
            
            # Auto-select best model if none selected
            current_model = self.session_manager.get('selected_model')
            if not current_model or not self.check_model_installed(current_model):
                best_model = analysis.get('best_model')
                if best_model:
                    self.session_manager.set('selected_model', best_model)
        
        except Exception as e:
            st.error(f"System analysis error: {str(e)}")
    
    def check_ollama_status(self) -> Dict:
        """Enhanced Ollama status check"""
        return self.system_analyzer.check_ollama_status()
    
    def check_model_installed(self, model_name: str) -> bool:
        """Check if a model is installed"""
        return self.ollama.check_model_installed(model_name)
    
    def install_ollama_model(self, model_name: str) -> bool:
        """Install an Ollama model with progress feedback"""
        return self.ollama.install_model(model_name)
    
    def get_system_recommendations(self) -> Dict:
        """Get system-based model recommendations"""
        return self.system_analyzer.recommend_models()
    
    def load_whisper_model(self, model_size: str = "base"):
        """Load Whisper model with caching"""
        cache_key = f'whisper_model_{model_size}'
        current_model = self.session_manager.get(cache_key)
        
        if current_model is None:
            model = self.whisper.load_model(model_size)
            if model:
                self.session_manager.set(cache_key, model)
                self.session_manager.set('whisper_model', model)
            return model
        return current_model
    
    def create_enhanced_knowledge_base(self, config: Dict = None) -> HybridKnowledgeBase:
        """Create enhanced knowledge base with hybrid search"""
        if config is None:
            config = {
                'chunk_size': self.session_manager.get('chunk_size', 1000),
                'overlap': self.session_manager.get('chunk_overlap', 200)
            }
        
        self.enhanced_knowledge_base = HybridKnowledgeBase(config)
        
        # Save to session state
        self._save_enhanced_knowledge_base(self.enhanced_knowledge_base)
        
        return self.enhanced_knowledge_base
    
    def process_documents_enhanced(self, texts: List[str], metadatas: List[Dict] = None) -> bool:
        """Process documents with enhanced knowledge base"""
        try:
            print(f"📄 Processing {len(texts)} documents with enhanced knowledge base...")
            
            if not self.enhanced_knowledge_base:
                print("🔧 Creating new enhanced knowledge base...")
                self.create_enhanced_knowledge_base()
            
            print(f"📚 Adding {len(texts)} documents to knowledge base...")
            self.enhanced_knowledge_base.add_documents(texts, metadatas)
            
            # Save the enhanced knowledge base to session state
            self._save_enhanced_knowledge_base(self.enhanced_knowledge_base)
            
            print("✅ Enhanced knowledge base processing completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"❌ Error processing documents: {str(e)}"
            st.error(error_msg)
            print(error_msg)
            return False
    
    def process_pdfs(self, uploaded_files) -> bool:
        """Process uploaded PDF files (compatibility method)"""
        try:
            print(f"📄 Processing {len(uploaded_files)} PDF files...")
            
            # Use the PDF processor
            pdf_processor = self.get_pdf_processor()
            
            texts = []
            metadatas = []
            
            for file in uploaded_files:
                print(f"📄 Processing file: {file.name}")
                
                try:
                    # Extract text using the PDF processor
                    result = pdf_processor.extract_text(file)
                    
                    if result and result['text']:
                        texts.append(result['text'])
                        metadatas.append(result['metadata'])
                        
                        # Also add to session manager for backward compatibility
                        self.session_manager.add_pdf_text(file.name, result['text'])
                        
                        print(f"✅ Successfully processed: {file.name}")
                    else:
                        print(f"⚠️ No text extracted from: {file.name}")
                        
                except Exception as e:
                    print(f"❌ Error processing {file.name}: {str(e)}")
            
            if texts:
                # Create knowledge base
                print(f"🧠 Creating knowledge base from {len(texts)} documents...")
                
                # Try enhanced knowledge base first
                if self.create_enhanced_knowledge_base():
                    success = self.process_documents_enhanced(texts, metadatas)
                    if success:
                        print("✅ Enhanced knowledge base created successfully")
                        return True
                
                # Fallback to basic knowledge base
                print("🔧 Falling back to basic knowledge base...")
                from services.knowledge_base import KnowledgeBase
                
                kb = KnowledgeBase(self.config.embedding)
                knowledge_base = kb.create_from_texts(texts)
                
                if knowledge_base:
                    self.session_manager.set('pdf_knowledge_base', knowledge_base)
                    print("✅ Basic knowledge base created successfully")
                    return True
                else:
                    print("❌ Failed to create basic knowledge base")
                    return False
            else:
                print("⚠️ No texts to process")
                return False
                
        except Exception as e:
            error_msg = f"❌ Error in process_pdfs: {str(e)}"
            print(error_msg)
            st.error(error_msg)
            return False
    
    def speech_to_text_enhanced(self, audio_data: bytes) -> Optional[Dict]:
        """Enhanced speech to text with confidence scores"""
        try:
            model = self.session_manager.get('whisper_model')
            if model is None:
                model_size = self.session_manager.get('whisper_model_size', 'base')
                model = self.load_whisper_model(model_size)
            
            # Enhanced transcription with metadata
            result = self.whisper.transcribe(audio_data, model)
            
            if result:
                return {
                    'text': result,
                    'confidence': 0.9,  # Placeholder - could be enhanced
                    'language': 'en',   # Could be detected
                    'timestamp': datetime.now().isoformat()
                }
            return None
            
        except Exception as e:
            st.error(f"Speech to text error: {str(e)}")
            return None
    
    def text_to_speech_enhanced(self, text: str, auto_play: bool = None) -> Optional[str]:
        """Enhanced text to speech with user preferences"""
        try:
            engine = self.session_manager.get('tts_engine')
            if not engine:
                return None
            
            # Check user preferences
            if auto_play is None:
                auto_play = self.session_manager.get('auto_play_responses', False)
            
            audio_file = self.tts.generate_audio(text, engine)
            
            if audio_file and auto_play:
                # Auto-play if enabled
                st.audio(audio_file, format='audio/wav', autoplay=True)
            
            return audio_file
            
        except Exception as e:
            st.error(f"Text to speech error: {str(e)}")
            return None
    
    def get_ai_response_enhanced(self, user_input: str, **kwargs) -> Dict:
        """Enhanced AI response with multiple modes and source citations"""
        try:
            model_name = kwargs.get('model_name', self.session_manager.get('selected_model'))
            mode = kwargs.get('mode', self.session_manager.get('chat_mode', 'chat'))
            
            print(f"🤖 get_ai_response_enhanced called:")
            print(f"   📝 User input: {user_input[:100]}...")
            print(f"   🎯 Model: {model_name}")
            print(f"   🎨 Mode: {mode}")
            
            # Get knowledge base with detailed logging
            enhanced_kb = self.enhanced_knowledge_base
            regular_kb = self.session_manager.get('pdf_knowledge_base')
            
            print(f"📚 Knowledge base status:")
            print(f"   🧠 Enhanced KB: {'✅ Available' if enhanced_kb else '❌ None'}")
            print(f"   📖 Regular KB: {'✅ Available' if regular_kb else '❌ None'}")
            
            knowledge_base = enhanced_kb or regular_kb
            
            if knowledge_base:
                print(f"🔍 Using knowledge base: {'Enhanced' if enhanced_kb else 'Regular'}")
                
                # Test knowledge base search
                if hasattr(knowledge_base, 'hybrid_search'):
                    print("🔎 Testing hybrid search...")
                    try:
                        test_results = knowledge_base.hybrid_search(user_input, k=3)
                        print(f"   📊 Search returned {len(test_results)} results")
                        for i, result in enumerate(test_results[:2]):  # Log first 2 results
                            if isinstance(result, tuple):
                                text, metadata, score = result[0], result[1], result[2] if len(result) > 2 else 0.0
                                print(f"   📄 Result {i+1}: {text[:100]}... (score: {score:.3f})")
                            else:
                                text = result.page_content if hasattr(result, 'page_content') else str(result)
                                print(f"   📄 Result {i+1}: {text[:100]}...")
                    except Exception as e:
                        print(f"   ❌ Search failed: {str(e)}")
                        
                elif hasattr(knowledge_base, 'similarity_search'):
                    print("🔎 Testing similarity search...")
                    try:
                        test_results = knowledge_base.similarity_search(user_input, k=3)
                        print(f"   📊 Search returned {len(test_results)} results")
                        for i, result in enumerate(test_results[:2]):
                            text = result.page_content if hasattr(result, 'page_content') else str(result)
                            print(f"   📄 Result {i+1}: {text[:100]}...")
                    except Exception as e:
                        print(f"   ❌ Search failed: {str(e)}")
                else:
                    print("   ❌ No search methods available")
            else:
                print("📭 No knowledge base available - will use general AI knowledge only")
            
            chat_history = self.session_manager.get('chat_history', [])
            print(f"💬 Chat history: {len(chat_history)} messages")
            
            # Generate response based on mode
            if mode in ['explain', 'quiz', 'practice', 'summarize']:
                print(f"🎓 Using educational mode: {mode}")
                response = self.ollama.generate_educational_response(
                    user_input, 
                    None,  # Will be set internally
                    knowledge_base, 
                    chat_history, 
                    mode
                )
                return {
                    'response': response,
                    'mode': mode,
                    'sources': [],
                    'model_used': model_name
                }
            
            elif knowledge_base and self.session_manager.get('show_source_citations', True):
                print("🔗 Using knowledge base with source citations")
                
                # Use enhanced search if available
                if hasattr(knowledge_base, 'hybrid_search'):
                    search_method = self.session_manager.get('search_method', 'hybrid')
                    print(f"🔍 Search method: {search_method}")
                    
                    try:
                        if search_method == 'hybrid':
                            results = knowledge_base.hybrid_search(user_input, k=5)
                        elif search_method == 'dense':
                            results = knowledge_base.similarity_search(user_input, k=5)
                        elif search_method == 'keyword':
                            results = knowledge_base.keyword_search(user_input, k=5)
                        else:
                            results = knowledge_base.hybrid_search(user_input, k=5)
                        
                        print(f"📊 Knowledge base search returned {len(results)} results")
                        
                        # Log search results
                        for i, result in enumerate(results[:3]):  # Log first 3 results
                            if isinstance(result, tuple):
                                text, metadata, score = result[0], result[1], result[2] if len(result) > 2 else 0.0
                                filename = metadata.get('filename', 'Unknown') if metadata else 'Unknown'
                                print(f"   📄 Result {i+1}: {filename} - {text[:150]}... (score: {score:.3f})")
                            else:
                                text = result.page_content if hasattr(result, 'page_content') else str(result)
                                filename = result.metadata.get('filename', 'Unknown') if hasattr(result, 'metadata') else 'Unknown'
                                print(f"   📄 Result {i+1}: {filename} - {text[:150]}...")
                        
                        if results:
                            # Generate response with sources
                            print("🎯 Generating response with knowledge base context")
                            response_data = self.ollama.generate_with_sources(
                                user_input, results, model_name
                            )
                            
                            response_data['mode'] = mode
                            response_data['model_used'] = model_name
                            print(f"✅ Generated response with {len(response_data.get('sources', []))} sources")
                            return response_data
                        else:
                            print("⚠️ No search results found - falling back to general knowledge")
                            
                    except Exception as e:
                        print(f"❌ Knowledge base search failed: {str(e)}")
                        st.error(f"Knowledge base search error: {str(e)}")
                
                # Fallback to regular response with basic knowledge base
                print("🔄 Falling back to basic knowledge base integration")
                response = self.ollama.generate_response(
                    user_input, model_name, knowledge_base, chat_history, mode
                )
                return {
                    'response': response,
                    'mode': mode,
                    'sources': [],
                    'model_used': model_name
                }
            
            else:
                # Standard response without knowledge base
                print("💭 Generating standard response without knowledge base")
                response = self.ollama.generate_response(
                    user_input, model_name, None, chat_history, mode
                )
                return {
                    'response': response,
                    'mode': mode,
                    'sources': [],
                    'model_used': model_name
                }
                
        except Exception as e:
            error_msg = f"AI response error: {str(e)}"
            print(f"❌ {error_msg}")
            st.error(error_msg)
            return {
                'response': f"I'm sorry, I encountered an error: {str(e)}",
                'mode': 'error',
                'sources': [],
                'model_used': 'error'
            }
    
    def save_conversation(self, chat_history: List[Dict], format: str = 'markdown') -> Optional[bytes]:
        """Save conversation in specified format"""
        try:
            from ui.modern_chat_components import export_chat_conversation
            return export_chat_conversation(chat_history, format)
        except Exception as e:
            st.error(f"Error saving conversation: {str(e)}")
            return None
    
    def get_quick_suggestions(self, context: str = "") -> List[str]:
        """Get context-aware quick reply suggestions"""
        base_suggestions = [
            "Explain this concept",
            "Give me an example",
            "Create a quiz question",
            "Summarize the key points",
            "What are the practical applications?"
        ]
        
        # Could be enhanced with AI-generated suggestions based on context
        return base_suggestions
    
    def get_performance_stats(self) -> Dict:
        """Get performance and usage statistics"""
        return {
            'system_info': self.session_manager.get('system_analysis', {}),
            'models_installed': self.check_ollama_status().get('installed_models', []),
            'knowledge_base_stats': self._get_knowledge_base_stats(),
            'session_stats': self._get_session_stats()
        }
    
    def _get_knowledge_base_stats(self) -> Dict:
        """Get knowledge base statistics"""
        if self.enhanced_knowledge_base:
            return self.enhanced_knowledge_base.get_stats()
        return {}
    
    def _get_session_stats(self) -> Dict:
        """Get current session statistics"""
        chat_history = self.session_manager.get('chat_history', [])
        return {
            'total_messages': len(chat_history),
            'user_messages': len([m for m in chat_history if m['role'] == 'user']),
            'assistant_messages': len([m for m in chat_history if m['role'] == 'assistant']),
            'session_duration': 'active'  # Could track actual duration
        }
    
    def cleanup_session(self):
        """Clean up session resources"""
        try:
            # Save settings if auto-save is enabled
            if self.session_manager.get('auto_save_conversations', True):
                current_profile = self.session_manager.get('current_profile', 'default')
                from services.settings_manager import get_current_settings
                settings = get_current_settings(self.session_manager)
                self.settings_manager.save_settings(settings, current_profile)
            
            # Clear temporary data
            self.session_manager.clear_temporary_data()
            
        except Exception as e:
            st.error(f"Cleanup error: {str(e)}")
    
    def get_voice_recorder(self) -> EnhancedVoiceRecorder:
        """Get voice recorder instance"""
        return self.voice_recorder
    
    def get_pdf_processor(self) -> EnhancedPDFProcessor:
        """Get PDF processor instance"""
        return self.pdf_processor
    
    def get_settings_manager(self) -> SettingsManager:
        """Get settings manager instance"""
        return self.settings_manager
    
    def get_system_analyzer(self) -> SystemAnalyzer:
        """Get system analyzer instance"""
        return self.system_analyzer

    def _restore_enhanced_knowledge_base(self):
        """Restore enhanced knowledge base from session state"""
        try:
            print("🔄 Attempting to restore enhanced knowledge base...")
            
            # Check if we have the flag indicating enhanced KB exists
            has_enhanced_kb = self.session_manager.get('has_enhanced_knowledge_base', False)
            
            if not has_enhanced_kb:
                print("   ❌ No enhanced knowledge base flag found")
                return None
            
            # Try to get the stored knowledge base object
            stored_kb = self.session_manager.get('enhanced_knowledge_base_object')
            
            if stored_kb:
                print(f"   ✅ Found stored enhanced KB: {type(stored_kb).__name__}")
                return stored_kb
            
            # If no object but we have documents, try to recreate
            pdf_documents = getattr(self.session_manager, 'pdf_documents', [])
            
            if pdf_documents:
                print(f"   🔄 Recreating enhanced KB from {len(pdf_documents)} stored documents...")
                
                # Recreate the knowledge base
                config = {
                    'chunk_size': self.session_manager.get('chunk_size', 1000),
                    'overlap': self.session_manager.get('chunk_overlap', 200)
                }
                
                enhanced_kb = HybridKnowledgeBase(config)
                
                # Extract texts and metadata
                texts = [doc['text'] for doc in pdf_documents if 'text' in doc]
                metadatas = [doc['metadata'] for doc in pdf_documents if 'metadata' in doc]
                
                if texts:
                    enhanced_kb.add_documents(texts, metadatas)
                    
                    # Store it back in session
                    self._save_enhanced_knowledge_base(enhanced_kb)
                    
                    print(f"   ✅ Successfully recreated enhanced KB with {len(texts)} documents")
                    return enhanced_kb
            
            print("   ❌ Could not restore enhanced knowledge base")
            return None
            
        except Exception as e:
            print(f"   ❌ Error restoring enhanced knowledge base: {str(e)}")
            return None
    
    def _save_enhanced_knowledge_base(self, knowledge_base):
        """Save enhanced knowledge base to session state"""
        try:
            print("💾 Saving enhanced knowledge base to session...")
            
            # Store the knowledge base object
            self.session_manager.set('enhanced_knowledge_base_object', knowledge_base)
            self.session_manager.set('has_enhanced_knowledge_base', True)
            
            print("   ✅ Enhanced knowledge base saved successfully")
            
        except Exception as e:
            print(f"   ❌ Error saving enhanced knowledge base: {str(e)}")
            st.error(f"Failed to save knowledge base: {str(e)}")
    