#!/usr/bin/env python3
"""
Enhanced Local AI Assistant - Main Application
"""

import os
import sys
from datetime import datetime
import json
from pathlib import Path

# Set environment variables BEFORE any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Suppress warnings before importing anything
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

# Import enhanced components
from core.session_manager import SessionManager
from core.enhanced_assistant import EnhancedLocalAIAssistant
from ui.modern_chat_components import apply_modern_chat_styles, render_modern_message, render_typing_indicator, get_feedback_analytics
from ui.enhanced_voice_recorder import EnhancedVoiceRecorder
from ui.enhanced_pdf_interface import render_pdf_upload_interface, render_pdf_upload_interface_simple
from services.settings_manager import render_profile_manager, render_advanced_settings
from services.system_analyzer import SystemAnalyzer

def render_enhanced_sidebar(assistant, session_manager):
    """Render enhanced sidebar with all new features"""
    
    st.sidebar.title("🤖 Enhanced AI Assistant")
    
    # System information
    with st.sidebar.expander("🖥️ System Status", expanded=False):
        system_analyzer = assistant.get_system_analyzer()
        analysis = system_analyzer.render_system_info_ui()
        
        # Ollama status
        ollama_status = assistant.check_ollama_status()
        if ollama_status['running']:
            st.success("✅ Ollama running")
            if ollama_status['installed_models']:
                st.write(f"📦 {len(ollama_status['installed_models'])} models installed")
        else:
            st.error("❌ Ollama not running")
            st.write(ollama_status.get('error', 'Unknown error'))
    
    # Model status and recommendations
    with st.sidebar.expander("🤖 AI Models", expanded=True):
        ollama_status = assistant.check_ollama_status()
        
        # Display Ollama status with debugging
        if ollama_status['running']:
            st.success(f"✅ Ollama is running")
            
            # Show detected models and allow selection
            detected_models = ollama_status.get('installed_models', [])
            if detected_models:
                st.write(f"**📋 Available Models ({len(detected_models)}):**")
                
                # Model selection dropdown
                current_model = session_manager.get('selected_model')
                
                # Auto-select first model if none selected
                if not current_model and detected_models:
                    session_manager.set('selected_model', detected_models[0])
                    current_model = detected_models[0]
                    st.success(f"🎯 Auto-selected: {current_model}")
                
                # Set default selection index
                default_index = 0
                if current_model and current_model in detected_models:
                    default_index = detected_models.index(current_model)
                
                selected_model = st.selectbox(
                    "Select Model:",
                    detected_models,
                    index=default_index,
                    key="model_selector"
                )
                
                # Update session when selection changes
                if selected_model != current_model:
                    session_manager.set('selected_model', selected_model)
                    st.success(f"✅ Selected: {selected_model}")
                    st.rerun()
                
                # Show current selection
                if selected_model:
                    st.info(f"🤖 Current model: {selected_model}")
                
                # Debug section if developer mode is on
                if session_manager.get('developer_mode', False):
                    st.write("**🔧 Debug Info:**")
                    st.write(f"- All detected: {detected_models}")
                    st.write(f"- Currently selected: {selected_model}")
                    st.write(f"- Raw status: {ollama_status}")
            else:
                st.warning("⚠️ No models detected")
                st.write("Install a model using the button below.")
        else:
            st.error(f"❌ Ollama not running: {ollama_status.get('error', 'Unknown error')}")
            st.info("Start Ollama by running 'ollama serve' in your terminal")
        
        # Install new model button
        if st.button("➕ Install New Model", use_container_width=True):
            st.session_state.show_model_installer = True
    
    # Voice settings
    with st.sidebar.expander("🎤 Voice Settings"):
        enable_voice_input = st.checkbox(
            "Enable Voice Input",
            value=session_manager.get('enable_speech_input', True),
            key="voice_input_enabled"
        )
        session_manager.set('enable_speech_input', enable_voice_input)
        
        enable_voice_output = st.checkbox(
            "Enable Voice Output",
            value=session_manager.get('enable_speech_output', True),
            key="voice_output_enabled"
        )
        session_manager.set('enable_speech_output', enable_voice_output)
        
        if enable_voice_input:
            whisper_size = st.selectbox(
                "Whisper Model Size",
                ["tiny", "base", "small", "medium", "large"],
                index=["tiny", "base", "small", "medium", "large"].index(
                    session_manager.get('whisper_model_size', 'base')
                ),
                key="whisper_model_select"
            )
            session_manager.set('whisper_model_size', whisper_size)
        
        if enable_voice_output:
            voice_speed = st.slider(
                "Voice Speed",
                min_value=0.5,
                max_value=2.0,
                value=session_manager.get('voice_speed', 1.0),
                step=0.1
            )
            session_manager.set('voice_speed', voice_speed)
            
            auto_play = st.checkbox(
                "Auto-play Responses",
                value=session_manager.get('auto_play_responses', False)
            )
            session_manager.set('auto_play_responses', auto_play)
    
    # Educational mode
    with st.sidebar.expander("🎓 Educational Mode"):
        educational_mode = st.checkbox(
            "Enable Educational Features",
            value=session_manager.get('enable_educational_mode', False)
        )
        session_manager.set('enable_educational_mode', educational_mode)
        
        if educational_mode:
            chat_mode = st.selectbox(
                "Learning Mode",
                ['chat', 'explain', 'quiz', 'practice', 'summarize'],
                index=['chat', 'explain', 'quiz', 'practice', 'summarize'].index(
                    session_manager.get('chat_mode', 'chat')
                )
            )
            session_manager.set('chat_mode', chat_mode)
    
    # PDF Processing
    with st.sidebar.expander("📄 Document Processing"):
        if st.button("📤 Upload PDFs", use_container_width=True):
            print("📄 PDF upload button clicked - setting show_pdf_interface to True")
            st.session_state.show_pdf_interface = True
            print(f"📊 Session state show_pdf_interface: {st.session_state.get('show_pdf_interface', False)}")
            st.rerun()
        
        # Show knowledge base status
        if session_manager.get('has_enhanced_knowledge_base'):
            st.success("✅ Enhanced knowledge base loaded")
            print("📚 Enhanced knowledge base is loaded")
        elif session_manager.get('pdf_knowledge_base'):
            st.info("📚 Basic knowledge base loaded")
            print("📚 Basic knowledge base is loaded")
        else:
            st.warning("📭 No documents loaded")
            print("📭 No knowledge base loaded")
    
    # Profile management
    with st.sidebar.expander("👤 Profile Management"):
        render_profile_manager(session_manager)
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        # Developer mode toggle
        developer_mode = st.checkbox(
            "🔧 Developer Mode (show debug info)",
            value=session_manager.get('developer_mode', False),
            key="developer_mode_toggle"
        )
        session_manager.set('developer_mode', developer_mode)
        
        render_advanced_settings(session_manager)
    
    # Performance stats
    if session_manager.get('developer_mode', False):
        with st.sidebar.expander("📊 Performance Stats"):
            stats = assistant.get_performance_stats()
            
            st.write("**System Info:**")
            system_info = stats.get('system_info', {}).get('system_info', {})
            if system_info:
                memory = system_info.get('memory', {})
                st.write(f"RAM: {memory.get('used_gb', 0):.1f} / {memory.get('total_gb', 0):.1f} GB")
                st.write(f"CPU: {system_info.get('cpu', {}).get('threads', 0)} threads")
            
            st.write("**Session Stats:**")
            session_stats = stats.get('session_stats', {})
            st.write(f"Messages: {session_stats.get('total_messages', 0)}")
            st.write(f"Models: {len(stats.get('models_installed', []))}")
            
            # Add feedback analytics
            feedback_stats = get_feedback_analytics()
            
            if feedback_stats['total_feedback'] > 0:
                st.write("**User Feedback:**")
                satisfaction_rate = feedback_stats['satisfaction_rate'] * 100
                st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("👍 Positive", feedback_stats['positive_count'])
                with col2:
                    st.metric("👎 Negative", feedback_stats['negative_count'])
                
                if feedback_stats['common_issues']:
                    st.write("**Top Issues:**")
                    for issue, count in feedback_stats['common_issues']:
                        st.write(f"- {issue}: {count}")
            else:
                st.info("No feedback collected yet")

    # Add knowledge base export section
    st.markdown("---")
    render_knowledge_base_export(assistant, session_manager)

def render_enhanced_chat_interface(assistant, session_manager):
    """Render enhanced chat interface with modern components"""
    
    # Apply modern styles
    apply_modern_chat_styles()
    
    # Main chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat history
    chat_history = session_manager.get('chat_history', [])
    
    if chat_history:
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        for i, message in enumerate(chat_history):
            render_modern_message(message, i)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px; opacity: 0.7;">
            <h2>👋 Welcome to Enhanced AI Assistant!</h2>
            <p>Start a conversation, upload documents, or ask me anything.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    render_input_interface(assistant, session_manager)

def render_input_interface(assistant, session_manager):
    """Render enhanced input interface"""
    
    # Voice recorder integration
    voice_recorder = assistant.get_voice_recorder()
    
    # Check if voice input is enabled
    voice_enabled = session_manager.get('enable_speech_input', True)
    
    # Use a form to handle input properly
    with st.form("user_input_form", clear_on_submit=True):
        # Text input
        user_input = st.text_input(
            "Message",
            placeholder="Type your message or use voice input...",
            label_visibility="collapsed"
        )
        
        # Submit button (invisible but necessary for form)
        submitted = st.form_submit_button("Send", use_container_width=True)
    
    # Voice controls (outside form to avoid conflicts)
    if voice_enabled:
        st.markdown("**🎤 Voice Controls**")
        audio_data = voice_recorder.render_voice_controls()
        if audio_data:
            # Process voice input
            with st.spinner("Processing voice..."):
                result = assistant.speech_to_text_enhanced(audio_data)
                if result and result['text']:
                    # Add voice message directly without modifying session state
                    process_user_input(assistant, session_manager, result['text'])
                    return
        
        # Advanced voice controls
        voice_recorder.render_advanced_controls()
    
    # Quick suggestions
    educational_mode = session_manager.get('enable_educational_mode', False)
    chat_history = session_manager.get('chat_history', [])
    if educational_mode and chat_history:
        from ui.modern_chat_components import render_message_suggestions
        suggestions = assistant.get_quick_suggestions()
        selected_suggestion = render_message_suggestions(suggestions[:3])
        if selected_suggestion:
            # Process suggestion directly
            process_user_input(assistant, session_manager, selected_suggestion)
            return
    
    # Process input when form is submitted
    if submitted and user_input:
        process_user_input(assistant, session_manager, user_input)

def process_user_input(assistant, session_manager, user_input):
    """Process user input and generate response"""
    
    # Add user message to history
    user_message = {
        'role': 'user',
        'content': user_input,
        'timestamp': st.session_state.get('current_time', ''),
        'id': f"user_{len(session_manager.get('chat_history', []))}"
    }
    
    session_manager.add_message(user_message)
    
    # Show typing indicator
    with st.empty():
        render_typing_indicator()
        
        # Generate AI response
        try:
            response_data = assistant.get_ai_response_enhanced(
                user_input,
                model_name=session_manager.get('selected_model'),
                mode=session_manager.get('chat_mode', 'chat')
            )
            
            # Add assistant message
            assistant_message = {
                'role': 'assistant',
                'content': response_data['response'],
                'timestamp': st.session_state.get('current_time', ''),
                'id': f"assistant_{len(session_manager.get('chat_history', []))}", 
                'sources': response_data.get('sources', []),
                'mode': response_data.get('mode', 'chat'),
                'model_used': response_data.get('model_used', 'unknown')
            }
            
            session_manager.add_message(assistant_message)
            
            # Text-to-speech if enabled
            if session_manager.get('enable_speech_output', True):
                assistant.text_to_speech_enhanced(response_data['response'])
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    
    # Rerun to update the interface (no need to clear input, form handles it)
    st.rerun()

def render_modal_interfaces(assistant, session_manager):
    """Render modal interfaces for PDF upload, model installation, etc."""
    
    print(f"🔍 Checking modal interfaces - show_pdf_interface: {st.session_state.get('show_pdf_interface', False)}")
    print(f"🔍 Checking modal interfaces - show_model_installer: {st.session_state.get('show_model_installer', False)}")
    
    # PDF upload interface
    if st.session_state.get('show_pdf_interface', False):
        print("📄 Rendering PDF upload interface")
        
        with st.container():
            st.markdown("---")
            render_pdf_upload_interface_simple(assistant, session_manager)
            
            if st.button("❌ Close PDF Interface", use_container_width=True):
                print("📄 Closing PDF interface")
                st.session_state.show_pdf_interface = False
                st.rerun()
    else:
        print("📄 PDF interface not shown (show_pdf_interface is False)")
    
    # Model installer
    if st.session_state.get('show_model_installer', False):
        print("🤖 Rendering model installer interface")
        
        with st.container():
            st.markdown("---")
            st.subheader("📥 Install New AI Model")
            
            st.markdown("""
            **Choose a model based on your needs:**
            - 🚀 **Fast & Light**: Good for quick responses, basic tasks
            - ⚖️ **Balanced**: Good quality and reasonable speed
            - 🧠 **High Quality**: Best responses, slower but more accurate
            """)
            
            # Organize models by categories
            model_categories = {
                "🚀 Fast & Light Models": {
                    "qwen2.5:3b": {
                        "name": "Qwen2.5 3B",
                        "description": "Fast Chinese & English model",
                        "size": "~2GB", "speed": "Very Fast", "quality": "Good"
                    },
                    "llama3.2:1b": {
                        "name": "Llama 3.2 1B", 
                        "description": "Smallest Llama model",
                        "size": "~1.3GB", "speed": "Very Fast", "quality": "Good"
                    },
                    "phi3:mini": {
                        "name": "Phi-3 Mini",
                        "description": "Microsoft's efficient model",
                        "size": "~2.3GB", "speed": "Fast", "quality": "Good"
                    },
                    "gemma2:2b": {
                        "name": "Gemma2 2B",
                        "description": "Google's fast model", 
                        "size": "~1.6GB", "speed": "Fast", "quality": "Good"
                    }
                },
                "⚖️ Balanced Models": {
                    "qwen2.5:latest": {
                        "name": "Qwen2.5 Latest (7B)",
                        "description": "Latest Qwen model - excellent all-around",
                        "size": "~4.7GB", "speed": "Medium", "quality": "Very Good"
                    },
                    "llama3.2:3b": {
                        "name": "Llama 3.2 3B",
                        "description": "Balanced Llama model",
                        "size": "~2.0GB", "speed": "Medium", "quality": "Good"
                    },
                    "llama3.2": {
                        "name": "Llama 3.2 (8B)",
                        "description": "Standard Llama model",
                        "size": "~4.7GB", "speed": "Medium", "quality": "Very Good"
                    },
                    "mistral:7b": {
                        "name": "Mistral 7B",
                        "description": "Excellent for reasoning tasks",
                        "size": "~4.1GB", "speed": "Medium", "quality": "Very Good"
                    }
                },
                "🧠 High Quality Models": {
                    "qwen2.5:14b": {
                        "name": "Qwen2.5 14B",
                        "description": "Large Qwen model - best quality",
                        "size": "~8.5GB", "speed": "Slow", "quality": "Excellent"
                    },
                    "llama3.1": {
                        "name": "Llama 3.1 (8B)",
                        "description": "Advanced Llama model",
                        "size": "~4.7GB", "speed": "Medium", "quality": "Excellent"
                    },
                    "codellama": {
                        "name": "Code Llama",
                        "description": "Specialized for coding tasks",
                        "size": "~3.8GB", "speed": "Medium", "quality": "Excellent for Code"
                    },
                    "mixtral:8x7b": {
                        "name": "Mixtral 8x7B",
                        "description": "Mixture of Experts - highest quality",
                        "size": "~26GB", "speed": "Slow", "quality": "Excellent"
                    }
                }
            }
            
            # Let user select category first
            selected_category = st.selectbox(
                "1️⃣ Choose a category:",
                list(model_categories.keys()),
                index=1  # Default to Balanced
            )
            
            # Show models in selected category
            if selected_category:
                st.write(f"**2️⃣ Choose a model from {selected_category}:**")
                
                category_models = model_categories[selected_category]
                
                # Create radio buttons for model selection
                model_options = []
                model_keys = []
                
                for model_key, model_info in category_models.items():
                    option_text = (
                        f"**{model_info['name']}** - {model_info['description']}\n"
                        f"   📏 Size: {model_info['size']} | "
                        f"⚡ Speed: {model_info['speed']} | "
                        f"🎯 Quality: {model_info['quality']}"
                    )
                    model_options.append(option_text)
                    model_keys.append(model_key)
                
                if model_options:
                    selected_idx = st.radio(
                        "Select model:",
                        range(len(model_options)),
                        format_func=lambda x: model_options[x],
                        key="model_installer_radio"
                    )
                    
                    selected_model = model_keys[selected_idx]
                    selected_info = category_models[selected_model]
                    
                    # Show detailed info for selected model
                    st.markdown("---")
                    st.markdown(f"### 📋 Selected: {selected_info['name']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📏 Download Size", selected_info['size'])
                    with col2:
                        st.metric("⚡ Speed", selected_info['speed'])
                    with col3:
                        st.metric("🎯 Quality", selected_info['quality'])
                    
                    st.info(f"ℹ️ {selected_info['description']}")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        if st.button(f"📥 Install {selected_info['name']}", type="primary", use_container_width=True):
                            print(f"🤖 Installing model: {selected_model}")
                            
                            with st.spinner(f"Installing {selected_info['name']}... This may take a few minutes."):
                                success = assistant.install_ollama_model(selected_model)
                                if success:
                                    st.success(f"✅ {selected_info['name']} installed successfully!")
                                    # Update the selected model to the newly installed one
                                    session_manager.set('selected_model', selected_model)
                                    st.session_state.show_model_installer = False
                                    print(f"✅ Model {selected_model} installed and set as active")
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to install {selected_info['name']}")
                                    print(f"❌ Failed to install model: {selected_model}")
                    
                    with col2:
                        if st.button("❌ Cancel", use_container_width=True):
                            print("🤖 Model installer cancelled")
                            st.session_state.show_model_installer = False
                            st.rerun()
                    
                    with col3:
                        if st.button("🔄 Refresh", use_container_width=True):
                            print("🤖 Model installer refreshed")
                            st.rerun()
            
            # Add helpful tips
            with st.expander("💡 Tips for choosing a model"):
                st.markdown("""
                **🚀 Fast & Light models** are good if you:
                - Want quick responses
                - Have limited RAM or storage
                - Use basic chat and Q&A
                
                **⚖️ Balanced models** are good if you:
                - Want good quality responses
                - Have moderate system resources
                - Need reliable performance
                
                **🧠 High Quality models** are good if you:
                - Want the best possible responses
                - Have powerful hardware
                - Do complex reasoning or coding
                
                **💾 Storage space:** Models are downloaded once and stored locally.
                **🔄 Switching:** You can install multiple models and switch between them anytime.
                """)
    else:
        print("🤖 Model installer not shown (show_model_installer is False)")

def render_knowledge_base_import(assistant, session_manager):
    """Render knowledge base import interface"""
    
    st.subheader("📥 Import Knowledge Base")
    
    st.markdown("""
    **Import a pre-built knowledge base:**
    - Upload a ZIP file exported from this or another Enhanced AI Assistant
    - Instantly load all documents, embeddings, and search indices
    - No need to reprocess PDFs or rebuild embeddings
    """)
    
    # File uploader for knowledge base ZIP
    uploaded_kb = st.file_uploader(
        "Choose Knowledge Base ZIP file",
        type=['zip'],
        key="kb_import_uploader",
        help="Select a knowledge base ZIP file to import"
    )
    
    if uploaded_kb:
        # Show file info
        file_size = len(uploaded_kb.getvalue()) / (1024 * 1024)  # MB
        st.write(f"**File:** {uploaded_kb.name} ({file_size:.1f} MB)")
        
        # Import button
        if st.button("📦 Import Knowledge Base", type="primary", use_container_width=True):
            try:
                with st.spinner("Importing knowledge base..."):
                    # Create temporary directory for extraction
                    import tempfile
                    import zipfile
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Extract ZIP file
                        zip_path = Path(temp_dir) / "imported_kb.zip"
                        with open(zip_path, 'wb') as f:
                            f.write(uploaded_kb.getvalue())
                        
                        # Extract and find knowledge base directory
                        with zipfile.ZipFile(zip_path, 'r') as zip_file:
                            zip_file.extractall(temp_dir)
                        
                        # Find the knowledge base directory (first directory in ZIP)
                        extracted_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
                        
                        if not extracted_dirs:
                            st.error("❌ Invalid knowledge base ZIP file - no directories found")
                            return
                        
                        kb_dir = extracted_dirs[0]
                        
                        # Check if it looks like a valid knowledge base
                        required_files = ['documents.json', 'config.json']
                        missing_files = [f for f in required_files if not (kb_dir / f).exists()]
                        
                        if missing_files:
                            st.error(f"❌ Invalid knowledge base - missing files: {missing_files}")
                            return
                        
                        # Load the knowledge base
                        from services.enhanced_knowledge_base import HybridKnowledgeBase
                        
                        new_kb = HybridKnowledgeBase()
                        success = new_kb.load_index(str(kb_dir))
                        
                        if success:
                            # Replace current knowledge base
                            assistant.enhanced_knowledge_base = new_kb
                            session_manager.set('has_enhanced_knowledge_base', True)
                            
                            # Get imported stats
                            stats = new_kb.get_stats()
                            
                            st.success("✅ Knowledge base imported successfully!")
                            
                            # Show imported stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Documents", stats.get('total_documents', 0))
                            with col2:
                                st.metric("Embeddings", "✅" if stats.get('has_embeddings') else "❌")
                            with col3:
                                st.metric("Search Indices", "✅" if stats.get('has_bm25') and stats.get('has_dense_index') else "❌")
                            
                            # Show import info if available
                            try:
                                with open(kb_dir / 'export_info.json', 'r') as f:
                                    export_info = json.load(f)
                                    
                                st.markdown("**Import Details:**")
                                info = export_info.get('export_info', {})
                                st.write(f"- **Original Export:** {info.get('name', 'Unknown')}")
                                st.write(f"- **Export Date:** {info.get('export_date', 'Unknown')}")
                                st.write(f"- **Version:** {info.get('version', 'Unknown')}")
                                
                            except:
                                pass  # No export info available
                            
                            st.info("💡 You can now use the imported knowledge base for questions and searches!")
                            
                            # Clear the uploaded file
                            st.session_state.kb_import_uploader = None
                            
                        else:
                            st.error("❌ Failed to load knowledge base - file may be corrupted")
                            
            except Exception as e:
                st.error(f"❌ Import failed: {str(e)}")
                st.write("Please ensure the ZIP file is a valid knowledge base export.")

def render_knowledge_base_export(assistant, session_manager):
    """Render knowledge base export interface"""
    
    # Check if we have a knowledge base
    has_kb = session_manager.get('has_enhanced_knowledge_base', False)
    kb = assistant.enhanced_knowledge_base
    
    if not has_kb or not kb or not kb.documents:
        st.info("📚 No knowledge base available to export. Upload and process some PDFs first!")
        
        # Show import option if no knowledge base exists
        st.markdown("---")
        render_knowledge_base_import(assistant, session_manager)
        return
    
    st.subheader("📦 Export Knowledge Base")
    
    # Get knowledge base statistics
    try:
        stats = kb.get_export_stats()
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", stats.get('total_documents', 0))
        with col2:
            st.metric("Source Files", stats.get('unique_source_files', 0))
        with col3:
            st.metric("Total Words", f"{stats.get('total_words', 0):,}")
        
        # Export name input
        export_name = st.text_input(
            "Export Name",
            value=f"enterprise_kb_{datetime.now().strftime('%Y%m%d')}",
            help="Name for your knowledge base export"
        )
        
        # Export description
        st.markdown("""
        **What gets exported:**
        - 📄 All processed documents and metadata
        - 🧠 Pre-computed embeddings (no re-processing needed)
        - 🔍 Search indices (BM25 + FAISS)
        - ⚙️ Configuration settings
        - 📖 Usage instructions and README
        """)
        
        # File size estimation
        estimated_size = estimate_export_size(stats)
        st.write(f"**Estimated ZIP size:** ~{estimated_size}")
        
        # Export button
        if st.button("📥 Create & Download Knowledge Base", type="primary", use_container_width=True):
            try:
                with st.spinner("Creating knowledge base export..."):
                    # Create the ZIP file
                    zip_data = kb.export_knowledge_base(export_name)
                    
                    # Get actual file size
                    file_size = len(zip_data) / (1024 * 1024)  # MB
                    
                    st.success(f"✅ Knowledge base exported successfully! ({file_size:.1f} MB)")
                    
                    # Download button
                    st.download_button(
                        label="💾 Download ZIP File",
                        data=zip_data,
                        file_name=f"{export_name}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                    # Show success message with instructions
                    st.markdown("""
                    ### 🎉 Export Complete!
                    
                    Your knowledge base has been packaged and is ready for download. 
                    
                    **Next steps:**
                    1. Click "Download ZIP File" above
                    2. Extract the ZIP in your target project
                    3. Use the included instructions to load it
                    
                    **Usage in other projects:**
                    ```python
                    from services.enhanced_knowledge_base import HybridKnowledgeBase
                    
                    kb = HybridKnowledgeBase()
                    kb.load_index(f"./{export_name}")
                    
                    # Ready to use!
                    results = kb.hybrid_search("your query", k=5)
                    ```
                    """)
                    
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                st.write("Please ensure you have sufficient disk space and try again.")
    
    except Exception as e:
        st.error(f"Error getting knowledge base statistics: {str(e)}")
    
    # Also show import option 
    st.markdown("---")
    render_knowledge_base_import(assistant, session_manager)

def estimate_export_size(stats: dict) -> str:
    """Estimate the size of the export ZIP file"""
    try:
        # Rough estimation based on content
        total_chars = stats.get('total_characters', 0)
        total_docs = stats.get('total_documents', 0)
        
        # Estimates (rough):
        # - JSON documents: ~1 byte per character
        # - Embeddings: ~4 bytes per dimension per document (float32)
        # - FAISS index: ~same as embeddings
        # - BM25: ~smaller, text-based
        # - Compression: ~50-70% reduction
        
        json_size = total_chars  # bytes
        embeddings_size = total_docs * 384 * 4  # 384-dim embeddings, 4 bytes each
        faiss_size = embeddings_size * 0.8  # roughly similar to embeddings
        bm25_size = total_chars * 0.1  # much smaller
        
        total_size = json_size + embeddings_size + faiss_size + bm25_size
        compressed_size = total_size * 0.4  # 60% compression estimate
        
        # Convert to human readable
        if compressed_size < 1024 * 1024:  # < 1 MB
            return f"{compressed_size / 1024:.1f} KB"
        elif compressed_size < 1024 * 1024 * 1024:  # < 1 GB
            return f"{compressed_size / (1024 * 1024):.1f} MB"
        else:
            return f"{compressed_size / (1024 * 1024 * 1024):.1f} GB"
            
    except:
        return "Unknown"

def main():
    # Page configuration
    st.set_page_config(
        page_title="Enhanced Local AI Assistant",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    print("🚀 Starting Enhanced Local AI Assistant...")
    print(f"📊 Initial session state keys: {list(st.session_state.keys())}")
    
    # Initialize session manager
    session_manager = SessionManager()
    session_manager.initialize()
    print("✅ Session manager initialized")
    
    # Initialize enhanced assistant
    try:
        assistant = EnhancedLocalAIAssistant(session_manager)
        print("✅ Enhanced assistant initialized")
    except Exception as e:
        error_msg = f"Failed to initialize assistant: {str(e)}"
        st.error(error_msg)
        print(f"❌ {error_msg}")
        st.stop()
    
    # Header with enhanced styling
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🚀 Enhanced Local AI Assistant</h1>
        <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">
            Advanced RAG • Voice Interface • Educational Mode • Smart Recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    print("📱 Creating main layout...")
    
    # Create main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        print("💬 Rendering main chat interface...")
        # Main chat interface
        render_enhanced_chat_interface(assistant, session_manager)
        
        print("🎛️ Rendering modal interfaces...")
        # Modal interfaces
        render_modal_interfaces(assistant, session_manager)
    
    with col2:
        print("🎛️ Rendering enhanced sidebar...")
        # Enhanced sidebar
        render_enhanced_sidebar(assistant, session_manager)
    
    print("🏁 Main layout rendering complete")
    
    # Cleanup on session end (if supported)
    if hasattr(st.session_state, 'cleanup_registered'):
        assistant.cleanup_session()

if __name__ == "__main__":
    main() 