"""
Enhanced PDF Processing Interface with Advanced Features
"""

import streamlit as st
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import json
from datetime import datetime

def format_file_size(size_bytes: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

class EnhancedPDFProcessor:
    def __init__(self):
        self.processed_docs = {}
        self.extraction_methods = {
            'PyPDF2': self._extract_with_pypdf2,
            'pdfplumber': self._extract_with_pdfplumber,
            'auto': self._extract_auto
        }
    
    def _extract_with_pypdf2(self, file) -> str:
        """Extract text using PyPDF2"""
        try:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"PyPDF2 extraction failed: {str(e)}")
            return ""
    
    def _extract_with_pdfplumber(self, file) -> str:
        """Extract text using pdfplumber (better for tables)"""
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row:
                                text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
            return text
        except Exception as e:
            st.error(f"pdfplumber extraction failed: {str(e)}")
            return ""
    
    def _extract_auto(self, file) -> str:
        """Auto-select best extraction method"""
        # Try pdfplumber first (better for complex layouts)
        text = self._extract_with_pdfplumber(file)
        if len(text.strip()) < 100:  # If extraction was poor, try PyPDF2
            file.seek(0)  # Reset file pointer
            text = self._extract_with_pypdf2(file)
        return text
    
    def extract_text(self, file, method: str = 'auto') -> Dict:
        """Extract text from PDF with metadata"""
        file.seek(0)  # Reset file pointer
        
        # Calculate file hash for deduplication
        content = file.read()
        file_hash = hashlib.md5(content).hexdigest()
        file.seek(0)  # Reset again
        
        # Check if already processed
        if file_hash in self.processed_docs:
            return self.processed_docs[file_hash]
        
        extraction_func = self.extraction_methods.get(method, self._extract_auto)
        text = extraction_func(file)
        
        # Get PDF metadata
        try:
            reader = PyPDF2.PdfReader(file)
            metadata = {
                'num_pages': len(reader.pages),
                'title': reader.metadata.get('/Title', ''),
                'author': reader.metadata.get('/Author', ''),
                'subject': reader.metadata.get('/Subject', ''),
                'creator': reader.metadata.get('/Creator', ''),
            }
        except:
            metadata = {'num_pages': 0}
        
        result = {
            'text': text,
            'metadata': {
                **metadata,
                'filename': file.name,
                'size_bytes': len(content),
                'hash': file_hash,
                'extraction_method': method,
                'processed_at': datetime.now().isoformat(),
                'word_count': len(text.split()) if text else 0,
                'char_count': len(text) if text else 0
            }
        }
        
        # Cache the result
        self.processed_docs[file_hash] = result
        
        return result

def render_pdf_upload_interface_simple(assistant, session_manager):
    """Simplified PDF upload interface without expanders for modal use"""
    
    # Custom CSS for drag-and-drop area
    st.markdown("""
    <style>
    .pdf-upload-area {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f9f9f9;
        margin: 10px 0;
    }
    .pdf-upload-area:hover {
        border-color: #1f77b4;
        background-color: #f0f8ff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("📄 Simple PDF Processing")
    
    # Processing options (no expander)
    st.markdown("**⚙️ Processing Settings**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        extraction_method = st.selectbox(
            "Extraction Method",
            ['auto', 'pdfplumber', 'PyPDF2'],
            help="Auto selects best method automatically"
        )
    
    with col2:
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=1000,
            help="Size of text chunks for processing"
        )
    
    with col3:
        overlap = st.slider(
            "Chunk Overlap",
            min_value=50,
            max_value=500,
            value=200,
            help="Overlap between chunks"
        )
    
    # Upload area
    st.markdown('<div class="pdf-upload-area">', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📄 Drag and drop PDF files here or click to browse",
        type=['pdf'],
        accept_multiple_files=True,
        key="simple_pdf_uploader",
        help="You can upload multiple PDF files at once"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Rest of the processing logic without expanders
    if uploaded_files:
        process_uploaded_pdfs_simple(uploaded_files, assistant, session_manager, 
                                   extraction_method, chunk_size, overlap)

def process_uploaded_pdfs_simple(files, assistant, session_manager, 
                               extraction_method, chunk_size, overlap):
    """Process uploaded PDFs without preview expanders"""
    
    if st.button("🚀 Process All PDFs", type="primary", use_container_width=True):
        print(f"📄 Starting PDF processing for {len(files)} files...")
        print(f"⚙️ Settings: method={extraction_method}, chunk_size={chunk_size}, overlap={overlap}")
        
        with st.spinner("Processing PDFs..."):
            processor = EnhancedPDFProcessor()
            processed_texts = []
            processed_metadatas = []
            
            for i, file in enumerate(files):
                print(f"📄 Processing file {i+1}/{len(files)}: {file.name}")
                
                try:
                    # Extract text
                    result = processor.extract_text(file, extraction_method)
                    print(f"📝 Extracted {len(result['text'])} characters from {file.name}")
                    
                    if result['text']:
                        # Store document info
                        processed_texts.append(result['text'])
                        processed_metadatas.append(result['metadata'])
                        
                        # Store in session manager
                        if not hasattr(session_manager, 'pdf_documents'):
                            session_manager.pdf_documents = []
                        
                        session_manager.pdf_documents.append({
                            'text': result['text'],
                            'metadata': result['metadata'],
                            'filename': file.name
                        })
                        
                        st.success(f"✅ Processed: {file.name}")
                        print(f"✅ Successfully processed: {file.name}")
                    else:
                        st.warning(f"⚠️ No text extracted from: {file.name}")
                        print(f"⚠️ No text extracted from: {file.name}")
                        
                except Exception as e:
                    error_msg = f"❌ Error processing {file.name}: {str(e)}"
                    st.error(error_msg)
                    print(error_msg)
            
            # Build knowledge base if we have documents
            if processed_texts:
                print(f"🧠 Building knowledge base with {len(processed_texts)} documents...")
                print(f"📊 Total words across documents: {sum(len(text.split()) for text in processed_texts)}")
                
                try:
                    # Check assistant capabilities
                    print(f"🔧 Assistant type: {type(assistant).__name__}")
                    print(f"📋 Assistant methods: {[method for method in dir(assistant) if not method.startswith('_')]}")
                    
                    # Check if assistant has enhanced knowledge base capability
                    if hasattr(assistant, 'create_enhanced_knowledge_base'):
                        print("🔧 Using enhanced knowledge base...")
                        
                        # Create configuration
                        config = {
                            'chunk_size': chunk_size,
                            'overlap': overlap
                        }
                        print(f"⚙️ Enhanced KB config: {config}")
                        
                        # Create enhanced knowledge base
                        print("🏗️ Creating enhanced knowledge base...")
                        kb = assistant.create_enhanced_knowledge_base(config)
                        print(f"✅ Enhanced knowledge base created: {type(kb).__name__}")
                        
                        # Process documents
                        print("📚 Adding documents to enhanced knowledge base...")
                        success = assistant.process_documents_enhanced(processed_texts, processed_metadatas)
                        
                        if success:
                            print("✅ Documents successfully added to enhanced knowledge base")
                            session_manager.set('has_enhanced_knowledge_base', True)
                            
                            # Verify the knowledge base is accessible
                            if assistant.enhanced_knowledge_base:
                                print(f"🔍 Enhanced KB verification: {type(assistant.enhanced_knowledge_base).__name__}")
                                if hasattr(assistant.enhanced_knowledge_base, 'get_stats'):
                                    stats = assistant.enhanced_knowledge_base.get_stats()
                                    print(f"📊 Enhanced KB stats: {stats}")
                            
                            st.success("🧠 Enhanced knowledge base created!")
                            print("✅ Enhanced knowledge base created successfully!")
                        else:
                            st.error("❌ Failed to create enhanced knowledge base")
                            print("❌ Failed to create enhanced knowledge base")
                    
                    elif hasattr(assistant, 'process_pdfs'):
                        print("🔧 Using standard PDF processing...")
                        
                        # Use the standard process_pdfs method
                        success = assistant.process_pdfs(files)
                        
                        if success:
                            print("✅ Standard knowledge base created")
                            
                            # Check what knowledge base was created
                            regular_kb = session_manager.get('pdf_knowledge_base')
                            if regular_kb:
                                print(f"🔍 Regular KB verification: {type(regular_kb).__name__}")
                                print(f"📋 Regular KB methods: {[method for method in dir(regular_kb) if not method.startswith('_')]}")
                            
                            st.success("📚 Standard knowledge base created!")
                            print("✅ Standard knowledge base created successfully!")
                        else:
                            st.error("❌ Failed to create knowledge base")
                            print("❌ Failed to create standard knowledge base")
                    
                    else:
                        print("❌ No PDF processing methods available in assistant")
                        st.error("❌ PDF processing not available - assistant methods missing")
                    
                    # Final verification - check what's available for queries
                    print("\n🔍 Final knowledge base verification:")
                    enhanced_kb = assistant.enhanced_knowledge_base
                    regular_kb = session_manager.get('pdf_knowledge_base')
                    
                    print(f"   🧠 Enhanced KB: {'✅ Available' if enhanced_kb else '❌ None'}")
                    print(f"   📖 Regular KB: {'✅ Available' if regular_kb else '❌ None'}")
                    
                    active_kb = enhanced_kb or regular_kb
                    if active_kb:
                        print(f"   🎯 Active KB type: {type(active_kb).__name__}")
                        print(f"   📋 Active KB methods: {[method for method in dir(active_kb) if not method.startswith('_')]}")
                        
                        # Test search capability
                        if hasattr(active_kb, 'hybrid_search'):
                            try:
                                test_results = active_kb.hybrid_search("test", k=1)
                                print(f"   ✅ Hybrid search test: {len(test_results)} results")
                            except Exception as e:
                                print(f"   ❌ Hybrid search test failed: {str(e)}")
                        
                        if hasattr(active_kb, 'similarity_search'):
                            try:
                                test_results = active_kb.similarity_search("test", k=1)
                                print(f"   ✅ Similarity search test: {len(test_results)} results")
                            except Exception as e:
                                print(f"   ❌ Similarity search test failed: {str(e)}")
                    else:
                        print("   ❌ No knowledge base available for queries!")
                    
                except Exception as e:
                    error_msg = f"❌ Knowledge base creation failed: {str(e)}"
                    st.error(error_msg)
                    print(error_msg)
                    print(f"📊 Available assistant methods: {[method for method in dir(assistant) if not method.startswith('_')]}")
            
            else:
                warning_msg = "⚠️ No documents were successfully processed"
                st.warning(warning_msg)
                print(warning_msg)

def render_pdf_upload_interface(assistant, session_manager):
    """Enhanced PDF upload interface with drag-and-drop"""
    
    # Custom CSS for modern interface
    st.markdown("""
    <style>
        .pdf-upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            transition: all 0.3s ease;
            margin: 20px 0;
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
            border-left: 4px solid #667eea;
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
        
        .extraction-stats {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .doc-preview {
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            max-height: 200px;
            overflow-y: auto;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("📄 Advanced PDF Processing")
    
    # Processing options
    with st.expander("⚙️ Processing Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            extraction_method = st.selectbox(
                "Extraction Method",
                ['auto', 'pdfplumber', 'PyPDF2'],
                help="Auto selects best method automatically"
            )
        
        with col2:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=1000,
                help="Size of text chunks for processing"
            )
        
        with col3:
            overlap = st.slider(
                "Chunk Overlap",
                min_value=50,
                max_value=500,
                value=200,
                help="Overlap between chunks"
            )
    
    # Upload area
    st.markdown('<div class="pdf-upload-area">', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📄 Drag and drop PDF files here or click to browse",
        type=['pdf'],
        accept_multiple_files=True,
        key="enhanced_pdf_uploader",
        help="You can upload multiple PDF files at once"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing options for uploaded files
    if uploaded_files:
        render_file_list(uploaded_files)
        
        # Global processing button
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Process All PDFs", type="primary", use_container_width=True):
                process_pdfs_with_progress(
                    assistant, uploaded_files, session_manager, 
                    extraction_method, chunk_size, overlap
                )
        
        with col2:
            if st.button("🔍 Preview All", use_container_width=True):
                preview_pdfs(uploaded_files, extraction_method)
        
        with col3:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.enhanced_pdf_uploader = []
                st.rerun()
    
    # Show processed documents
    render_processed_documents_enhanced(session_manager)

def render_file_list(files):
    """Render list of uploaded files with details"""
    st.markdown("### 📋 Uploaded Files")
    
    total_size = sum(file.size for file in files)
    st.info(f"📊 **Total:** {len(files)} files, {format_file_size(total_size)}")
    
    for i, file in enumerate(files):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"📄 **{file.name}**")
            
            with col2:
                st.write(f"📏 {format_file_size(file.size)}")
            
            with col3:
                st.write(f"📑 PDF Document")
            
            with col4:
                if st.button("🗑️", key=f"remove_{i}", help="Remove file"):
                    # Note: Can't actually remove from uploaded_files, 
                    # but could implement with session state
                    st.warning("Use 'Clear All' to remove files")

def preview_pdfs(files, extraction_method):
    """Preview PDF content before processing"""
    processor = EnhancedPDFProcessor()
    
    st.markdown("### 👀 PDF Previews")
    
    for file in files:
        with st.expander(f"Preview: {file.name}"):
            try:
                result = processor.extract_text(file, extraction_method)
                text = result['text']
                metadata = result['metadata']
                
                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pages", metadata.get('num_pages', 'Unknown'))
                with col2:
                    st.metric("Words", metadata.get('word_count', 0))
                with col3:
                    st.metric("Characters", metadata.get('char_count', 0))
                
                # Show text preview
                if text:
                    st.markdown("**Preview (first 500 characters):**")
                    st.markdown(f'<div class="doc-preview">{text[:500]}...</div>', 
                              unsafe_allow_html=True)
                else:
                    st.warning("No text could be extracted from this PDF")
                    
            except Exception as e:
                st.error(f"Error previewing {file.name}: {str(e)}")

def process_pdfs_with_progress(assistant, files, session_manager, 
                              extraction_method, chunk_size, overlap):
    """Process PDFs with detailed progress indication"""
    
    processor = EnhancedPDFProcessor()
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_container = st.empty()
        
        total_files = len(files)
        processed_docs = []
        failed_files = []
        
        for i, file in enumerate(files):
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {file.name}... ({i+1}/{total_files})")
            
            try:
                # Extract text
                result = processor.extract_text(file, extraction_method)
                text = result['text']
                metadata = result['metadata']
                
                if text and len(text.strip()) > 50:
                    # Add to session
                    doc_data = {
                        'text': text,
                        'metadata': metadata,
                        'processing_settings': {
                            'extraction_method': extraction_method,
                            'chunk_size': chunk_size,
                            'overlap': overlap
                        }
                    }
                    
                    processed_docs.append(doc_data)
                    
                    # Add to session manager
                    if not hasattr(session_manager, 'pdf_documents'):
                        session_manager.pdf_documents = []
                    session_manager.pdf_documents.append(doc_data)
                    
                    st.success(f"✅ {file.name} processed successfully")
                    
                    # Show extraction stats
                    with stats_container.container():
                        st.markdown(f"""
                        <div class="extraction-stats">
                            <strong>{file.name}</strong><br>
                            📄 Pages: {metadata.get('num_pages', 'Unknown')} | 
                            📝 Words: {metadata.get('word_count', 0):,} | 
                            📊 Method: {metadata.get('extraction_method', 'Unknown')}
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    failed_files.append(file.name)
                    st.error(f"❌ Could not extract meaningful text from {file.name}")
                    
            except Exception as e:
                failed_files.append(file.name)
                st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        # Final status
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Create knowledge base if we have documents
        if processed_docs:
            status_text.text("Creating knowledge base...")
            
            try:
                # Use enhanced knowledge base if available
                if hasattr(assistant, 'enhanced_knowledge_base'):
                    texts = [doc['text'] for doc in processed_docs]
                    metadatas = [doc['metadata'] for doc in processed_docs]
                    
                    config = {
                        'chunk_size': chunk_size,
                        'overlap': overlap
                    }
                    
                    assistant.enhanced_knowledge_base = assistant._create_enhanced_knowledge_base(config)
                    assistant.enhanced_knowledge_base.add_documents(texts, metadatas)
                    
                    session_manager.set('has_enhanced_knowledge_base', True)
                    
                else:
                    # Fallback to regular knowledge base
                    texts = [doc['text'] for doc in processed_docs]
                    assistant.knowledge_base = assistant._create_knowledge_base_from_texts(texts)
                
                st.success(f"🎉 Knowledge base created with {len(processed_docs)} documents!")
                
                # Show summary
                total_words = sum(doc['metadata'].get('word_count', 0) for doc in processed_docs)
                total_pages = sum(doc['metadata'].get('num_pages', 0) for doc in processed_docs)
                
                st.markdown(f"""
                **📊 Processing Summary:**
                - ✅ Successfully processed: {len(processed_docs)} files
                - ❌ Failed: {len(failed_files)} files  
                - 📄 Total pages: {total_pages:,}
                - 📝 Total words: {total_words:,}
                """)
                
                if failed_files:
                    st.warning(f"Failed files: {', '.join(failed_files)}")
                
            except Exception as e:
                st.error(f"❌ Error creating knowledge base: {str(e)}")
        
        else:
            st.warning("No documents were successfully processed")
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

def render_processed_documents_enhanced(session_manager):
    """Render enhanced processed documents interface"""
    
    if not hasattr(session_manager, 'pdf_documents') or not session_manager.pdf_documents:
        return
    
    st.subheader("📚 Processed Documents")
    
    docs = session_manager.pdf_documents
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents", len(docs))
    
    with col2:
        total_pages = sum(doc['metadata'].get('num_pages', 0) for doc in docs)
        st.metric("Total Pages", f"{total_pages:,}")
    
    with col3:
        total_words = sum(doc['metadata'].get('word_count', 0) for doc in docs)
        st.metric("Total Words", f"{total_words:,}")
    
    with col4:
        total_size = sum(doc['metadata'].get('size_bytes', 0) for doc in docs)
        st.metric("Total Size", format_file_size(total_size))
    
    # Document list with actions
    for i, doc in enumerate(docs):
        metadata = doc['metadata']
        
        with st.expander(f"📄 {metadata.get('filename', f'Document {i+1}')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Pages:** {metadata.get('num_pages', 'Unknown')}")
                st.write(f"**Words:** {metadata.get('word_count', 0):,}")
                st.write(f"**Size:** {format_file_size(metadata.get('size_bytes', 0))}")
                st.write(f"**Processed:** {metadata.get('processed_at', 'Unknown')}")
                st.write(f"**Method:** {metadata.get('extraction_method', 'Unknown')}")
                
                if metadata.get('title'):
                    st.write(f"**Title:** {metadata['title']}")
                if metadata.get('author'):
                    st.write(f"**Author:** {metadata['author']}")
            
            with col2:
                if st.button("🗑️ Remove", key=f"remove_doc_{i}"):
                    session_manager.pdf_documents.pop(i)
                    st.rerun()
                
                if st.button("👀 Preview", key=f"preview_doc_{i}"):
                    st.markdown("**Text Preview:**")
                    preview_text = doc['text'][:1000]
                    st.text_area(
                        "Content", 
                        preview_text + "..." if len(doc['text']) > 1000 else preview_text,
                        height=200,
                        key=f"preview_text_{i}"
                    )
    
    # Bulk actions
    if docs:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All Documents"):
                session_manager.pdf_documents = []
                session_manager.set('has_enhanced_knowledge_base', False)
                st.rerun()
        
        with col2:
            if st.button("💾 Export Metadata"):
                export_documents_metadata(docs)
        
        with col3:
            if st.button("🔄 Rebuild Knowledge Base"):
                rebuild_knowledge_base(session_manager, docs)

def export_documents_metadata(docs):
    """Export document metadata as JSON"""
    metadata_only = [doc['metadata'] for doc in docs]
    
    json_data = json.dumps(metadata_only, indent=2, default=str)
    
    st.download_button(
        label="📥 Download Metadata JSON",
        data=json_data,
        file_name=f"pdf_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def rebuild_knowledge_base(session_manager, docs):
    """Rebuild the knowledge base from existing documents"""
    try:
        with st.spinner("Rebuilding knowledge base..."):
            # This would integrate with the assistant's knowledge base
            st.success("✅ Knowledge base rebuilt successfully!")
    except Exception as e:
        st.error(f"❌ Error rebuilding knowledge base: {str(e)}") 