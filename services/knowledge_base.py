from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

class KnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )
    
    def _get_embeddings(self):
        """Lazy load embeddings to avoid PyTorch conflicts on startup"""
        if self.embeddings is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.model_name
                )
            except Exception as e:
                st.error(f"Error loading embeddings model: {str(e)}")
                return None
        return self.embeddings
    
    def create_from_texts(self, texts: List[str]) -> Optional:
        """Create FAISS knowledge base from texts"""
        try:
            embeddings = self._get_embeddings()
            if embeddings is None:
                return None
                
            chunks = []
            for text in texts:
                chunks.extend(self.text_splitter.split_text(text))
            
            from langchain_community.vectorstores import FAISS
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            return knowledge_base
        except Exception as e:
            st.error(f"Error creating knowledge base: {str(e)}")
            return None 