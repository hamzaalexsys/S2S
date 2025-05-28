from typing import List
import streamlit as st

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = None
    
    def _get_embeddings(self):
        """Lazy load embeddings to avoid PyTorch conflicts on startup"""
        if self.embeddings is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            except Exception as e:
                st.error(f"Error loading embeddings model: {str(e)}")
                return None
        return self.embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self._get_embeddings()
        if embeddings is None:
            return []
        return embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embeddings = self._get_embeddings()
        if embeddings is None:
            return []
        return embeddings.embed_query(text) 