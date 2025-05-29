"""
Enhanced Knowledge Base with Hybrid Search Implementation
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import faiss
import json
import pickle
from pathlib import Path
import streamlit as st
import zipfile
from datetime import datetime
import tempfile
import shutil
import io

class HybridKnowledgeBase:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.documents = []
        self.metadata = []
        self.bm25 = None
        self.dense_index = None
        self.embeddings = None
        
        # Initialize models
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Upgraded to better model
        
        # Initialize reranker with proper error handling
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Cross-encoder reranker loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load cross-encoder: {e}. Using fallback scoring.")
            self.reranker = None
        
        # Index parameters
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.overlap = self.config.get('overlap', 200)
    
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """Add documents to the knowledge base"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Chunk documents with context
        chunked_docs = []
        chunked_metadata = []
        
        for text, metadata in zip(texts, metadatas):
            chunks = self._smart_chunk_with_context(text, metadata)
            chunked_docs.extend([chunk['text'] for chunk in chunks])
            chunked_metadata.extend([chunk['metadata'] for chunk in chunks])
        
        # Add to existing documents
        self.documents.extend(chunked_docs)
        self.metadata.extend(chunked_metadata)
        
        # Rebuild indices
        self._build_indices()
    
    def _smart_chunk_with_context(self, text: str, metadata: dict) -> List[dict]:
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
                        'context_after': sentences[i+1:i+3] if i+1 < len(sentences) else [],
                        'total_sentences': len(sentences)
                    }
                }
                chunks.append(chunk_data)
                
                # Overlap handling
                overlap_sentences = int(self.overlap / (current_size / len(current_chunk)))
                current_chunk = current_chunk[-overlap_sentences:]
                current_size = sum(len(s) for s in current_chunk)
        
        # Handle remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_data = {
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': len(chunks),
                    'start_sentence': len(sentences) - len(current_chunk),
                    'end_sentence': len(sentences) - 1,
                    'context_before': [],
                    'context_after': [],
                    'total_sentences': len(sentences)
                }
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using basic rules"""
        import re
        
        # Basic sentence splitting - could be enhanced with NLTK or spaCy
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _build_indices(self):
        """Build BM25 and dense indices"""
        if not self.documents:
            return
        
        # Build BM25 index
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Build dense index
        st.info("Building dense embeddings...")
        self.embeddings = self.embedding_model.encode(
            self.documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.dense_index.add(self.embeddings)
        
        st.success(f"Knowledge base built with {len(self.documents)} chunks")
    
    def hybrid_search(self, query: str, k: int = 10, 
                     alpha: float = 0.5) -> List[Tuple[str, dict, float]]:
        """
        Combine BM25 and dense retrieval with optional reranking
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for BM25 vs dense search (0 = only dense, 1 = only BM25)
        """
        if not self.documents:
            return []
        
        # Get more candidates for reranking
        candidates_k = min(k * 3, len(self.documents))  # Increase candidates for better reranking
        
        # BM25 search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_indices = np.argsort(bm25_scores)[-candidates_k:][::-1]
        
        # Dense search
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        dense_scores, dense_indices = self.dense_index.search(query_embedding, candidates_k)
        dense_scores = dense_scores[0]
        dense_indices = dense_indices[0]
        
        # Combine results
        candidates = self._merge_results(
            bm25_top_indices, bm25_scores, 
            dense_indices, dense_scores, 
            alpha
        )
        
        # Apply reranking if available
        if self.reranker is not None and len(candidates) > 1:
            try:
                reranked_results = self._rerank_results(query, candidates[:min(20, len(candidates))])
                final_results = reranked_results[:k]
                print(f"✅ Reranking applied: {len(candidates)} → {len(final_results)} results")
            except Exception as e:
                print(f"⚠️ Reranking failed: {e}. Using merged results.")
                final_results = self._convert_candidates_to_results(candidates[:k])
        else:
            final_results = self._convert_candidates_to_results(candidates[:k])
        
        return final_results
    
    def _convert_candidates_to_results(self, candidates: List[Tuple[int, float]]) -> List[Tuple[str, dict, float]]:
        """Convert candidate indices to final result format"""
        results = []
        for doc_idx, score in candidates:
            if doc_idx < len(self.documents):
                results.append((
                    self.documents[doc_idx],
                    self.metadata[doc_idx] if doc_idx < len(self.metadata) else {},
                    float(score)
                ))
        return results
    
    def _rerank_results(self, query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[str, dict, float]]:
        """Rerank results using cross-encoder"""
        if not candidates or self.reranker is None:
            return self._convert_candidates_to_results(candidates)
        
        # Prepare query-document pairs for reranking
        query_doc_pairs = []
        candidate_info = []
        
        for doc_idx, original_score in candidates:
            if doc_idx < len(self.documents):
                doc_text = self.documents[doc_idx]
                # Truncate document if too long (cross-encoders have token limits)
                if len(doc_text) > 512:
                    doc_text = doc_text[:512] + "..."
                
                query_doc_pairs.append([query, doc_text])
                candidate_info.append({
                    'doc_idx': doc_idx,
                    'original_score': original_score,
                    'document': self.documents[doc_idx],
                    'metadata': self.metadata[doc_idx] if doc_idx < len(self.metadata) else {}
                })
        
        if not query_doc_pairs:
            return []
        
        try:
            # Get reranking scores using CrossEncoder
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # Combine with original scores (weighted average)
            final_results = []
            for i, (score, candidate) in enumerate(zip(rerank_scores, candidate_info)):
                # Weight: 70% reranking score, 30% original score
                combined_score = 0.7 * float(score) + 0.3 * candidate['original_score']
                
                final_results.append((
                    candidate['document'],
                    candidate['metadata'],
                    combined_score
                ))
            
            # Sort by combined score
            final_results.sort(key=lambda x: x[2], reverse=True)
            return final_results
            
        except Exception as e:
            print(f"❌ Reranking error: {e}")
            return self._convert_candidates_to_results(candidates)
    
    def _merge_results(self, bm25_indices: np.ndarray, bm25_scores: np.ndarray,
                      dense_indices: np.ndarray, dense_scores: np.ndarray,
                      alpha: float) -> List[Tuple[int, float]]:
        """Merge BM25 and dense search results"""
        # Normalize scores
        bm25_scores_norm = self._normalize_scores(bm25_scores[bm25_indices])
        dense_scores_norm = self._normalize_scores(dense_scores)
        
        # Create score mapping
        score_map = {}
        
        # Add BM25 results
        for idx, score in zip(bm25_indices, bm25_scores_norm):
            score_map[idx] = alpha * score
        
        # Add dense results
        for idx, score in zip(dense_indices, dense_scores_norm):
            if idx in score_map:
                score_map[idx] += (1 - alpha) * score
            else:
                score_map[idx] = (1 - alpha) * score
        
        # Sort by combined score
        sorted_results = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range"""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def similarity_search(self, query: str, k: int = 10) -> List[Tuple[str, dict, float]]:
        """Simple similarity search using only dense embeddings"""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.dense_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    self.metadata[idx] if idx < len(self.metadata) else {},
                    float(score)
                ))
        
        return results
    
    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[str, dict, float]]:
        """Simple keyword search using only BM25"""
        if not self.documents:
            return []
        
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    self.metadata[idx] if idx < len(self.metadata) else {},
                    float(scores[idx])
                ))
        
        return results
    
    def save_index(self, path: str):
        """Save the knowledge base to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save documents and metadata
        with open(path / 'documents.json', 'w') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(path / 'embeddings.npy', self.embeddings)
        
        # Save FAISS index
        if self.dense_index is not None:
            faiss.write_index(self.dense_index, str(path / 'faiss_index.bin'))
        
        # Save BM25 index
        if self.bm25 is not None:
            with open(path / 'bm25_index.pkl', 'wb') as f:
                pickle.dump(self.bm25, f)
        
        # Save config
        with open(path / 'config.json', 'w') as f:
            json.dump(self.config, f)
    
    def load_index(self, path: str) -> bool:
        """Load the knowledge base from disk"""
        path = Path(path)
        
        if not path.exists():
            return False
        
        try:
            # Load documents and metadata
            with open(path / 'documents.json', 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            
            # Load embeddings
            if (path / 'embeddings.npy').exists():
                self.embeddings = np.load(path / 'embeddings.npy')
            
            # Load FAISS index
            if (path / 'faiss_index.bin').exists():
                self.dense_index = faiss.read_index(str(path / 'faiss_index.bin'))
            
            # Load BM25 index
            if (path / 'bm25_index.pkl').exists():
                with open(path / 'bm25_index.pkl', 'rb') as f:
                    self.bm25 = pickle.load(f)
            
            # Load config
            if (path / 'config.json').exists():
                with open(path / 'config.json', 'r') as f:
                    self.config.update(json.load(f))
            
            return True
        
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")
            return False
    
    def get_stats(self) -> dict:
        """Get statistics about the knowledge base"""
        return {
            'total_documents': len(self.documents),
            'total_metadata': len(self.metadata),
            'has_bm25': self.bm25 is not None,
            'has_dense_index': self.dense_index is not None,
            'has_embeddings': self.embeddings is not None,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'config': self.config
        }
    
    def clear(self):
        """Clear all data from the knowledge base"""
        self.documents = []
        self.metadata = []
        self.bm25 = None
        self.dense_index = None
        self.embeddings = None
    
    def export_knowledge_base(self, export_name: str = None) -> bytes:
        """Export the complete knowledge base as a ZIP file"""
        if not self.documents:
            raise ValueError("No documents in knowledge base to export")
        
        if not export_name:
            export_name = f"knowledge_base_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            kb_dir = Path(temp_dir) / export_name
            kb_dir.mkdir(parents=True, exist_ok=True)
            
            # Save all knowledge base components
            self.save_index(str(kb_dir))
            
            # Add export metadata
            export_metadata = {
                'export_info': {
                    'name': export_name,
                    'export_date': datetime.now().isoformat(),
                    'version': '1.0',
                    'description': 'Enhanced RAG Knowledge Base Export',
                    'compatibility': 'Enhanced Local AI Assistant v1.0+'
                },
                'statistics': self.get_stats(),
                'usage_instructions': {
                    'load_command': f'kb.load_index("./{export_name}")',
                    'requirements': ['sentence-transformers', 'faiss-cpu', 'rank-bm25'],
                    'python_version': '>=3.8'
                }
            }
            
            with open(kb_dir / 'export_info.json', 'w') as f:
                json.dump(export_metadata, f, indent=2)
            
            # Create README file
            readme_content = f"""# {export_name}

## Enhanced RAG Knowledge Base Export

**Export Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Documents**: {len(self.documents):,}
**Total Chunks**: {len(self.documents):,}

## Files Included:
- `documents.json` - All text chunks and metadata
- `embeddings.npy` - Dense vector representations  
- `faiss_index.bin` - FAISS search index
- `bm25_index.pkl` - BM25 keyword search index
- `config.json` - Configuration settings
- `export_info.json` - Export metadata and statistics
- `README.md` - This file

## How to Use:

1. Extract this ZIP file to your project directory
2. Install requirements: `pip install sentence-transformers faiss-cpu rank-bm25`
3. Load the knowledge base:

```python
from services.enhanced_knowledge_base import HybridKnowledgeBase

kb = HybridKnowledgeBase()
success = kb.load_index("./{export_name}")

if success:
    # Ready to use!
    results = kb.hybrid_search("your query here", k=5)
    print("Knowledge base loaded successfully!")
```

## Search Methods Available:
- `hybrid_search()` - Best quality (BM25 + Dense + Reranking)
- `similarity_search()` - Semantic search only
- `keyword_search()` - Keyword search only

Generated by Enhanced Local AI Assistant
"""
            
            with open(kb_dir / 'README.md', 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            # Create ZIP file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add all files from the knowledge base directory
                for file_path in kb_dir.rglob('*'):
                    if file_path.is_file():
                        # Get relative path for ZIP
                        arc_name = file_path.relative_to(kb_dir.parent)
                        zip_file.write(file_path, arc_name)
            
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
    
    def get_export_stats(self) -> Dict:
        """Get detailed statistics for export"""
        if not self.documents:
            return {}
        
        total_chars = sum(len(doc) for doc in self.documents)
        total_words = sum(len(doc.split()) for doc in self.documents)
        
        # Get unique source files
        source_files = set()
        for metadata in self.metadata:
            if 'filename' in metadata:
                source_files.add(metadata['filename'])
        
        return {
            'total_documents': len(self.documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_chunk_size': total_chars // len(self.documents) if self.documents else 0,
            'unique_source_files': len(source_files),
            'source_files': list(source_files),
            'has_embeddings': self.embeddings is not None,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'has_bm25_index': self.bm25 is not None,
            'has_dense_index': self.dense_index is not None,
            'config': self.config
        } 