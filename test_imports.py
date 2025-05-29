#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import os
import sys

# Set environment variables before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        print("  ✓ Testing streamlit...")
        import streamlit as st
        print("  ✅ Streamlit imported successfully")
        
        print("  ✓ Testing torch...")
        import torch
        print("  ✅ Torch imported successfully")
        
        print("  ✓ Testing whisper...")
        import whisper
        print("  ✅ Whisper imported successfully")
        
        print("  ✓ Testing langchain...")
        import langchain
        print("  ✅ LangChain imported successfully")
        
        print("  ✓ Testing langchain-community...")
        from langchain_community.llms import Ollama
        print("  ✅ LangChain Community imported successfully")
        
        print("  ✓ Testing sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        print("  ✅ Sentence Transformers imported successfully")
        
        print("  ✓ Testing faiss...")
        import faiss
        print("  ✅ FAISS imported successfully")
        
        print("  ✓ Testing pyttsx3...")
        import pyttsx3
        print("  ✅ pyttsx3 imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 