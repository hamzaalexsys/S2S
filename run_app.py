#!/usr/bin/env python3
"""
Startup script for Local AI Assistant
This script handles environment setup and runs the Streamlit app safely
"""

import os
import sys
import subprocess

def setup_environment():
    """Setup environment variables to prevent conflicts"""
    # Prevent torch from interfering with Streamlit
    # Remove TORCH_LOGS as it was causing ValueError
    # os.environ["TORCH_LOGS"] = "0"  # This was causing the error
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Streamlit specific settings
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

def main():
    """Main function to run the app"""
    print("🚀 Starting Local AI Assistant...")
    print("📋 Setting up environment...")
    
    setup_environment()
    
    print("✅ Environment configured")
    print("🌐 Starting Streamlit server...")
    
    try:
        # Run streamlit with specific configuration
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.fileWatcherType", "none",
            "--server.runOnSave", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 