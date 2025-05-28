import tempfile
import os
from typing import Optional

def save_audio_to_temp_file(audio_data: bytes, suffix: str = ".wav") -> str:
    """Save audio data to a temporary file and return the path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio_data)
        return tmp_file.name

def cleanup_temp_file(file_path: str) -> bool:
    """Clean up a temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            return True
        return False
    except Exception:
        return False

def validate_audio_format(audio_data: bytes) -> bool:
    """Basic validation for audio data"""
    if not audio_data or len(audio_data) < 100:  # Minimum size check
        return False
    return True 