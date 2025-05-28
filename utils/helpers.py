import re
from typing import List, Dict, Any, Optional

def sanitize_text(text: str) -> str:
    """Sanitize text for display in UI"""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def validate_model_name(model_name: str) -> bool:
    """Validate if model name follows expected format"""
    if not model_name:
        return False
    # Basic validation for model names
    pattern = r'^[a-zA-Z0-9._:-]+$'
    return bool(re.match(pattern, model_name))

def extract_error_message(exception: Exception) -> str:
    """Extract a clean error message from an exception"""
    error_msg = str(exception)
    # Remove common prefixes that aren't user-friendly
    prefixes_to_remove = [
        "Error: ",
        "Exception: ",
        "RuntimeError: ",
        "ValueError: "
    ]
    
    for prefix in prefixes_to_remove:
        if error_msg.startswith(prefix):
            error_msg = error_msg[len(prefix):]
            break
    
    return error_msg.strip() 