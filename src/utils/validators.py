import re
from urllib.parse import urlparse, parse_qs
from typing import Optional
import os

def validate_youtube_url(url: str) -> bool:
    """
    Validate if the provided URL is a valid YouTube video URL
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid YouTube URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    # YouTube URL patterns
    patterns = [
        r'^https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'^https?://(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'^https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'^https?://(?:m\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})'
    ]
    
    return any(re.match(pattern, url) for pattern in patterns)

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID if found, None otherwise
    """
    if not validate_youtube_url(url):
        return None
    
    # Extract from different URL formats
    patterns = [
        r'(?:v=|/)([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        r'embed/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    max_length = 200
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized or "untitled"

def format_number(num: int) -> str:
    """
    Format large numbers with K, M, B suffixes
    
    Args:
        num: Number to format
        
    Returns:
        Formatted number string
    """
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K"
    elif num < 1000000000:
        return f"{num/1000000:.1f}M"
    else:
        return f"{num/1000000000:.1f}B"

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."
