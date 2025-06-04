"""
Utility functions for the YouTube Comment Sentiment Analysis project.
"""
import re
import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse, parse_qs
import logging

logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID if found, None otherwise
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)',
        r'youtube\.com\/.*[?&]v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            # Validate video ID format (11 characters, alphanumeric with - and _)
            if re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
                return video_id
    
    logger.warning(f"Could not extract video ID from URL: {url}")
    return None

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove @ mentions but keep the text after
    text = re.sub(r'@\w+', '', text)
    
    # Remove excessive punctuation (more than 3 consecutive)
    text = re.sub(r'([.!?]){4,}', r'\1\1\1', text)
    
    return text.strip()

def validate_comment(comment: str, min_length: int = 3, max_length: int = 1000) -> bool:
    """
    Validate if a comment meets basic criteria.
    
    Args:
        comment: Comment text
        min_length: Minimum comment length
        max_length: Maximum comment length
        
    Returns:
        True if valid, False otherwise
    """
    if not comment or not isinstance(comment, str):
        return False
    
    # Check length
    if len(comment.strip()) < min_length or len(comment) > max_length:
        return False
    
    # Check if comment is mostly punctuation or numbers
    alphanumeric_ratio = sum(c.isalnum() for c in comment) / len(comment)
    if alphanumeric_ratio < 0.3:
        return False
    
    # Check for spam patterns
    spam_patterns = [
        r'^(.)\1{10,}$',  # Repeated characters
        r'^\d+$',  # Only numbers
        r'^[^\w\s]+$',  # Only punctuation
    ]
    
    for pattern in spam_patterns:
        if re.match(pattern, comment):
            return False
    
    return True

def create_cache_key(*args) -> str:
    """
    Create a cache key from arguments.
    
    Args:
        *args: Arguments to create key from
        
    Returns:
        MD5 hash as cache key
    """
    key_string = '|'.join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

def save_to_cache(key: str, data: Any, cache_dir: Union[str, Path]) -> None:
    """
    Save data to cache file.
    
    Args:
        key: Cache key
        data: Data to cache
        cache_dir: Cache directory
    """
    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"Data cached to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

def load_from_cache(key: str, cache_dir: Union[str, Path]) -> Optional[Any]:
    """
    Load data from cache file.
    
    Args:
        key: Cache key
        cache_dir: Cache directory
        
    Returns:
        Cached data if found, None otherwise
    """
    try:
        cache_dir = Path(cache_dir)
        cache_file = cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Data loaded from cache: {cache_file}")
            return data
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
    
    return None

def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise

def load_json(filepath: Union[str, Path]) -> Optional[Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data if successful, None otherwise
    """
    try:
        filepath = Path(filepath)
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
    
    return None

def format_number(num: Union[int, float], precision: int = 1) -> str:
    """
    Format number with appropriate suffix (K, M, B).
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if num < 1000:
        return str(int(num))
    elif num < 1000000:
        return f"{num/1000:.{precision}f}K"
    elif num < 1000000000:
        return f"{num/1000000:.{precision}f}M"
    else:
        return f"{num/1000000000:.{precision}f}B"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

class ProgressTracker:
    """Simple progress tracker for operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        if self.current % max(1, self.total // 20) == 0:  # Update every 5%
            percentage = (self.current / self.total) * 100
            logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self) -> None:
        """Mark as finished."""
        logger.info(f"{self.description} completed: {self.current}/{self.total}")

def validate_video_id(video_id: str) -> bool:
    """
    Validate YouTube video ID format.
    
    Args:
        video_id: Video ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not video_id or not isinstance(video_id, str):
        return False
    
    # YouTube video IDs are 11 characters long and contain alphanumeric, hyphen, and underscore
    if len(video_id) != 11:
        return False
    
    # Check if it matches the pattern
    if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        return False
    
    return True
