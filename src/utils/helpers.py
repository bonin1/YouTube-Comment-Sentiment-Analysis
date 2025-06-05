"""
Utility functions and helpers for the sentiment analysis system.
"""

import os
import sys
import threading
import time
from typing import Callable, Any, Optional
from functools import wraps
from pathlib import Path

from config import get_logger


def ensure_directories(*dirs: Path) -> None:
    """Ensure directories exist, create them if they don't."""
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def safe_execute(func: Callable, *args, **kwargs) -> tuple[Any, Optional[Exception]]:
    """
    Safely execute a function and return result and any exception.
    
    Returns:
        Tuple of (result, exception) where exception is None if successful
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        return None, e


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function execution on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between attempts in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(__name__)
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {str(e)}")
                        raise
                    else:
                        logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {str(e)}")
                        time.sleep(delay)
            
        return wrapper
    return decorator


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_number(number: int) -> str:
    """Format large numbers with appropriate suffixes."""
    if number < 1000:
        return str(number)
    elif number < 1000000:
        return f"{number/1000:.1f}K"
    else:
        return f"{number/1000000:.1f}M"


def validate_url(url: str) -> bool:
    """Validate if URL is a valid YouTube URL."""
    import re
    
    youtube_patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
    ]
    
    for pattern in youtube_patterns:
        if re.search(pattern, url):
            return True
    
    return False


class ProgressTracker:
    """Thread-safe progress tracking for long-running operations."""
    
    def __init__(self, total: int, callback: Optional[Callable] = None):
        """Initialize progress tracker."""
        self.total = total
        self.current = 0
        self.callback = callback
        self._lock = threading.Lock()
        self.start_time = time.time()
        
    def update(self, increment: int = 1, message: str = ""):
        """Update progress."""
        with self._lock:
            self.current += increment
            if self.current > self.total:
                self.current = self.total
            
            percentage = (self.current / self.total) * 100 if self.total > 0 else 0
            elapsed = time.time() - self.start_time
            
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                eta_str = format_duration(eta)
            else:
                eta_str = "Unknown"
            
            if self.callback:
                self.callback(percentage, self.current, self.total, message, eta_str)
    
    def set_total(self, total: int):
        """Update total count."""
        with self._lock:
            self.total = total
    
    def get_progress(self) -> dict:
        """Get current progress information."""
        with self._lock:
            percentage = (self.current / self.total) * 100 if self.total > 0 else 0
            elapsed = time.time() - self.start_time
            
            return {
                'current': self.current,
                'total': self.total,
                'percentage': percentage,
                'elapsed': elapsed,
                'elapsed_str': format_duration(elapsed)
            }


class ThreadSafeQueue:
    """Simple thread-safe queue implementation."""
    
    def __init__(self):
        """Initialize queue."""
        self._queue = []
        self._lock = threading.Lock()
    
    def put(self, item: Any):
        """Add item to queue."""
        with self._lock:
            self._queue.append(item)
    
    def get(self) -> Any:
        """Get item from queue."""
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            return None
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0
    
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)


def setup_environment():
    """Setup environment for the application."""
    logger = get_logger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    
    # Set up matplotlib backend for GUI
    try:
        import matplotlib
        matplotlib.use('TkAgg')
    except ImportError:
        logger.warning("Matplotlib not available")
    
    # Set environment variables
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings from transformers
    
    logger.info("Environment setup complete")


def get_system_info() -> dict:
    """Get system information for debugging."""
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': sys.version,
        'cpu_count': os.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
    }


def clean_filename(filename: str) -> str:
    """Clean filename to be safe for file system."""
    import re
    
    # Remove or replace invalid characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Limit length
    if len(cleaned) > 100:
        cleaned = cleaned[:100]
    
    return cleaned


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using simple methods."""
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0
