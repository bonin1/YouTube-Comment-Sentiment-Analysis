"""
Utils package for YouTube Comment Sentiment Analysis.

This package provides utility functions, data management, and helper classes.
"""

from .data_manager import DataManager
from .helpers import (
    ensure_directories, safe_execute, retry_on_failure,
    format_duration, format_number, validate_url, 
    ProgressTracker, ThreadSafeQueue, setup_environment,
    get_system_info, clean_filename, calculate_text_similarity
)

__all__ = [
    'DataManager',
    'ensure_directories', 
    'safe_execute', 
    'retry_on_failure',
    'format_duration', 
    'format_number', 
    'validate_url',
    'ProgressTracker', 
    'ThreadSafeQueue', 
    'setup_environment',
    'get_system_info', 
    'clean_filename', 
    'calculate_text_similarity'
]
