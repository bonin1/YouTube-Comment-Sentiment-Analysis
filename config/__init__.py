"""
Configuration package for YouTube Comment Sentiment Analysis.

This package provides centralized configuration management, logging setup,
and settings validation for the entire application.
"""

from .settings import settings, Settings
from .logging_config import setup_logging, get_logger, get_gui_logs, set_log_level

__all__ = [
    'settings',
    'Settings', 
    'setup_logging',
    'get_logger',
    'get_gui_logs',
    'set_log_level'
]

__version__ = "1.0.0"
