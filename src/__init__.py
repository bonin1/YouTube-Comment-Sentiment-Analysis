"""
Source package for YouTube Comment Sentiment Analysis.

This package contains all core functionality including scrapers, analyzers,
processors, visualizers, and utilities.
"""

from .scrapers import CommentScraper
from .analyzers import SentimentAnalyzer
from .processors import TextPreprocessor
from .visualizers import AdvancedVisualizer
from .utils import DataManager, ProgressTracker

__all__ = [
    'CommentScraper',
    'SentimentAnalyzer', 
    'TextPreprocessor',
    'AdvancedVisualizer',
    'DataManager',
    'ProgressTracker'
]

__version__ = "1.0.0"
