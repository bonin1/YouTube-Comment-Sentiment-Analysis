import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from src.core.comment_scraper import YouTubeCommentScraper
from src.core.sentiment_analyzer import SentimentAnalyzer
from src.core.data_processor import DataProcessor
from src.gui.main_window import SentimentAnalysisGUI
from src.utils.logger import setup_logger
from src.config.settings import get_settings

def main():
    """Main entry point for the application"""
    settings = get_settings()
    logger = setup_logger()
    
    logger.info("Starting YouTube Comment Sentiment Analysis Application")
    
    try:
        # Initialize GUI
        app = SentimentAnalysisGUI()
        app.run()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()
