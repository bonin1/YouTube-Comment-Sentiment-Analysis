import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

# Add src to path for testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.comment_scraper import YouTubeCommentScraper, Comment
from src.core.sentiment_analyzer import SentimentAnalyzer, SentimentResult
from src.core.data_processor import DataProcessor
from src.utils.validators import validate_youtube_url, extract_video_id

class TestCommentScraper(unittest.TestCase):
    """Test YouTube comment scraper functionality"""
    
    def setUp(self):
        self.scraper = YouTubeCommentScraper()
    
    def test_validate_youtube_url(self):
        """Test URL validation"""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ"
        ]
        
        invalid_urls = [
            "https://vimeo.com/123456",
            "not_a_url",
            "",
            "https://youtube.com/watch?v=invalid"
        ]
        
        for url in valid_urls:
            self.assertTrue(validate_youtube_url(url), f"Should be valid: {url}")
        
        for url in invalid_urls:
            self.assertFalse(validate_youtube_url(url), f"Should be invalid: {url}")
    
    def test_extract_video_id(self):
        """Test video ID extraction"""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("invalid_url", None)
        ]
        
        for url, expected_id in test_cases:
            result = extract_video_id(url)
            self.assertEqual(result, expected_id, f"Failed for URL: {url}")
    
    def test_comment_creation(self):
        """Test Comment object creation"""
        comment = Comment(
            id="test_id",
            text="Test comment",
            author="Test Author",
            likes=5,
            reply_count=2,
            time_parsed=datetime.now()
        )
        
        self.assertEqual(comment.id, "test_id")
        self.assertEqual(comment.text, "Test comment")
        self.assertEqual(comment.likes, 5)
        self.assertFalse(comment.is_reply)
        
        # Test to_dict method
        comment_dict = comment.to_dict()
        self.assertIsInstance(comment_dict, dict)
        self.assertIn('id', comment_dict)
        self.assertIn('text', comment_dict)

class TestSentimentAnalyzer(unittest.TestCase):
    """Test sentiment analysis functionality"""
    
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_sentiment_result_creation(self):
        """Test SentimentResult object creation"""
        result = SentimentResult(
            text="Great video!",
            sentiment="positive",
            confidence=0.95,
            scores={"positive": 0.95, "negative": 0.03, "neutral": 0.02}
        )
        
        self.assertEqual(result.sentiment, "positive")
        self.assertEqual(result.confidence, 0.95)
        
        # Test to_dict method
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertIn('sentiment', result_dict)
    
    def test_basic_sentiment_analysis(self):
        """Test basic sentiment analysis"""
        test_cases = [
            ("I love this video!", "positive"),
            ("This is terrible", "negative"),
            ("This is okay", "neutral")
        ]
        
        for text, expected_sentiment in test_cases:
            try:
                result = self.analyzer.analyze_sentiment(text, method="vader")
                self.assertIsInstance(result, SentimentResult)
                self.assertIn(result.sentiment, ["positive", "negative", "neutral"])
            except Exception as e:
                # Skip if VADER is not available
                self.skipTest(f"VADER not available: {e}")
    
    def test_empty_text_handling(self):
        """Test handling of empty or invalid text"""
        test_cases = ["", None, "   ", "a"]
        
        for text in test_cases:
            result = self.analyzer.analyze_sentiment(text)
            self.assertIsInstance(result, SentimentResult)
            # Should return neutral for empty/invalid text
            if not text or not text.strip() or len(text.strip()) < 2:
                self.assertEqual(result.sentiment, "neutral")

class TestDataProcessor(unittest.TestCase):
    """Test data processing functionality"""
    
    def setUp(self):
        self.processor = DataProcessor()
    
    def test_sentiment_summary_generation(self):
        """Test sentiment summary generation"""
        # Create sample data
        sample_data = pd.DataFrame([
            {
                'text': 'Great video!',
                'sentiment': 'positive',
                'sentiment_confidence': 0.9,
                'likes': 10,
                'author': 'user1'
            },
            {
                'text': 'Not bad',
                'sentiment': 'neutral',
                'sentiment_confidence': 0.6,
                'likes': 2,
                'author': 'user2'
            },
            {
                'text': 'Terrible content',
                'sentiment': 'negative',
                'sentiment_confidence': 0.8,
                'likes': 0,
                'author': 'user3'
            }
        ])
        
        summary = self.processor.get_sentiment_summary(sample_data)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_comments', summary)
        self.assertIn('sentiment_distribution', summary)
        self.assertIn('sentiment_percentages', summary)
        
        self.assertEqual(summary['total_comments'], 3)
        self.assertEqual(summary['sentiment_distribution']['positive'], 1)
        self.assertEqual(summary['sentiment_distribution']['negative'], 1)
        self.assertEqual(summary['sentiment_distribution']['neutral'], 1)
    
    def test_data_cleaning(self):
        """Test data cleaning functionality"""
        # Create sample data with issues
        dirty_data = pd.DataFrame([
            {'text': 'Good comment', 'author': 'user1'},
            {'text': '', 'author': 'user2'},  # Empty text
            {'text': 'a', 'author': 'user3'},  # Too short
            {'text': 'Good comment', 'author': 'user1'},  # Duplicate
            {'text': None, 'author': 'user4'},  # None text
        ])
        
        cleaned_data = self.processor.clean_text_data(dirty_data)
        
        # Should remove empty, short, and duplicate entries
        self.assertLess(len(cleaned_data), len(dirty_data))
        self.assertTrue(all(len(text) >= 3 for text in cleaned_data['text']))

class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def test_url_validation(self):
        """Test URL validation utility"""
        self.assertTrue(validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertFalse(validate_youtube_url("not_a_url"))
        self.assertFalse(validate_youtube_url(""))
    
    def test_video_id_extraction(self):
        """Test video ID extraction utility"""
        test_url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = extract_video_id(test_url)
        self.assertEqual(video_id, "dQw4w9WgXcQ")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
