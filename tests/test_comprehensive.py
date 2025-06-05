"""
Comprehensive test suite for YouTube Comment Sentiment Analysis system.

This module contains unit tests, integration tests, and end-to-end tests
for all components of the sentiment analysis system.

Author: GitHub Copilot
Date: 2025
"""

import pytest
import asyncio
import tempfile
import sqlite3
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import Settings
from src.scrapers.youtube_scraper import CommentScraper
from src.processors.text_processor import TextPreprocessor
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.visualizers.advanced_visualizer import AdvancedVisualizer
from src.utils.data_manager import DataManager
from src.utils.helpers import ProgressTracker, validate_url


class TestYouTubeScraper:
    """Test cases for YouTube comment scraper."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scraper = CommentScraper()
    
    def test_extract_video_id_standard_url(self):
        """Test video ID extraction from standard YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = self.scraper.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_video_id_short_url(self):
        """Test video ID extraction from short YouTube URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        video_id = self.scraper.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_video_id_embed_url(self):
        """Test video ID extraction from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        video_id = self.scraper.extract_video_id(url)
        assert video_id == "dQw4w9WgXcQ"
    
    def test_extract_video_id_invalid_url(self):
        """Test video ID extraction from invalid URL."""
        url = "https://example.com/invalid"
        video_id = self.scraper.extract_video_id(url)
        assert video_id is None
    
    @patch('youtube_comment_downloader.YoutubeCommentDownloader')
    async def test_scrape_comments_success(self, mock_downloader):
        """Test successful comment scraping."""
        # Mock comment data
        mock_comments = [
            {
                'text': 'This is a great video!',
                'author': 'TestUser1',
                'time': '2 hours ago',
                'likes': 5,
                'cid': 'comment1'
            },
            {
                'text': 'Not so good...',
                'author': 'TestUser2', 
                'time': '1 hour ago',
                'likes': 2,
                'cid': 'comment2'
            }
        ]
        
        mock_downloader.return_value.get_comments.return_value = mock_comments
        
        comments = await self.scraper.scrape_comments(
            "https://www.youtube.com/watch?v=test123",
            limit=2
        )
        
        assert len(comments) == 2
        assert comments[0]['text'] == 'This is a great video!'
        assert comments[1]['text'] == 'Not so good...'
    
    @patch('youtube_comment_downloader.YoutubeCommentDownloader')
    async def test_scrape_comments_with_progress(self, mock_downloader):
        """Test comment scraping with progress callback."""
        mock_comments = [{'text': f'Comment {i}', 'cid': f'c{i}'} for i in range(5)]
        mock_downloader.return_value.get_comments.return_value = mock_comments
        
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        comments = await self.scraper.scrape_comments(
            "https://www.youtube.com/watch?v=test123",
            limit=5,
            progress_callback=progress_callback
        )
        
        assert len(comments) == 5
        assert len(progress_calls) > 0


class TestTextProcessor:
    """Test cases for text processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = TextPreprocessor()
    
    def test_basic_text_cleaning(self):
        """Test basic text cleaning functionality."""
        text = "This is a GREAT video!!! 😊😊😊 Check it out at https://example.com"
        result = self.processor.process_text(text)
        
        assert 'cleaned_text' in result
        assert 'features' in result
        assert len(result['cleaned_text']) > 0
    
    def test_emoji_handling(self):
        """Test emoji processing."""
        text = "I love this! 😍❤️🔥"
        result = self.processor.process_text(text)
        
        # Should extract emoji features
        assert 'emoji_count' in result['features']
        assert result['features']['emoji_count'] > 0
    
    def test_url_removal(self):
        """Test URL removal from text."""
        text = "Check this out: https://example.com and also http://test.org"
        result = self.processor.process_text(text)
        
        # URLs should be removed or replaced
        assert 'https://example.com' not in result['cleaned_text']
        assert 'http://test.org' not in result['cleaned_text']
    
    def test_feature_extraction(self):
        """Test feature extraction from text."""
        text = "This is an AMAZING video with great quality!"
        result = self.processor.process_text(text)
        
        features = result['features']
        
        # Check expected features
        assert 'word_count' in features
        assert 'char_count' in features
        assert 'exclamation_count' in features
        assert 'caps_ratio' in features
        assert features['word_count'] > 0
        assert features['char_count'] > 0
    
    def test_empty_text(self):
        """Test processing of empty text."""
        result = self.processor.process_text("")
        
        assert result['cleaned_text'] == ""
        assert result['features']['word_count'] == 0
        assert result['features']['char_count'] == 0


class TestSentimentAnalyzer:
    """Test cases for sentiment analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
    
    def test_vader_positive_sentiment(self):
        """Test VADER analyzer with positive text."""
        text = "I absolutely love this video! It's amazing and wonderful!"
        result = self.analyzer.analyze_sentiment(text, model="vader")
        
        assert 'label' in result
        assert 'confidence' in result
        assert 'scores' in result
        assert result['label'] in ['positive', 'negative', 'neutral']
        assert 0 <= result['confidence'] <= 1
    
    def test_vader_negative_sentiment(self):
        """Test VADER analyzer with negative text."""
        text = "This is terrible and awful. I hate it completely!"
        result = self.analyzer.analyze_sentiment(text, model="vader")
        
        assert result['label'] in ['positive', 'negative', 'neutral']
        assert 0 <= result['confidence'] <= 1
    
    def test_textblob_analyzer(self):
        """Test TextBlob analyzer."""
        text = "This is a good video with nice content."
        result = self.analyzer.analyze_sentiment(text, model="textblob")
        
        assert 'label' in result
        assert 'confidence' in result
        assert 'scores' in result
        assert result['label'] in ['positive', 'negative', 'neutral']
    
    def test_ensemble_analyzer(self):
        """Test ensemble analyzer."""
        text = "This is a great video! Really enjoyed it."
        result = self.analyzer.analyze_sentiment(text, model="ensemble")
        
        assert 'label' in result
        assert 'confidence' in result
        assert 'scores' in result
        assert 'individual_predictions' in result
        assert len(result['individual_predictions']) > 1
    
    def test_batch_analysis(self):
        """Test batch sentiment analysis."""
        texts = [
            "This is amazing!",
            "This is terrible.",
            "This is okay, I guess."
        ]
        
        results = self.analyzer.analyze_batch(texts, model="vader")
        
        assert len(results) == 3
        for result in results:
            assert 'label' in result
            assert 'confidence' in result
    
    def test_empty_text_analysis(self):
        """Test analysis of empty text."""
        result = self.analyzer.analyze_sentiment("", model="vader")
        
        assert result['label'] == 'neutral'
        assert result['confidence'] >= 0


class TestDataManager:
    """Test cases for data management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Initialize data manager with test database
        self.data_manager = DataManager(db_path=self.temp_db.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_save_and_load_comments(self):
        """Test saving and loading comments."""
        video_id = "test123"
        comments = [
            {
                'text': 'Great video!',
                'author': 'TestUser',
                'timestamp': datetime.now().isoformat(),
                'likes': 5,
                'cid': 'comment1'
            },
            {
                'text': 'Not bad',
                'author': 'TestUser2',
                'timestamp': datetime.now().isoformat(),
                'likes': 2,
                'cid': 'comment2'
            }
        ]
        
        # Save comments
        saved_count = self.data_manager.save_comments(video_id, comments)
        assert saved_count == 2
        
        # Load comments
        loaded_comments = self.data_manager.load_comments(video_id)
        assert len(loaded_comments) == 2
        assert loaded_comments[0]['text'] == 'Great video!'
    
    def test_save_and_load_sentiment_results(self):
        """Test saving and loading sentiment results."""
        video_id = "test123"
        results = [
            {
                'text': 'Great video!',
                'sentiment': {
                    'label': 'positive',
                    'confidence': 0.9,
                    'scores': {'positive': 0.9, 'neutral': 0.1, 'negative': 0.0}
                },
                'cid': 'comment1'
            }
        ]
        
        # Save results
        self.data_manager.save_sentiment_results(video_id, results)
        
        # Load results
        loaded_results = self.data_manager.load_sentiment_results(video_id)
        assert len(loaded_results) == 1
        assert loaded_results[0]['sentiment']['label'] == 'positive'
    
    def test_export_comments_json(self):
        """Test exporting comments to JSON."""
        comments = [
            {'text': 'Test comment', 'author': 'TestUser', 'cid': 'c1'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            self.data_manager.export_comments(comments, export_path)
            
            # Verify export
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert len(exported_data) == 1
            assert exported_data[0]['text'] == 'Test comment'
        
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_export_comments_csv(self):
        """Test exporting comments to CSV."""
        comments = [
            {'text': 'Test comment', 'author': 'TestUser', 'likes': 5, 'cid': 'c1'}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            export_path = f.name
        
        try:
            self.data_manager.export_comments(comments, export_path)
            
            # Verify export
            with open(export_path, 'r') as f:
                content = f.read()
            
            assert 'Test comment' in content
            assert 'TestUser' in content
        
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestAdvancedVisualizer:
    """Test cases for advanced visualizations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = AdvancedVisualizer()
        self.sample_results = [
            {
                'text': 'I love this video!',
                'sentiment': {
                    'label': 'positive',
                    'confidence': 0.9,
                    'scores': {'positive': 0.9, 'neutral': 0.1, 'negative': 0.0}
                },
                'timestamp': datetime.now().isoformat()
            },
            {
                'text': 'This is okay',
                'sentiment': {
                    'label': 'neutral',
                    'confidence': 0.7,
                    'scores': {'positive': 0.2, 'neutral': 0.7, 'negative': 0.1}
                },
                'timestamp': datetime.now().isoformat()
            },
            {
                'text': 'Not good at all',
                'sentiment': {
                    'label': 'negative',
                    'confidence': 0.8,
                    'scores': {'positive': 0.1, 'neutral': 0.1, 'negative': 0.8}
                },
                'timestamp': datetime.now().isoformat()
            }
        ]
    
    def test_create_sentiment_pie_chart(self):
        """Test sentiment pie chart creation."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            result_path = self.visualizer.create_sentiment_pie_chart(
                self.sample_results, output_path
            )
            
            assert result_path == output_path
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_create_sentiment_timeline(self):
        """Test sentiment timeline creation."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            result_path = self.visualizer.create_sentiment_timeline(
                self.sample_results, output_path
            )
            
            assert result_path == output_path
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_create_sentiment_treemap(self):
        """Test sentiment treemap creation."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            result_path = self.visualizer.create_sentiment_treemap(
                self.sample_results, output_path
            )
            
            assert result_path == output_path
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            result_path = self.visualizer.create_interactive_dashboard(
                self.sample_results, output_path
            )
            
            assert result_path == output_path
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Verify HTML content
            with open(output_path, 'r') as f:
                content = f.read()
            assert '<html>' in content
            assert 'Sentiment Analysis Dashboard' in content
        
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestHelpers:
    """Test cases for utility functions."""
    
    def test_validate_youtube_url_valid(self):
        """Test validation of valid YouTube URLs."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ"        ]
        for url in valid_urls:
            assert validate_url(url), f"URL should be valid: {url}"
    
    def test_validate_youtube_url_invalid(self):
        """Test validation of invalid YouTube URLs."""
        invalid_urls = [
            "https://example.com/video",
            "https://vimeo.com/123456",
            "not_a_url",
            "",
            "https://youtube.com/invalid"        ]
        for url in invalid_urls:
            assert not validate_url(url), f"URL should be invalid: {url}"
    
    def test_extract_video_id_helper(self):
        """Test video ID extraction helper function."""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("invalid_url", None)
        ]
        
        for url, expected in test_cases:
            result = extract_video_id(url)
            assert result == expected, f"Expected {expected}, got {result} for URL: {url}"
    
    def test_progress_tracker(self):
        """Test progress tracker functionality."""
        tracker = ProgressTracker(total=10, description="Test progress")
        
        # Test initial state
        assert tracker.current == 0
        assert tracker.total == 10
        assert tracker.description == "Test progress"
        
        # Test updates
        tracker.update(5)
        assert tracker.current == 5
        
        tracker.update()  # Should increment by 1
        assert tracker.current == 6
        
        # Test completion
        tracker.update(10)
        assert tracker.current == 10
        assert tracker.is_complete()


class TestIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)  
        self.scraper = CommentScraper()
        self.processor = TextPreprocessor()
        self.analyzer = SentimentAnalyzer()
        self.visualizer = AdvancedVisualizer()
        self.data_manager = DataManager(db_path=self.temp_db.name)
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_complete_pipeline_mock(self):
        """Test complete analysis pipeline with mocked data."""
        # Mock comments data
        video_id = "test123"
        mock_comments = [
            {
                'text': 'This is an amazing video! I love it so much!',
                'author': 'HappyViewer',
                'timestamp': datetime.now().isoformat(),
                'likes': 15,
                'cid': 'comment1'
            },
            {
                'text': 'Not really impressed with this content.',
                'author': 'CriticalViewer',
                'timestamp': datetime.now().isoformat(),
                'likes': 3,
                'cid': 'comment2'
            },
            {
                'text': 'This is okay, nothing special.',
                'author': 'NeutralViewer',
                'timestamp': datetime.now().isoformat(),
                'likes': 1,
                'cid': 'comment3'
            }
        ]
        
        # Step 1: Save comments
        saved_count = self.data_manager.save_comments(video_id, mock_comments)
        assert saved_count == 3
        
        # Step 2: Load and process comments
        loaded_comments = self.data_manager.load_comments(video_id)
        processed_comments = []
        
        for comment in loaded_comments:
            processed = self.processor.process_text(comment['text'])
            processed_comments.append({
                **comment,
                'processed_text': processed['cleaned_text'],
                'features': processed['features']
            })
        
        assert len(processed_comments) == 3
        assert all('processed_text' in c for c in processed_comments)
        
        # Step 3: Analyze sentiment
        sentiment_results = []
        for comment in processed_comments:
            sentiment = self.analyzer.analyze_sentiment(
                comment['processed_text'], 
                model="vader"
            )
            sentiment_results.append({
                **comment,
                'sentiment': sentiment
            })
        
        assert len(sentiment_results) == 3
        assert all('sentiment' in r for r in sentiment_results)
        
        # Step 4: Save sentiment results
        self.data_manager.save_sentiment_results(video_id, sentiment_results)
        
        # Step 5: Load results and verify
        loaded_results = self.data_manager.load_sentiment_results(video_id)
        assert len(loaded_results) == 3
        
        # Step 6: Generate visualizations
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test pie chart
            pie_path = os.path.join(temp_dir, 'pie.png')
            result = self.visualizer.create_sentiment_pie_chart(loaded_results, pie_path)
            assert result == pie_path
            assert os.path.exists(pie_path)
            
            # Test treemap
            treemap_path = os.path.join(temp_dir, 'treemap.png')
            result = self.visualizer.create_sentiment_treemap(loaded_results, treemap_path)
            assert result == treemap_path
            assert os.path.exists(treemap_path)
            
            # Test dashboard
            dashboard_path = os.path.join(temp_dir, 'dashboard.html')
            result = self.visualizer.create_interactive_dashboard(loaded_results, dashboard_path)
            assert result == dashboard_path
            assert os.path.exists(dashboard_path)
    
    def test_error_handling(self):
        """Test error handling throughout the pipeline."""
        # Test with invalid video ID
        comments = self.data_manager.load_comments("nonexistent_video")
        assert comments == []
        
        # Test with empty text processing
        result = self.processor.process_text("")
        assert result['cleaned_text'] == ""
        assert result['features']['word_count'] == 0
        
        # Test sentiment analysis with empty text
        sentiment = self.analyzer.analyze_sentiment("", model="vader")
        assert sentiment['label'] == 'neutral'
        
        # Test visualization with empty data
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            # Should handle empty data gracefully
            result = self.visualizer.create_sentiment_pie_chart([], output_path)
            # Should either return None or create a default chart
            assert result is None or os.path.exists(output_path)
        
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


# Test fixtures and utilities
@pytest.fixture
def sample_comments():
    """Fixture providing sample comment data."""
    return [
        {
            'text': 'This is an amazing video! Great work!',
            'author': 'TestUser1',
            'timestamp': '2024-01-01T12:00:00',
            'likes': 10,
            'cid': 'comment1'
        },
        {
            'text': 'Not impressed with this content.',
            'author': 'TestUser2',
            'timestamp': '2024-01-01T12:05:00',
            'likes': 2,
            'cid': 'comment2'
        },
        {
            'text': 'This is okay, nothing special really.',
            'author': 'TestUser3',
            'timestamp': '2024-01-01T12:10:00',
            'likes': 5,
            'cid': 'comment3'
        }
    ]


@pytest.fixture
def sample_sentiment_results():
    """Fixture providing sample sentiment analysis results."""
    return [
        {
            'text': 'This is amazing!',
            'sentiment': {
                'label': 'positive',
                'confidence': 0.9,
                'scores': {'positive': 0.9, 'neutral': 0.1, 'negative': 0.0}
            },
            'timestamp': '2024-01-01T12:00:00',
            'cid': 'comment1'
        },
        {
            'text': 'This is terrible.',
            'sentiment': {
                'label': 'negative',
                'confidence': 0.8,
                'scores': {'positive': 0.1, 'neutral': 0.1, 'negative': 0.8}
            },
            'timestamp': '2024-01-01T12:05:00',
            'cid': 'comment2'
        }
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
