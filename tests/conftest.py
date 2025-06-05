"""
Pytest configuration and fixtures for YouTube Comment Sentiment Analysis tests.

This module provides shared fixtures, configuration, and utilities
for the test suite.

Author: GitHub Copilot
Date: 2025
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path for tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.data_manager import DataManager
from src.scrapers.youtube_scraper import CommentScraper
from src.processors.text_processor import TextPreprocessor
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.visualizers.advanced_visualizer import AdvancedVisualizer

# Configure pytest
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


@pytest.fixture(scope="session")
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="function")
def temp_database():
    """Create a temporary database for testing."""
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    
    yield temp_db.name
    
    # Cleanup
    if os.path.exists(temp_db.name):
        os.unlink(temp_db.name)


@pytest.fixture(scope="function")
def data_manager(temp_database):
    """Create a DataManager instance with temporary database."""
    return DataManager(db_path=temp_database)


@pytest.fixture(scope="session")
def youtube_scraper():
    """Create a CommentScraper instance."""
    return CommentScraper()


@pytest.fixture(scope="session")
def text_processor():
    """Create a TextPreprocessor instance."""
    return TextPreprocessor()


@pytest.fixture(scope="session")
def sentiment_analyzer():
    """Create a SentimentAnalyzer instance."""
    return SentimentAnalyzer()


@pytest.fixture(scope="session")
def visualizer():
    """Create an AdvancedVisualizer instance."""
    return AdvancedVisualizer()


@pytest.fixture
def sample_video_urls():
    """Provide sample YouTube video URLs for testing."""
    return [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
    ]


@pytest.fixture
def sample_comments():
    """Provide sample comment data for testing."""
    base_time = datetime.now()
    
    return [
        {
            'text': 'This is an absolutely amazing video! I love everything about it! 😍❤️',
            'author': 'HappyViewer123',
            'timestamp': (base_time - timedelta(hours=2)).isoformat(),
            'likes': 25,
            'cid': 'comment_positive_1',
            'reply_count': 3
        },
        {
            'text': 'This video is terrible and boring. Complete waste of time! 😡',
            'author': 'CriticalUser456',
            'timestamp': (base_time - timedelta(hours=1, minutes=30)).isoformat(),
            'likes': 5,
            'cid': 'comment_negative_1',
            'reply_count': 8
        },
        {
            'text': 'This is okay I guess. Nothing special but not bad either.',
            'author': 'NeutralWatcher789',
            'timestamp': (base_time - timedelta(hours=1)).isoformat(),
            'likes': 12,
            'cid': 'comment_neutral_1',
            'reply_count': 1
        },
        {
            'text': 'AMAZING CONTENT!!! Keep up the great work! 🔥🔥🔥',
            'author': 'EnthusiasticFan',
            'timestamp': (base_time - timedelta(minutes=45)).isoformat(),
            'likes': 18,
            'cid': 'comment_positive_2',
            'reply_count': 5
        },
        {
            'text': 'Not impressed at all. Expected much better quality.',
            'author': 'DisappointedViewer',
            'timestamp': (base_time - timedelta(minutes=30)).isoformat(),
            'likes': 3,
            'cid': 'comment_negative_2',
            'reply_count': 2
        },
        {
            'text': 'Thanks for sharing this information. Very helpful.',
            'author': 'GratefulUser',
            'timestamp': (base_time - timedelta(minutes=15)).isoformat(),
            'likes': 8,
            'cid': 'comment_positive_3',
            'reply_count': 0
        },
        {
            'text': 'The video quality could be better. Audio is fine though.',
            'author': 'TechnicalReviewer',
            'timestamp': (base_time - timedelta(minutes=10)).isoformat(),
            'likes': 4,
            'cid': 'comment_neutral_2',
            'reply_count': 1
        },
        {
            'text': 'Love the editing style! Really well done! 👏✨',
            'author': 'CreativeAppreciator',
            'timestamp': (base_time - timedelta(minutes=5)).isoformat(),
            'likes': 15,
            'cid': 'comment_positive_4',
            'reply_count': 0
        }
    ]


@pytest.fixture
def sample_processed_comments(sample_comments, text_processor):
    """Provide sample processed comments with features."""
    processed = []
    
    for comment in sample_comments:
        result = text_processor.process_text(comment['text'])
        processed.append({
            **comment,
            'processed_text': result['cleaned_text'],
            'features': result['features']
        })
    
    return processed


@pytest.fixture
def sample_sentiment_results(sample_processed_comments, sentiment_analyzer):
    """Provide sample sentiment analysis results."""
    results = []
    
    for comment in sample_processed_comments:
        sentiment = sentiment_analyzer.analyze_sentiment(
            comment['processed_text'], 
            model="vader"
        )
        results.append({
            **comment,
            'sentiment': sentiment
        })
    
    return results


@pytest.fixture
def large_comment_dataset():
    """Provide a large dataset for performance testing."""
    comments = []
    base_time = datetime.now()
    
    sentiment_templates = {
        'positive': [
            "This is amazing! Love it so much! {emoji}",
            "Fantastic work! Keep it up! {emoji}",
            "Brilliant content, thank you for sharing! {emoji}",
            "Outstanding quality and presentation! {emoji}",
            "This made my day! Absolutely wonderful! {emoji}"
        ],
        'negative': [
            "This is terrible and disappointing. {emoji}",
            "Not good at all. Complete waste of time. {emoji}",
            "Very poor quality and boring content. {emoji}",
            "Terrible execution and bad audio quality. {emoji}",
            "Not worth watching. Really disappointed. {emoji}"
        ],
        'neutral': [
            "This is okay, nothing special though.",
            "Average content, could be better.",
            "Not bad but not great either.",
            "It's fine, just what I expected.",
            "Decent video, nothing more to say."
        ]
    }
    
    emojis = {
        'positive': ['😍', '❤️', '🔥', '👏', '✨', '🎉', '😊', '👍'],
        'negative': ['😡', '😒', '👎', '😞', '🙄', '😤', '💔', '😠'],
        'neutral': ['😐', '🤔', '😑', '🤷', '😌']
    }
    
    for i in range(1000):
        # Random sentiment distribution
        if i % 3 == 0:
            sentiment_type = 'positive'
        elif i % 3 == 1:
            sentiment_type = 'negative'
        else:
            sentiment_type = 'neutral'
        
        template = sentiment_templates[sentiment_type][i % len(sentiment_templates[sentiment_type])]
        emoji = emojis[sentiment_type][i % len(emojis[sentiment_type])] if sentiment_type != 'neutral' else ''
        
        text = template.format(emoji=emoji)
        
        comments.append({
            'text': text,
            'author': f'TestUser{i}',
            'timestamp': (base_time - timedelta(minutes=i)).isoformat(),
            'likes': i % 50,
            'cid': f'large_comment_{i}',
            'reply_count': i % 10
        })
    
    return comments


@pytest.fixture
def test_config():
    """Provide test configuration settings."""
    return {
        'test_video_id': 'test_video_123',
        'test_output_dir': 'test_output',
        'max_comments': 100,
        'timeout_seconds': 30,
        'models_to_test': ['vader', 'textblob'],
        'visualization_types': ['sentiment_pie', 'timeline', 'wordcloud', 'treemap']
    }


@pytest.fixture
def mock_youtube_api_response():
    """Provide mock YouTube API response data."""
    return {
        'items': [
            {
                'id': 'test_video_123',
                'snippet': {
                    'title': 'Test Video Title',
                    'description': 'Test video description',
                    'publishedAt': '2024-01-01T12:00:00Z',
                    'channelTitle': 'Test Channel',
                    'statistics': {
                        'viewCount': '10000',
                        'likeCount': '500',
                        'commentCount': '50'
                    }
                }
            }
        ]
    }


@pytest.fixture(scope="session")
def test_assets_dir():
    """Create test assets directory structure."""
    assets_dir = Path(__file__).parent.parent / "assets" / "test"
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample test data files
    sample_data = {
        'comments': [
            {
                'text': 'Sample comment for testing',
                'author': 'TestUser',
                'timestamp': '2024-01-01T12:00:00',
                'likes': 5,
                'cid': 'sample_comment_1'
            }
        ],
        'sentiment_results': [
            {
                'text': 'Sample comment for testing',
                'sentiment': {
                    'label': 'positive',
                    'confidence': 0.75,
                    'scores': {'positive': 0.75, 'neutral': 0.20, 'negative': 0.05}
                }
            }
        ]
    }
    
    # Save test data files
    for filename, data in sample_data.items():
        filepath = assets_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    return assets_dir


class TestHelpers:
    """Helper methods for tests."""
    
    @staticmethod
    def assert_sentiment_result(result):
        """Assert that a sentiment result has the expected structure."""
        assert isinstance(result, dict)
        assert 'label' in result
        assert 'confidence' in result
        assert 'scores' in result
        assert result['label'] in ['positive', 'negative', 'neutral']
        assert 0 <= result['confidence'] <= 1
        assert isinstance(result['scores'], dict)
        assert 'positive' in result['scores']
        assert 'negative' in result['scores']
        assert 'neutral' in result['scores']
    
    @staticmethod
    def assert_processed_text_result(result):
        """Assert that a processed text result has the expected structure."""
        assert isinstance(result, dict)
        assert 'cleaned_text' in result
        assert 'features' in result
        assert isinstance(result['cleaned_text'], str)
        assert isinstance(result['features'], dict)
        
        # Check common features
        expected_features = [
            'word_count', 'char_count', 'sentence_count',
            'exclamation_count', 'question_count', 'caps_ratio',
            'emoji_count', 'url_count', 'mention_count'
        ]
        
        for feature in expected_features:
            assert feature in result['features'], f"Missing feature: {feature}"
            assert isinstance(result['features'][feature], (int, float))
    
    @staticmethod
    def assert_visualization_file(filepath, expected_extension=None):
        """Assert that a visualization file was created correctly."""
        assert os.path.exists(filepath), f"Visualization file not found: {filepath}"
        assert os.path.getsize(filepath) > 0, f"Visualization file is empty: {filepath}"
        
        if expected_extension:
            assert filepath.endswith(expected_extension), f"Wrong file extension: {filepath}"
    
    @staticmethod
    def create_temp_output_dir():
        """Create a temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp(prefix="sentiment_test_")
        return Path(temp_dir)
    
    @staticmethod
    def cleanup_temp_files(*filepaths):
        """Clean up temporary test files."""
        for filepath in filepaths:
            if filepath and os.path.exists(filepath):
                try:
                    if os.path.isfile(filepath):
                        os.unlink(filepath)
                    elif os.path.isdir(filepath):
                        import shutil
                        shutil.rmtree(filepath)
                except Exception as e:
                    print(f"Warning: Could not clean up {filepath}: {e}")


# Make TestHelpers available at module level
test_helpers = TestHelpers()
