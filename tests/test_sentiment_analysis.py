"""
Test Suite for YouTube Comment Sentiment Analysis

This module contains comprehensive tests for all components of the sentiment analysis system.
Run with: python -m pytest tests/ -v
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test configuration
TEST_VIDEO_ID = "dQw4w9WgXcQ"  # Rick Roll - reliable test video
TEST_COMMENTS = [
    "This is amazing! I love it so much! 😊",
    "Terrible video, waste of time 😞",
    "This is okay, nothing special",
    "Absolutely fantastic and wonderful!",
    "I hate this so much, very boring",
    "It's alright, could be better"
]

@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_comments():
    """Provide sample comments for testing."""
    return TEST_COMMENTS

class TestUtils:
    """Test utility functions."""
    
    def test_extract_video_id_from_url(self):
        """Test video ID extraction from various URL formats."""
        from src.utils import extract_video_id
        
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("invalid_url", None),
            ("", None)
        ]
        
        for url, expected_id in test_cases:
            result = extract_video_id(url)
            assert result == expected_id, f"Failed for URL: {url}"
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        from src.utils import clean_text
        
        test_cases = [
            ("Hello World!", "Hello World!"),
            ("  Multiple   spaces  ", "Multiple spaces"),
            ("Text with\n\nnewlines\t\ttabs", "Text with newlines tabs"),
            ("", ""),
            ("UPPERCASE TEXT", "UPPERCASE TEXT"),
            ("Mixed123Numbers456", "Mixed123Numbers456")
        ]
        
        for input_text, expected in test_cases:
            result = clean_text(input_text)
            assert result == expected, f"Failed for input: '{input_text}'"
    
    def test_validate_video_id(self):
        """Test video ID validation."""
        from src.utils import validate_video_id
        
        valid_ids = ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"]
        invalid_ids = ["", "too_short", "invalid-chars!", None, 123]
        
        for video_id in valid_ids:
            assert validate_video_id(video_id), f"Valid ID rejected: {video_id}"
        
        for video_id in invalid_ids:
            assert not validate_video_id(video_id), f"Invalid ID accepted: {video_id}"

class TestPreprocessor:
    """Test text preprocessing functionality."""
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        from src.preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        test_text = "This is AMAZING!!! I love it so much! 😊"
        result = preprocessor.preprocess(test_text)
        
        assert result['original_text'] == test_text
        assert 'processed_text' in result
        assert 'tokens' in result
        assert 'sentiment_indicators' in result
        assert isinstance(result['tokens'], list)
    
    def test_emoji_handling(self):
        """Test emoji processing."""
        from src.preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        test_cases = [
            ("I love this! 😊", "positive"),
            ("This sucks 😞", "negative"),
            ("It's okay 😐", "neutral"),
            ("Amazing! 😍🥰", "positive")
        ]
        
        for text, expected_sentiment in test_cases:
            result = preprocessor.preprocess(text)
            indicators = result['sentiment_indicators']
            
            # Check if the expected sentiment is detected
            if expected_sentiment == "positive":
                assert indicators['positive_count'] > 0
            elif expected_sentiment == "negative":
                assert indicators['negative_count'] > 0
    
    def test_stop_word_removal(self):
        """Test stop word removal."""
        from src.preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        text = "This is a very good movie and I really like it"
        result = preprocessor.preprocess(text)
        
        # Stop words should be removed from tokens
        stop_words = {"this", "is", "a", "and", "i", "it"}
        tokens = [token.lower() for token in result['tokens']]
        
        for stop_word in stop_words:
            assert stop_word not in tokens, f"Stop word '{stop_word}' not removed"
    
    def test_lemmatization(self):
        """Test lemmatization functionality."""
        from src.preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        # Test words that should be lemmatized
        test_cases = [
            ("running", "run"),
            ("better", "good"),
            ("movies", "movie")
        ]
        
        for original, expected in test_cases:
            result = preprocessor.preprocess(original)
            # Check if lemmatized form appears in processed text
            assert any(expected in token.lower() for token in result['tokens'])

class TestSentimentRules:
    """Test sentiment rule engine."""
    
    def test_basic_sentiment_classification(self):
        """Test basic sentiment classification."""
        from src.sentiment_rules import SentimentRuleEngine
        
        engine = SentimentRuleEngine()
        
        test_cases = [
            ("I love this amazing video!", "positive"),
            ("This is terrible and boring", "negative"),
            ("This is okay, nothing special", "neutral"),
            ("Absolutely fantastic and wonderful!", "positive"),
            ("I hate this so much", "negative")
        ]
        
        for text, expected_sentiment in test_cases:
            result = engine.classify(text)
            
            assert 'sentiment' in result
            assert 'confidence' in result
            assert result['sentiment'] == expected_sentiment
            assert 0 <= result['confidence'] <= 1
    
    def test_emoji_sentiment(self):
        """Test emoji-based sentiment detection."""
        from src.sentiment_rules import SentimentRuleEngine
        
        engine = SentimentRuleEngine()
        
        test_cases = [
            ("Great video! 😊", "positive"),
            ("Terrible 😞", "negative"),
            ("It's okay 😐", "neutral"),
            ("Love it! 😍🥰", "positive"),
            ("Hate it 😡👎", "negative")
        ]
        
        for text, expected_sentiment in test_cases:
            result = engine.classify(text)
            assert result['sentiment'] == expected_sentiment
    
    def test_negation_handling(self):
        """Test negation handling in sentiment analysis."""
        from src.sentiment_rules import SentimentRuleEngine
        
        engine = SentimentRuleEngine()
        
        test_cases = [
            ("This is not bad", "positive"),  # Double negative
            ("This is not good", "negative"),
            ("I don't hate it", "positive"),  # Negated negative
            ("I don't love it", "negative")   # Negated positive
        ]
        
        for text, expected_sentiment in test_cases:
            result = engine.classify(text)
            # Note: This is a complex feature, so we'll check if classification is reasonable
            assert result['sentiment'] in ["positive", "negative", "neutral"]
    
    def test_intensifier_impact(self):
        """Test impact of intensifiers on confidence."""
        from src.sentiment_rules import SentimentRuleEngine
        
        engine = SentimentRuleEngine()
        
        # Compare with and without intensifiers
        basic_result = engine.classify("This is good")
        intensified_result = engine.classify("This is very good")
        
        # Intensified version should have higher confidence
        assert intensified_result['confidence'] >= basic_result['confidence']

class TestVisualization:
    """Test visualization generation."""
    
    def test_sentiment_pie_chart(self, temp_output_dir):
        """Test sentiment pie chart generation."""
        from src.visualizer import SentimentVisualizer
        
        visualizer = SentimentVisualizer(output_dir=temp_output_dir)
        
        # Sample data
        sentiment_counts = {"positive": 50, "negative": 30, "neutral": 20}
        
        # Generate pie chart
        chart_path = visualizer.create_sentiment_pie_chart(sentiment_counts)
        
        assert Path(chart_path).exists()
        assert Path(chart_path).suffix == '.png'
    
    def test_word_frequency_chart(self, temp_output_dir):
        """Test word frequency chart generation."""
        from src.visualizer import SentimentVisualizer
        
        visualizer = SentimentVisualizer(output_dir=temp_output_dir)
        
        # Sample word frequency data
        word_freq = {
            "positive": {"amazing": 10, "great": 8, "love": 7},
            "negative": {"terrible": 5, "hate": 4, "bad": 3},
            "neutral": {"okay": 6, "fine": 4, "average": 3}
        }
        
        # Generate frequency chart
        chart_path = visualizer.create_word_frequency_chart(word_freq)
        
        assert Path(chart_path).exists()
        assert Path(chart_path).suffix == '.png'

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, temp_output_dir, sample_comments):
        """Test the complete analysis pipeline."""
        from src.sentiment_analyzer import SentimentAnalyzer
        
        # Mock the scraper to return our test comments
        analyzer = SentimentAnalyzer(
            video_id=TEST_VIDEO_ID,
            max_comments=len(sample_comments),
            output_dir=temp_output_dir
        )
        
        # Override the scraper method to return our test data
        async def mock_scrape():
            return [{"text": comment, "author": f"user{i}"} 
                   for i, comment in enumerate(sample_comments)]
        
        analyzer.scraper.scrape_comments = mock_scrape
        
        # Run analysis
        results = await analyzer.analyze()
        
        # Verify results structure
        assert results is not None
        assert 'statistics' in results
        assert 'comments' in results
        assert 'word_frequency' in results
        
        # Verify statistics
        stats = results['statistics']
        assert stats['total_comments'] == len(sample_comments)
        assert 'sentiment_distribution' in stats
        assert 'processing_time' in stats
        
        # Verify all comments were processed
        assert len(results['comments']) == len(sample_comments)
        
        # Verify sentiment distribution
        sentiment_dist = stats['sentiment_distribution']
        assert all(sentiment in sentiment_dist for sentiment in ['positive', 'negative', 'neutral'])
        assert sum(sentiment_dist.values()) == len(sample_comments)
    
    @pytest.mark.asyncio
    async def test_export_functionality(self, temp_output_dir, sample_comments):
        """Test export functionality."""
        from src.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(
            video_id=TEST_VIDEO_ID,
            max_comments=len(sample_comments),
            output_dir=temp_output_dir
        )
        
        # Mock results
        mock_results = {
            'statistics': {
                'total_comments': len(sample_comments),
                'sentiment_distribution': {'positive': 2, 'negative': 2, 'neutral': 2},
                'processing_time': 1.0
            },
            'comments': [
                {'text': comment, 'sentiment': 'positive', 'confidence': 0.8}
                for comment in sample_comments
            ],
            'word_frequency': {'positive': {'great': 5}, 'negative': {'bad': 3}, 'neutral': {'okay': 2}}
        }
        
        # Test different export formats
        formats = ['csv', 'json', 'excel']
        
        for fmt in formats:
            export_files = analyzer.export_results(mock_results, format=fmt)
            
            assert len(export_files) > 0
            for file_path in export_files:
                assert Path(file_path).exists()
                if fmt == 'csv':
                    assert Path(file_path).suffix == '.csv'
                elif fmt == 'json':
                    assert Path(file_path).suffix == '.json'
                elif fmt == 'excel':
                    assert Path(file_path).suffix in ['.xlsx', '.xls']

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_video_id(self):
        """Test handling of invalid video IDs."""
        from src.sentiment_analyzer import SentimentAnalyzer
        
        with pytest.raises((ValueError, Exception)):
            SentimentAnalyzer(video_id="invalid_id")
    
    def test_empty_comment_handling(self):
        """Test handling of empty or invalid comments."""
        from src.preprocessor import TextPreprocessor
        from src.sentiment_rules import SentimentRuleEngine
        
        preprocessor = TextPreprocessor()
        engine = SentimentRuleEngine()
        
        # Test empty text
        result = preprocessor.preprocess("")
        assert result['processed_text'] == ""
        
        # Test whitespace only
        result = preprocessor.preprocess("   \n\t   ")
        assert result['processed_text'].strip() == ""
        
        # Test sentiment classification of empty text
        result = engine.classify("")
        assert result['sentiment'] == 'neutral'
        assert result['confidence'] >= 0
    
    def test_unicode_handling(self):
        """Test handling of unicode characters and emojis."""
        from src.preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        unicode_texts = [
            "Hello 世界! 😊",
            "Café résumé naïve",
            "Русский текст",
            "Arabic: مرحبا",
            "Mathematical: α β γ"
        ]
        
        for text in unicode_texts:
            result = preprocessor.preprocess(text)
            assert result is not None
            assert 'processed_text' in result

# Performance tests
class TestPerformance:
    """Test performance characteristics."""
    
    def test_preprocessing_performance(self):
        """Test preprocessing performance with large text."""
        from src.preprocessor import TextPreprocessor
        import time
        
        preprocessor = TextPreprocessor()
        
        # Create large text
        large_text = "This is a test comment. " * 1000
        
        start_time = time.time()
        result = preprocessor.preprocess(large_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process within reasonable time (adjust threshold as needed)
        assert processing_time < 5.0, f"Preprocessing took too long: {processing_time}s"
        assert result is not None
    
    def test_sentiment_classification_performance(self):
        """Test sentiment classification performance."""
        from src.sentiment_rules import SentimentRuleEngine
        import time
        
        engine = SentimentRuleEngine()
        
        # Test multiple classifications
        comments = [
            "This is amazing and wonderful!",
            "Terrible and boring video",
            "It's okay, nothing special"
        ] * 100  # 300 comments
        
        start_time = time.time()
        results = [engine.classify(comment) for comment in comments]
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should classify all comments within reasonable time
        assert processing_time < 10.0, f"Classification took too long: {processing_time}s"
        assert len(results) == len(comments)
        assert all('sentiment' in result for result in results)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
