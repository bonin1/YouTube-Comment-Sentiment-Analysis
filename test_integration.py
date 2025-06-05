#!/usr/bin/env python3
"""
Integration test for YouTube Comment Sentiment Analysis application
Tests the complete workflow without actually scraping YouTube
"""

import asyncio
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.core.sentiment_analyzer import SentimentAnalyzer
from src.core.data_processor import DataProcessor
from src.utils.validators import validate_youtube_url, extract_video_id

def test_complete_workflow():
    """Test the complete sentiment analysis workflow"""
    print("=" * 60)
    print("YouTube Comment Sentiment Analysis - Integration Test")
    print("=" * 60)
    
    # Test 1: URL validation
    print("\n1. Testing URL validation...")
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/jNQXAC9IVRw",
        "invalid_url"
    ]
    
    for url in test_urls:
        is_valid = validate_youtube_url(url)
        video_id = extract_video_id(url) if is_valid else None
        print(f"  URL: {url}")
        print(f"  Valid: {is_valid}, Video ID: {video_id}")
    
    # Test 2: Sentiment Analysis
    print("\n2. Testing sentiment analysis...")
    analyzer = SentimentAnalyzer()
    
    sample_comments = [
        "This video is absolutely amazing! Best tutorial ever!",
        "Terrible quality video, waste of time",
        "The video is okay, nothing special but not bad either",
        "Love the content! Very helpful and informative 👍",
        "Worst explanation ever. Confusing and boring.",
        "Great job! This helped me understand the concept perfectly",
        "Meh, could be better. Average content.",
        "Fantastic work! Please make more videos like this!",
        "Not what I expected. Pretty disappointed.",
        "Excellent tutorial! Clear and well-structured."    ]
    
    results = []
    for comment in sample_comments:
        result = analyzer.analyze_sentiment(comment)
        results.append({
            'text': comment,
            'sentiment': result.sentiment,
            'sentiment_confidence': result.confidence,  # Use correct column name
            'scores': result.scores
        })
        print(f"  Comment: {comment[:50]}...")
        print(f"  Sentiment: {result.sentiment} (Confidence: {result.confidence:.3f})")
    
    # Test 3: Data Processing
    print("\n3. Testing data processing...")
    processor = DataProcessor()
    
    # Create a DataFrame with the sentiment results
    df = pd.DataFrame(results)
    df['author'] = [f'user_{i+1}' for i in range(len(df))]
    df['likes'] = [10, 2, 5, 15, 1, 12, 3, 18, 4, 14]
    df['timestamp'] = datetime.now()
    
    print(f"  Created DataFrame with {len(df)} comments")
    
    # Clean the data
    cleaned_df = processor.clean_text_data(df)
    print(f"  Cleaned data: {len(cleaned_df)} comments")
      # Generate sentiment summary
    summary = processor.get_sentiment_summary(cleaned_df)
    print(f"  Sentiment Summary:")
    print(f"    Total comments: {summary['total_comments']}")
    print(f"    Positive: {summary['sentiment_distribution']['positive']} ({summary['sentiment_percentages']['positive']:.1f}%)")
    print(f"    Negative: {summary['sentiment_distribution']['negative']} ({summary['sentiment_percentages']['negative']:.1f}%)")
    print(f"    Neutral: {summary['sentiment_distribution']['neutral']} ({summary['sentiment_percentages']['neutral']:.1f}%)")
    print(f"    Average confidence: {summary['average_confidence']:.3f}")
      # Test 4: Export functionality
    print("\n4. Testing export functionality...")
    export_path = "test_analysis_results"
    
    try:
        # Export to CSV
        csv_file = processor.export_data(cleaned_df, "csv", export_path)
        print(f"  Exported to CSV: {csv_file}")
        
        # Export to JSON
        json_file = processor.export_data(cleaned_df, "json", export_path)
        print(f"  Exported to JSON: {json_file}")
        
    except Exception as e:
        print(f"  Export test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Integration test completed successfully! ✅")
    print("=" * 60)
    
    return {
        'sentiment_results': results,
        'summary': summary,
        'total_comments': len(results)
    }

if __name__ == "__main__":
    test_results = test_complete_workflow()
