"""
Basic Usage Example - YouTube Comment Sentiment Analysis

This example demonstrates the basic usage of the sentiment analysis tool
for analyzing YouTube comments with default settings.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.sentiment_analyzer import SentimentAnalyzer
from config.logging_config import setup_logging

async def basic_analysis_example():
    """Example of basic sentiment analysis."""
    print("🚀 Basic Sentiment Analysis Example")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Example video ID (Rick Astley - Never Gonna Give You Up)
    video_id = "dQw4w9WgXcQ"
    
    print(f"📹 Analyzing video: {video_id}")
    
    # Initialize analyzer with basic settings
    analyzer = SentimentAnalyzer(
        video_id=video_id,
        max_comments=100,  # Analyze 100 comments for quick demo
        scraping_method="auto",
        output_dir="data/outputs/basic_example"
    )
    
    try:
        # Run analysis
        print("🔍 Starting analysis...")
        results = await analyzer.analyze()
        
        if results:
            # Print basic statistics
            stats = results['statistics']
            print(f"\n✅ Analysis Complete!")
            print(f"📊 Comments analyzed: {stats['total_comments']}")
            print(f"⏱️  Processing time: {stats['processing_time']:.2f}s")
            
            # Print sentiment distribution
            sentiment_dist = stats['sentiment_distribution']
            print(f"\n💭 Sentiment Distribution:")
            for sentiment, count in sentiment_dist.items():
                percentage = (count / stats['total_comments']) * 100
                print(f"   {sentiment.title()}: {count} ({percentage:.1f}%)")
            
            # Export results
            export_files = analyzer.export_results(results, format="csv")
            print(f"\n📄 Results exported to: {export_files[0]}")
            
        else:
            print("❌ Analysis failed!")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(basic_analysis_example())
