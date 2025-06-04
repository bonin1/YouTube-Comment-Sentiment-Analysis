"""
Advanced Usage Example - YouTube Comment Sentiment Analysis

This example demonstrates advanced features including:
- Custom configuration
- Multiple export formats
- Rich visualizations
- Detailed analysis
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.sentiment_analyzer import SentimentAnalyzer
from config.logging_config import setup_logging

async def advanced_analysis_example():
    """Example of advanced sentiment analysis with all features."""
    print("🚀 Advanced Sentiment Analysis Example")
    print("=" * 50)
    
    # Setup verbose logging
    setup_logging(level="INFO")
    
    # Example videos for comparison
    videos = [
        {
            "id": "jNQXAC9IVRw",  # Me at the zoo (first YouTube video)
            "name": "First YouTube Video"
        },
        {
            "id": "dQw4w9WgXcQ",  # Rick Roll
            "name": "Rick Astley - Never Gonna Give You Up"
        }
    ]
    
    for video in videos:
        print(f"\n📹 Analyzing: {video['name']} ({video['id']})")
        print("-" * 60)
        
        # Initialize analyzer with advanced settings
        analyzer = SentimentAnalyzer(
            video_id=video['id'],
            max_comments=500,  # More comments for better analysis
            scraping_method="selenium",  # Use Selenium for more reliable scraping
            sort_by="top",  # Get top comments
            output_dir=f"data/outputs/advanced_example/{video['id']}",
            use_cache=True  # Enable caching for faster re-runs
        )
        
        try:
            # Run comprehensive analysis
            print("🔍 Running comprehensive analysis...")
            results = await analyzer.analyze()
            
            if results:
                # Print detailed statistics
                print_detailed_stats(results, video['name'])
                
                # Export in multiple formats
                print("💾 Exporting results in multiple formats...")
                formats = ["csv", "json", "excel"]
                for fmt in formats:
                    export_files = analyzer.export_results(results, format=fmt)
                    print(f"   ✅ {fmt.upper()}: {export_files[0]}")
                
                # Generate comprehensive visualizations
                print("📊 Generating visualizations...")
                viz_files = analyzer.generate_visualizations(
                    results, 
                    interactive=True  # Generate interactive plots
                )
                
                for viz_file in viz_files:
                    print(f"   📈 {Path(viz_file).name}")
                
                # Custom analysis examples
                await perform_custom_analysis(results)
                
            else:
                print("❌ Analysis failed!")
                
        except Exception as e:
            print(f"❌ Error analyzing {video['name']}: {str(e)}")
            continue
    
    print(f"\n🎉 Advanced analysis complete!")
    print(f"📁 Results saved in: data/outputs/advanced_example/")

def print_detailed_stats(results: dict, video_name: str):
    """Print detailed analysis statistics."""
    stats = results['statistics']
    
    print(f"✅ Analysis Complete for {video_name}!")
    print(f"📊 Comments analyzed: {stats['total_comments']:,}")
    print(f"⏱️  Processing time: {stats['processing_time']:.2f}s")
    
    # Sentiment distribution with details
    sentiment_dist = stats['sentiment_distribution']
    print(f"\n💭 Detailed Sentiment Distribution:")
    total = stats['total_comments']
    
    for sentiment, count in sentiment_dist.items():
        percentage = (count / total) * 100
        bar = "█" * int(percentage / 2)  # Simple bar chart
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}[sentiment]
        print(f"   {emoji} {sentiment.title():>8}: {count:>4} ({percentage:>5.1f}%) {bar}")
    
    # Confidence statistics
    conf_stats = stats.get('confidence_statistics', {})
    if conf_stats:
        print(f"\n🎯 Classification Confidence:")
        print(f"   📊 Average: {conf_stats.get('mean', 0):.3f}")
        print(f"   📈 Median:  {conf_stats.get('median', 0):.3f}")
        print(f"   📉 Std Dev: {conf_stats.get('std', 0):.3f}")
        print(f"   🔻 Min:     {conf_stats.get('min', 0):.3f}")
        print(f"   🔺 Max:     {conf_stats.get('max', 0):.3f}")

async def perform_custom_analysis(results: dict):
    """Perform additional custom analysis on results."""
    print(f"\n🔬 Custom Analysis Insights:")
    
    comments = results.get('comments', [])
    if not comments:
        return
    
    # Find most confident predictions
    most_confident = sorted(
        comments, 
        key=lambda x: x.get('confidence', 0), 
        reverse=True
    )[:3]
    
    print(f"   🎯 Most Confident Predictions:")
    for i, comment in enumerate(most_confident, 1):
        sentiment = comment.get('sentiment', 'unknown')
        confidence = comment.get('confidence', 0)
        text = comment.get('processed_text', '')[:50] + "..."
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "❓")
        print(f"      {i}. {emoji} {sentiment.title()} ({confidence:.3f}): {text}")
    
    # Analyze comment lengths by sentiment
    sentiment_lengths = {}
    for comment in comments:
        sentiment = comment.get('sentiment', 'unknown')
        length = len(comment.get('text', ''))
        if sentiment not in sentiment_lengths:
            sentiment_lengths[sentiment] = []
        sentiment_lengths[sentiment].append(length)
    
    print(f"   📏 Average Comment Length by Sentiment:")
    for sentiment, lengths in sentiment_lengths.items():
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "❓")
            print(f"      {emoji} {sentiment.title()}: {avg_length:.1f} characters")
    
    # Find comments with emojis
    emoji_comments = [c for c in comments if any(ord(char) > 127 for char in c.get('text', ''))]
    emoji_percentage = (len(emoji_comments) / len(comments)) * 100
    print(f"   😀 Comments with emojis: {len(emoji_comments)} ({emoji_percentage:.1f}%)")

if __name__ == "__main__":
    asyncio.run(advanced_analysis_example())
