#!/usr/bin/env python3
"""
YouTube Comment Sentiment Analysis - Main Entry Point

This is the main script for running the YouTube Comment Sentiment Analysis tool.
It provides a command-line interface for analyzing sentiment in YouTube video comments.

Usage:
    python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --max-comments 1000
    python main.py --video-id "VIDEO_ID" --output-dir "custom_output"
    python main.py --help

Author: YouTube Comment Sentiment Analysis Project
Date: 2025
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.sentiment_analyzer import SentimentAnalyzer
from config.settings import Settings
from config.logging_config import setup_logging
import logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YouTube Comment Sentiment Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --max-comments 500
  python main.py --video-id "dQw4w9WgXcQ" --method selenium --visualize
  python main.py --url "https://youtu.be/dQw4w9WgXcQ" --output-dir results --export-format excel
        """
    )
    
    # Video specification (required)
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument(
        "--url",
        type=str,
        help="YouTube video URL to analyze"
    )
    video_group.add_argument(
        "--video-id",
        type=str,
        help="YouTube video ID to analyze"
    )
    
    # Scraping options
    parser.add_argument(
        "--max-comments",
        type=int,
        default=1000,
        help="Maximum number of comments to scrape (default: 1000)"
    )
    
    parser.add_argument(
        "--method",
        choices=["requests", "selenium", "auto"],
        default="auto",
        help="Scraping method to use (default: auto)"
    )
    
    parser.add_argument(
        "--sort-by",
        choices=["top", "new"],
        default="top",
        help="Comment sorting method (default: top)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Output directory for results (default: data/outputs)"
    )
    
    parser.add_argument(
        "--export-format",
        choices=["csv", "json", "excel", "all"],
        default="csv",
        help="Export format for results (default: csv)"
    )
    
    # Analysis options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive plots"
    )
    
    # Processing options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def setup_environment(args):
    """Setup environment and logging based on arguments."""
    # Set logging level
    if args.debug:
        log_level = "DEBUG"
    elif args.verbose:
        log_level = "INFO"
    else:
        log_level = "WARNING"
    
    # Setup logging
    setup_logging(level=log_level)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                  YouTube Comment Sentiment Analysis Tool                     ║
║                                                                              ║
║  A comprehensive rule-based sentiment analysis tool for YouTube comments     ║
║  Features: Advanced NLP preprocessing, Rule-based classification,            ║
║           Rich visualizations, Export capabilities                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def main():
    """Main application entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print banner
        print_banner()
        
        # Setup environment
        output_dir = setup_environment(args)
        
        # Get logger
        logger = logging.getLogger(__name__)
        
        # Extract video ID
        if args.url:
            from src.utils import extract_video_id
            video_id = extract_video_id(args.url)
            if not video_id:
                logger.error(f"Could not extract video ID from URL: {args.url}")
                return 1
        else:
            video_id = args.video_id
        
        logger.info(f"Starting analysis for video ID: {video_id}")
        print(f"🎬 Analyzing video: {video_id}")
        
        # Initialize analyzer
        analyzer = SentimentAnalyzer(
            video_id=video_id,
            max_comments=args.max_comments,
            scraping_method=args.method,
            sort_by=args.sort_by,
            output_dir=str(output_dir),
            use_cache=not args.no_cache
        )
        
        # Run analysis
        print("🔍 Starting sentiment analysis...")
        results = await analyzer.analyze()
        
        if not results:
            logger.error("Analysis failed - no results generated")
            print("❌ Analysis failed!")
            return 1
        
        # Print summary
        print_results_summary(results)
        
        # Export results
        print("📊 Exporting results...")
        export_files = analyzer.export_results(
            results, 
            format=args.export_format
        )
        
        for file_path in export_files:
            print(f"✅ Exported: {file_path}")
        
        # Generate visualizations
        if args.visualize:
            print("📈 Generating visualizations...")
            viz_files = analyzer.generate_visualizations(
                results,
                interactive=args.interactive
            )
            
            for file_path in viz_files:
                print(f"✅ Visualization saved: {file_path}")
        
        print(f"\n🎉 Analysis complete! Results saved to: {output_dir}")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        return 1

def print_results_summary(results: dict):
    """Print a summary of analysis results."""
    stats = results.get('statistics', {})
    
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"📝 Total Comments Analyzed: {stats.get('total_comments', 0):,}")
    print(f"⏱️  Processing Time: {stats.get('processing_time', 0):.2f} seconds")
    
    # Sentiment distribution
    sentiment_dist = stats.get('sentiment_distribution', {})
    if sentiment_dist:
        print(f"\n💭 Sentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            percentage = (count / stats.get('total_comments', 1)) * 100
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(sentiment, "📊")
            print(f"   {emoji} {sentiment.title()}: {count:,} ({percentage:.1f}%)")
    
    # Confidence statistics
    confidence_stats = stats.get('confidence_statistics', {})
    if confidence_stats:
        print(f"\n🎯 Confidence Statistics:")
        print(f"   📊 Average: {confidence_stats.get('mean', 0):.3f}")
        print(f"   📈 Median: {confidence_stats.get('median', 0):.3f}")
        print(f"   📉 Std Dev: {confidence_stats.get('std', 0):.3f}")
    
    # Top words
    word_freq = results.get('word_frequency', {})
    if word_freq:
        print(f"\n🔤 Top Words by Sentiment:")
        for sentiment in ['positive', 'negative', 'neutral']:
            words = word_freq.get(sentiment, {})
            if words:
                top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:5]
                word_list = ", ".join([f"{word}({count})" for word, count in top_words])
                emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}[sentiment]
                print(f"   {emoji} {sentiment.title()}: {word_list}")
    
    print("="*60)

if __name__ == "__main__":
    # Run the main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
