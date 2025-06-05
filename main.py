#!/usr/bin/env python3
"""
YouTube Comment Sentiment Analysis - Command Line Interface

This module provides a command-line interface for the YouTube Comment Sentiment Analysis system.
It supports various operations including comment scraping, sentiment analysis, visualization,
and data export with comprehensive logging and error handling.

Author: GitHub Copilot
Date: 2025
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import Settings, get_settings
from config.logging_config import setup_logging
from src.scrapers.youtube_scraper import CommentScraper
from src.processors.text_processor import TextPreprocessor
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.visualizers.advanced_visualizer import AdvancedVisualizer
from src.utils.data_manager import DataManager
from src.utils.helpers import ProgressTracker, validate_url
import logging

# Initialize logger
logger = logging.getLogger(__name__)


class YouTubeCommentCLI:
    """Command-line interface for YouTube Comment Sentiment Analysis."""
    
    def __init__(self):
        """Initialize CLI with all components."""
        self.settings = get_settings()
        setup_logging()
          # Initialize components
        self.scraper = CommentScraper()
        self.processor = TextPreprocessor()
        self.analyzer = SentimentAnalyzer()
        self.visualizer = AdvancedVisualizer()
        self.data_manager = DataManager()
        
    async def scrape_comments(self, video_url: str, limit: int = 100, 
                            output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Scrape comments from a YouTube video.
        
        Args:
            video_url: YouTube video URL
            limit: Maximum comments to scrape
            output_file: Optional output file path
            
        Returns:
            Dictionary containing scraping results
        """
        logger.info(f"Starting comment scraping for: {video_url}")
        
        try:            # Validate URL
            if not validate_url(video_url):
                raise ValueError("Invalid YouTube URL format")
              # Create progress tracker
            progress = ProgressTracker(total=limit)
            
            # Scrape comments
            comments = await self.scraper.scrape_comments(
                video_url=video_url,
                limit=limit,
                progress_callback=progress.update
            )
            
            if not comments:
                logger.warning("No comments found")
                return {"success": False, "message": "No comments found"}
            
            # Save to database
            video_id = self.scraper.extract_video_id(video_url)
            saved_count = self.data_manager.save_comments(video_id, comments)
            
            # Export if requested
            if output_file:
                self.data_manager.export_comments(comments, output_file)
                logger.info(f"Comments exported to: {output_file}")
            
            result = {
                "success": True,
                "video_id": video_id,
                "comments_scraped": len(comments),
                "comments_saved": saved_count,
                "output_file": output_file
            }
            
            print(f"\n✅ Successfully scraped {len(comments)} comments")
            if output_file:
                print(f"📁 Exported to: {output_file}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error scraping comments: {e}")
            print(f"❌ Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def analyze_sentiment(self, video_id: str, model: str = "ensemble",
                              output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze sentiment of scraped comments.
        
        Args:
            video_id: YouTube video ID
            model: Sentiment analysis model to use
            output_dir: Optional output directory for results
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting sentiment analysis for video: {video_id}")
        
        try:
            # Load comments from database
            comments = self.data_manager.load_comments(video_id)
            if not comments:
                raise ValueError(f"No comments found for video ID: {video_id}")
            
            print(f"📊 Analyzing sentiment for {len(comments)} comments...")
            
            # Process comments
            processed_comments = []
            progress = ProgressTracker(total=len(comments))
            
            for comment in comments:
                processed = self.processor.process_text(comment['text'])
                processed_comments.append({
                    **comment,
                    'processed_text': processed['cleaned_text'],
                    'features': processed['features']
                })
                progress.update()
            
            # Analyze sentiment
            results = []
            progress = ProgressTracker(total=len(processed_comments))
            
            for comment in processed_comments:
                sentiment = self.analyzer.analyze_sentiment(
                    comment['processed_text'],
                    model=model
                )
                results.append({
                    **comment,
                    'sentiment': sentiment
                })
                progress.update()
            
            # Save results
            self.data_manager.save_sentiment_results(video_id, results)
            
            # Generate summary statistics
            summary = self._generate_summary_stats(results)
            
            # Export results if requested
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Export detailed results
                self.data_manager.export_sentiment_results(
                    results, 
                    str(output_path / f"{video_id}_sentiment_results.json")
                )
                
                # Export summary
                with open(output_path / f"{video_id}_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"📁 Results exported to: {output_dir}")
            
            result = {
                "success": True,
                "video_id": video_id,
                "comments_analyzed": len(results),
                "summary": summary,
                "output_dir": output_dir
            }
            
            # Print summary
            print(f"\n✅ Analysis complete!")
            print(f"📈 Sentiment Distribution:")
            print(f"   Positive: {summary['sentiment_distribution']['positive']:.1%}")
            print(f"   Neutral:  {summary['sentiment_distribution']['neutral']:.1%}")
            print(f"   Negative: {summary['sentiment_distribution']['negative']:.1%}")
            print(f"📊 Average Confidence: {summary['average_confidence']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            print(f"❌ Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_visualizations(self, video_id: str, output_dir: str,
                                    viz_types: list = None) -> Dict[str, Any]:
        """
        Generate visualizations for sentiment analysis results.
        
        Args:
            video_id: YouTube video ID
            output_dir: Output directory for visualizations
            viz_types: List of visualization types to generate
            
        Returns:
            Dictionary containing generation results
        """
        logger.info(f"Generating visualizations for video: {video_id}")
        
        try:
            # Load sentiment results
            results = self.data_manager.load_sentiment_results(video_id)
            if not results:
                raise ValueError(f"No sentiment results found for video ID: {video_id}")
            
            if viz_types is None:
                viz_types = ["sentiment_pie", "timeline", "wordcloud", "treemap", "dashboard"]
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            generated_files = []
            
            print(f"🎨 Generating {len(viz_types)} visualization types...")
            
            for viz_type in viz_types:
                try:
                    if viz_type == "sentiment_pie":
                        file_path = self.visualizer.create_sentiment_pie_chart(
                            results, str(output_path / f"{video_id}_sentiment_pie.png")
                        )
                    elif viz_type == "timeline":
                        file_path = self.visualizer.create_sentiment_timeline(
                            results, str(output_path / f"{video_id}_timeline.png")
                        )
                    elif viz_type == "wordcloud":
                        file_path = self.visualizer.create_sentiment_wordclouds(
                            results, str(output_path / f"{video_id}_wordclouds.png")
                        )
                    elif viz_type == "treemap":
                        file_path = self.visualizer.create_sentiment_treemap(
                            results, str(output_path / f"{video_id}_treemap.png")
                        )
                    elif viz_type == "dashboard":
                        file_path = self.visualizer.create_interactive_dashboard(
                            results, str(output_path / f"{video_id}_dashboard.html")
                        )
                    else:
                        logger.warning(f"Unknown visualization type: {viz_type}")
                        continue
                    
                    if file_path:
                        generated_files.append(file_path)
                        print(f"   ✅ {viz_type}: {Path(file_path).name}")
                    
                except Exception as e:
                    logger.error(f"Error generating {viz_type}: {e}")
                    print(f"   ❌ {viz_type}: {e}")
            
            result = {
                "success": True,
                "video_id": video_id,
                "generated_files": generated_files,
                "output_dir": str(output_path)
            }
            
            print(f"\n🎨 Generated {len(generated_files)} visualizations")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            print(f"❌ Error: {e}")
            return {"success": False, "error": str(e)}
    
    async def full_analysis(self, video_url: str, limit: int = 500,
                          model: str = "ensemble", output_dir: str = "output",
                          generate_viz: bool = True) -> Dict[str, Any]:
        """
        Perform complete analysis pipeline: scrape, analyze, visualize.
        
        Args:
            video_url: YouTube video URL
            limit: Maximum comments to scrape
            model: Sentiment analysis model
            output_dir: Output directory for all results
            generate_viz: Whether to generate visualizations
            
        Returns:
            Dictionary containing complete analysis results
        """
        logger.info(f"Starting full analysis pipeline for: {video_url}")
        
        try:
            print("🚀 Starting YouTube Comment Sentiment Analysis Pipeline")
            print(f"📺 Video URL: {video_url}")
            print(f"💬 Comment limit: {limit}")
            print(f"🧠 Model: {model}")
            print(f"📁 Output directory: {output_dir}")
            print("-" * 60)
            
            # Step 1: Scrape comments
            print("\n1️⃣ Scraping comments...")
            scrape_result = await self.scrape_comments(video_url, limit)
            if not scrape_result["success"]:
                return scrape_result
            
            video_id = scrape_result["video_id"]
            
            # Step 2: Analyze sentiment
            print("\n2️⃣ Analyzing sentiment...")
            analysis_result = await self.analyze_sentiment(video_id, model, output_dir)
            if not analysis_result["success"]:
                return analysis_result
            
            # Step 3: Generate visualizations
            viz_result = None
            if generate_viz:
                print("\n3️⃣ Generating visualizations...")
                viz_result = await self.generate_visualizations(video_id, output_dir)
            
            # Combine results
            result = {
                "success": True,
                "video_id": video_id,
                "video_url": video_url,
                "scraping": scrape_result,
                "analysis": analysis_result,
                "visualizations": viz_result,
                "output_dir": output_dir
            }
            
            print("\n" + "=" * 60)
            print("🎉 ANALYSIS COMPLETE!")
            print(f"📊 Comments analyzed: {analysis_result['comments_analyzed']}")
            print(f"📈 Sentiment summary: {analysis_result['summary']['sentiment_distribution']}")
            if viz_result and viz_result["success"]:
                print(f"🎨 Visualizations: {len(viz_result['generated_files'])}")
            print(f"📁 Results saved to: {output_dir}")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in full analysis: {e}")
            print(f"❌ Pipeline failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_summary_stats(self, results: list) -> Dict[str, Any]:
        """Generate summary statistics from sentiment results."""
        if not results:
            return {}
        
        # Count sentiments
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        confidence_scores = []
        
        for result in results:
            sentiment = result.get("sentiment", {})
            label = sentiment.get("label", "neutral").lower()
            confidence = sentiment.get("confidence", 0.0)
            
            if label in sentiment_counts:
                sentiment_counts[label] += 1
            confidence_scores.append(confidence)
        
        total = len(results)
        sentiment_distribution = {
            k: v / total for k, v in sentiment_counts.items()
        }
        
        return {
            "total_comments": total,
            "sentiment_counts": sentiment_counts,
            "sentiment_distribution": sentiment_distribution,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            "max_confidence": max(confidence_scores) if confidence_scores else 0.0,
            "min_confidence": min(confidence_scores) if confidence_scores else 0.0
        }


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="YouTube Comment Sentiment Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis pipeline
  python main.py full-analysis "https://youtube.com/watch?v=VIDEO_ID" --limit 1000

  # Just scrape comments
  python main.py scrape "https://youtube.com/watch?v=VIDEO_ID" --limit 500 --output comments.json

  # Analyze existing comments
  python main.py analyze VIDEO_ID --model ensemble --output-dir results/

  # Generate visualizations
  python main.py visualize VIDEO_ID --output-dir viz/ --types sentiment_pie timeline wordcloud
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Full analysis command
    full_parser = subparsers.add_parser("full-analysis", help="Complete analysis pipeline")
    full_parser.add_argument("video_url", help="YouTube video URL")
    full_parser.add_argument("--limit", type=int, default=500, help="Maximum comments to scrape")
    full_parser.add_argument("--model", default="ensemble", choices=["vader", "textblob", "transformers", "ensemble"],
                           help="Sentiment analysis model")
    full_parser.add_argument("--output-dir", default="output", help="Output directory")
    full_parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape comments from YouTube video")
    scrape_parser.add_argument("video_url", help="YouTube video URL")
    scrape_parser.add_argument("--limit", type=int, default=100, help="Maximum comments to scrape")
    scrape_parser.add_argument("--output", help="Output file path")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze sentiment of scraped comments")
    analyze_parser.add_argument("video_id", help="YouTube video ID")
    analyze_parser.add_argument("--model", default="ensemble", choices=["vader", "textblob", "transformers", "ensemble"],
                              help="Sentiment analysis model")
    analyze_parser.add_argument("--output-dir", help="Output directory for results")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("video_id", help="YouTube video ID")
    viz_parser.add_argument("--output-dir", required=True, help="Output directory")
    viz_parser.add_argument("--types", nargs="+", 
                          choices=["sentiment_pie", "timeline", "wordcloud", "treemap", "dashboard"],
                          help="Visualization types to generate")
    
    return parser


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = YouTubeCommentCLI()
    
    try:
        if args.command == "full-analysis":
            result = await cli.full_analysis(
                video_url=args.video_url,
                limit=args.limit,
                model=args.model,
                output_dir=args.output_dir,
                generate_viz=not args.no_viz
            )
        
        elif args.command == "scrape":
            result = await cli.scrape_comments(
                video_url=args.video_url,
                limit=args.limit,
                output_file=args.output
            )
        
        elif args.command == "analyze":
            result = await cli.analyze_sentiment(
                video_id=args.video_id,
                model=args.model,
                output_dir=args.output_dir
            )
        
        elif args.command == "visualize":
            result = await cli.generate_visualizations(
                video_id=args.video_id,
                output_dir=args.output_dir,
                viz_types=args.types
            )
        
        else:
            print(f"Unknown command: {args.command}")
            return
        
        # Exit with appropriate code
        sys.exit(0 if result.get("success", False) else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
