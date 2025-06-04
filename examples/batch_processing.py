"""
Batch Processing Example - YouTube Comment Sentiment Analysis

This example demonstrates how to process multiple videos in batch
and generate comparative analysis reports.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.sentiment_analyzer import SentimentAnalyzer
from config.logging_config import setup_logging

# Popular videos for demonstration
DEMO_VIDEOS = [
    {"id": "jNQXAC9IVRw", "title": "Me at the zoo", "category": "Historic"},
    {"id": "dQw4w9WgXcQ", "title": "Never Gonna Give You Up", "category": "Music"},
    {"id": "9bZkp7q19f0", "title": "PSY - GANGNAM STYLE", "category": "Music"},
    {"id": "kffacxfA7G4", "title": "Baby Shark Dance", "category": "Kids"},
    {"id": "fC7oUOUEEi4", "title": "Despacito", "category": "Music"}
]

async def batch_processing_example():
    """Example of batch processing multiple videos."""
    print("🚀 Batch Processing Example")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Create batch output directory
    batch_dir = Path("data/outputs/batch_example")
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    batch_results = []
    failed_videos = []
    
    print(f"📋 Processing {len(DEMO_VIDEOS)} videos...")
    
    for i, video in enumerate(DEMO_VIDEOS, 1):
        print(f"\n[{i}/{len(DEMO_VIDEOS)}] 📹 {video['title']} ({video['id']})")
        print("-" * 60)
        
        try:
            # Initialize analyzer
            analyzer = SentimentAnalyzer(
                video_id=video['id'],
                max_comments=200,  # Reasonable number for batch processing
                scraping_method="auto",
                output_dir=str(batch_dir / video['id']),
                use_cache=True
            )
            
            # Run analysis
            results = await analyzer.analyze()
            
            if results:
                # Add metadata
                results['metadata'] = video
                results['analysis_timestamp'] = datetime.now().isoformat()
                batch_results.append(results)
                
                # Print quick summary
                stats = results['statistics']
                sentiment_dist = stats['sentiment_distribution']
                
                print(f"✅ Completed: {stats['total_comments']} comments")
                print(f"   😊 Positive: {sentiment_dist.get('positive', 0)}")
                print(f"   😞 Negative: {sentiment_dist.get('negative', 0)}")
                print(f"   😐 Neutral:  {sentiment_dist.get('neutral', 0)}")
                
                # Export individual results
                analyzer.export_results(results, format="csv")
                
            else:
                print("❌ Analysis failed")
                failed_videos.append(video)
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            failed_videos.append(video)
            continue
    
    # Generate batch report
    if batch_results:
        print(f"\n📊 Generating batch analysis report...")
        await generate_batch_report(batch_results, batch_dir)
    
    # Print final summary
    print_batch_summary(batch_results, failed_videos)

async def generate_batch_report(batch_results: list, output_dir: Path):
    """Generate comprehensive batch analysis report."""
    
    # Aggregate statistics
    aggregated_stats = aggregate_batch_statistics(batch_results)
    
    # Save aggregated results
    batch_report = {
        "summary": aggregated_stats,
        "individual_results": batch_results,
        "generated_at": datetime.now().isoformat()
    }
    
    # Save JSON report
    report_file = output_dir / "batch_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(batch_report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Batch report saved: {report_file}")
    
    # Generate comparison visualizations
    await create_comparison_visualizations(batch_results, output_dir)

def aggregate_batch_statistics(batch_results: list) -> dict:
    """Aggregate statistics across all videos."""
    total_comments = sum(r['statistics']['total_comments'] for r in batch_results)
    total_time = sum(r['statistics']['processing_time'] for r in batch_results)
    
    # Aggregate sentiment distribution
    aggregated_sentiment = {"positive": 0, "negative": 0, "neutral": 0}
    category_sentiment = {}
    
    for result in batch_results:
        # Overall sentiment
        for sentiment, count in result['statistics']['sentiment_distribution'].items():
            aggregated_sentiment[sentiment] += count
        
        # By category
        category = result['metadata']['category']
        if category not in category_sentiment:
            category_sentiment[category] = {"positive": 0, "negative": 0, "neutral": 0}
        
        for sentiment, count in result['statistics']['sentiment_distribution'].items():
            category_sentiment[category][sentiment] += count
    
    return {
        "total_videos": len(batch_results),
        "total_comments": total_comments,
        "total_processing_time": total_time,
        "average_comments_per_video": total_comments / len(batch_results),
        "overall_sentiment_distribution": aggregated_sentiment,
        "sentiment_by_category": category_sentiment,
        "video_rankings": rank_videos_by_sentiment(batch_results)
    }

def rank_videos_by_sentiment(batch_results: list) -> dict:
    """Rank videos by sentiment scores."""
    video_scores = []
    
    for result in batch_results:
        sentiment_dist = result['statistics']['sentiment_distribution']
        total = sum(sentiment_dist.values())
        
        if total > 0:
            # Calculate sentiment score (-1 to 1)
            positive_ratio = sentiment_dist.get('positive', 0) / total
            negative_ratio = sentiment_dist.get('negative', 0) / total
            sentiment_score = positive_ratio - negative_ratio
            
            video_scores.append({
                "video_id": result['metadata']['id'],
                "title": result['metadata']['title'],
                "category": result['metadata']['category'],
                "sentiment_score": sentiment_score,
                "total_comments": total,
                "positive_percentage": positive_ratio * 100,
                "negative_percentage": negative_ratio * 100
            })
    
    # Sort by sentiment score
    video_scores.sort(key=lambda x: x['sentiment_score'], reverse=True)
    
    return {
        "most_positive": video_scores[:3],
        "most_negative": video_scores[-3:],
        "all_rankings": video_scores
    }

async def create_comparison_visualizations(batch_results: list, output_dir: Path):
    """Create comparison visualizations for batch results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create comparison charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Batch Analysis Comparison', fontsize=16, fontweight='bold')
        
        # Data preparation
        video_names = [r['metadata']['title'][:20] + "..." if len(r['metadata']['title']) > 20 
                      else r['metadata']['title'] for r in batch_results]
        categories = [r['metadata']['category'] for r in batch_results]
        
        # 1. Comments count comparison
        comment_counts = [r['statistics']['total_comments'] for r in batch_results]
        ax1.bar(range(len(video_names)), comment_counts)
        ax1.set_title('Comments Analyzed per Video')
        ax1.set_ylabel('Number of Comments')
        ax1.set_xticks(range(len(video_names)))
        ax1.set_xticklabels(video_names, rotation=45, ha='right')
        
        # 2. Sentiment distribution
        sentiments = ['positive', 'negative', 'neutral']
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        
        sentiment_data = []
        for sentiment in sentiments:
            sentiment_data.append([
                r['statistics']['sentiment_distribution'].get(sentiment, 0) 
                for r in batch_results
            ])
        
        bottom = np.zeros(len(video_names))
        for i, (sentiment, color) in enumerate(zip(sentiments, colors)):
            ax2.bar(range(len(video_names)), sentiment_data[i], 
                   bottom=bottom, label=sentiment.title(), color=color)
            bottom += sentiment_data[i]
        
        ax2.set_title('Sentiment Distribution by Video')
        ax2.set_ylabel('Number of Comments')
        ax2.set_xticks(range(len(video_names)))
        ax2.set_xticklabels(video_names, rotation=45, ha='right')
        ax2.legend()
        
        # 3. Processing time comparison
        processing_times = [r['statistics']['processing_time'] for r in batch_results]
        ax3.bar(range(len(video_names)), processing_times, color='orange')
        ax3.set_title('Processing Time per Video')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xticks(range(len(video_names)))
        ax3.set_xticklabels(video_names, rotation=45, ha='right')
        
        # 4. Category sentiment comparison
        category_data = {}
        for result in batch_results:
            category = result['metadata']['category']
            if category not in category_data:
                category_data[category] = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            for sentiment, count in result['statistics']['sentiment_distribution'].items():
                category_data[category][sentiment] += count
        
        categories_list = list(category_data.keys())
        category_sentiment_data = []
        for sentiment in sentiments:
            category_sentiment_data.append([
                category_data[cat][sentiment] for cat in categories_list
            ])
        
        bottom = np.zeros(len(categories_list))
        for i, (sentiment, color) in enumerate(zip(sentiments, colors)):
            ax4.bar(range(len(categories_list)), category_sentiment_data[i],
                   bottom=bottom, label=sentiment.title(), color=color)
            bottom += category_sentiment_data[i]
        
        ax4.set_title('Sentiment Distribution by Category')
        ax4.set_ylabel('Number of Comments')
        ax4.set_xticks(range(len(categories_list)))
        ax4.set_xticklabels(categories_list)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save comparison chart
        comparison_file = output_dir / "batch_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison chart saved: {comparison_file}")
        
    except ImportError:
        print("⚠️  Matplotlib not available - skipping comparison visualizations")

def print_batch_summary(batch_results: list, failed_videos: list):
    """Print final batch processing summary."""
    print(f"\n" + "="*60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("="*60)
    
    total_videos = len(batch_results) + len(failed_videos)
    success_rate = (len(batch_results) / total_videos) * 100 if total_videos > 0 else 0
    
    print(f"🎬 Total Videos: {total_videos}")
    print(f"✅ Successful: {len(batch_results)}")
    print(f"❌ Failed: {len(failed_videos)}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if batch_results:
        total_comments = sum(r['statistics']['total_comments'] for r in batch_results)
        total_time = sum(r['statistics']['processing_time'] for r in batch_results)
        
        print(f"\n📊 Processing Statistics:")
        print(f"   💬 Total Comments: {total_comments:,}")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        print(f"   📊 Avg Comments/Video: {total_comments/len(batch_results):.0f}")
        print(f"   ⚡ Avg Time/Video: {total_time/len(batch_results):.2f}s")
        
        # Overall sentiment
        overall_sentiment = {"positive": 0, "negative": 0, "neutral": 0}
        for result in batch_results:
            for sentiment, count in result['statistics']['sentiment_distribution'].items():
                overall_sentiment[sentiment] += count
        
        print(f"\n💭 Overall Sentiment Distribution:")
        for sentiment, count in overall_sentiment.items():
            percentage = (count / total_comments) * 100 if total_comments > 0 else 0
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}[sentiment]
            print(f"   {emoji} {sentiment.title()}: {count:,} ({percentage:.1f}%)")
    
    if failed_videos:
        print(f"\n❌ Failed Videos:")
        for video in failed_videos:
            print(f"   - {video['title']} ({video['id']})")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(batch_processing_example())
