"""
Main YouTube Comment Sentiment Analyzer class that orchestrates the entire analysis pipeline.
"""
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd

from src.scraper import YouTubeCommentScraper, Comment
from src.preprocessor import TextPreprocessor
from src.sentiment_rules import SentimentRuleEngine, SentimentLabel, SentimentResult
from src.visualizer import SentimentVisualizer
from src.utils import (
    extract_video_id, create_cache_key, save_to_cache, load_from_cache,
    save_json, format_number, ProgressTracker
)
from config.settings import (
    DEFAULT_SCRAPING_METHOD, MAX_COMMENTS, ENABLE_CACHING, CACHE_DIR,
    OUTPUT_DIR, MIN_COMMENT_LENGTH, MAX_COMMENT_LENGTH
)
from config.logging_config import logger

class YouTubeSentimentAnalyzer:
    """
    Main class for YouTube comment sentiment analysis.
    
    This class orchestrates the entire pipeline:
    1. Comment scraping
    2. Text preprocessing
    3. Sentiment classification
    4. Visualization generation
    5. Results export
    """
    
    def __init__(
        self,
        scraping_method: str = DEFAULT_SCRAPING_METHOD,
        cache_enabled: bool = ENABLE_CACHING,
        output_dir: Union[str, Path] = OUTPUT_DIR
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            scraping_method: Method for scraping comments ('selenium' or 'requests')
            cache_enabled: Enable caching of results
            output_dir: Directory for saving outputs
        """
        self.scraping_method = scraping_method
        self.cache_enabled = cache_enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.sentiment_engine = SentimentRuleEngine()
        self.visualizer = SentimentVisualizer(self.output_dir)
        
        logger.info(f"YouTubeSentimentAnalyzer initialized with method: {scraping_method}")
        logger.info(f"Cache enabled: {cache_enabled}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def analyze_video(
        self,
        video_url: str,
        max_comments: int = MAX_COMMENTS,
        include_replies: bool = False,
        filter_spam: bool = True,
        force_refresh: bool = False
    ) -> Dict:
        """
        Perform complete sentiment analysis on a YouTube video's comments.
        
        Args:
            video_url: YouTube video URL
            max_comments: Maximum number of comments to analyze
            include_replies: Include comment replies
            filter_spam: Filter out spam comments
            force_refresh: Force refresh and ignore cache
            
        Returns:
            Dictionary containing analysis results
        """
        video_id = extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")
        
        logger.info(f"Starting sentiment analysis for video: {video_id}")
        logger.info(f"Parameters: max_comments={max_comments}, include_replies={include_replies}")
        
        # Check cache
        cache_key = create_cache_key(video_id, max_comments, include_replies, filter_spam)
        if self.cache_enabled and not force_refresh:
            cached_result = load_from_cache(cache_key, CACHE_DIR)
            if cached_result:
                logger.info("Loaded results from cache")
                return cached_result
        
        try:
            # Step 1: Scrape comments
            logger.info("Step 1: Scraping comments...")
            comments = self._scrape_comments(video_url, max_comments, include_replies)
            
            if not comments:
                raise ValueError("No comments found or scraped")
            
            logger.info(f"Scraped {len(comments)} comments")
            
            # Step 2: Filter and preprocess
            logger.info("Step 2: Preprocessing comments...")
            if filter_spam:
                comments = self._filter_spam_comments(comments)
                logger.info(f"After spam filtering: {len(comments)} comments")
            
            processed_comments = self._preprocess_comments(comments)
            
            # Step 3: Sentiment classification
            logger.info("Step 3: Classifying sentiment...")
            sentiment_results = self._classify_sentiments(processed_comments)
            
            # Step 4: Aggregate results
            logger.info("Step 4: Aggregating results...")
            analysis_results = self._aggregate_results(
                video_id, video_url, comments, processed_comments, sentiment_results
            )
            
            # Step 5: Cache results
            if self.cache_enabled:
                save_to_cache(cache_key, analysis_results, CACHE_DIR)
                logger.info("Results cached for future use")
            
            logger.info("Sentiment analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            raise
    
    def _scrape_comments(
        self,
        video_url: str,
        max_comments: int,
        include_replies: bool
    ) -> List[Comment]:
        """Scrape comments from YouTube video."""
        with YouTubeCommentScraper(method=self.scraping_method) as scraper:
            return scraper.scrape_comments(
                video_url=video_url,
                max_comments=max_comments,
                include_replies=include_replies
            )
    
    def _filter_spam_comments(self, comments: List[Comment]) -> List[Comment]:
        """Filter out spam and low-quality comments."""
        filtered_comments = []
        spam_patterns = [
            r'^(.)\1{5,}$',  # Repeated characters
            r'^[^\w\s]+$',   # Only punctuation
            r'^\d+$',        # Only numbers
            r'bit\.ly|tinyurl|goo\.gl',  # Short URLs
            r'subscribe.*subscribe',     # Subscribe spam
            r'check.*out.*channel',      # Channel promotion
        ]
        
        for comment in comments:
            # Check length
            if len(comment.text) < MIN_COMMENT_LENGTH or len(comment.text) > MAX_COMMENT_LENGTH:
                continue
            
            # Check spam patterns
            is_spam = any(
                __import__('re').search(pattern, comment.text.lower())
                for pattern in spam_patterns
            )
            
            if not is_spam:
                filtered_comments.append(comment)
        
        return filtered_comments
    
    def _preprocess_comments(self, comments: List[Comment]) -> List[Dict]:
        """Preprocess all comments."""
        processed_comments = []
        progress = ProgressTracker(len(comments), "Preprocessing comments")
        
        for comment in comments:
            try:
                processed_result = self.preprocessor.preprocess(
                    comment.text,
                    remove_stop_words=True,
                    lemmatize=True,
                    handle_emojis=True,
                    expand_contractions=True,
                    handle_slang=True
                )
                
                processed_comments.append({
                    'original_comment': comment,
                    'preprocessing_result': processed_result
                })
                
                progress.update()
                
            except Exception as e:
                logger.warning(f"Error preprocessing comment: {e}")
                continue
        
        progress.finish()
        return processed_comments
    
    def _classify_sentiments(self, processed_comments: List[Dict]) -> List[SentimentResult]:
        """Classify sentiment for all processed comments."""
        sentiment_results = []
        progress = ProgressTracker(len(processed_comments), "Classifying sentiment")
        
        for processed_comment in processed_comments:
            try:
                preprocessing_result = processed_comment['preprocessing_result']
                
                sentiment_result = self.sentiment_engine.classify_sentiment(
                    text=preprocessing_result['original'],
                    processed_tokens=preprocessing_result['tokens'],
                    sentiment_indicators=preprocessing_result['sentiment_indicators']
                )
                
                sentiment_results.append(sentiment_result)
                progress.update()
                
            except Exception as e:
                logger.warning(f"Error classifying sentiment: {e}")
                continue
        
        progress.finish()
        return sentiment_results
    
    def _aggregate_results(
        self,
        video_id: str,
        video_url: str,
        original_comments: List[Comment],
        processed_comments: List[Dict],
        sentiment_results: List[SentimentResult]
    ) -> Dict:
        """Aggregate all analysis results."""
        # Basic statistics
        total_comments = len(sentiment_results)
        sentiment_counts = Counter(result.label for result in sentiment_results)
        
        # Word frequency analysis by sentiment
        word_frequencies_by_sentiment = self._calculate_word_frequencies_by_sentiment(
            processed_comments, sentiment_results
        )
        
        # Confidence statistics
        confidence_stats = self._calculate_confidence_statistics(sentiment_results)
        
        # Create detailed results
        detailed_results = []
        for i, (processed_comment, sentiment_result) in enumerate(
            zip(processed_comments, sentiment_results)
        ):
            original_comment = processed_comment['original_comment']
            preprocessing_result = processed_comment['preprocessing_result']
            
            detailed_results.append({
                'comment_id': i + 1,
                'original_text': original_comment.text,
                'processed_text': preprocessing_result['processed'],
                'author': original_comment.author,
                'likes': original_comment.likes,
                'timestamp': original_comment.timestamp,
                'sentiment_label': sentiment_result.label.value,
                'confidence': sentiment_result.confidence,
                'positive_score': sentiment_result.positive_score,
                'negative_score': sentiment_result.negative_score,
                'neutral_score': sentiment_result.neutral_score,
                'matched_rules': sentiment_result.matched_rules,
                'sentiment_words': sentiment_result.sentiment_words,
                'word_count': preprocessing_result['word_count'],
                'emoji_count': preprocessing_result['emoji_count']
            })
        
        # Compile final results
        analysis_results = {
            'metadata': {
                'video_id': video_id,
                'video_url': video_url,
                'analysis_timestamp': datetime.now().isoformat(),
                'total_comments_scraped': len(original_comments),
                'total_comments_analyzed': total_comments,
                'scraping_method': self.scraping_method
            },
            'summary': {
                'sentiment_distribution': {
                    label.value: count for label, count in sentiment_counts.items()
                },
                'sentiment_percentages': {
                    label.value: (count / total_comments) * 100 
                    for label, count in sentiment_counts.items()
                },
                'confidence_statistics': confidence_stats,
                'total_words_analyzed': sum(
                    len(result['preprocessing_result']['tokens']) 
                    for result in processed_comments
                ),
                'average_comment_length': sum(
                    len(result['preprocessing_result']['tokens']) 
                    for result in processed_comments
                ) / len(processed_comments) if processed_comments else 0
            },
            'word_frequencies': {
                label.value: frequencies 
                for label, frequencies in word_frequencies_by_sentiment.items()
            },
            'detailed_results': detailed_results,
            'sentiment_rules_stats': self.sentiment_engine.get_sentiment_stats()
        }
        
        return analysis_results
    
    def _calculate_word_frequencies_by_sentiment(
        self,
        processed_comments: List[Dict],
        sentiment_results: List[SentimentResult]
    ) -> Dict[SentimentLabel, List[Tuple[str, int]]]:
        """Calculate word frequencies grouped by sentiment."""
        word_counts_by_sentiment = {
            SentimentLabel.POSITIVE: Counter(),
            SentimentLabel.NEGATIVE: Counter(),
            SentimentLabel.NEUTRAL: Counter()
        }
        
        for processed_comment, sentiment_result in zip(processed_comments, sentiment_results):
            tokens = processed_comment['preprocessing_result']['tokens']
            word_counts_by_sentiment[sentiment_result.label].update(tokens)
        
        # Convert to sorted lists
        word_frequencies_by_sentiment = {}
        for sentiment, counter in word_counts_by_sentiment.items():
            word_frequencies_by_sentiment[sentiment] = counter.most_common(50)
        
        return word_frequencies_by_sentiment
    
    def _calculate_confidence_statistics(self, sentiment_results: List[SentimentResult]) -> Dict:
        """Calculate confidence score statistics."""
        confidences = [result.confidence for result in sentiment_results]
        
        if not confidences:
            return {}
        
        import numpy as np
        
        return {
            'mean': float(np.mean(confidences)),
            'median': float(np.median(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'q25': float(np.percentile(confidences, 25)),
            'q75': float(np.percentile(confidences, 75))
        }
    
    def generate_visualizations(self, analysis_results: Dict) -> Dict[str, str]:
        """
        Generate all visualizations for the analysis results.
        
        Args:
            analysis_results: Results from analyze_video()
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        logger.info("Generating visualizations...")
        
        # Extract data for visualizations
        sentiment_counts = {
            SentimentLabel(label): count 
            for label, count in analysis_results['summary']['sentiment_distribution'].items()
        }
        
        word_frequencies_by_sentiment = {
            SentimentLabel(label): frequencies 
            for label, frequencies in analysis_results['word_frequencies'].items()
        }
        
        # Reconstruct sentiment results for confidence analysis
        sentiment_results = []
        for detail in analysis_results['detailed_results']:
            # Create a minimal SentimentResult for visualization
            from dataclasses import dataclass
            
            @dataclass
            class MinimalSentimentResult:
                label: SentimentLabel
                confidence: float
                sentiment_words: List[str]
            
            sentiment_results.append(MinimalSentimentResult(
                label=SentimentLabel(detail['sentiment_label']),
                confidence=detail['confidence'],
                sentiment_words=detail['sentiment_words']
            ))
        
        visualization_paths = {}
        
        try:
            # 1. Sentiment pie chart
            pie_chart_path = self.visualizer.create_sentiment_pie_chart(
                sentiment_counts,
                title=f"Sentiment Distribution ({sum(sentiment_counts.values()):,} comments)"
            )
            visualization_paths['pie_chart'] = pie_chart_path
            
            # 2. Word frequency analysis
            word_freq_path = self.visualizer.create_word_frequency_analysis(
                word_frequencies_by_sentiment
            )
            visualization_paths['word_frequency'] = word_freq_path
            
            # 3. Word clouds
            word_clouds_path = self.visualizer.create_word_clouds(
                word_frequencies_by_sentiment
            )
            visualization_paths['word_clouds'] = word_clouds_path
            
            # 4. Confidence distribution
            confidence_path = self.visualizer.create_confidence_distribution(
                sentiment_results
            )
            visualization_paths['confidence_distribution'] = confidence_path
            
            # 5. Comprehensive dashboard
            dashboard_path = self.visualizer.create_comprehensive_dashboard(
                sentiment_counts,
                word_frequencies_by_sentiment,
                sentiment_results
            )
            visualization_paths['dashboard'] = dashboard_path
            
            # 6. Summary statistics
            summary_path = self.visualizer.save_summary_stats(
                sentiment_counts,
                word_frequencies_by_sentiment,
                sentiment_results
            )
            visualization_paths['summary'] = summary_path
            
            logger.info(f"Generated {len(visualization_paths)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise
        
        return visualization_paths
    
    def export_results(
        self,
        analysis_results: Dict,
        format: str = 'csv',
        filename: Optional[str] = None
    ) -> str:
        """
        Export analysis results to file.
        
        Args:
            analysis_results: Results from analyze_video()
            format: Export format ('csv', 'json', 'excel')
            filename: Custom filename (without extension)
            
        Returns:
            Path to exported file
        """
        if not filename:
            video_id = analysis_results['metadata']['video_id']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sentiment_analysis_{video_id}_{timestamp}"
        
        if format.lower() == 'csv':
            return self._export_to_csv(analysis_results, filename)
        elif format.lower() == 'json':
            return self._export_to_json(analysis_results, filename)
        elif format.lower() == 'excel':
            return self._export_to_excel(analysis_results, filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_to_csv(self, analysis_results: Dict, filename: str) -> str:
        """Export results to CSV file."""
        filepath = self.output_dir / f"{filename}.csv"
        
        # Convert detailed results to DataFrame
        df = pd.DataFrame(analysis_results['detailed_results'])
        
        # Add summary information as comments
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Write summary as comments
            f.write(f"# YouTube Comment Sentiment Analysis Results\n")
            f.write(f"# Video ID: {analysis_results['metadata']['video_id']}\n")
            f.write(f"# Analysis Date: {analysis_results['metadata']['analysis_timestamp']}\n")
            f.write(f"# Total Comments: {analysis_results['metadata']['total_comments_analyzed']}\n")
            f.write(f"# Sentiment Distribution:\n")
            
            for sentiment, count in analysis_results['summary']['sentiment_distribution'].items():
                percentage = analysis_results['summary']['sentiment_percentages'][sentiment]
                f.write(f"#   {sentiment.title()}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"#\n")
            
            # Write DataFrame
            df.to_csv(f, index=False)
        
        logger.info(f"Results exported to CSV: {filepath}")
        return str(filepath)
    
    def _export_to_json(self, analysis_results: Dict, filename: str) -> str:
        """Export results to JSON file."""
        filepath = self.output_dir / f"{filename}.json"
        save_json(analysis_results, filepath)
        logger.info(f"Results exported to JSON: {filepath}")
        return str(filepath)
    
    def _export_to_excel(self, analysis_results: Dict, filename: str) -> str:
        """Export results to Excel file with multiple sheets."""
        filepath = self.output_dir / f"{filename}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Video ID',
                    'Total Comments Analyzed',
                    'Analysis Date',
                    'Positive Comments',
                    'Negative Comments',
                    'Neutral Comments',
                    'Average Confidence',
                    'Total Words Analyzed'
                ],
                'Value': [
                    analysis_results['metadata']['video_id'],
                    analysis_results['metadata']['total_comments_analyzed'],
                    analysis_results['metadata']['analysis_timestamp'],
                    analysis_results['summary']['sentiment_distribution'].get('positive', 0),
                    analysis_results['summary']['sentiment_distribution'].get('negative', 0),
                    analysis_results['summary']['sentiment_distribution'].get('neutral', 0),
                    analysis_results['summary']['confidence_statistics'].get('mean', 0),
                    analysis_results['summary']['total_words_analyzed']
                ]
            }
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results sheet
            df_detailed = pd.DataFrame(analysis_results['detailed_results'])
            df_detailed.to_excel(writer, sheet_name='Detailed Results', index=False)
            
            # Word frequencies sheets
            for sentiment, frequencies in analysis_results['word_frequencies'].items():
                if frequencies:
                    df_words = pd.DataFrame(frequencies, columns=['Word', 'Frequency'])
                    sheet_name = f'{sentiment.title()} Words'
                    df_words.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Results exported to Excel: {filepath}")
        return str(filepath)
    
    def get_analysis_summary(self, analysis_results: Dict) -> str:
        """
        Get a human-readable summary of the analysis results.
        
        Args:
            analysis_results: Results from analyze_video()
            
        Returns:
            Formatted summary string
        """
        metadata = analysis_results['metadata']
        summary = analysis_results['summary']
        
        total_comments = metadata['total_comments_analyzed']
        sentiment_dist = summary['sentiment_distribution']
        
        summary_text = f"""
YouTube Comment Sentiment Analysis Summary
==========================================

Video ID: {metadata['video_id']}
Analysis Date: {metadata['analysis_timestamp']}
Total Comments Analyzed: {format_number(total_comments)}

Sentiment Distribution:
  • Positive: {sentiment_dist.get('positive', 0):,} ({summary['sentiment_percentages'].get('positive', 0):.1f}%)
  • Negative: {sentiment_dist.get('negative', 0):,} ({summary['sentiment_percentages'].get('negative', 0):.1f}%)
  • Neutral: {sentiment_dist.get('neutral', 0):,} ({summary['sentiment_percentages'].get('neutral', 0):.1f}%)

Confidence Statistics:
  • Average: {summary['confidence_statistics'].get('mean', 0):.3f}
  • Median: {summary['confidence_statistics'].get('median', 0):.3f}
  • Range: {summary['confidence_statistics'].get('min', 0):.3f} - {summary['confidence_statistics'].get('max', 0):.3f}

Analysis Details:
  • Total Words Analyzed: {format_number(summary['total_words_analyzed'])}
  • Average Comment Length: {summary['average_comment_length']:.1f} words
  • Scraping Method: {metadata['scraping_method']}
"""
        
        return summary_text.strip()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup if needed
        pass

# Alias for backward compatibility with tests
SentimentAnalyzer = YouTubeSentimentAnalyzer
