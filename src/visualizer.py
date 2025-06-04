"""
Advanced data visualization for sentiment analysis results.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.sentiment_rules import SentimentLabel, SentimentResult
from config.settings import FIGURE_SIZE, DPI, COLOR_PALETTE, OUTPUT_DIR

logger = logging.getLogger(__name__)

class SentimentVisualizer:
    """Advanced visualization for sentiment analysis results."""
    
    def __init__(self, output_dir: Union[str, Path] = OUTPUT_DIR):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette(COLOR_PALETTE)
        
        # Color mapping for sentiments
        self.sentiment_colors = {
            SentimentLabel.POSITIVE: '#2E8B57',  # Sea Green
            SentimentLabel.NEGATIVE: '#DC143C',  # Crimson
            SentimentLabel.NEUTRAL: '#4682B4'    # Steel Blue
        }
        
        # Set up font sizes
        self.font_sizes = {
            'title': 16,
            'label': 12,
            'tick': 10,
            'legend': 11
        }
    
    def create_sentiment_pie_chart(
        self,
        sentiment_counts: Dict[SentimentLabel, int],
        title: str = "Comment Sentiment Distribution",
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> str:
        """
        Create a pie chart showing sentiment distribution.
        
        Args:
            sentiment_counts: Dictionary of sentiment counts
            title: Chart title
            save_path: Custom save path
            interactive: Create interactive plotly chart
            
        Returns:
            Path to saved chart
        """
        if interactive:
            return self._create_interactive_pie_chart(sentiment_counts, title, save_path)
        else:
            return self._create_static_pie_chart(sentiment_counts, title, save_path)
    
    def _create_interactive_pie_chart(
        self,
        sentiment_counts: Dict[SentimentLabel, int],
        title: str,
        save_path: Optional[str]
    ) -> str:
        """Create interactive pie chart using Plotly."""
        labels = [label.value.title() for label in sentiment_counts.keys()]
        values = list(sentiment_counts.values())
        colors = [self.sentiment_colors[label] for label in sentiment_counts.keys()]
        
        # Calculate percentages
        total = sum(values)
        percentages = [f"{(v/total)*100:.1f}%" for v in values]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=colors,
            textinfo='label+percent+value',
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>' +
                         'Count: %{value}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            ),
            margin=dict(t=60, b=40, l=40, r=120),
            width=800,
            height=500
        )
        
        # Add annotation in center
        fig.add_annotation(
            text=f"Total<br>{total:,}<br>Comments",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )
        
        # Save chart
        if not save_path:
            save_path = self.output_dir / "sentiment_pie_chart_interactive.html"
        
        fig.write_html(str(save_path))
        logger.info(f"Interactive pie chart saved to {save_path}")
        return str(save_path)
    
    def _create_static_pie_chart(
        self,
        sentiment_counts: Dict[SentimentLabel, int],
        title: str,
        save_path: Optional[str]
    ) -> str:
        """Create static pie chart using Matplotlib."""
        labels = [label.value.title() for label in sentiment_counts.keys()]
        values = list(sentiment_counts.values())
        colors = [self.sentiment_colors[label] for label in sentiment_counts.keys()]
        
        # Calculate percentages
        total = sum(values)
        percentages = [f"{(v/total)*100:.1f}%" for v in values]
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05 if label == SentimentLabel.POSITIVE else 0 for label in sentiment_counts.keys()],
            shadow=True,
            textprops={'fontsize': self.font_sizes['label']}
        )
        
        # Customize text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Add title
        ax.set_title(title, fontsize=self.font_sizes['title'], fontweight='bold', pad=20)
        
        # Add total count annotation
        ax.text(0, -1.3, f"Total Comments: {total:,}", 
               ha='center', fontsize=self.font_sizes['label'], 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save chart
        if not save_path:
            save_path = self.output_dir / "sentiment_pie_chart.png"
        
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Static pie chart saved to {save_path}")
        return str(save_path)
    
    def create_word_frequency_analysis(
        self,
        word_frequencies_by_sentiment: Dict[SentimentLabel, List[Tuple[str, int]]],
        top_n: int = 20,
        save_path: Optional[str] = None
    ) -> str:
        """
        Create word frequency analysis visualization.
        
        Args:
            word_frequencies_by_sentiment: Word frequencies grouped by sentiment
            top_n: Number of top words to show
            save_path: Custom save path
            
        Returns:
            Path to saved chart
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=DPI)
        
        for i, (sentiment, frequencies) in enumerate(word_frequencies_by_sentiment.items()):
            if not frequencies:
                continue
            
            # Get top N words
            top_words = frequencies[:top_n]
            words, counts = zip(*top_words) if top_words else ([], [])
            
            # Create horizontal bar chart
            ax = axes[i]
            bars = ax.barh(range(len(words)), counts, 
                          color=self.sentiment_colors[sentiment], alpha=0.7)
            
            # Customize axis
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=self.font_sizes['tick'])
            ax.set_xlabel('Frequency', fontsize=self.font_sizes['label'])
            ax.set_title(f'{sentiment.value.title()} Words', 
                        fontsize=self.font_sizes['title'], fontweight='bold')
            
            # Add value labels on bars
            for j, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                       str(count), ha='left', va='center', fontsize=self.font_sizes['tick'])
            
            # Invert y-axis to show highest frequency at top
            ax.invert_yaxis()
            
            # Add grid
            ax.grid(axis='x', alpha=0.3)
            ax.set_axisbelow(True)
        
        plt.suptitle('Top Words by Sentiment Category', 
                    fontsize=self.font_sizes['title'] + 2, fontweight='bold')
        plt.tight_layout()
        
        # Save chart
        if not save_path:
            save_path = self.output_dir / "word_frequency_analysis.png"
        
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Word frequency analysis saved to {save_path}")
        return str(save_path)
    
    def create_word_clouds(
        self,
        word_frequencies_by_sentiment: Dict[SentimentLabel, List[Tuple[str, int]]],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create word clouds for each sentiment category.
        
        Args:
            word_frequencies_by_sentiment: Word frequencies grouped by sentiment
            save_path: Custom save path
            
        Returns:
            Path to saved chart
        """
        fig, axes = plt.subplots(1, 3, figsize=(21, 7), dpi=DPI)
        
        for i, (sentiment, frequencies) in enumerate(word_frequencies_by_sentiment.items()):
            ax = axes[i]
            
            if not frequencies:
                ax.text(0.5, 0.5, f'No {sentiment.value} words found', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=self.font_sizes['label'])
                ax.set_title(f'{sentiment.value.title()} Words', 
                           fontsize=self.font_sizes['title'], fontweight='bold')
                ax.axis('off')
                continue
            
            # Create word frequency dictionary
            word_freq_dict = dict(frequencies[:100])  # Limit to top 100 words
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=600, height=400,
                background_color='white',
                colormap=self._get_colormap_for_sentiment(sentiment),
                max_words=100,
                relative_scaling=0.5,
                random_state=42
            ).generate_from_frequencies(word_freq_dict)
            
            # Display word cloud
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'{sentiment.value.title()} Words', 
                        fontsize=self.font_sizes['title'], fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('Word Clouds by Sentiment Category', 
                    fontsize=self.font_sizes['title'] + 2, fontweight='bold')
        plt.tight_layout()
        
        # Save chart
        if not save_path:
            save_path = self.output_dir / "word_clouds.png"
        
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Word clouds saved to {save_path}")
        return str(save_path)
    
    def create_confidence_distribution(
        self,
        sentiment_results: List[SentimentResult],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create confidence score distribution visualization.
        
        Args:
            sentiment_results: List of sentiment analysis results
            save_path: Custom save path
            
        Returns:
            Path to saved chart
        """
        # Group confidence scores by sentiment
        confidence_by_sentiment = {
            SentimentLabel.POSITIVE: [],
            SentimentLabel.NEGATIVE: [],
            SentimentLabel.NEUTRAL: []
        }
        
        for result in sentiment_results:
            confidence_by_sentiment[result.label].append(result.confidence)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=DPI)
        
        for i, (sentiment, confidences) in enumerate(confidence_by_sentiment.items()):
            ax = axes[i]
            
            if not confidences:
                ax.text(0.5, 0.5, f'No {sentiment.value} predictions', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{sentiment.value.title()} Confidence')
                continue
            
            # Create histogram
            ax.hist(confidences, bins=20, alpha=0.7, 
                   color=self.sentiment_colors[sentiment], edgecolor='black')
            
            # Add statistics
            mean_conf = np.mean(confidences)
            ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_conf:.3f}')
            
            ax.set_xlabel('Confidence Score', fontsize=self.font_sizes['label'])
            ax.set_ylabel('Frequency', fontsize=self.font_sizes['label'])
            ax.set_title(f'{sentiment.value.title()} Confidence Distribution', 
                        fontsize=self.font_sizes['title'], fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle('Confidence Score Distributions by Sentiment', 
                    fontsize=self.font_sizes['title'] + 2, fontweight='bold')
        plt.tight_layout()
        
        # Save chart
        if not save_path:
            save_path = self.output_dir / "confidence_distribution.png"
        
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confidence distribution chart saved to {save_path}")
        return str(save_path)
    
    def create_sentiment_timeline(
        self,
        comments_with_timestamps: List[Dict],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create sentiment timeline visualization.
        
        Args:
            comments_with_timestamps: List of comments with timestamps and sentiment
            save_path: Custom save path
            
        Returns:
            Path to saved chart
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(comments_with_timestamps)
        
        if 'timestamp' not in df.columns or df.empty:
            logger.warning("No timestamp data available for timeline")
            return ""
        
        # Parse timestamps and group by time periods
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        if df.empty:
            logger.warning("No valid timestamps found")
            return ""
        
        # Group by hour or day depending on data span
        time_span = df['timestamp'].max() - df['timestamp'].min()
        if time_span.days > 7:
            freq = 'D'  # Daily
            time_label = 'Date'
        else:
            freq = 'H'  # Hourly
            time_label = 'Hour'
        
        # Group and count sentiments
        df['time_period'] = df['timestamp'].dt.floor(freq)
        sentiment_timeline = df.groupby(['time_period', 'sentiment']).size().unstack(fill_value=0)
        
        # Create stacked area chart
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
        
        colors = [self.sentiment_colors[SentimentLabel(col)] for col in sentiment_timeline.columns]
        sentiment_timeline.plot(kind='area', stacked=True, ax=ax, color=colors, alpha=0.7)
        
        ax.set_xlabel(time_label, fontsize=self.font_sizes['label'])
        ax.set_ylabel('Number of Comments', fontsize=self.font_sizes['label'])
        ax.set_title('Sentiment Timeline', fontsize=self.font_sizes['title'], fontweight='bold')
        ax.legend(title='Sentiment', title_fontsize=self.font_sizes['legend'])
        ax.grid(alpha=0.3)
        
        # Format x-axis
        if freq == 'D':
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        if not save_path:
            save_path = self.output_dir / "sentiment_timeline.png"
        
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sentiment timeline saved to {save_path}")
        return str(save_path)
    
    def create_comprehensive_dashboard(
        self,
        sentiment_counts: Dict[SentimentLabel, int],
        word_frequencies_by_sentiment: Dict[SentimentLabel, List[Tuple[str, int]]],
        sentiment_results: List[SentimentResult],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            sentiment_counts: Sentiment distribution counts
            word_frequencies_by_sentiment: Word frequencies by sentiment
            sentiment_results: Sentiment analysis results
            save_path: Custom save path
            
        Returns:
            Path to saved dashboard
        """
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Sentiment Distribution', 'Top Positive Words', 'Top Negative Words',
                'Confidence Scores', 'Sentiment Summary', 'Word Count Distribution'
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "table"}, {"type": "histogram"}]
            ]
        )
        
        # 1. Pie chart
        labels = [label.value.title() for label in sentiment_counts.keys()]
        values = list(sentiment_counts.values())
        colors = [self.sentiment_colors[label] for label in sentiment_counts.keys()]
        
        fig.add_trace(go.Pie(
            labels=labels, values=values, marker_colors=colors,
            name="Sentiment Distribution"
        ), row=1, col=1)
        
        # 2. Top positive words
        if SentimentLabel.POSITIVE in word_frequencies_by_sentiment:
            pos_words = word_frequencies_by_sentiment[SentimentLabel.POSITIVE][:10]
            if pos_words:
                words, counts = zip(*pos_words)
                fig.add_trace(go.Bar(
                    x=list(counts), y=list(words), orientation='h',
                    marker_color=self.sentiment_colors[SentimentLabel.POSITIVE],
                    name="Positive Words"
                ), row=1, col=2)
        
        # 3. Top negative words
        if SentimentLabel.NEGATIVE in word_frequencies_by_sentiment:
            neg_words = word_frequencies_by_sentiment[SentimentLabel.NEGATIVE][:10]
            if neg_words:
                words, counts = zip(*neg_words)
                fig.add_trace(go.Bar(
                    x=list(counts), y=list(words), orientation='h',
                    marker_color=self.sentiment_colors[SentimentLabel.NEGATIVE],
                    name="Negative Words"
                ), row=1, col=3)
        
        # 4. Confidence distribution
        confidences = [result.confidence for result in sentiment_results]
        fig.add_trace(go.Histogram(
            x=confidences, nbinsx=20,
            marker_color='lightblue',
            name="Confidence Distribution"
        ), row=2, col=1)
        
        # 5. Summary table
        total_comments = sum(sentiment_counts.values())
        avg_confidence = np.mean(confidences) if confidences else 0
        
        summary_data = [
            ['Total Comments', f'{total_comments:,}'],
            ['Positive %', f'{(sentiment_counts.get(SentimentLabel.POSITIVE, 0)/total_comments)*100:.1f}%'],
            ['Negative %', f'{(sentiment_counts.get(SentimentLabel.NEGATIVE, 0)/total_comments)*100:.1f}%'],
            ['Neutral %', f'{(sentiment_counts.get(SentimentLabel.NEUTRAL, 0)/total_comments)*100:.1f}%'],
            ['Avg Confidence', f'{avg_confidence:.3f}']
        ]
        
        fig.add_trace(go.Table(
            header=dict(values=['Metric', 'Value']),
            cells=dict(values=list(zip(*summary_data)))
        ), row=2, col=2)
        
        # 6. Word count distribution
        word_counts = [len(result.sentiment_words) for result in sentiment_results]
        fig.add_trace(go.Histogram(
            x=word_counts, nbinsx=15,
            marker_color='lightgreen',
            name="Words per Comment"
        ), row=2, col=3)
        
        # Update layout
        fig.update_layout(
            title_text="YouTube Comment Sentiment Analysis Dashboard",
            showlegend=False,
            height=800,
            width=1400
        )
        
        # Save dashboard
        if not save_path:
            save_path = self.output_dir / "sentiment_dashboard.html"
        
        fig.write_html(str(save_path))
        logger.info(f"Comprehensive dashboard saved to {save_path}")
        return str(save_path)
    
    def _get_colormap_for_sentiment(self, sentiment: SentimentLabel) -> str:
        """Get appropriate colormap for sentiment."""
        if sentiment == SentimentLabel.POSITIVE:
            return 'Greens'
        elif sentiment == SentimentLabel.NEGATIVE:
            return 'Reds'
        else:
            return 'Blues'
    
    def save_summary_stats(
        self,
        sentiment_counts: Dict[SentimentLabel, int],
        word_frequencies_by_sentiment: Dict[SentimentLabel, List[Tuple[str, int]]],
        sentiment_results: List[SentimentResult],
        save_path: Optional[str] = None
    ) -> str:
        """
        Save summary statistics to a text file.
        
        Args:
            sentiment_counts: Sentiment distribution counts
            word_frequencies_by_sentiment: Word frequencies by sentiment
            sentiment_results: Sentiment analysis results
            save_path: Custom save path
            
        Returns:
            Path to saved summary
        """
        total_comments = sum(sentiment_counts.values())
        
        summary_lines = [
            "YouTube Comment Sentiment Analysis Summary",
            "=" * 50,
            "",
            f"Total Comments Analyzed: {total_comments:,}",
            "",
            "Sentiment Distribution:",
            "-" * 25
        ]
        
        # Add sentiment percentages
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_comments) * 100 if total_comments > 0 else 0
            summary_lines.append(f"{sentiment.value.title()}: {count:,} ({percentage:.1f}%)")
        
        summary_lines.extend([
            "",
            "Confidence Statistics:",
            "-" * 22
        ])
        
        # Add confidence stats
        confidences = [result.confidence for result in sentiment_results]
        if confidences:
            summary_lines.extend([
                f"Average Confidence: {np.mean(confidences):.3f}",
                f"Median Confidence: {np.median(confidences):.3f}",
                f"Min Confidence: {np.min(confidences):.3f}",
                f"Max Confidence: {np.max(confidences):.3f}"
            ])
        
        # Add top words for each sentiment
        for sentiment, frequencies in word_frequencies_by_sentiment.items():
            if frequencies:
                summary_lines.extend([
                    "",
                    f"Top {sentiment.value.title()} Words:",
                    "-" * (len(f"Top {sentiment.value.title()} Words:"))
                ])
                
                for i, (word, count) in enumerate(frequencies[:10], 1):
                    summary_lines.append(f"{i:2d}. {word} ({count})")
        
        # Save summary
        if not save_path:
            save_path = self.output_dir / "analysis_summary.txt"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Summary statistics saved to {save_path}")
        return str(save_path)
    
    def create_word_frequency_chart(self, word_freq_data: Dict[str, Dict[str, int]]) -> str:
        """Alias for create_word_frequency_analysis for test compatibility."""
        return self.create_word_frequency_analysis(word_freq_data)

    def _normalize_sentiment_counts(self, sentiment_counts: Dict) -> Dict[SentimentLabel, int]:
        """Normalize sentiment counts to use SentimentLabel enum."""
        from .sentiment_rules import SentimentLabel
        
        normalized = {}
        for key, value in sentiment_counts.items():
            if isinstance(key, str):
                # Convert string to SentimentLabel
                if key.lower() == 'positive':
                    normalized[SentimentLabel.POSITIVE] = value
                elif key.lower() == 'negative':
                    normalized[SentimentLabel.NEGATIVE] = value
                elif key.lower() == 'neutral':
                    normalized[SentimentLabel.NEUTRAL] = value
            else:
                normalized[key] = value
        
        return normalized
