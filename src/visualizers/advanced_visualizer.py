"""
Advanced Visualization Module

This module provides comprehensive visualization capabilities including
squarify treemaps, interactive dashboards, and various chart types for
sentiment analysis results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import squarify
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from wordcloud import WordCloud
import base64
from io import BytesIO
from pathlib import Path

from config import get_logger, settings


class AdvancedVisualizer:
    """Advanced visualization system with interactive charts and dashboards."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.logger = get_logger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color schemes
        self.color_schemes = {
            'sentiment': {
                'positive': '#2E8B57',    # Sea Green
                'negative': '#DC143C',    # Crimson
                'neutral': '#4682B4'      # Steel Blue
            },
            'confidence': {
                'high': '#228B22',        # Forest Green
                'medium': '#FFD700',      # Gold
                'low': '#FF6347'          # Tomato
            },
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'inferno': plt.cm.inferno
        }
        
        # Default figure size
        self.fig_size = (settings.CHART_WIDTH/100, settings.CHART_HEIGHT/100)
        
        self.logger.info("Advanced visualizer initialized")
    
    def create_sentiment_distribution_pie(self, 
                                        results: List[Dict], 
                                        title: str = "Sentiment Distribution",
                                        save_path: Optional[Path] = None) -> plt.Figure:
        """Create a pie chart showing sentiment distribution."""
        sentiments = [r['sentiment'] for r in results if 'sentiment' in r]
        sentiment_counts = Counter(sentiments)
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Data for pie chart
        labels = list(sentiment_counts.keys())
        sizes = list(sentiment_counts.values())
        colors = [self.color_schemes['sentiment'].get(label, '#808080') for label in labels]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05] * len(labels)  # Slightly separate slices
        )
        
        # Customize text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontsize(14)
            text.set_fontweight('bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add total count
        total = sum(sizes)
        ax.text(0, -1.3, f'Total Comments: {total}', 
                ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_squarify_treemap(self, 
                              results: List[Dict],
                              title: str = "Sentiment Analysis Treemap",
                              save_path: Optional[Path] = None) -> plt.Figure:
        """Create a squarify treemap visualization."""
        # Prepare data
        sentiments = [r['sentiment'] for r in results if 'sentiment' in r]
        confidences = [r.get('confidence', 0) for r in results if 'sentiment' in r]
        
        # Group by sentiment and calculate metrics
        sentiment_data = {}
        for sentiment, confidence in zip(sentiments, confidences):
            if sentiment not in sentiment_data:
                sentiment_data[sentiment] = []
            sentiment_data[sentiment].append(confidence)
        
        # Calculate sizes and colors
        labels = []
        sizes = []
        colors = []
        
        for sentiment, conf_list in sentiment_data.items():
            count = len(conf_list)
            avg_confidence = np.mean(conf_list)
            
            labels.append(f'{sentiment.title()}\n{count} comments\nAvg Conf: {avg_confidence:.2f}')
            sizes.append(count)
            
            # Color intensity based on average confidence
            base_color = self.color_schemes['sentiment'].get(sentiment, '#808080')
            # Convert hex to RGB and adjust intensity
            if base_color.startswith('#'):
                r = int(base_color[1:3], 16)
                g = int(base_color[3:5], 16)
                b = int(base_color[5:7], 16)
                
                # Adjust intensity based on confidence
                intensity = 0.5 + (avg_confidence * 0.5)  # 0.5 to 1.0
                r = int(r * intensity)
                g = int(g * intensity)
                b = int(b * intensity)
                
                colors.append(f'#{r:02x}{g:02x}{b:02x}')
            else:
                colors.append(base_color)
        
        # Create treemap
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        squarify.plot(
            sizes=sizes,
            label=labels,
            color=colors,
            alpha=0.8,
            text_kwargs={'fontsize': 10, 'weight': 'bold'}
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_confidence_distribution(self, 
                                     results: List[Dict],
                                     title: str = "Confidence Distribution",
                                     save_path: Optional[Path] = None) -> plt.Figure:
        """Create confidence distribution histogram."""
        confidences = [r.get('confidence', 0) for r in results if 'confidence' in r]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_size[0]*2, self.fig_size[1]))
        
        # Histogram
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Score Distribution')
        ax1.axvline(settings.CONFIDENCE_THRESHOLD, color='red', linestyle='--', 
                   label=f'Threshold ({settings.CONFIDENCE_THRESHOLD})')
        ax1.legend()
        
        # Box plot by sentiment
        sentiments = [r['sentiment'] for r in results if 'sentiment' in r and 'confidence' in r]
        conf_by_sentiment = {}
        for sentiment, confidence in zip(sentiments, confidences):
            if sentiment not in conf_by_sentiment:
                conf_by_sentiment[sentiment] = []
            conf_by_sentiment[sentiment].append(confidence)
        
        if conf_by_sentiment:
            box_data = [conf_by_sentiment[sentiment] for sentiment in conf_by_sentiment.keys()]
            box_labels = list(conf_by_sentiment.keys())
            
            box_colors = [self.color_schemes['sentiment'].get(label, '#808080') for label in box_labels]
            
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence by Sentiment')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_word_cloud(self, 
                         texts: List[str], 
                         sentiment: str = 'all',
                         title: str = "Word Cloud",
                         save_path: Optional[Path] = None) -> plt.Figure:
        """Create word cloud visualization."""
        # Combine all texts
        combined_text = ' '.join(texts)
        
        if not combined_text.strip():
            # Create empty figure if no text
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.text(0.5, 0.5, 'No text data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.axis('off')
            return fig
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=600,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(combined_text)
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(f'{title} ({sentiment.title()})', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, results: List[Dict]) -> str:
        """Create an interactive Plotly dashboard."""
        if not results:
            return "<p>No data available for visualization.</p>"
        
        # Prepare data
        df = pd.DataFrame(results)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Sentiment Distribution', 'Confidence vs Agreement', 
                          'Sentiment Over Time', 'Model Comparison'],
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Sentiment Distribution Pie Chart
        sentiment_counts = df['sentiment'].value_counts()
        colors = [self.color_schemes['sentiment'].get(sent, '#808080') 
                 for sent in sentiment_counts.index]
        
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker_colors=colors,
                name="Sentiment Distribution"
            ),
            row=1, col=1
        )
        
        # 2. Confidence vs Agreement Scatter
        if 'agreement_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['confidence'],
                    y=df['agreement_score'],
                    mode='markers',
                    marker=dict(
                        color=df['sentiment'].map(self.color_schemes['sentiment']),
                        size=8,
                        opacity=0.6
                    ),
                    text=df['sentiment'],
                    name="Confidence vs Agreement"
                ),
                row=1, col=2
            )
        
        # 3. Sentiment distribution over batch indices (simulating time)
        if 'batch_index' in df.columns:
            sentiment_over_time = df.groupby(['batch_index', 'sentiment']).size().unstack(fill_value=0)
            
            for sentiment in sentiment_over_time.columns:
                fig.add_trace(
                    go.Bar(
                        x=sentiment_over_time.index,
                        y=sentiment_over_time[sentiment],
                        name=sentiment,
                        marker_color=self.color_schemes['sentiment'].get(sentiment, '#808080')
                    ),
                    row=2, col=1
                )
        
        # 4. Model Performance Comparison
        if 'individual_results' in df.columns:
            model_accuracy = {}
            for _, row in df.iterrows():
                if 'individual_results' in row and row['individual_results']:
                    for model, result in row['individual_results'].items():
                        if not result.get('error', False):
                            if model not in model_accuracy:
                                model_accuracy[model] = []
                            model_accuracy[model].append(result['confidence'])
            
            if model_accuracy:
                models = list(model_accuracy.keys())
                avg_confidences = [np.mean(model_accuracy[model]) for model in models]
                
                fig.add_trace(
                    go.Bar(
                        x=models,
                        y=avg_confidences,
                        marker_color='lightblue',
                        name="Average Model Confidence"
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="YouTube Comments Sentiment Analysis Dashboard",
            title_x=0.5,
            showlegend=True,
            height=800,
            font=dict(size=12)
        )
        
        # Convert to HTML
        return fig.to_html(include_plotlyjs='cdn')
    
    def create_comprehensive_report(self, 
                                  results: List[Dict], 
                                  video_info: Dict,
                                  stats: Dict,
                                  export_dir: Path) -> Path:
        """Create a comprehensive HTML report with all visualizations."""
        export_dir.mkdir(exist_ok=True)
        
        # Create individual plots
        plots = {}
        
        # Sentiment distribution pie chart
        fig1 = self.create_sentiment_distribution_pie(results)
        pie_path = export_dir / "sentiment_pie.png"
        fig1.savefig(pie_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        plots['pie'] = pie_path
        
        # Squarify treemap
        fig2 = self.create_squarify_treemap(results)
        treemap_path = export_dir / "sentiment_treemap.png"
        fig2.savefig(treemap_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        plots['treemap'] = treemap_path
        
        # Confidence distribution
        fig3 = self.create_confidence_distribution(results)
        conf_path = export_dir / "confidence_distribution.png"
        fig3.savefig(conf_path, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        plots['confidence'] = conf_path
        
        # Word clouds by sentiment
        texts_by_sentiment = {}
        for result in results:
            sentiment = result.get('sentiment', 'neutral')
            if sentiment not in texts_by_sentiment:
                texts_by_sentiment[sentiment] = []
            # Get original text if available
            if 'text' in result:
                texts_by_sentiment[sentiment].append(result['text'])
        
        wordcloud_paths = {}
        for sentiment, texts in texts_by_sentiment.items():
            if texts:
                fig4 = self.create_word_cloud(texts, sentiment, f"{sentiment.title()} Word Cloud")
                wc_path = export_dir / f"wordcloud_{sentiment}.png"
                fig4.savefig(wc_path, dpi=300, bbox_inches='tight')
                plt.close(fig4)
                wordcloud_paths[sentiment] = wc_path
        
        # Create interactive dashboard
        dashboard_html = self.create_interactive_dashboard(results)
        
        # Generate HTML report
        html_content = self._generate_html_report(
            video_info, stats, plots, wordcloud_paths, dashboard_html
        )
        
        # Save HTML report
        report_path = export_dir / "sentiment_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
        return report_path
    
    def _generate_html_report(self, 
                            video_info: Dict, 
                            stats: Dict, 
                            plots: Dict,
                            wordcloud_paths: Dict,
                            dashboard_html: str) -> str:
        """Generate HTML report content."""
        def image_to_base64(image_path: Path) -> str:
            """Convert image to base64 string."""
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YouTube Comments Sentiment Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #007acc;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .stat-number {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .stat-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                .chart-section {{
                    margin: 40px 0;
                    text-align: center;
                }}
                .chart-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 30px;
                    margin: 20px 0;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                }}
                .dashboard {{
                    margin: 40px 0;
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    overflow: hidden;
                }}
                .wordcloud-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>YouTube Comments Sentiment Analysis Report</h1>
                    <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <section>
                    <h2>Video Information</h2>
                    <p><strong>Video ID:</strong> {video_info.get('video_id', 'N/A')}</p>
                    <p><strong>URL:</strong> <a href="{video_info.get('url', '#')}" target="_blank">{video_info.get('url', 'N/A')}</a></p>
                </section>
                
                <section>
                    <h2>Analysis Summary</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">{stats.get('total_comments', 0)}</div>
                            <div class="stat-label">Total Comments</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{stats.get('average_confidence', 0):.2f}</div>
                            <div class="stat-label">Average Confidence</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{stats.get('high_confidence_percentage', 0):.1f}%</div>
                            <div class="stat-label">High Confidence</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{stats.get('sentiment_percentages', {}).get('positive', 0):.1f}%</div>
                            <div class="stat-label">Positive Sentiment</div>
                        </div>
                    </div>
                </section>
                
                <section class="chart-section">
                    <h2>Sentiment Analysis Visualizations</h2>
                    <div class="chart-grid">
                        <div class="chart-container">
                            <h3>Sentiment Distribution</h3>
                            <img src="data:image/png;base64,{image_to_base64(plots['pie'])}" alt="Sentiment Pie Chart">
                        </div>
                        <div class="chart-container">
                            <h3>Sentiment Treemap</h3>
                            <img src="data:image/png;base64,{image_to_base64(plots['treemap'])}" alt="Sentiment Treemap">
                        </div>
                        <div class="chart-container">
                            <h3>Confidence Analysis</h3>
                            <img src="data:image/png;base64,{image_to_base64(plots['confidence'])}" alt="Confidence Distribution">
                        </div>
                    </div>
                </section>
                
                <section class="chart-section">
                    <h2>Word Clouds by Sentiment</h2>
                    <div class="wordcloud-grid">
        """
        
        # Add word clouds
        for sentiment, path in wordcloud_paths.items():
            html += f"""
                        <div class="chart-container">
                            <h3>{sentiment.title()} Comments</h3>
                            <img src="data:image/png;base64,{image_to_base64(path)}" alt="{sentiment} Word Cloud">
                        </div>
            """
        
        html += f"""
                    </div>
                </section>
                
                <section>
                    <h2>Interactive Dashboard</h2>
                    <div class="dashboard">
                        {dashboard_html}
                    </div>
                </section>
                
                <section>
                    <h2>Detailed Statistics</h2>
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                        <tr style="background-color: #f2f2f2;">
                            <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">Metric</th>
                            <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">Value</th>
                        </tr>
        """
        
        # Add statistics table
        for key, value in stats.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    html += f"""
                        <tr>
                            <td style="padding: 12px; border: 1px solid #ddd;">{key} - {sub_key}</td>
                            <td style="padding: 12px; border: 1px solid #ddd;">{sub_value}</td>
                        </tr>
                    """
            else:
                html += f"""
                    <tr>
                        <td style="padding: 12px; border: 1px solid #ddd;">{key.replace('_', ' ').title()}</td>
                        <td style="padding: 12px; border: 1px solid #ddd;">{value}</td>
                    </tr>
                """
        
        html += """
                    </table>
                </section>
                
                <footer style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;">
                    <p>Generated by YouTube Comment Sentiment Analysis System</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html
