"""
Advanced YouTube Comment Sentiment Analysis with Streamlit
===========================================================

A comprehensive web application for analyzing YouTube comment sentiment
with interactive visualizations and detailed analytics.

Features:
- YouTube comment scraping and validation
- Advanced sentiment analysis using transformer models
- Interactive dashboards with multiple visualization types
- Data export capabilities
- Real-time progress tracking
- Comprehensive analytics and insights

Author: AI Assistant
Date: June 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import asyncio
import logging
from datetime import datetime
import io
import base64
from typing import Dict, Any, List, Optional
import time
import json

# Import our custom modules
from src.core.comment_scraper import YouTubeCommentScraper, Comment
from src.core.sentiment_analyzer import SentimentAnalyzer
from src.core.data_processor import DataProcessor
from src.utils.validators import validate_youtube_url, extract_video_id
from src.config.settings import get_settings

# Configure page
st.set_page_config(
    page_title="YouTube Comment Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .comment-box {
        background-color: #808080;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-positive {
        border-left: 4px solid #28a745;
    }
    .sentiment-negative {
        border-left: 4px solid #dc3545;
    }
    .sentiment-neutral {
        border-left: 4px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitYouTubeSentimentApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = self._setup_logging()
        
        # Initialize session state
        self._init_session_state()
        
        # Initialize components
        self._init_components()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'analyzed_data' not in st.session_state:
            st.session_state.analyzed_data = None
        if 'analysis_summary' not in st.session_state:
            st.session_state.analysis_summary = None
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'comments_data' not in st.session_state:
            st.session_state.comments_data = []
        if 'video_info' not in st.session_state:
            st.session_state.video_info = {}
    
    def _init_components(self):
        """Initialize analysis components"""
        try:
            with st.spinner("Initializing analysis components..."):
                self.scraper = YouTubeCommentScraper()
                self.analyzer = SentimentAnalyzer()
                self.processor = DataProcessor()
            st.success("✅ Analysis components initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {str(e)}")
            self.logger.error(f"Component initialization failed: {e}")
    
    def run(self):
        """Main application entry point"""
        # Application header
        self._render_header()
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content area
        if st.session_state.processing_complete and st.session_state.analyzed_data is not None:
            self._render_dashboard()
        else:
            self._render_input_section()
    
    def _render_header(self):
        """Render application header"""
        st.title("🎬 YouTube Comment Sentiment Analysis")
        st.markdown("""
        **Analyze YouTube video comments with advanced ML-powered sentiment analysis**
        
        This application scrapes YouTube comments, analyzes their sentiment using state-of-the-art 
        transformer models, and provides comprehensive visualizations and insights.
        """)
        
        # Quick stats if data is available
        if st.session_state.analyzed_data is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            data = st.session_state.analyzed_data
            with col1:
                st.metric("📝 Total Comments", len(data))
            with col2:
                positive_pct = (data['sentiment'] == 'positive').mean() * 100
                st.metric("😊 Positive", f"{positive_pct:.1f}%")
            with col3:
                negative_pct = (data['sentiment'] == 'negative').mean() * 100
                st.metric("😢 Negative", f"{negative_pct:.1f}%")
            with col4:
                avg_confidence = data['sentiment_confidence'].mean()
                st.metric("🎯 Avg Confidence", f"{avg_confidence:.3f}")
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("⚙️ Analysis Settings")
            
            # YouTube URL input
            url = st.text_input(
                "🔗 YouTube Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Enter a valid YouTube video URL"
            )
            
            # Analysis parameters
            st.subheader("📊 Parameters")
            
            limit = st.slider(
                "Number of Comments",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Maximum number of comments to analyze"
            )
            include_replies = st.checkbox(
                "Include Replies",
                value=False,
                help="Include reply comments in the analysis"
            )
            
            sort_by = st.selectbox(
                "Sort Comments By",
                ["Top Liked", "Most Recent", "Oldest"],
                index=0,
                help="Choose how to sort comments before analysis"
            )
            
            # Analysis button
            if st.button("🚀 Start Analysis", type="primary"):
                if url:
                    if validate_youtube_url(url):
                        self._run_analysis(url, limit, include_replies, sort_by)
                    else:
                        st.error("❌ Please enter a valid YouTube URL")
                else:
                    st.error("❌ Please enter a YouTube URL")
            
            # Clear data button
            if st.session_state.processing_complete:
                if st.button("🗑️ Clear Data"):
                    self._clear_data()
                    st.rerun()
            
            # Export section
            if st.session_state.analyzed_data is not None:
                st.subheader("💾 Export Data")
                
                export_format = st.selectbox(
                    "Format",
                    ["CSV", "JSON", "Excel"],
                    help="Choose export format"
                )
                
                if st.button("📥 Download Data"):
                    self._download_data(export_format)
    
    def _render_input_section(self):
        """Render input and getting started section"""
        st.header("🎯 Getting Started")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### How to use this application:
            
            1. **Enter YouTube URL** - Paste a valid YouTube video URL in the sidebar
            2. **Set Parameters** - Choose number of comments and whether to include replies
            3. **Start Analysis** - Click the "Start Analysis" button
            4. **View Results** - Explore interactive dashboards and visualizations
            5. **Export Data** - Download results in your preferred format
            
            #### Features:
            - 🤖 **Advanced ML Models** - Uses transformer-based sentiment analysis
            - 📊 **Interactive Visualizations** - Multiple chart types and dashboards
            - 💭 **Comment Insights** - Detailed analysis of individual comments
            - 📈 **Sentiment Trends** - Track sentiment patterns over time
            - ☁️ **Word Clouds** - Visual representation of common terms
            - 📋 **Data Export** - Download results in multiple formats
            """)
        
        with col2:
            st.info("""
            **💡 Tips:**
            
            - Use popular videos for better results
            - Start with 50-100 comments for quick analysis
            - Include replies for deeper insights
            - Try different videos to compare sentiment
            """)
            
            st.warning("""
            **⚠️ Note:**
            
            Analysis may take a few minutes depending on the number of comments and your internet connection.
            """)
    def _run_analysis(self, url: str, limit: int, include_replies: bool, sort_by: str = "Top Liked"):
        """Run the complete sentiment analysis"""
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Validate and extract video info
            status_text.text("🔍 Validating video...")
            progress_bar.progress(10)
            
            if not self.scraper.validate_video(url):
                st.error("❌ Video is not accessible or doesn't exist")
                return
            
            video_info = self.scraper.get_video_info(url)
            st.session_state.video_info = video_info
            
            # Step 2: Scrape comments
            status_text.text("📥 Scraping comments...")
            progress_bar.progress(30)
            
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                comments = loop.run_until_complete(
                    self.scraper.scrape_comments(
                        url=url,
                        limit=limit,
                        include_replies=include_replies,
                        sort_by=sort_by
                    )
                )
                
                if not comments:
                    st.error("❌ No comments found or video is not accessible")
                    return
                
                progress_bar.progress(60)
                status_text.text("🤖 Analyzing sentiment...")
                
                # Step 3: Process comments with sentiment analysis
                st.session_state.analyzed_data = loop.run_until_complete(
                    self.processor.process_comments(
                        comments=comments,
                        video_id=video_info['video_id']
                    )
                )
                
                # Step 4: Generate summary
                progress_bar.progress(90)
                status_text.text("📊 Generating insights...")
                
                st.session_state.analysis_summary = self.processor.get_sentiment_summary(
                    st.session_state.analyzed_data
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                st.session_state.processing_complete = True
                st.session_state.comments_data = [comment.to_dict() for comment in comments]
                
                # Show success message
                st.success(f"🎉 Successfully analyzed {len(comments)} comments!")
                
                time.sleep(1)  # Brief pause to show completion
                st.rerun()  # Refresh to show dashboard
                
            finally:
                loop.close()
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            self.logger.error(f"Analysis failed: {e}")
    
    def _render_dashboard(self):
        """Render the main dashboard with all visualizations"""
        st.header("📊 Analysis Dashboard")
        
        data = st.session_state.analyzed_data
        summary = st.session_state.analysis_summary
        
        # Video information
        self._render_video_info()
        
        # Overview metrics
        self._render_overview_metrics(summary)
        
        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Sentiment Overview",
            "💬 Comment Analysis", 
            "☁️ Word Clouds",
            "📋 Data Table",
            "📊 Advanced Charts"
        ])
        
        with tab1:
            self._render_sentiment_overview(data, summary)
        
        with tab2:
            self._render_comment_analysis(data)
        
        with tab3:
            self._render_word_clouds(data)
        
        with tab4:
            self._render_data_table(data)
        
        with tab5:
            self._render_advanced_charts(data)
    
    def _render_video_info(self):
        """Render video information section"""
        if st.session_state.video_info:
            video_info = st.session_state.video_info
            
            st.subheader("🎥 Video Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Video ID:** {video_info.get('video_id', 'N/A')}")
            with col2:
                st.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            with col3:
                st.write(f"**Status:** {video_info.get('status', 'Unknown').title()}")
    
    def _render_overview_metrics(self, summary: Dict[str, Any]):
        """Render overview metrics cards"""
        st.subheader("📈 Overview Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total = summary.get('total_comments', 0)
            st.metric("📝 Total Comments", f"{total:,}")
        
        with col2:
            pos_pct = summary.get('sentiment_percentages', {}).get('positive', 0)
            st.metric("😊 Positive", f"{pos_pct:.1f}%", 
                     delta=f"{pos_pct - 33.3:.1f}%" if pos_pct != 0 else None)
        
        with col3:
            neg_pct = summary.get('sentiment_percentages', {}).get('negative', 0)
            st.metric("😢 Negative", f"{neg_pct:.1f}%",
                     delta=f"{neg_pct - 33.3:.1f}%" if neg_pct != 0 else None)
        
        with col4:
            neu_pct = summary.get('sentiment_percentages', {}).get('neutral', 0)
            st.metric("😐 Neutral", f"{neu_pct:.1f}%",
                     delta=f"{neu_pct - 33.3:.1f}%" if neu_pct != 0 else None)
        
        with col5:
            avg_conf = summary.get('average_confidence', 0)
            st.metric("🎯 Avg Confidence", f"{avg_conf:.3f}")
    
    def _render_sentiment_overview(self, data: pd.DataFrame, summary: Dict[str, Any]):
        """Render sentiment overview visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            st.subheader("📊 Sentiment Distribution")
            
            sentiment_counts = data['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution",
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545',
                    'neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Sentiment confidence distribution
            st.subheader("🎯 Confidence Distribution")
            
            fig_hist = px.histogram(
                data,
                x='sentiment_confidence',
                color='sentiment',
                title="Confidence Score Distribution",
                nbins=20,
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545',
                    'neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Sentiment vs Likes analysis
        st.subheader("👍 Sentiment vs Engagement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment vs likes scatter plot
            fig_scatter = px.scatter(
                data,
                x='likes',
                y='sentiment_confidence',
                color='sentiment',
                title="Sentiment Confidence vs Likes",
                hover_data=['author', 'text'],
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545',
                    'neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Average likes by sentiment
            avg_likes = data.groupby('sentiment')['likes'].mean().reset_index()
            fig_bar = px.bar(
                avg_likes,
                x='sentiment',
                y='likes',
                title="Average Likes by Sentiment",
                color='sentiment',
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545',
                    'neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def _render_comment_analysis(self, data: pd.DataFrame):
        """Render detailed comment analysis"""
        st.subheader("💬 Comment Analysis")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                ["All", "Positive", "Negative", "Neutral"]
            )
        
        with col2:
            min_likes = st.number_input(
                "Minimum Likes",
                min_value=0,
                value=0,
                step=1
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Likes", "Confidence", "Text Length"]
            )
        
        # Apply filters
        filtered_data = data.copy()
        
        if sentiment_filter != "All":
            filtered_data = filtered_data[
                filtered_data['sentiment'] == sentiment_filter.lower()
            ]
        
        if min_likes > 0:
            filtered_data = filtered_data[filtered_data['likes'] >= min_likes]
        
        # Sort data
        if sort_by == "Likes":
            filtered_data = filtered_data.sort_values('likes', ascending=False)
        elif sort_by == "Confidence":
            filtered_data = filtered_data.sort_values('sentiment_confidence', ascending=False)
        else:  # Text Length
            filtered_data['text_length'] = filtered_data['text'].str.len()
            filtered_data = filtered_data.sort_values('text_length', ascending=False)
        
        # Display comments
        st.write(f"**Showing {len(filtered_data)} comments**")
        
        for idx, row in filtered_data.head(20).iterrows():
            sentiment_class = f"sentiment-{row['sentiment']}"
            
            st.markdown(f"""
            <div class="comment-box {sentiment_class}">
                <strong>👤 {row['author']}</strong> 
                <span style="float: right;">
                    👍 {row['likes']} | 🎯 {row['sentiment_confidence']:.3f} | 
                    <span style="color: {'#28a745' if row['sentiment'] == 'positive' else '#dc3545' if row['sentiment'] == 'negative' else '#6c757d'}">
                        {row['sentiment'].upper()}
                    </span>
                </span>
                <br><br>
                {row['text'][:500]}{'...' if len(row['text']) > 500 else ''}
            </div>
            """, unsafe_allow_html=True)
    
    def _render_word_clouds(self, data: pd.DataFrame):
        """Render word cloud visualizations"""
        st.subheader("☁️ Word Clouds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Positive sentiment word cloud
            st.write("**😊 Positive Comments**")
            positive_text = ' '.join(
                data[data['sentiment'] == 'positive']['text'].astype(str)
            )
            
            if positive_text.strip():
                wordcloud_pos = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='Greens'
                ).generate(positive_text)
                
                fig_pos, ax_pos = plt.subplots(figsize=(8, 6))
                ax_pos.imshow(wordcloud_pos, interpolation='bilinear')
                ax_pos.axis('off')
                st.pyplot(fig_pos)
            else:
                st.write("No positive comments to display")
        
        with col2:
            # Negative sentiment word cloud
            st.write("**😢 Negative Comments**")
            negative_text = ' '.join(
                data[data['sentiment'] == 'negative']['text'].astype(str)
            )
            
            if negative_text.strip():
                wordcloud_neg = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='Reds'
                ).generate(negative_text)
                
                fig_neg, ax_neg = plt.subplots(figsize=(8, 6))
                ax_neg.imshow(wordcloud_neg, interpolation='bilinear')
                ax_neg.axis('off')
                st.pyplot(fig_neg)
            else:
                st.write("No negative comments to display")
        
        # Overall word cloud
        st.write("**🌈 All Comments**")
        all_text = ' '.join(data['text'].astype(str))
        
        if all_text.strip():
            wordcloud_all = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis'
            ).generate(all_text)
            
            fig_all, ax_all = plt.subplots(figsize=(12, 6))
            ax_all.imshow(wordcloud_all, interpolation='bilinear')
            ax_all.axis('off')
            st.pyplot(fig_all)
    
    def _render_data_table(self, data: pd.DataFrame):
        """Render data table with filtering and search"""
        st.subheader("📋 Data Table")
        
        # Search functionality
        search_term = st.text_input("🔍 Search comments", placeholder="Enter search term...")
        
        # Filter data based on search
        display_data = data.copy()
        if search_term:
            display_data = display_data[
                display_data['text'].str.contains(search_term, case=False, na=False)
            ]
        
        # Select columns to display
        columns_to_show = st.multiselect(
            "Select columns to display",
            options=list(data.columns),
            default=['author', 'text', 'sentiment', 'sentiment_confidence', 'likes']
        )
        
        if columns_to_show:
            st.dataframe(
                display_data[columns_to_show],
                use_container_width=True,
                height=400
            )
        
        # Data summary
        st.subheader("📊 Data Summary")
        st.write(data.describe())
    
    def _render_advanced_charts(self, data: pd.DataFrame):
        """Render advanced visualization charts"""
        st.subheader("📊 Advanced Analytics")
        
        # Text length analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📏 Comment Length Analysis**")
            data['text_length'] = data['text'].str.len()
            
            fig_length = px.box(
                data,
                x='sentiment',
                y='text_length',
                title="Comment Length by Sentiment",
                color='sentiment',
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545',
                    'neutral': '#6c757d'
                }
            )
            st.plotly_chart(fig_length, use_container_width=True)
        
        with col2:
            st.write("**🔤 Text Length Distribution**")
            fig_hist_length = px.histogram(
                data,
                x='text_length',
                title="Distribution of Comment Lengths",
                nbins=30
            )
            st.plotly_chart(fig_hist_length, use_container_width=True)
        
        # Correlation heatmap
        st.write("**🔗 Correlation Analysis**")
        numeric_columns = ['likes', 'reply_count', 'sentiment_confidence', 'text_length']
        correlation_data = data[numeric_columns].corr()
        
        fig_heatmap = px.imshow(
            correlation_data,
            title="Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Top authors analysis
        st.write("**👥 Top Authors by Engagement**")
        author_stats = data.groupby('author').agg({
            'likes': 'sum',
            'text': 'count',
            'sentiment_confidence': 'mean'
        }).rename(columns={'text': 'comment_count'}).reset_index()
        
        author_stats = author_stats.sort_values('likes', ascending=False).head(10)
        
        fig_authors = px.bar(
            author_stats,
            x='author',
            y='likes',
            title="Top 10 Authors by Total Likes",
            hover_data=['comment_count', 'sentiment_confidence']
        )
        fig_authors.update_xaxes(tickangle=45)
        st.plotly_chart(fig_authors, use_container_width=True)
    
    def _download_data(self, format_type: str):
        """Generate download link for data"""
        data = st.session_state.analyzed_data
        
        if format_type == "CSV":
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="youtube_comments_sentiment.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        elif format_type == "JSON":
            json_str = data.to_json(orient='records', indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="youtube_comments_sentiment.json">Download JSON File</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        elif format_type == "Excel":
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                data.to_excel(writer, sheet_name='Sentiment_Analysis', index=False)
            
            b64 = base64.b64encode(buffer.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="youtube_comments_sentiment.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    def _clear_data(self):
        """Clear all session data"""
        st.session_state.analyzed_data = None
        st.session_state.analysis_summary = None
        st.session_state.processing_complete = False
        st.session_state.comments_data = []
        st.session_state.video_info = {}

def main():
    """Main application entry point"""
    app = StreamlitYouTubeSentimentApp()
    app.run()

if __name__ == "__main__":
    main()
