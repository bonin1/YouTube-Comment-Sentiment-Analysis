import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import asyncio

from .comment_scraper import Comment
from .sentiment_analyzer import SentimentResult, SentimentAnalyzer
from ..config.settings import get_settings
from ..utils.decorators import timing

class DataProcessor:
    """Data processing and storage manager"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        db_path = self.settings.DATA_DIR / "comments.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = str(db_path)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    id TEXT PRIMARY KEY,
                    video_id TEXT,
                    text TEXT,
                    author TEXT,
                    likes INTEGER,
                    reply_count INTEGER,
                    time_parsed TEXT,
                    is_reply BOOLEAN,
                    parent_id TEXT,
                    sentiment TEXT,
                    sentiment_confidence REAL,
                    sentiment_scores TEXT,
                    created_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    url TEXT,
                    title TEXT,
                    scraped_at TEXT,
                    total_comments INTEGER,
                    sentiment_summary TEXT
                )
            """)
            
            conn.commit()
    
    @timing
    async def process_comments(
        self,
        comments: List[Comment],
        video_id: str,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """
        Process comments with sentiment analysis
        
        Args:
            comments: List of Comment objects
            video_id: Video ID for grouping
            progress_callback: Optional progress callback
            
        Returns:
            DataFrame with processed comments
        """
        if not comments:
            return pd.DataFrame()
        
        self.logger.info(f"Processing {len(comments)} comments for video {video_id}")
        
        # Convert comments to DataFrame
        df = pd.DataFrame([comment.to_dict() for comment in comments])
        
        # Analyze sentiment for all comments
        texts = df['text'].tolist()
        sentiment_results = self.sentiment_analyzer.analyze_batch(texts)
        
        # Add sentiment data to DataFrame
        df['sentiment'] = [result.sentiment for result in sentiment_results]
        df['sentiment_confidence'] = [result.confidence for result in sentiment_results]
        df['sentiment_scores'] = [json.dumps(result.scores) for result in sentiment_results]
        df['video_id'] = video_id
        df['created_at'] = datetime.now().isoformat()
        
        # Store in database
        await self._store_comments(df)
        
        self.logger.info(f"Successfully processed {len(df)} comments")
        return df
    
    async def _store_comments(self, df: pd.DataFrame):
        """Store comments in database"""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('comments', conn, if_exists='append', index=False)
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate sentiment analysis summary
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        sentiment_counts = df['sentiment'].value_counts()
        total_comments = len(df)
        
        summary = {
            'total_comments': total_comments,
            'sentiment_distribution': {
                'positive': sentiment_counts.get('positive', 0),
                'negative': sentiment_counts.get('negative', 0),
                'neutral': sentiment_counts.get('neutral', 0)
            },
            'sentiment_percentages': {
                'positive': (sentiment_counts.get('positive', 0) / total_comments) * 100,
                'negative': (sentiment_counts.get('negative', 0) / total_comments) * 100,
                'neutral': (sentiment_counts.get('neutral', 0) / total_comments) * 100
            },
            'average_confidence': df['sentiment_confidence'].mean(),
            'top_positive_comments': self._get_top_comments(df, 'positive', 5),
            'top_negative_comments': self._get_top_comments(df, 'negative', 5),
            'most_liked_comments': df.nlargest(10, 'likes')[['text', 'likes', 'sentiment']].to_dict('records')
        }
        
        return summary
    
    def _get_top_comments(self, df: pd.DataFrame, sentiment: str, limit: int = 5) -> List[Dict]:
        """Get top comments by sentiment and confidence"""
        filtered_df = df[df['sentiment'] == sentiment]
        if filtered_df.empty:
            return []
        
        top_comments = filtered_df.nlargest(limit, 'sentiment_confidence')[
            ['text', 'sentiment_confidence', 'likes']
        ].to_dict('records')
        
        return top_comments
    
    def export_data(
        self,
        df: pd.DataFrame,
        format: str = 'csv',
        filename: Optional[str] = None
    ) -> str:
        """
        Export processed data to file
        
        Args:
            df: DataFrame to export
            format: Export format ('csv', 'json', 'xlsx')
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_comments_{timestamp}"
        
        export_dir = self.settings.DEFAULT_EXPORT_DIR
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            filepath = export_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False, encoding='utf-8')
        elif format.lower() == 'json':
            filepath = export_dir / f"{filename}.json"
            df.to_json(filepath, orient='records', indent=2, force_ascii=False)
        elif format.lower() == 'xlsx':
            filepath = export_dir / f"{filename}.xlsx"
            df.to_excel(filepath, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Data exported to {filepath}")
        return str(filepath)
    
    def load_stored_data(self, video_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load stored data from database
        
        Args:
            video_id: Optional video ID to filter by
            
        Returns:
            DataFrame with stored comments
        """
        query = "SELECT * FROM comments"
        params = []
        
        if video_id:
            query += " WHERE video_id = ?"
            params.append(video_id)
        
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def get_video_history(self) -> pd.DataFrame:
        """Get history of processed videos"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT DISTINCT video_id, COUNT(*) as comment_count, MIN(created_at) as first_scraped "
                "FROM comments GROUP BY video_id ORDER BY first_scraped DESC",
                conn
            )
        
        return df
    
    def clean_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate text data
        
        Args:
            df: DataFrame with comment data
            
        Returns:
            Cleaned DataFrame
        """
        # Remove empty or invalid text
        df = df[df['text'].notna() & (df['text'].str.len() > 0)]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'], keep='first')
        
        # Filter out very short comments (likely spam)
        df = df[df['text'].str.len() >= 3]
        
        # Basic text cleaning
        df['text'] = df['text'].str.strip()
        
        return df
