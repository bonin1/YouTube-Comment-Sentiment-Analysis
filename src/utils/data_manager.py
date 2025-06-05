"""
Data Management Utilities

This module provides data persistence, caching, and export functionality.
"""

import sqlite3
import json
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd

from config import get_logger, settings


class DataManager:
    """Comprehensive data management with SQLite persistence and caching."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize data manager."""
        self.logger = get_logger(__name__)
        self.db_path = db_path or settings.get_database_path()
        self.cache_enabled = settings.CACHE_ENABLED
        self.cache_expiry = timedelta(hours=settings.CACHE_EXPIRY_HOURS)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"Data manager initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Videos table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT UNIQUE NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Comments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    comment_id TEXT UNIQUE,
                    author TEXT,
                    text TEXT NOT NULL,
                    likes INTEGER DEFAULT 0,
                    timestamp TIMESTAMP,
                    is_reply BOOLEAN DEFAULT 0,
                    parent_id TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id)
                )
            ''')
            
            # Sentiment analysis results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comment_id TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence REAL,
                    agreement_score REAL,
                    model_results TEXT,
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (comment_id) REFERENCES comments (comment_id)
                )
            ''')
            
            # Analysis sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    video_id TEXT NOT NULL,
                    total_comments INTEGER,
                    analysis_settings TEXT,
                    statistics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments (video_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_comment_id ON sentiment_results (comment_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_video_id ON analysis_sessions (video_id)')
            
            conn.commit()
    
    def save_video_info(self, video_info: Dict) -> int:
        """Save video information to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO videos (video_id, url, title, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (
                    video_info.get('video_id'),
                    video_info.get('url'),
                    video_info.get('title', ''),
                    json.dumps(video_info)
                ))
                
                video_db_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Saved video info for {video_info.get('video_id')}")
                return video_db_id
                
            except Exception as e:
                self.logger.error(f"Error saving video info: {str(e)}")
                raise
    
    def save_comments(self, comments: List[Dict], video_id: str) -> List[int]:
        """Save comments to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            comment_ids = []
            
            try:
                for comment in comments:
                    cursor.execute('''
                        INSERT OR REPLACE INTO comments 
                        (video_id, comment_id, author, text, likes, timestamp, is_reply, parent_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        video_id,
                        comment.get('id'),
                        comment.get('author'),
                        comment.get('text'),
                        comment.get('likes', 0),
                        comment.get('timestamp'),
                        comment.get('is_reply', False),
                        comment.get('parent_id')
                    ))
                    comment_ids.append(cursor.lastrowid)
                
                conn.commit()
                
                self.logger.info(f"Saved {len(comments)} comments for video {video_id}")
                return comment_ids
                
            except Exception as e:
                self.logger.error(f"Error saving comments: {str(e)}")
                raise
    
    def save_sentiment_results(self, results: List[Dict], comments: List[Dict]) -> List[int]:
        """Save sentiment analysis results to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            result_ids = []
            
            try:
                for result, comment in zip(results, comments):
                    cursor.execute('''
                        INSERT OR REPLACE INTO sentiment_results
                        (comment_id, sentiment, confidence, agreement_score, model_results)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        comment.get('id'),
                        result.get('sentiment'),
                        result.get('confidence'),
                        result.get('agreement_score'),
                        json.dumps(result.get('individual_results', {}))
                    ))
                    result_ids.append(cursor.lastrowid)
                
                conn.commit()
                
                self.logger.info(f"Saved {len(results)} sentiment analysis results")
                return result_ids
                
            except Exception as e:
                self.logger.error(f"Error saving sentiment results: {str(e)}")
                raise
    
    def save_analysis_session(self, session_data: Dict) -> str:
        """Save complete analysis session."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO analysis_sessions
                    (session_id, video_id, total_comments, analysis_settings, statistics)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    session_data.get('video_id'),
                    session_data.get('total_comments'),
                    json.dumps(session_data.get('settings', {})),
                    json.dumps(session_data.get('statistics', {}))
                ))
                
                conn.commit()
                
                self.logger.info(f"Saved analysis session: {session_id}")
                return session_id
                
            except Exception as e:
                self.logger.error(f"Error saving analysis session: {str(e)}")
                raise
    
    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """Retrieve video information from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM videos WHERE video_id = ?', (video_id,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                video_info = dict(zip(columns, row))
                
                # Parse metadata
                if video_info.get('metadata'):
                    video_info['metadata'] = json.loads(video_info['metadata'])
                
                return video_info
            
            return None
    
    def get_comments(self, video_id: str) -> List[Dict]:
        """Retrieve comments for a video."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM comments WHERE video_id = ?', (video_id,))
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description]
            comments = [dict(zip(columns, row)) for row in rows]
            
            return comments
    
    def get_sentiment_results(self, video_id: str) -> List[Dict]:
        """Retrieve sentiment analysis results for a video."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT sr.*, c.text, c.author 
                FROM sentiment_results sr
                JOIN comments c ON sr.comment_id = c.comment_id
                WHERE c.video_id = ?
            ''', (video_id,))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in rows:
                result = dict(zip(columns, row))
                
                # Parse model results
                if result.get('model_results'):
                    result['model_results'] = json.loads(result['model_results'])
                
                results.append(result)
            
            return results
    
    def export_to_csv(self, video_id: str, output_path: Path) -> Path:
        """Export analysis results to CSV."""
        # Get complete data
        video_info = self.get_video_info(video_id)
        comments = self.get_comments(video_id)
        sentiment_results = self.get_sentiment_results(video_id)
        
        # Create DataFrame
        data = []
        for comment in comments:
            # Find corresponding sentiment result
            sentiment_result = next(
                (r for r in sentiment_results if r['comment_id'] == comment['comment_id']), 
                {}
            )
            
            row = {
                'video_id': video_id,
                'comment_id': comment.get('comment_id'),
                'author': comment.get('author'),
                'text': comment.get('text'),
                'likes': comment.get('likes'),
                'timestamp': comment.get('timestamp'),
                'is_reply': comment.get('is_reply'),
                'sentiment': sentiment_result.get('sentiment'),
                'confidence': sentiment_result.get('confidence'),
                'agreement_score': sentiment_result.get('agreement_score'),
                'analyzed_at': sentiment_result.get('analyzed_at')
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        self.logger.info(f"Exported data to CSV: {output_path}")
        return output_path
    
    def export_to_json(self, video_id: str, output_path: Path) -> Path:
        """Export analysis results to JSON."""
        video_info = self.get_video_info(video_id)
        comments = self.get_comments(video_id)
        sentiment_results = self.get_sentiment_results(video_id)
        
        export_data = {
            'video_info': video_info,
            'comments': comments,
            'sentiment_results': sentiment_results,
            'export_metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_comments': len(comments),
                'total_results': len(sentiment_results)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported data to JSON: {output_path}")
        return output_path
    
    def get_analysis_history(self) -> List[Dict]:
        """Get history of analysis sessions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT s.*, v.url, v.title
                FROM analysis_sessions s
                JOIN videos v ON s.video_id = v.video_id
                ORDER BY s.created_at DESC
            ''')
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            sessions = []
            
            for row in rows:
                session = dict(zip(columns, row))
                
                # Parse JSON fields
                if session.get('analysis_settings'):
                    session['analysis_settings'] = json.loads(session['analysis_settings'])
                if session.get('statistics'):
                    session['statistics'] = json.loads(session['statistics'])
                
                sessions.append(session)
            
            return sessions
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old data from database."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old sentiment results
            cursor.execute('DELETE FROM sentiment_results WHERE analyzed_at < ?', (cutoff_date,))
            deleted_results = cursor.rowcount
            
            # Delete old sessions
            cursor.execute('DELETE FROM analysis_sessions WHERE created_at < ?', (cutoff_date,))
            deleted_sessions = cursor.rowcount
            
            conn.commit()
            
            total_deleted = deleted_results + deleted_sessions
            self.logger.info(f"Cleaned up {total_deleted} old records")
            
            return total_deleted
