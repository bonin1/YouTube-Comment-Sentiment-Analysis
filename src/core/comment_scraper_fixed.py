import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from youtube_comment_downloader import YoutubeCommentDownloader
from dataclasses import dataclass
from datetime import datetime
import time

from ..utils.validators import validate_youtube_url, extract_video_id
from ..utils.decorators import timing
from ..config.settings import get_settings

@dataclass
class Comment:
    """Data class for YouTube comment"""
    id: str
    text: str
    author: str
    likes: int
    reply_count: int
    time_parsed: datetime
    is_reply: bool = False
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert comment to dictionary"""
        return {
            'id': self.id,
            'text': self.text,
            'author': self.author,
            'likes': self.likes,
            'reply_count': self.reply_count,
            'time_parsed': self.time_parsed.isoformat(),
            'is_reply': self.is_reply,
            'parent_id': self.parent_id
        }

class YouTubeCommentScraper:
    """Advanced YouTube comment scraper with rate limiting and error handling"""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize the scraper
        
        Args:
            progress_callback: Optional callback for progress updates
        """
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.downloader = YoutubeCommentDownloader()
        self.progress_callback = progress_callback
        
        # Rate limiting
        self.last_request_time = 0
    
    @timing
    def validate_video(self, url: str) -> bool:
        """
        Validate if the video exists and has comments enabled
        
        Args:
            url: YouTube video URL
            
        Returns:
            True if video is valid and accessible
        """
        if not validate_youtube_url(url):
            raise ValueError("Invalid YouTube URL format")
        
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID")
        
        try:
            # Try to get just one comment to test accessibility
            self.logger.info(f"Validating video: {video_id}")
            comments = self.downloader.get_comments(video_id)
            first_comment = next(comments)  # Try to get first comment
            self.logger.info(f"Video validation successful. Found comment: {first_comment.get('text', '')[:50]}...")
            return True
        except StopIteration:
            self.logger.warning(f"Video {video_id} has no comments")
            return False
        except Exception as e:
            self.logger.error(f"Video validation failed: {e}")
            # Don't fail validation completely - some videos might have restricted comments but still be valid
            return True
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.settings.RATE_LIMIT_DELAY:
            sleep_time = self.settings.RATE_LIMIT_DELAY - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @timing
    async def scrape_comments(
        self,
        url: str,
        limit: int = 100,
        include_replies: bool = False
    ) -> List[Comment]:
        """
        Scrape comments from YouTube video
        
        Args:
            url: YouTube video URL
            limit: Maximum number of comments to scrape
            include_replies: Whether to include reply comments
            
        Returns:
            List of Comment objects
        """
        if not self.validate_video(url):
            raise ValueError("Invalid or inaccessible video")
        
        video_id = extract_video_id(url)
        comments_list = []
        
        self.logger.info(f"Starting to scrape {limit} comments from video {video_id}")
        
        try:
            comments_generator = self.downloader.get_comments(video_id)
            
            processed_count = 0
            
            for comment_data in comments_generator:
                if processed_count >= limit:
                    break
                
                # Apply rate limiting
                self._rate_limit()
                
                # Parse comment data
                comment = self._parse_comment(comment_data)
                if comment:
                    comments_list.append(comment)
                    processed_count += 1
                    
                    # Update progress
                    if self.progress_callback:
                        progress = processed_count / limit
                        await self.progress_callback(progress, f"Scraped {processed_count}/{limit} comments")
                
                # Handle replies if requested
                if include_replies and 'replies' in comment_data:
                    reply_comments = self._process_replies(comment_data['replies'], comment.id)
                    comments_list.extend(reply_comments)
                
                # Small delay between comments
                await asyncio.sleep(0.1)
        
        except Exception as e:
            self.logger.error(f"Error scraping comments: {e}")
            raise
        
        self.logger.info(f"Successfully scraped {len(comments_list)} comments")
        return comments_list
    
    def _parse_comment(self, comment_data: Dict[str, Any]) -> Optional[Comment]:
        """
        Parse raw comment data into Comment object
        
        Args:
            comment_data: Raw comment data from YouTube API
            
        Returns:
            Comment object or None if parsing fails
        """
        try:
            return Comment(
                id=comment_data.get('cid', ''),
                text=comment_data.get('text', ''),
                author=comment_data.get('author', ''),
                likes=int(comment_data.get('votes', 0)),
                reply_count=int(comment_data.get('replies', 0)),
                time_parsed=datetime.now(),
                is_reply=False
            )
        except Exception as e:
            self.logger.warning(f"Failed to parse comment: {e}")
            return None
    
    def _process_replies(self, replies_data: List[Dict], parent_id: str) -> List[Comment]:
        """
        Process reply comments
        
        Args:
            replies_data: List of reply comment data
            parent_id: Parent comment ID
            
        Returns:
            List of reply Comment objects
        """
        reply_comments = []
        
        for reply_data in replies_data:
            try:
                reply = Comment(
                    id=reply_data.get('cid', ''),
                    text=reply_data.get('text', ''),
                    author=reply_data.get('author', ''),
                    likes=int(reply_data.get('votes', 0)),
                    reply_count=0,  # Replies to replies are not typically handled
                    time_parsed=datetime.now(),
                    is_reply=True,
                    parent_id=parent_id
                )
                reply_comments.append(reply)
            except Exception as e:
                self.logger.warning(f"Failed to parse reply: {e}")
                continue
        
        return reply_comments
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get basic video information
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with video information
        """
        video_id = extract_video_id(url)
        
        return {
            'video_id': video_id,
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'status': 'ready' if self.validate_video(url) else 'unavailable'
        }
