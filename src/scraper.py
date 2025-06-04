"""
YouTube comment scraping functionality with multiple methods.
"""
import time
import random
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from src.utils import extract_video_id, clean_text, validate_comment, ProgressTracker
from config.settings import (
    MAX_COMMENTS, TIMEOUT, MAX_RETRIES, SELENIUM_HEADLESS,
    SELENIUM_IMPLICIT_WAIT, SELENIUM_PAGE_LOAD_TIMEOUT
)

logger = logging.getLogger(__name__)

@dataclass
class Comment:
    """Data class for YouTube comment."""
    text: str
    author: str
    likes: int
    timestamp: Optional[str] = None
    replies: int = 0
    is_reply: bool = False
    video_id: Optional[str] = None

class YouTubeCommentScraper:
    """Main scraper class for YouTube comments."""
    
    def __init__(self, method: str = 'selenium', headless: bool = True):
        """
        Initialize the scraper.
        
        Args:
            method: Scraping method ('selenium' or 'requests')
            headless: Run browser in headless mode
        """
        self.method = method.lower()
        self.headless = headless
        self.driver = None
        
        if self.method == 'selenium':
            self._setup_selenium()
    
    def _setup_selenium(self) -> None:
        """Setup Selenium WebDriver."""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument('--headless')
            
            # Performance and stability options
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-logging')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            # Install and setup ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Set timeouts
            self.driver.implicitly_wait(SELENIUM_IMPLICIT_WAIT)
            self.driver.set_page_load_timeout(SELENIUM_PAGE_LOAD_TIMEOUT)
            
            logger.info("Selenium WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Selenium: {e}")
            raise
    
    def scrape_comments(
        self,
        video_url: str,
        max_comments: int = MAX_COMMENTS,
        include_replies: bool = False,
        sort_by: str = 'top'
    ) -> List[Comment]:
        """
        Scrape comments from a YouTube video.
        
        Args:
            video_url: YouTube video URL
            max_comments: Maximum number of comments to scrape
            include_replies: Include comment replies
            sort_by: Sort order ('top' or 'new')
            
        Returns:
            List of Comment objects
        """
        video_id = extract_video_id(video_url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {video_url}")
        
        logger.info(f"Scraping comments for video ID: {video_id}")
        logger.info(f"Method: {self.method}, Max comments: {max_comments}")
        
        if self.method == 'selenium':
            return self._scrape_with_selenium(video_url, video_id, max_comments, include_replies, sort_by)
        else:
            return self._scrape_with_requests(video_url, video_id, max_comments)
    
    def _scrape_with_selenium(
        self,
        video_url: str,
        video_id: str,
        max_comments: int,
        include_replies: bool,
        sort_by: str
    ) -> List[Comment]:
        """Scrape comments using Selenium."""
        comments = []
        
        try:
            # Navigate to video
            self.driver.get(video_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "ytd-comment-thread-renderer"))
            )
            
            # Scroll to comments section
            self._scroll_to_comments()
            
            # Set sort order if needed
            if sort_by == 'new':
                self._set_sort_order()
            
            # Progress tracker
            progress = ProgressTracker(max_comments, "Scraping comments")
            
            # Scroll and collect comments
            last_comment_count = 0
            scroll_attempts = 0
            max_scroll_attempts = 50
            
            while len(comments) < max_comments and scroll_attempts < max_scroll_attempts:
                # Get current comments
                comment_elements = self.driver.find_elements(
                    By.CSS_SELECTOR, "ytd-comment-thread-renderer"
                )
                
                # Extract new comments
                for element in comment_elements[last_comment_count:]:
                    if len(comments) >= max_comments:
                        break
                    
                    comment = self._extract_comment_from_element(element, video_id)
                    if comment and validate_comment(comment.text):
                        comments.append(comment)
                        progress.update()
                
                # Check if we got new comments
                if len(comment_elements) == last_comment_count:
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0
                    last_comment_count = len(comment_elements)
                
                # Scroll down to load more comments
                self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(random.uniform(1, 2))  # Random delay to avoid detection
            
            progress.finish()
            logger.info(f"Successfully scraped {len(comments)} comments")
            
        except TimeoutException:
            logger.warning("Timeout waiting for comments to load")
        except Exception as e:
            logger.error(f"Error scraping with Selenium: {e}")
            raise
        
        return comments[:max_comments]
    
    def _scrape_with_requests(self, video_url: str, video_id: str, max_comments: int) -> List[Comment]:
        """Scrape comments using requests (basic method)."""
        comments = []
        
        try:
            # This is a simplified implementation
            # In practice, you might need to handle YouTube's API or use youtube-comment-scraper package
            logger.warning("Requests-based scraping is limited. Consider using Selenium or YouTube API.")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(video_url, headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract comments from page source (limited)
            # This is a basic implementation - real scraping would require more sophisticated parsing
            comment_scripts = soup.find_all('script')
            
            for script in comment_scripts:
                if script.string and 'commentRenderer' in script.string:
                    # Basic comment extraction (would need more sophisticated parsing)
                    logger.info("Found comment data in page source")
                    break
            
            logger.warning(f"Requests method returned {len(comments)} comments (limited implementation)")
            
        except Exception as e:
            logger.error(f"Error scraping with requests: {e}")
            raise
        
        return comments
    
    def _scroll_to_comments(self) -> None:
        """Scroll to comments section."""
        try:
            # Scroll down to trigger comment loading
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
                time.sleep(2)
            
            # Wait for comments to appear
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-comment-thread-renderer"))
            )
            
        except TimeoutException:
            logger.warning("Comments section not found or took too long to load")
    
    def _set_sort_order(self) -> None:
        """Set comment sort order to newest first."""
        try:
            # Click on sort dropdown
            sort_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[aria-label*='Sort']"))
            )
            sort_button.click()
            
            # Select "Newest first"
            newest_option = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//tp-yt-paper-listbox//a[contains(text(), 'Newest')]"))
            )
            newest_option.click()
            
            time.sleep(2)  # Wait for comments to reload
            
        except (TimeoutException, NoSuchElementException):
            logger.warning("Could not change sort order")
    
    def _extract_comment_from_element(self, element, video_id: str) -> Optional[Comment]:
        """Extract comment data from a web element."""
        try:
            # Extract comment text
            text_element = element.find_element(By.CSS_SELECTOR, "#content-text")
            text = clean_text(text_element.text.strip())
            
            if not text:
                return None
            
            # Extract author
            try:
                author_element = element.find_element(By.CSS_SELECTOR, "#author-text")
                author = author_element.text.strip()
            except:
                author = "Unknown"
            
            # Extract likes
            try:
                likes_element = element.find_element(By.CSS_SELECTOR, "#vote-count-middle")
                likes_text = likes_element.text.strip()
                likes = self._parse_like_count(likes_text)
            except:
                likes = 0
            
            # Extract timestamp
            try:
                time_element = element.find_element(By.CSS_SELECTOR, ".published-time-text a")
                timestamp = time_element.text.strip()
            except:
                timestamp = None
            
            # Extract reply count
            try:
                replies_element = element.find_element(By.CSS_SELECTOR, "#more-replies")
                replies_text = replies_element.text
                replies = int(''.join(filter(str.isdigit, replies_text))) if replies_text else 0
            except:
                replies = 0
            
            return Comment(
                text=text,
                author=author,
                likes=likes,
                timestamp=timestamp,
                replies=replies,
                video_id=video_id
            )
            
        except Exception as e:
            logger.debug(f"Error extracting comment: {e}")
            return None
    
    def _parse_like_count(self, likes_text: str) -> int:
        """Parse like count from text."""
        if not likes_text:
            return 0
        
        likes_text = likes_text.lower().replace(',', '')
        
        if 'k' in likes_text:
            return int(float(likes_text.replace('k', '')) * 1000)
        elif 'm' in likes_text:
            return int(float(likes_text.replace('m', '')) * 1000000)
        else:
            try:
                return int(''.join(filter(str.isdigit, likes_text)))
            except:
                return 0
    
    def close(self) -> None:
        """Close the scraper and clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.warning(f"Error closing WebDriver: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
