"""
YouTube Comment Scraper
"""

import re
import time
import asyncio
from typing import List, Dict, Optional, Iterator, Callable

from config import get_logger, settings


class CommentScraper:
    def __init__(self):
        self.logger = get_logger(__name__)
        
    async def scrape_comments(self, video_url: str, limit: int = None, progress_callback=None):
        return []
