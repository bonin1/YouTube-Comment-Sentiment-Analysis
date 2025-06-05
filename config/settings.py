"""
Configuration Management for YouTube Comment Sentiment Analysis

This module provides comprehensive configuration management with environment variable support,
validation, and default values using Pydantic for type safety and validation.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support and validation."""
    
    # Project Paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent.absolute())
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    LOGS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    EXPORTS_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "exports")
    CACHE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "cache")
    
    # API Configuration
    YOUTUBE_API_KEY: Optional[str] = Field(None, description="YouTube Data API key")
    
    # Analysis Settings
    MAX_COMMENTS: int = Field(1000, ge=1, le=50000, description="Maximum number of comments to scrape")
    SENTIMENT_THRESHOLD: float = Field(0.1, ge=0.0, le=1.0, description="Sentiment classification threshold")
    CONFIDENCE_THRESHOLD: float = Field(0.7, ge=0.0, le=1.0, description="Model confidence threshold")
    BATCH_SIZE: int = Field(100, ge=1, le=1000, description="Processing batch size")
    
    # GUI Settings
    THEME: str = Field("modern", description="GUI theme")
    WINDOW_WIDTH: int = Field(1200, ge=800, le=2560, description="Default window width")
    WINDOW_HEIGHT: int = Field(800, ge=600, le=1440, description="Default window height")
    THEME_MODE: str = Field("light", description="Theme mode: light or dark")
    
    # Database Settings
    DATABASE_URL: str = Field("sqlite:///data/sentiment_analysis.db", description="Database connection URL")
    CACHE_ENABLED: bool = Field(True, description="Enable caching")
    CACHE_EXPIRY_HOURS: int = Field(24, ge=1, le=168, description="Cache expiry in hours")
    
    # Logging Configuration
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_FORMAT: str = Field("structured", description="Log format type")
    LOG_TO_FILE: bool = Field(True, description="Enable file logging")
    LOG_FILE_PATH: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "logs" / "app.log")
    
    # Text Processing Settings
    REMOVE_STOPWORDS: bool = Field(True, description="Remove stopwords during preprocessing")
    ENABLE_LEMMATIZATION: bool = Field(True, description="Enable lemmatization")
    HANDLE_EMOJIS: bool = Field(True, description="Process emojis in text")
    MIN_COMMENT_LENGTH: int = Field(5, ge=1, le=100, description="Minimum comment length")
    MAX_COMMENT_LENGTH: int = Field(1000, ge=100, le=10000, description="Maximum comment length")
    
    # Visualization Settings
    CHART_WIDTH: int = Field(800, ge=400, le=2000, description="Default chart width")
    CHART_HEIGHT: int = Field(600, ge=300, le=1500, description="Default chart height")
    COLOR_SCHEME: str = Field("viridis", description="Default color scheme")
    ENABLE_INTERACTIVE_PLOTS: bool = Field(True, description="Enable interactive plots")
    
    # Performance Settings
    USE_GPU: bool = Field(False, description="Use GPU for processing if available")
    NUM_WORKERS: int = Field(4, ge=1, le=16, description="Number of worker threads")
    CHUNK_SIZE: int = Field(1000, ge=100, le=10000, description="Data processing chunk size")
    TIMEOUT_SECONDS: int = Field(30, ge=5, le=300, description="Request timeout in seconds")
    
    # Export Settings
    DEFAULT_EXPORT_FORMAT: str = Field("csv", description="Default export format")
    INCLUDE_METADATA: bool = Field(True, description="Include metadata in exports")
    COMPRESS_EXPORTS: bool = Field(False, description="Compress export files")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    @validator('LOG_LEVEL')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of: {valid_levels}')
        return v.upper()
    
    @validator('THEME_MODE')
    def validate_theme_mode(cls, v):
        """Validate theme mode."""
        valid_modes = ['light', 'dark']
        if v.lower() not in valid_modes:
            raise ValueError(f'THEME_MODE must be one of: {valid_modes}')
        return v.lower()
    
    @validator('DEFAULT_EXPORT_FORMAT')
    def validate_export_format(cls, v):
        """Validate export format."""
        valid_formats = ['csv', 'json', 'xlsx', 'html']
        if v.lower() not in valid_formats:
            raise ValueError(f'DEFAULT_EXPORT_FORMAT must be one of: {valid_formats}')
        return v.lower()
    
    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.LOGS_DIR,
            self.EXPORTS_DIR,
            self.CACHE_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_database_path(self) -> Path:
        """Get the resolved database path."""
        if self.DATABASE_URL.startswith('sqlite:///'):
            db_path = self.DATABASE_URL.replace('sqlite:///', '')
            return self.PROJECT_ROOT / db_path
        return Path(self.DATABASE_URL)
    
    def get_log_file_path(self) -> Path:
        """Get the resolved log file path."""
        return self.LOGS_DIR / "app.log"


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


# Export commonly used settings for convenience
MAX_COMMENTS = settings.MAX_COMMENTS
SENTIMENT_THRESHOLD = settings.SENTIMENT_THRESHOLD
CONFIDENCE_THRESHOLD = settings.CONFIDENCE_THRESHOLD
BATCH_SIZE = settings.BATCH_SIZE
LOG_LEVEL = settings.LOG_LEVEL
THEME = settings.THEME
WINDOW_WIDTH = settings.WINDOW_WIDTH
WINDOW_HEIGHT = settings.WINDOW_HEIGHT
