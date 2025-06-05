from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Optional, List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "YouTube Comment Sentiment Analysis"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    
    # Paths
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).parent.parent.absolute())
    DATA_DIR: Path = Field(default_factory=lambda: Path("data"))
    MODELS_DIR: Path = Field(default_factory=lambda: Path("models"))
    LOGS_DIR: Path = Field(default_factory=lambda: Path("logs"))
    
    # Scraping settings
    DEFAULT_COMMENT_LIMIT: int = Field(default=100, ge=1, le=10000)
    REQUEST_TIMEOUT: int = Field(default=30, ge=5, le=120)
    RATE_LIMIT_DELAY: float = Field(default=1.0, ge=0.1, le=5.0)
    MAX_RETRIES: int = Field(default=3, ge=1, le=10)
    
    # ML Model settings
    MODEL_NAME: str = Field(default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    BATCH_SIZE: int = Field(default=32, ge=1, le=256)
    MAX_LENGTH: int = Field(default=512, ge=50, le=1024)
    CONFIDENCE_THRESHOLD: float = Field(default=0.7, ge=0.5, le=1.0)
    
    # GUI settings
    WINDOW_WIDTH: int = Field(default=1200, ge=800, le=2000)
    WINDOW_HEIGHT: int = Field(default=800, ge=600, le=1200)
    THEME: str = Field(default="light", pattern="^(light|dark)$")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///data/comments.db")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/app.log"
      # Export settings
    EXPORT_FORMATS: List[str] = Field(default=["csv", "json", "xlsx"])
    DEFAULT_EXPORT_DIR: Path = Field(default_factory=lambda: Path("exports"))
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()
    
    @field_validator('DATA_DIR', 'MODELS_DIR', 'LOGS_DIR', 'DEFAULT_EXPORT_DIR')
    @classmethod
    def create_directories(cls, v):
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get the global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
