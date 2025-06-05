"""
Logging Configuration for YouTube Comment Sentiment Analysis

This module provides comprehensive logging setup with multiple handlers,
formatters, and integration with the GUI application.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

from .settings import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_data['extra_' + key] = value
        
        return json.dumps(log_data, default=str)


class GUIHandler(logging.Handler):
    """Custom logging handler for GUI integration."""
    
    def __init__(self, gui_callback=None):
        """Initialize GUI handler."""
        super().__init__()
        self.gui_callback = gui_callback
        self.logs = []
        
    def emit(self, record: logging.LogRecord):
        """Emit log record to GUI."""
        try:
            msg = self.format(record)
            self.logs.append({
                'timestamp': datetime.fromtimestamp(record.created),
                'level': record.levelname,
                'message': record.getMessage(),
                'logger': record.name
            })
            
            # Keep only last 1000 logs to prevent memory issues
            if len(self.logs) > 1000:
                self.logs = self.logs[-500:]
            
            if self.gui_callback:
                self.gui_callback(record.levelname, record.getMessage())
                
        except Exception:
            self.handleError(record)
    
    def get_logs(self) -> list:
        """Get stored logs."""
        return self.logs.copy()


class LoggerManager:
    """Centralized logger management."""
    
    def __init__(self):
        """Initialize logger manager."""
        self.loggers: Dict[str, logging.Logger] = {}
        self.gui_handler: Optional[GUIHandler] = None
        self.setup_complete = False
    
    def setup_logging(self, 
                     level: str = None, 
                     console_output: bool = True,
                     file_output: bool = None,
                     gui_callback=None) -> None:
        """Setup comprehensive logging configuration."""
        
        # Use settings defaults if not provided
        if level is None:
            level = settings.LOG_LEVEL
        if file_output is None:
            file_output = settings.LOG_TO_FILE
            
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        if settings.LOG_FORMAT == 'structured':
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if file_output:
            log_file = settings.get_log_file_path()
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # GUI handler
        if gui_callback:
            self.gui_handler = GUIHandler(gui_callback)
            self.gui_handler.setLevel(logging.INFO)
            # Use simple formatter for GUI
            gui_formatter = logging.Formatter('%(levelname)s - %(message)s')
            self.gui_handler.setFormatter(gui_formatter)
            root_logger.addHandler(self.gui_handler)
        
        self.setup_complete = True
        
        # Log setup completion
        logger = self.get_logger(__name__)
        logger.info(f"Logging setup complete - Level: {level}, File: {file_output}, Console: {console_output}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the specified name."""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
            
            # Ensure logging is setup
            if not self.setup_complete:
                self.setup_logging()
                
        return self.loggers[name]
    
    def get_gui_logs(self) -> list:
        """Get logs from GUI handler."""
        if self.gui_handler:
            return self.gui_handler.get_logs()
        return []
    
    def set_level(self, level: str) -> None:
        """Set logging level for all loggers."""
        log_level = getattr(logging, level.upper())
        
        # Update root logger
        logging.getLogger().setLevel(log_level)
        
        # Update all handlers
        for handler in logging.getLogger().handlers:
            handler.setLevel(log_level)
        
        logger = self.get_logger(__name__)
        logger.info(f"Logging level changed to {level}")


# Global logger manager instance
_logger_manager = LoggerManager()


def setup_logging(level: str = None, 
                 console_output: bool = True,
                 file_output: bool = None,
                 gui_callback=None) -> None:
    """Setup logging configuration."""
    _logger_manager.setup_logging(level, console_output, file_output, gui_callback)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return _logger_manager.get_logger(name)


def get_gui_logs() -> list:
    """Get GUI logs."""
    return _logger_manager.get_gui_logs()


def set_log_level(level: str) -> None:
    """Set logging level."""
    _logger_manager.set_level(level)
