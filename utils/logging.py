"""Enhanced logging configuration for the trading system."""
import logging
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from config.settings import AppConfig, LogLevel


class CorrelationIDFilter(logging.Filter):
    """Log filter to add correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to log record if present in extra."""
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = '-'
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id') and record.correlation_id != '-':
            log_data["correlation_id"] = record.correlation_id
        
        # Add any extra fields from the log call
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName', 'relativeCreated',
                'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                'correlation_id', 'asctime'
            ]:
                if isinstance(value, (str, int, float, bool, type(None))):
                    log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(config: Optional[AppConfig] = None) -> None:
    """
    Set up logging for the trading system.
    
    Args:
        config: Application configuration. If None, creates a basic config.
    """
    if config is None:
        # Create basic config if not provided
        log_level = LogLevel.INFO
        use_json = False
    else:
        log_level = config.log_level
        use_json = os.getenv("LOG_FORMAT", "text").lower() == "json"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.value)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level.value)
    
    # Add correlation ID filter
    correlation_filter = CorrelationIDFilter()
    console_handler.addFilter(correlation_filter)
    
    # Set formatter
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set levels for third-party libraries
    logging.getLogger("alpaca").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

