"""Enhanced logging configuration for the trading system."""
import logging
import json
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
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
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create file handler with daily rotation
    log_filename = os.path.join(logs_dir, f"trading_system_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=30,  # Keep 30 days of logs
        encoding='utf-8'
    )
    file_handler.setLevel(log_level.value)
    file_handler.addFilter(correlation_filter)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Also create a general log file (all logs, no rotation by date)
    general_log_file = os.path.join(logs_dir, "trading_system.log")
    general_file_handler = RotatingFileHandler(
        general_log_file,
        maxBytes=50 * 1024 * 1024,  # 50 MB per file
        backupCount=5,  # Keep 5 backup files
        encoding='utf-8'
    )
    general_file_handler.setLevel(log_level.value)
    general_file_handler.addFilter(correlation_filter)
    general_file_handler.setFormatter(formatter)
    root_logger.addHandler(general_file_handler)
    
    # Log where logs are being written
    logging.info(f"Logging to console and files in: {logs_dir}")
    
    # Set levels for third-party libraries
    logging.getLogger("alpaca").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

