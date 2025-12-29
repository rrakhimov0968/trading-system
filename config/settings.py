"""Configuration management for the trading system."""
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

# Load environment variables once at module level
load_dotenv()


class TradingMode(str, Enum):
    """Trading mode options."""
    PAPER = "paper"
    LIVE = "live"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str
    secret_key: str
    base_url: Optional[str] = None
    paper: bool = True
    
    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        """Load Alpaca config from environment variables."""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL")
        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        
        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment"
            )
        
        return cls(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
            paper=paper
        )


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str  # "openai", "anthropic", "groq"
    api_key: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    
    @classmethod
    def openai_from_env(cls) -> Optional["LLMConfig"]:
        """Load OpenAI config from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return cls(
            provider="openai",
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
        )
    
    @classmethod
    def anthropic_from_env(cls) -> Optional["LLMConfig"]:
        """Load Anthropic config from environment."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        return cls(
            provider="anthropic",
            api_key=api_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))
        )
    
    @classmethod
    def groq_from_env(cls) -> Optional["LLMConfig"]:
        """Load Groq config from environment."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        return cls(
            provider="groq",
            api_key=api_key,
            model=os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
            base_url=os.getenv("GROQ_BASE_URL")
        )


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    
    @classmethod
    def from_env(cls) -> Optional["DatabaseConfig"]:
        """Load database config from environment."""
        url = os.getenv("DATABASE_URL")
        if not url:
            return None
        return cls(
            url=url,
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
        )


@dataclass
class AppConfig:
    """Main application configuration."""
    trading_mode: TradingMode
    log_level: LogLevel
    alpaca: AlpacaConfig
    openai: Optional[LLMConfig] = None
    anthropic: Optional[LLMConfig] = None
    groq: Optional[LLMConfig] = None
    database: Optional[DatabaseConfig] = None
    correlation_id_header: str = "X-Correlation-ID"
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load all configuration from environment variables."""
        trading_mode = TradingMode(
            os.getenv("TRADING_MODE", "paper").lower()
        )
        log_level = LogLevel(
            os.getenv("LOG_LEVEL", "INFO").upper()
        )
        
        return cls(
            trading_mode=trading_mode,
            log_level=log_level,
            alpaca=AlpacaConfig.from_env(),
            openai=LLMConfig.openai_from_env(),
            anthropic=LLMConfig.anthropic_from_env(),
            groq=LLMConfig.groq_from_env(),
            database=DatabaseConfig.from_env()
        )


# Global config instance (lazy-loaded)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global application configuration."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config

