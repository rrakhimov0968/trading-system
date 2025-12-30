"""Configuration management for the trading system."""
import os
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
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
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),  # Current working Claude model
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
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),  # Current working Groq model
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
        # Default to SQLite if no DATABASE_URL is set
        if not url:
            # Use default SQLite database in project root
            default_db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trading_system.db"
            )
            url = f"sqlite:///{default_db_path}"
            logger = logging.getLogger(__name__)
            logger.info(f"Using default SQLite database: {url}")
        
        return cls(
            url=url,
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
        )


class DataProvider(str, Enum):
    """Data provider options."""
    ALPACA = "alpaca"
    POLYGON = "polygon"
    YAHOO = "yahoo"  # yfinance


@dataclass
class DataProviderConfig:
    """Data provider configuration."""
    provider: DataProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_per_minute: int = 200  # Default rate limit
    cache_ttl_seconds: int = 60  # Cache TTL in seconds
    
    @classmethod
    def alpaca_from_env(cls) -> Optional["DataProviderConfig"]:
        """Load Alpaca data config from environment."""
        # Reuse Alpaca trading credentials if available
        api_key = os.getenv("ALPACA_API_KEY")
        if not api_key:
            return None
        return cls(
            provider=DataProvider.ALPACA,
            api_key=api_key,
            base_url=os.getenv("ALPACA_DATA_BASE_URL"),
            rate_limit_per_minute=int(os.getenv("ALPACA_DATA_RATE_LIMIT", "200")),
            cache_ttl_seconds=int(os.getenv("ALPACA_DATA_CACHE_TTL", "60"))
        )
    
    @classmethod
    def polygon_from_env(cls) -> Optional["DataProviderConfig"]:
        """Load Polygon config from environment."""
        api_key = os.getenv("POLYGON_API_KEY")
        if not api_key:
            return None
        return cls(
            provider=DataProvider.POLYGON,
            api_key=api_key,
            base_url=os.getenv("POLYGON_BASE_URL", "https://api.polygon.io"),
            rate_limit_per_minute=int(os.getenv("POLYGON_RATE_LIMIT", "5")),  # Free tier limit
            cache_ttl_seconds=int(os.getenv("POLYGON_CACHE_TTL", "60"))
        )
    
    @classmethod
    def yahoo_from_env(cls) -> "DataProviderConfig":
        """Load Yahoo Finance config (no API key needed)."""
        return cls(
            provider=DataProvider.YAHOO,
            rate_limit_per_minute=int(os.getenv("YAHOO_RATE_LIMIT", "10")),  # Conservative
            cache_ttl_seconds=int(os.getenv("YAHOO_CACHE_TTL", "300"))  # 5 minutes
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
    data_provider: Optional[DataProviderConfig] = None
    correlation_id_header: str = "X-Correlation-ID"
    # Orchestration settings
    loop_interval_seconds: int = 60  # How often to run the main loop
    symbols: List[str] = None  # Symbols to monitor
    # Quant agent settings
    quant_min_sharpe: float = 1.5  # Minimum Sharpe ratio threshold
    quant_max_drawdown: float = 0.08  # Maximum drawdown threshold (8%)
    quant_max_vif: float = 10.0  # Maximum VIF for multicollinearity check
    quant_use_llm: bool = False  # Enable LLM review for quant analysis
    # Risk agent settings
    risk_max_per_trade: float = 0.02  # Maximum risk per trade (2%)
    risk_max_daily_loss: float = 0.05  # Maximum daily loss (5%)
    risk_min_confidence: float = 0.3  # Minimum confidence to approve signal
    risk_max_qty: int = 1000  # Maximum shares per trade
    risk_default_account_balance: float = 10000.0  # Default if can't fetch from broker
    risk_use_llm_advisor: bool = False  # Enable LLM advisor for explanations
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load all configuration from environment variables."""
        trading_mode = TradingMode(
            os.getenv("TRADING_MODE", "paper").lower()
        )
        log_level = LogLevel(
            os.getenv("LOG_LEVEL", "INFO").upper()
        )
        
        # Determine data provider (priority: Polygon > Alpaca > Yahoo)
        data_provider = (
            DataProviderConfig.polygon_from_env() or
            DataProviderConfig.alpaca_from_env() or
            DataProviderConfig.yahoo_from_env()
        )
        
        # Parse symbols list
        symbols_str = os.getenv("SYMBOLS", "AAPL,MSFT,GOOGL")
        symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
        
        return cls(
            trading_mode=trading_mode,
            log_level=log_level,
            alpaca=AlpacaConfig.from_env(),
            openai=LLMConfig.openai_from_env(),
            anthropic=LLMConfig.anthropic_from_env(),
            groq=LLMConfig.groq_from_env(),
            database=DatabaseConfig.from_env(),
            data_provider=data_provider,
            loop_interval_seconds=int(os.getenv("LOOP_INTERVAL_SECONDS", "60")),
            symbols=symbols,
            quant_min_sharpe=float(os.getenv("QUANT_MIN_SHARPE", "1.5")),
            quant_max_drawdown=float(os.getenv("QUANT_MAX_DRAWDOWN", "0.08")),
            quant_max_vif=float(os.getenv("QUANT_MAX_VIF", "10.0")),
            quant_use_llm=os.getenv("QUANT_USE_LLM", "false").lower() == "true",
            risk_max_per_trade=float(os.getenv("RISK_MAX_PER_TRADE", "0.02")),
            risk_max_daily_loss=float(os.getenv("RISK_MAX_DAILY_LOSS", "0.05")),
            risk_min_confidence=float(os.getenv("RISK_MIN_CONFIDENCE", "0.3")),
            risk_max_qty=int(os.getenv("RISK_MAX_QTY", "1000")),
            risk_default_account_balance=float(os.getenv("RISK_DEFAULT_ACCOUNT_BALANCE", "10000.0")),
            risk_use_llm_advisor=os.getenv("RISK_USE_LLM_ADVISOR", "false").lower() == "true"
        )


# Global config instance (lazy-loaded)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global application configuration."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config

