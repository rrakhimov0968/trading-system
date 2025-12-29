"""Market data models."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import pandas as pd


@dataclass
class Bar:
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str
    timeframe: str = "1Min"  # 1Min, 5Min, 15Min, 1Hour, 1Day, etc.


@dataclass
class Quote:
    """Latest quote data."""
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime


@dataclass
class Trade:
    """Trade tick data."""
    symbol: str
    price: float
    size: int
    timestamp: datetime
    conditions: Optional[List[str]] = None


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    bars: Optional[List[Bar]] = None
    quote: Optional[Quote] = None
    latest_trade: Optional[Trade] = None
    dataframe: Optional[pd.DataFrame] = None  # Pandas DataFrame for analysis
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert bars to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with OHLCV data
        """
        if self.dataframe is not None:
            return self.dataframe
        
        if not self.bars:
            return pd.DataFrame()
        
        data = {
            'timestamp': [bar.timestamp for bar in self.bars],
            'open': [bar.open for bar in self.bars],
            'high': [bar.high for bar in self.bars],
            'low': [bar.low for bar in self.bars],
            'close': [bar.close for bar in self.bars],
            'volume': [bar.volume for bar in self.bars]
        }
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        self.dataframe = df
        return df

