"""Technical analysis indicators for strategies."""
import pandas as pd
import numpy as np
from typing import Optional


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        close: Series of closing prices
        period: RSI period (default 14)
    
    Returns:
        Series of RSI values
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period (default 14)
    
    Returns:
        Series of ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    atr = true_range.rolling(period).mean()
    return atr


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        close: Series of closing prices
        period: Moving average period (default 20)
        num_std: Number of standard deviations (default 2.0)
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return upper, middle, lower


def calculate_donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = 20
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Donchian Channels (breakout indicator).
    
    Args:
        high: Series of high prices
        low: Series of low prices
        period: Lookback period (default 20)
    
    Returns:
        Tuple of (upper_channel, lower_channel)
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    
    return upper, lower


def calculate_momentum(close: pd.Series, period: int = 126) -> pd.Series:
    """
    Calculate momentum (percentage change over period).
    
    Args:
        close: Series of closing prices
        period: Momentum period (default 126 = ~6 months)
    
    Returns:
        Series of momentum values as percentages
    """
    return close.pct_change(periods=period) * 100

