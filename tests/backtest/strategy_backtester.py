"""Strategy backtester that wraps existing strategy classes."""
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime

from core.strategies import STRATEGY_REGISTRY
from tests.backtest.backtest_engine import BacktestEngine
import logging

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Wrapper for backtesting strategies using existing strategy classes.
    
    This class bridges your existing strategy architecture with the backtesting engine.
    """
    
    def __init__(
        self,
        strategy_name: str,
        engine: BacktestEngine,
        strategy_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize strategy backtester.
        
        Args:
            strategy_name: Name from STRATEGY_REGISTRY (e.g., "TrendFollowing")
            engine: BacktestEngine instance
            strategy_config: Optional strategy configuration dict
        """
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(STRATEGY_REGISTRY.keys())}"
            )
        
        self.strategy_name = strategy_name
        self.strategy_class = STRATEGY_REGISTRY[strategy_name]
        self.engine = engine
        self.strategy_config = strategy_config or {}
        
        logger.info(f"StrategyBacktester initialized for {strategy_name}")
    
    def backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        per_symbol: bool = True,
        timeframe: str = "1Day"
    ) -> Dict[str, Any]:
        """
        Run backtest for strategy on given symbols.
        
        Args:
            symbols: List of stock symbols to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD, defaults to today)
            per_symbol: If True, returns results per symbol. If False, portfolio-level
            timeframe: Bar timeframe. Options: "1Day" (default), "1Hour", "15Min", "5Min", "1Min"
                      Use "1Hour" for intraday trading and better exit timing
        
        Returns:
            Dictionary with backtest results
        """
        # Fetch data with specified timeframe
        price_data = self.engine.fetch_data(symbols, start_date, end_date, timeframe=timeframe)
        
        if price_data.empty:
            raise ValueError("No data fetched for backtesting")
        
        print(f"\n📈 DATA LOADED ({timeframe} timeframe):")
        print(f"  Symbols: {list(price_data.columns)}")
        print(f"  Period: {price_data.index[0]} to {price_data.index[-1]}")
        print(f"  Total Bars: {len(price_data)}")
        if timeframe == "1Day":
            print(f"  Trading Days: {len(price_data)}")
        elif timeframe == "1Hour":
            days = (price_data.index[-1] - price_data.index[0]).days
            print(f"  Calendar Days: ~{days}")
            print(f"  Hours: {len(price_data)}")
        else:
            print(f"  Bars: {len(price_data)}")
        
        # Generate signals using existing strategy class
        entries, exits = self.engine.generate_signals_from_strategy(
            self.strategy_class,
            price_data,
            self.strategy_config
        )
        
        # DEBUG: Call debug method
        if hasattr(self.engine, 'debug_signals'):
            self.engine.debug_signals(price_data, entries, exits)
        
        # Run backtest
        portfolio = self.engine.run_backtest(price_data, entries, exits)
        
        # Analyze results
        if per_symbol:
            # Analyze each symbol separately
            # VectorBT Portfolio automatically handles per-symbol metrics
            results = {}
            for symbol in symbols:
                if symbol not in price_data.columns:
                    logger.warning(f"Symbol {symbol} not in price data, skipping")
                    continue
                
                # VectorBT Portfolio methods return Series when multi-symbol
                # The analyze_results method will extract symbol-specific metrics
                try:
                    symbol_results = self.engine.analyze_results(
                        portfolio=portfolio,
                        strategy_name=self.strategy_name,
                        symbol=symbol,
                        start_date=pd.to_datetime(start_date),
                        end_date=pd.to_datetime(end_date) if end_date else datetime.now(),
                        parameters=self.strategy_config,
                        price_data=price_data
                    )
                    results[symbol] = symbol_results
                except Exception as e:
                    logger.exception(f"Error analyzing {symbol}: {e}")
                    continue
            
            return results
        else:
            # Portfolio-level analysis (all symbols combined)
            # Note: VectorBT portfolio handles multi-symbol automatically
            combined_results = self.engine.analyze_results(
                portfolio=portfolio,
                strategy_name=self.strategy_name,
                symbol="PORTFOLIO",
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date) if end_date else datetime.now(),
                parameters=self.strategy_config,
                price_data=price_data
            )
            return {"PORTFOLIO": combined_results}

