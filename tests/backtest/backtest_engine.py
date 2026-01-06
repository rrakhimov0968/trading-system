"""Core backtesting engine using VectorBT."""
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import warnings

from config.settings import AppConfig
from utils.database import DatabaseManager, BacktestResults

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Core backtesting engine that integrates with existing strategy classes.
    
    This engine uses VectorBT for fast vectorized backtesting while maintaining
    compatibility with the existing strategy architecture.
    """
    
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        initial_cash: float = 10000.0,
        commission: float = 0.001,  # 0.1% per trade
        risk_free_rate: float = 0.04,  # 4% annual risk-free rate
        database_manager: Optional[DatabaseManager] = None
    ):
        """
        Initialize backtest engine.
        
        Args:
            config: Application configuration (for thresholds)
            initial_cash: Initial capital
            commission: Commission rate (0.001 = 0.1%)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            database_manager: Optional database manager for persisting results
        """
        self.config = config
        self.initial_cash = initial_cash
        self.commission = commission
        self.risk_free_rate = risk_free_rate
        self.db_manager = database_manager
        
        # Load thresholds from config or use defaults
        if config:
            self.min_sharpe = config.quant_min_sharpe
            self.max_drawdown_threshold = config.quant_max_drawdown
        else:
            self.min_sharpe = float(os.getenv("QUANT_MIN_SHARPE", "1.5"))
            self.max_drawdown_threshold = float(os.getenv("QUANT_MAX_DRAWDOWN", "0.08"))
        
        self.min_win_rate = 0.45  # 45% minimum win rate
        
        logger.info(
            f"BacktestEngine initialized: "
            f"Initial Cash=${initial_cash:,.2f}, "
            f"Commission={commission:.3%}, "
            f"Risk-Free Rate={risk_free_rate:.2%}, "
            f"Min Sharpe={self.min_sharpe}, "
            f"Max DD={self.max_drawdown_threshold:.1%}"
        )
    
    def fetch_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        timeframe: str = "1Day"
    ) -> pd.DataFrame:
        """
        Fetch historical price data using yfinance via VectorBT.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime, defaults to today)
            timeframe: Bar timeframe (currently only supports "1Day")
        
        Returns:
            DataFrame with Close prices (symbols as columns, dates as index)
        
        Raises:
            ValueError: If data fetching fails
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        try:
            # Use existing DataAgent which has working yfinance integration
            # This is more reliable than calling yfinance directly
            from agents.data_agent import DataAgent
            from config.settings import DataProviderConfig
            
            logger.info(f"Fetching data using DataAgent for {len(symbols)} symbols...")
            
            # Initialize DataAgent with Yahoo provider
            yahoo_config = DataProviderConfig.yahoo_from_env()
            from config.settings import AppConfig
            
            # Create minimal config for DataAgent
            if self.config:
                data_config = self.config
            else:
                # Create minimal config
                import os
                from config.settings import TradingMode, LogLevel
                data_config = AppConfig(
                    trading_mode=TradingMode.PAPER,
                    log_level=LogLevel.INFO,
                    data_provider=yahoo_config
                )
            
            data_agent = DataAgent(config=data_config)
            
            # For backtesting, we need to fetch historical data with lenient validation
            # DataAgent's process() doesn't support strict_validation parameter,
            # so we call _fetch_data() directly for each symbol with strict_validation=False
            from datetime import datetime as dt
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) if end_date else dt.now()
            
            # Calculate limit based on date range (rough estimate: 252 trading days per year)
            days_diff = (end_dt - start_dt).days
            limit = min(int(days_diff * 1.5), 1000)  # Add buffer, cap at 1000
            
            # Fetch each symbol individually with strict_validation=False for backtesting
            # This allows stale historical data which is normal for backtesting
            data_frames = {}
            for symbol in symbols:
                try:
                    # Use _fetch_data() directly with strict_validation=False
                    # This allows historical/stale data which is expected for backtesting
                    market_data = data_agent._fetch_data(
                        symbol=symbol,
                        timeframe="1Day",
                        start_date=start_dt,
                        end_date=end_dt,
                        limit=limit,
                        strict_validation=False  # Critical: allow stale data for backtesting
                    )
                    
                    if market_data and market_data.bars:
                        df = market_data.to_dataframe()
                        if not df.empty:
                            data_frames[symbol] = df
                            logger.debug(f"Fetched {len(df)} bars for {symbol} via DataAgent (strict_validation=False)")
                        else:
                            logger.warning(f"Empty DataFrame for {symbol}")
                    else:
                        logger.warning(f"No bars returned for {symbol}")
                except Exception as symbol_error:
                    logger.warning(f"Failed to fetch {symbol}: {symbol_error}")
                    continue
            
            if not data_frames:
                raise ValueError("No data fetched for any symbol")
            
            # Combine all symbols into a single DataFrame with MultiIndex columns
            # First, get all unique dates across all symbols
            all_dates = set()
            for df in data_frames.values():
                all_dates.update(df.index)
            all_dates = sorted(all_dates)
            
            # Build MultiIndex DataFrame structure
            price_data_dict = {}
            ohlc_dict = {metric.lower(): {} for metric in ['Open', 'High', 'Low', 'Close', 'Volume']}
            
            for symbol, df in data_frames.items():
                # Reindex to align all dates, forward-fill missing values
                reindexed = df.reindex(all_dates, method='ffill')
                
                # DataAgent returns lowercase column names: 'open', 'high', 'low', 'close', 'volume'
                # Handle both lowercase and uppercase
                close_col = 'close' if 'close' in reindexed.columns else 'Close'
                
                # Extract Close prices for portfolio simulation
                price_data_dict[symbol] = reindexed[close_col]
                
                # Extract OHLC for strategies
                # Map both lowercase (DataAgent) and uppercase (yfinance direct) column names
                column_map = {
                    'open': 'open' if 'open' in reindexed.columns else 'Open',
                    'high': 'high' if 'high' in reindexed.columns else 'High',
                    'low': 'low' if 'low' in reindexed.columns else 'Low',
                    'close': 'close' if 'close' in reindexed.columns else 'Close',
                    'volume': 'volume' if 'volume' in reindexed.columns else 'Volume'
                }
                
                for metric_lower, metric_col in column_map.items():
                    if metric_col in reindexed.columns:
                        ohlc_dict[metric_lower][symbol] = reindexed[metric_col]
            
            # Create price_data DataFrame (Close prices only)
            price_data = pd.DataFrame(price_data_dict)
            
            # Create OHLC data dictionaries
            self.ohlc_data = {
                metric: pd.DataFrame(dict_data) for metric, dict_data in ohlc_dict.items()
                if dict_data
            }
            
            # Handle data gaps: forward-fill and drop NaNs
            price_data = price_data.ffill().dropna(axis=1)  # Drop columns (symbols) with all NaN
            
            # Clean OHLC data
            for metric in self.ohlc_data.keys():
                self.ohlc_data[metric] = self.ohlc_data[metric].ffill().dropna(axis=1)
            
            logger.info(f"✅ Fetched {len(price_data)} days of data for {len(price_data.columns)} symbols")
            
            if price_data.empty:
                raise ValueError("No data returned from yfinance")
            
            return price_data
            
        except Exception as e:
            logger.exception(f"Error fetching data: {e}")
            raise ValueError(f"Failed to fetch data: {str(e)}")
    
    def generate_signals_from_strategy(
        self,
        strategy_class,
        price_data: pd.DataFrame,
        strategy_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate entry/exit signals using an existing strategy class.
        
        Fixed version that properly handles strategy state and signal conversion.
        
        Args:
            strategy_class: Strategy class from core.strategies (e.g., TrendFollowing)
            price_data: DataFrame with Close prices
            strategy_config: Optional strategy configuration dict
        
        Returns:
            Tuple of (entries, exits) DataFrames (True/False values)
        """
        from models.market_data import MarketData, Bar
        from models.signal import SignalAction
        
        # Initialize DataFrames with False values
        entries = pd.DataFrame(False, index=price_data.index, columns=price_data.columns)
        exits = pd.DataFrame(False, index=price_data.index, columns=price_data.columns)
        
        # Initialize strategy once per symbol
        for symbol in price_data.columns:
            logger.info(f"Generating signals for {symbol}...")
            
            # Get symbol data
            symbol_close = price_data[symbol].dropna()
            
            if len(symbol_close) < 30:  # Minimum bars needed
                logger.warning(f"Insufficient data for {symbol}: {len(symbol_close)} bars")
                continue
            
            # Create MarketData object with all bars
            bars = []
            for idx, close_price in symbol_close.items():
                timestamp = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
                
                # Try to get OHLC data if available
                if hasattr(self, 'ohlc_data') and self.ohlc_data:
                    try:
                        # Get OHLC values
                        close_val = self.ohlc_data['close'].loc[idx, symbol] if symbol in self.ohlc_data['close'].columns else close_price
                        open_val = self.ohlc_data['open'].loc[idx, symbol] if 'open' in self.ohlc_data and symbol in self.ohlc_data['open'].columns else close_val
                        high_val = self.ohlc_data['high'].loc[idx, symbol] if 'high' in self.ohlc_data and symbol in self.ohlc_data['high'].columns else close_val
                        low_val = self.ohlc_data['low'].loc[idx, symbol] if 'low' in self.ohlc_data and symbol in self.ohlc_data['low'].columns else close_val
                        volume_val = int(self.ohlc_data['volume'].loc[idx, symbol]) if 'volume' in self.ohlc_data and symbol in self.ohlc_data['volume'].columns else 0
                    except Exception:
                        # Fallback
                        open_val = high_val = low_val = close_price
                        volume_val = 0
                else:
                    open_val = high_val = low_val = close_price
                    volume_val = 0
                
                bars.append(Bar(
                    timestamp=timestamp,
                    open=float(open_val),
                    high=float(high_val),
                    low=float(low_val),
                    close=float(close_val),
                    volume=int(volume_val),
                    symbol=symbol,
                    timeframe="1Day"
                ))
            
            # Create single MarketData object with ALL bars
            market_data = MarketData(symbol=symbol, bars=bars)
            
            # Initialize strategy
            strategy = strategy_class(config=strategy_config or {})
            
            # **DEBUG: Print strategy type and config**
            logger.debug(f"Using strategy: {strategy.__class__.__name__}")
            
            # Generate signals for ALL bars at once
            # Most strategies internally handle rolling windows
            try:
                signal_data = []
                
                # Get lookback period (default to 200 for strategies that need MA200, or 50 minimum)
                lookback_period = getattr(strategy, 'lookback_period', 200)
                min_bars = max(lookback_period, 50)  # At least 50 bars, or strategy's lookback
                
                # Process each bar sequentially to simulate live trading
                for i in range(min_bars, len(bars)):
                    # Get current bar
                    current_bar = bars[i]
                    
                    # Create market data up to current point
                    current_market_data = MarketData(
                        symbol=symbol,
                        bars=bars[:i+1]  # All bars up to current
                    )
                    
                    # Generate signal
                    try:
                        signal = strategy.generate_signal(current_market_data)
                        signal_data.append((current_bar.timestamp, signal))
                        
                        # **DEBUG: Log first few signals**
                        if i < min_bars + 10:
                            logger.debug(f"  {current_bar.timestamp}: {signal}")
                            
                    except (ValueError, Exception) as e:
                        logger.debug(f"Signal generation error at {current_bar.timestamp}: {e}")
                        continue
                
                # **DEBUG: Count signal types**
                buy_signals = sum(1 for _, s in signal_data if s == SignalAction.BUY)
                sell_signals = sum(1 for _, s in signal_data if s == SignalAction.SELL)
                hold_signals = sum(1 for _, s in signal_data if s == SignalAction.HOLD)
                
                logger.info(f"{symbol}: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD signals")
                
                # Convert signals to entries/exits
                # VectorBT needs: entries=True for BUY, exits=True for SELL
                for timestamp, signal in signal_data:
                    if signal == SignalAction.BUY:
                        entries.loc[timestamp, symbol] = True
                    elif signal == SignalAction.SELL:
                        exits.loc[timestamp, symbol] = True
                    # HOLD signals are ignored
                
            except Exception as e:
                logger.exception(f"Error generating signals for {symbol}: {e}")
                continue
        
        # **DEBUG: Print signal statistics**
        logger.info(f"Generated {entries.sum().sum()} entry signals and {exits.sum().sum()} exit signals")
        
        # **CRITICAL: Check if any signals exist**
        if entries.sum().sum() == 0 and exits.sum().sum() == 0:
            logger.warning("⚠️ NO SIGNALS GENERATED! Check strategy logic.")
        else:
            # Print first few signals for debugging
            for symbol in entries.columns:
                symbol_entries = entries[symbol]
                symbol_exits = exits[symbol]
                if symbol_entries.any():
                    first_entry = symbol_entries[symbol_entries].index[0]
                    logger.info(f"{symbol}: First entry at {first_entry}")
                if symbol_exits.any():
                    first_exit = symbol_exits[symbol_exits].index[0]
                    logger.info(f"{symbol}: First exit at {first_exit}")
        
        return entries, exits
    
    def run_backtest(
        self,
        price_data: pd.DataFrame,
        entries: pd.DataFrame,
        exits: pd.DataFrame
    ) -> vbt.Portfolio:
        """
        Run backtest simulation using VectorBT.
        
        Args:
            price_data: DataFrame with Close prices
            entries: Boolean DataFrame indicating entry points
            exits: Boolean DataFrame indicating exit points
        
        Returns:
            VectorBT Portfolio object with backtest results
        """
        logger.info("Running backtest simulation...")
        
        # DEBUG: Print signal stats
        print(f"\n📊 SIGNAL STATISTICS:")
        print(f"Price data shape: {price_data.shape}")
        print(f"Entries shape: {entries.shape}")
        print(f"Total entry signals: {entries.sum().sum()}")
        print(f"Total exit signals: {exits.sum().sum()}")
        
        # Check for signal issues
        if entries.sum().sum() == 0:
            print("⚠️ NO ENTRY SIGNALS - Check strategy logic!")
        
        if exits.sum().sum() == 0:
            print("⚠️ NO EXIT SIGNALS - Check strategy logic!")
        
        # Check if entries and exits are properly paired
        # VectorBT needs at least one entry and one exit to create a trade
        for symbol in price_data.columns:
            if symbol in entries.columns and symbol in exits.columns:
                entry_count = entries[symbol].sum()
                exit_count = exits[symbol].sum()
                if entry_count > 0 and exit_count == 0:
                    print(f"⚠️ {symbol}: Has entries but NO exits!")
                elif exit_count > 0 and entry_count == 0:
                    print(f"⚠️ {symbol}: Has exits but NO entries!")
        
        # DEBUG: Show sample of first entry and exit
        for symbol in price_data.columns[:2]:  # First 2 symbols
            if entries[symbol].any():
                first_entry = entries[symbol][entries[symbol]].index[0]
                print(f"{symbol} first entry: {first_entry}")
            if exits[symbol].any():
                first_exit = exits[symbol][exits[symbol]].index[0]
                print(f"{symbol} first exit: {first_exit}")
        
        # Run VectorBT backtest
        portfolio = vbt.Portfolio.from_signals(
            price_data,
            entries,
            exits,
            init_cash=self.initial_cash,
            fees=self.commission,
            freq='1D'
        )
        
        # DEBUG: Check trades
        trades = portfolio.trades
        if trades is not None and len(trades) > 0:
            print(f"✅ Generated {len(trades)} trades")
        else:
            print("❌ NO TRADES GENERATED!")
            # Try to understand why
            if entries.sum().sum() > 0 and exits.sum().sum() > 0:
                print("   - Signals exist but no trades created")
                print("   - Possible issue: Entry/exit signals not properly paired")
                print("   - Check if exit comes after entry for each trade")
        
        logger.info("✅ Backtest complete")
        return portfolio
    
    def analyze_results(
        self,
        portfolio: vbt.Portfolio,
        strategy_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        parameters: Optional[Dict[str, Any]] = None,
        is_walk_forward: bool = False,
        walk_forward_period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze backtest results and validate against thresholds.
        
        Args:
            portfolio: VectorBT Portfolio object (can be multi-symbol or single symbol)
            strategy_name: Strategy name
            symbol: Stock symbol (or "PORTFOLIO" for combined)
            start_date: Backtest start date
            end_date: Backtest end date
            parameters: Strategy parameters
            is_walk_forward: Whether this is walk-forward analysis
            walk_forward_period: Period number if walk-forward
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Get portfolio statistics with risk-free rate adjustment
            daily_rf = self.risk_free_rate / 252  # Daily risk-free rate
            
            # VectorBT Portfolio returns Series/DataFrame for multi-symbol portfolios
            # For single symbol, returns scalar values
            # Handle per-symbol vs portfolio-level analysis
            if symbol != "PORTFOLIO":
                # Per-symbol analysis - extract metrics for specific symbol
                try:
                    # Get metrics - VectorBT returns Series for multi-column portfolios
                    total_return_series = portfolio.total_return()
                    sharpe_series = portfolio.sharpe_ratio(risk_free=daily_rf)
                    max_dd_series = portfolio.max_drawdown()
                    
                    # Extract symbol-specific values
                    if isinstance(total_return_series, pd.Series):
                        # Multi-symbol portfolio
                        if symbol in total_return_series.index:
                            total_return = float(total_return_series[symbol]) * 100
                        else:
                            # Try by position
                            try:
                                total_return = float(total_return_series.iloc[0]) * 100
                            except:
                                total_return = float(total_return_series) * 100 if hasattr(total_return_series, '__float__') else 0.0
                    else:
                        # Single symbol or scalar
                        total_return = float(total_return_series) * 100
                    
                    if isinstance(sharpe_series, pd.Series):
                        if symbol in sharpe_series.index:
                            sharpe = float(sharpe_series[symbol])
                        else:
                            sharpe = float(sharpe_series.iloc[0]) if len(sharpe_series) > 0 else 0.0
                    else:
                        sharpe = float(sharpe_series)
                    
                    if isinstance(max_dd_series, pd.Series):
                        if symbol in max_dd_series.index:
                            max_dd = float(max_dd_series[symbol]) * 100
                        else:
                            max_dd = float(max_dd_series.iloc[0]) * 100 if len(max_dd_series) > 0 else 0.0
                    else:
                        max_dd = float(max_dd_series) * 100
                    
                    # Get trades - filter by symbol if possible
                    trades = portfolio.trades
                    if hasattr(trades, 'records_readable') and trades.records_readable is not None:
                        # Try to filter trades by symbol (column index)
                        try:
                            symbol_idx = portfolio.wrapper.columns.get_loc(symbol) if hasattr(portfolio, 'wrapper') and hasattr(portfolio.wrapper, 'columns') else None
                            if symbol_idx is not None:
                                symbol_trades = trades.records_readable[trades.records_readable['Column'] == symbol_idx]
                                total_trades = len(symbol_trades)
                                if total_trades > 0:
                                    # Calculate win rate from PnL
                                    winning_trades = (symbol_trades['PnL'] > 0).sum() if 'PnL' in symbol_trades.columns else 0
                                    win_rate = (winning_trades / total_trades) * 100
                                    # Get average return
                                    if 'Return' in symbol_trades.columns:
                                        avg_trade_return = float(symbol_trades['Return'].mean()) * 100
                                    else:
                                        avg_trade_return = 0.0
                                else:
                                    win_rate = 0.0
                                    avg_trade_return = 0.0
                            else:
                                # Fallback - use all trades
                                total_trades = len(trades.records_readable) if trades.records_readable is not None else 0
                                if total_trades > 0:
                                    win_rate = float(trades.win_rate()) * 100
                                    avg_trade_return = float(trades.avg_return()) * 100
                                else:
                                    win_rate = 0.0
                                    avg_trade_return = 0.0
                        except:
                            # Fallback
                            total_trades = len(trades) if trades is not None else 0
                            win_rate = float(trades.win_rate()) * 100 if total_trades > 0 else 0.0
                            avg_trade_return = float(trades.avg_return()) * 100 if total_trades > 0 else 0.0
                    else:
                        # Fallback - use portfolio trades
                        total_trades = len(trades) if trades is not None else 0
                        win_rate = float(trades.win_rate()) * 100 if total_trades > 0 else 0.0
                        avg_trade_return = float(trades.avg_return()) * 100 if total_trades > 0 else 0.0
                        
                except Exception as e:
                    logger.warning(f"Error extracting per-symbol metrics for {symbol}: {e}. Using portfolio-level.")
                    # Fallback to portfolio-level
                    total_return = float(portfolio.total_return()) * 100 if hasattr(portfolio.total_return(), '__float__') else 0.0
                    sharpe = float(portfolio.sharpe_ratio(risk_free=daily_rf)) if hasattr(portfolio.sharpe_ratio(risk_free=daily_rf), '__float__') else 0.0
                    max_dd = float(portfolio.max_drawdown()) * 100 if hasattr(portfolio.max_drawdown(), '__float__') else 0.0
                    trades = portfolio.trades
                    total_trades = len(trades) if trades is not None else 0
                    win_rate = float(trades.win_rate()) * 100 if total_trades > 0 else 0.0
                    avg_trade_return = float(trades.avg_return()) * 100 if total_trades > 0 else 0.0
            else:
                # Portfolio-level analysis (combined across all symbols)
                total_return = float(portfolio.total_return()) * 100 if not isinstance(portfolio.total_return(), pd.Series) else float(portfolio.total_return().mean()) * 100
                sharpe_val = portfolio.sharpe_ratio(risk_free=daily_rf)
                sharpe = float(sharpe_val) if not isinstance(sharpe_val, pd.Series) else float(sharpe_val.mean())
                max_dd_val = portfolio.max_drawdown()
                max_dd = float(max_dd_val) * 100 if not isinstance(max_dd_val, pd.Series) else float(max_dd_val.mean()) * 100
                
                # Calculate win rate
                trades = portfolio.trades
                if trades is not None and len(trades) > 0:
                    win_rate = float(trades.win_rate()) * 100
                    total_trades = len(trades)
                    avg_trade_return = float(trades.avg_return()) * 100
                else:
                    win_rate = 0.0
                    total_trades = 0
                    avg_trade_return = 0.0
            
            # Validate against thresholds
            passed = (
                sharpe >= self.min_sharpe and
                max_dd <= self.max_drawdown_threshold * 100 and
                win_rate >= self.min_win_rate * 100
            )
            
            results = {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'total_return': float(total_return),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_dd),
                'win_rate': float(win_rate),
                'total_trades': int(total_trades),
                'avg_trade_return': float(avg_trade_return),
                'passed': passed,
                'min_sharpe': self.min_sharpe,
                'max_drawdown_threshold': self.max_drawdown_threshold * 100,
                'min_win_rate': self.min_win_rate * 100,
                'parameters': parameters,
                'risk_free_rate': self.risk_free_rate,
                'initial_cash': self.initial_cash,
                'commission': self.commission,
                'is_walk_forward': is_walk_forward,
                'walk_forward_period': walk_forward_period
            }
            
            # Log to database if available
            if self.db_manager:
                try:
                    self.db_manager.log_backtest_result(**results)
                    logger.debug(f"Logged backtest result to database for {strategy_name} on {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to log backtest result to database: {e}")
            
            return results
            
        except Exception as e:
            logger.exception(f"Error analyzing backtest results: {e}")
            raise
    
    def debug_signals(self, price_data: pd.DataFrame, entries: pd.DataFrame, exits: pd.DataFrame) -> None:
        """
        Debug signal generation issues.
        """
        print("\n" + "=" * 80)
        print("🔍 DEBUG SIGNAL ANALYSIS")
        print("=" * 80)
        
        for symbol in price_data.columns:
            print(f"\n{symbol}:")
            symbol_entries = entries[symbol].sum()
            symbol_exits = exits[symbol].sum()
            print(f"  Entry signals: {symbol_entries}")
            print(f"  Exit signals: {symbol_exits}")
            
            if symbol_entries > 0:
                entry_dates = entries[symbol][entries[symbol]].index
                print(f"  First entry: {entry_dates[0]}")
                print(f"  Last entry: {entry_dates[-1]}")
            
            if symbol_exits > 0:
                exit_dates = exits[symbol][exits[symbol]].index
                print(f"  First exit: {exit_dates[0]}")
                print(f"  Last exit: {exit_dates[-1]}")
            
            # Check for signal conflicts
            if symbol_entries > 0 and symbol_exits > 0:
                # Check if entry and exit signals overlap
                overlap = (entries[symbol] & exits[symbol]).sum()
                if overlap > 0:
                    print(f"  ⚠️ WARNING: {overlap} overlapping entry/exit signals!")
        
        print("\n" + "=" * 80)
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Print formatted backtest results.
        
        Args:
            results: Results dictionary from analyze_results
        """
        print("\n" + "=" * 80)
        print(f"📈 BACKTEST RESULTS: {results['strategy_name']} - {results['symbol']}")
        print("=" * 80)
        
        print(f"\nPeriod: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${results['initial_cash']:,.2f}")
        print(f"Commission: {results['commission']:.3%}")
        print(f"Risk-Free Rate: {results['risk_free_rate']:.2%} (annual)")
        
        print("\n--- PERFORMANCE METRICS ---")
        print(f"Total Return:    {results['total_return']:>10.2f}%")
        print(f"Sharpe Ratio:    {results['sharpe_ratio']:>10.2f} (min: {results['min_sharpe']:.2f})")
        print(f"Max Drawdown:    {results['max_drawdown']:>10.2f}% (max: {results['max_drawdown_threshold']:.2f}%)")
        print(f"Win Rate:        {results['win_rate']:>10.2f}% (min: {results['min_win_rate']:.2f}%)")
        print(f"Total Trades:    {results['total_trades']:>10d}")
        print(f"Avg Trade Return: {results['avg_trade_return']:>9.2f}%")
        
        print("\n--- VALIDATION ---")
        status = "✅ PASS" if results['passed'] else "❌ FAIL"
        print(f"Status: {status}")
        
        if not results['passed']:
            failures = []
            if results['sharpe_ratio'] < results['min_sharpe']:
                failures.append(f"Sharpe {results['sharpe_ratio']:.2f} < {results['min_sharpe']:.2f}")
            if results['max_drawdown'] > results['max_drawdown_threshold']:
                failures.append(f"Max DD {results['max_drawdown']:.2f}% > {results['max_drawdown_threshold']:.2f}%")
            if results['win_rate'] < results['min_win_rate']:
                failures.append(f"Win Rate {results['win_rate']:.2f}% < {results['min_win_rate']:.2f}%")
            
            if failures:
                print("Failures:")
                for failure in failures:
                    print(f"  - {failure}")
        
        print("=" * 80)

