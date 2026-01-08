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
        database_manager: Optional[DatabaseManager] = None,
        enable_risk_management: bool = True,
        default_stop_loss_pct: float = 0.15,  # 15% stop-loss
        default_take_profit_pct: float = 0.25  # 25% take-profit
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
        self.enable_risk_management = enable_risk_management
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.default_trailing_stop_pct = 0.08  # 8% trailing stop by default
        
        # Track highest prices for trailing stops (per symbol per entry)
        self._highest_price_tracking: Dict[str, float] = {}
        
        # Load thresholds from config or use defaults
        if config:
            self.min_sharpe = config.quant_min_sharpe
            self.max_drawdown_threshold = config.quant_max_drawdown
        else:
            self.min_sharpe = float(os.getenv("QUANT_MIN_SHARPE", "0.8"))  # Lowered from 1.5 for realistic validation
            self.max_drawdown_threshold = float(os.getenv("QUANT_MAX_DRAWDOWN", "0.15"))  # Increased from 0.08 (8%) to 0.15 (15%)
        
        self.min_win_rate = 0.45  # 45% minimum win rate
        
        logger.info(
            f"BacktestEngine initialized: "
            f"Initial Cash=${initial_cash:,.2f}, "
            f"Commission={commission:.3%}, "
            f"Risk-Free Rate={risk_free_rate:.2%}, "
            f"Min Sharpe={self.min_sharpe}, "
            f"Max DD={self.max_drawdown_threshold:.1%}"
        )
    
    @staticmethod
    def _normalize_symbol_for_provider(symbol: str, provider) -> str:
        """
        Normalize symbol format for the specific data provider.
        
        Different providers use different formats:
        - Alpaca: BRK.B (dot)
        - Yahoo Finance: BRK-B (hyphen)
        - Polygon: Usually dots
        
        Args:
            symbol: Original symbol (may have hyphen or dot)
            provider: DataProvider enum value
            
        Returns:
            Normalized symbol for the provider
        """
        from config.settings import DataProvider
        
        # Common symbols that need normalization
        # Map: {yahoo_format: alpaca_format}
        symbol_map = {
            'BRK-B': 'BRK.B',
            'BF-B': 'BF.B',
        }
        
        # If provider is Alpaca, convert hyphens to dots
        if provider == DataProvider.ALPACA:
            # Check if we have a known mapping
            if symbol in symbol_map:
                return symbol_map[symbol]
            # Otherwise, convert hyphen to dot (e.g., BRK-B -> BRK.B)
            if '-' in symbol:
                return symbol.replace('-', '.')
        
        # For Yahoo Finance, convert dots to hyphens
        elif provider == DataProvider.YAHOO:
            # Check reverse mapping
            reverse_map = {v: k for k, v in symbol_map.items()}
            if symbol in reverse_map:
                return reverse_map[symbol]
            # Otherwise, convert dot to hyphen (e.g., BRK.B -> BRK-B)
            if '.' in symbol:
                return symbol.replace('.', '-')
        
        # For Polygon or other providers, keep as-is or use dots
        # Polygon typically uses dots like Alpaca
        elif provider == DataProvider.POLYGON:
            if symbol in symbol_map:
                return symbol_map[symbol]
            if '-' in symbol:
                return symbol.replace('-', '.')
        
        # Default: return as-is
        return symbol
    
    def fetch_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        timeframe: str = "1Day"
    ) -> pd.DataFrame:
        """
        Fetch historical price data using DataAgent.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime, defaults to today)
            timeframe: Bar timeframe. Supported: "1Min", "5Min", "15Min", "1Hour", "1Day"
                      - "1Day" (default): Daily bars, uses end-of-day prices
                      - "1Hour": Hourly bars, allows intraday trading and better exit timing
                      - "15Min", "5Min", "1Min": Higher frequency for more granular analysis
        
        Returns:
            DataFrame with Close prices (symbols as columns, timestamps as index)
        
        Raises:
            ValueError: If data fetching fails
        
        Note:
            - Daily bars: ~252 bars per year (trading days only)
            - Hourly bars: ~6.5 hours per trading day × 252 days ≈ 1,638 bars per year
            - With hourly data, strategies can execute trades during market hours for better prices
            - Hourly data requires more API calls and storage but provides better backtest accuracy
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
            
            # Calculate limit based on date range and timeframe
            days_diff = (end_dt - start_dt).days
            
            # Adjust limit based on timeframe
            if timeframe == "1Day":
                # Daily: ~252 trading days per year
                limit = min(int(days_diff * 1.5), 1000)
            elif timeframe == "1Hour":
                # Hourly: ~6.5 hours per trading day × ~252 days ≈ 1,638 bars per year
                # But we'll fetch for all days (including weekends for hourly data)
                bars_per_day = 24  # Hourly bars per calendar day
                limit = min(int(days_diff * bars_per_day * 1.2), 10000)  # Cap at 10k for hourly
            elif timeframe in ["15Min", "5Min"]:
                # 15-min: 4 per hour × 24 hours = 96 per day
                # 5-min: 12 per hour × 24 hours = 288 per day
                bars_per_hour = 4 if timeframe == "15Min" else 12
                bars_per_day = bars_per_hour * 24
                limit = min(int(days_diff * bars_per_day * 1.2), 50000)
            elif timeframe == "1Min":
                # 1-min: 60 per hour × 24 hours = 1,440 per day
                bars_per_day = 60 * 24
                limit = min(int(days_diff * bars_per_day * 1.2), 100000)
            else:
                # Default to daily calculation
                limit = min(int(days_diff * 1.5), 1000)
            
            logger.info(f"Calculated limit for {timeframe} timeframe: {limit} bars")
            
            # Fetch each symbol individually with strict_validation=False for backtesting
            # This allows stale historical data which is normal for backtesting
            data_frames = {}
            for symbol in symbols:
                try:
                    # Normalize symbol for the data provider
                    # Alpaca uses dots (BRK.B), Yahoo uses hyphens (BRK-B)
                    normalized_symbol = self._normalize_symbol_for_provider(
                        symbol, 
                        data_agent.provider
                    )
                    
                    # Use _fetch_data() directly with strict_validation=False
                    # This allows historical/stale data which is expected for backtesting
                    market_data = data_agent._fetch_data(
                        symbol=normalized_symbol,
                        timeframe=timeframe,  # Use the specified timeframe (1Hour, 1Day, etc.)
                        start_date=start_dt,
                        end_date=end_dt,
                        limit=limit,
                        strict_validation=False  # Critical: allow stale data for backtesting
                    )
                    
                    if market_data and market_data.bars:
                        df = market_data.to_dataframe()
                        if not df.empty:
                            # Use original symbol as key (not normalized_symbol)
                            # This ensures consistency with the rest of the code
                            data_frames[symbol] = df
                            logger.debug(f"Fetched {len(df)} bars for {symbol} (normalized: {normalized_symbol}) via DataAgent")
                        else:
                            logger.warning(f"Empty DataFrame for {symbol} (normalized: {normalized_symbol})")
                    else:
                        logger.warning(f"No bars returned for {symbol} (normalized: {normalized_symbol})")
                except Exception as symbol_error:
                    logger.warning(f"Failed to fetch {symbol} (normalized: {normalized_symbol}): {symbol_error}")
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
                
                # Track position state and entry price for risk management
                in_position = False
                entry_price = 0.0
                entry_timestamp = None
                position_key = None  # Track unique position for trailing stop
                
                # Get risk management parameters from strategy config or use defaults
                stop_loss_pct = strategy.config.get('stop_loss_pct', self.default_stop_loss_pct) if self.enable_risk_management else None
                take_profit_pct = strategy.config.get('take_profit_pct', self.default_take_profit_pct) if self.enable_risk_management else None
                trailing_stop_pct = strategy.config.get('trailing_stop_pct', self.default_trailing_stop_pct) if self.enable_risk_management else None
                
                # Create unique key for tracking this position (symbol + entry timestamp)
                position_key = None
                
                # Get lookback period (default to 200 for strategies that need MA200, or 50 minimum)
                # For hourly data, adjust: 200 daily bars ≈ 200 trading days, but hourly bars need more
                lookback_period = getattr(strategy, 'lookback_period', 200)
                
                # Adjust min_bars based on timeframe
                # Daily: 200 bars = 200 trading days
                # Hourly: If strategy needs 200 daily bars, we need ~200 hours (assuming ~6.5 hours/day)
                # But strategies typically work with the same lookback regardless of timeframe
                # So if strategy needs 200 bars for MA200, use 200 bars at any timeframe
                min_bars = max(lookback_period, 50)  # At least 50 bars, or strategy's lookback
                
                # Process each bar sequentially to simulate live trading
                for i in range(min_bars, len(bars)):
                    # Get current bar
                    current_bar = bars[i]
                    current_price = current_bar.close
                    
                    # Create market data up to current point
                    current_market_data = MarketData(
                        symbol=symbol,
                        bars=bars[:i+1]  # All bars up to current
                    )
                    
                    # RISK MANAGEMENT: Check stop-loss, trailing stop, and take-profit BEFORE generating signal
                    if self.enable_risk_management and in_position and entry_price > 0:
                        # Create position tracking key
                        if position_key is None:
                            position_key = f"{symbol}_{entry_timestamp.isoformat() if entry_timestamp else 'unknown'}"
                        
                        # Initialize highest price tracking for this position
                        if position_key not in self._highest_price_tracking:
                            self._highest_price_tracking[position_key] = entry_price
                        
                        # Update highest price since entry (for trailing stop)
                        if current_price > self._highest_price_tracking[position_key]:
                            self._highest_price_tracking[position_key] = current_price
                            highest_price = current_price
                        else:
                            highest_price = self._highest_price_tracking[position_key]
                        
                        # Check 1: Fixed stop-loss (absolute loss limit)
                        if stop_loss_pct and current_price <= entry_price * (1 - stop_loss_pct):
                            # Stop-loss hit!
                            signal_data.append((current_bar.timestamp, SignalAction.SELL))
                            in_position = False
                            loss_pct = (current_price / entry_price - 1) * 100
                            logger.debug(f"  {current_bar.timestamp}: STOP-LOSS at ${current_price:.2f} (entry: ${entry_price:.2f}, loss: {loss_pct:.2f}%)")
                            entry_price = 0.0
                            entry_timestamp = None
                            # Clean up tracking
                            if position_key in self._highest_price_tracking:
                                del self._highest_price_tracking[position_key]
                            position_key = None
                            continue
                        
                        # Check 2: Trailing stop-loss (protects gains, moves up with price)
                        if trailing_stop_pct and highest_price > entry_price:
                            # Only apply trailing stop if we have gains (price above entry)
                            trailing_stop_price = highest_price * (1 - trailing_stop_pct)
                            if current_price <= trailing_stop_price:
                                # Trailing stop hit!
                                signal_data.append((current_bar.timestamp, SignalAction.SELL))
                                in_position = False
                                gain_pct = (current_price / entry_price - 1) * 100
                                peak_gain_pct = (highest_price / entry_price - 1) * 100
                                logger.debug(f"  {current_bar.timestamp}: TRAILING STOP at ${current_price:.2f} (entry: ${entry_price:.2f}, peak: ${highest_price:.2f}, gain: {gain_pct:.2f}%, peak gain: {peak_gain_pct:.2f}%)")
                                entry_price = 0.0
                                entry_timestamp = None
                                # Clean up tracking
                                if position_key in self._highest_price_tracking:
                                    del self._highest_price_tracking[position_key]
                                position_key = None
                                continue
                        
                        # Check 3: Take-profit (lock in large gains)
                        if take_profit_pct and current_price >= entry_price * (1 + take_profit_pct):
                            # Take-profit hit!
                            signal_data.append((current_bar.timestamp, SignalAction.SELL))
                            in_position = False
                            gain_pct = (current_price / entry_price - 1) * 100
                            logger.debug(f"  {current_bar.timestamp}: TAKE-PROFIT at ${current_price:.2f} (entry: ${entry_price:.2f}, gain: {gain_pct:.2f}%)")
                            entry_price = 0.0
                            entry_timestamp = None
                            # Clean up tracking
                            if position_key in self._highest_price_tracking:
                                del self._highest_price_tracking[position_key]
                            position_key = None
                            continue
                    
                    # Generate signal from strategy
                    try:
                        signal = strategy.generate_signal(current_market_data)
                        
                        # Handle original strategy signals
                        if signal == SignalAction.BUY and not in_position:
                            signal_data.append((current_bar.timestamp, SignalAction.BUY))
                            in_position = True
                            entry_price = current_price
                            entry_timestamp = current_bar.timestamp
                            position_key = f"{symbol}_{entry_timestamp.isoformat()}"
                            # Initialize highest price tracking
                            self._highest_price_tracking[position_key] = entry_price
                            if i < min_bars + 10:
                                logger.debug(f"  {current_bar.timestamp}: BUY at ${entry_price:.2f} (entering position)")
                        elif signal == SignalAction.SELL and in_position:
                            signal_data.append((current_bar.timestamp, SignalAction.SELL))
                            in_position = False
                            gain_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0
                            if i < min_bars + 10:
                                logger.debug(f"  {current_bar.timestamp}: SELL at ${current_price:.2f} (exiting position, entry: ${entry_price:.2f}, gain: {gain_pct:.2f}%)")
                            entry_price = 0.0
                            entry_timestamp = None
                            # Clean up tracking
                            if position_key and position_key in self._highest_price_tracking:
                                del self._highest_price_tracking[position_key]
                            position_key = None
                        elif signal == SignalAction.HOLD:
                            # HOLD signals are ignored
                            pass
                        else:
                            # Invalid signal combination (BUY when already in position, or SELL when not in position)
                            if i < min_bars + 10:
                                logger.debug(f"  {current_bar.timestamp}: {signal} IGNORED (in_position={in_position})")
                            
                    except (ValueError, Exception) as e:
                        logger.debug(f"Signal generation error at {current_bar.timestamp}: {e}")
                        continue
                
                # **DEBUG: Count signal types**
                buy_signals = sum(1 for _, s in signal_data if s == SignalAction.BUY)
                sell_signals = sum(1 for _, s in signal_data if s == SignalAction.SELL)
                
                logger.info(f"{symbol}: {buy_signals} BUY, {sell_signals} SELL signals")
                
                # Convert signals to entries/exits
                # VectorBT needs: entries=True for BUY, exits=True for SELL
                for timestamp, signal in signal_data:
                    if signal == SignalAction.BUY:
                        entries.loc[timestamp, symbol] = True
                    elif signal == SignalAction.SELL:
                        exits.loc[timestamp, symbol] = True
                
                # CRITICAL: If we end in a position, add a final exit at the last bar
                if in_position:
                    last_timestamp = bars[-1].timestamp
                    exits.loc[last_timestamp, symbol] = True
                    final_price = bars[-1].close
                    final_gain = (final_price / entry_price - 1) * 100 if entry_price > 0 else 0
                    logger.info(f"{symbol}: Added final exit at {last_timestamp} (position was still open, entry: ${entry_price:.2f}, final: ${final_price:.2f}, gain: {final_gain:.2f}%)")
                    # Clean up tracking
                    if position_key and position_key in self._highest_price_tracking:
                        del self._highest_price_tracking[position_key]
                
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
        if trades is not None:
            if hasattr(trades, '__len__'):
                print(f"✅ Generated {len(trades)} trades")
            elif hasattr(trades, 'records') and trades.records is not None:
                print(f"✅ Generated {len(trades.records)} trades")
            elif hasattr(trades, 'records_readable') and trades.records_readable is not None:
                print(f"✅ Generated {len(trades.records_readable)} trades")
            else:
                print("❌ Unable to determine number of trades")
            
            # Call debug method to inspect trades object structure
            self.debug_trades(portfolio)
        else:
            print("❌ NO TRADES GENERATED!")
            # Try to understand why
            if entries.sum().sum() > 0 and exits.sum().sum() > 0:
                print("   - Signals exist but no trades created")
                print("   - Possible issue: Entry/exit signals not properly paired")
                print("   - Check if exit comes after entry for each trade")
        
        logger.info("✅ Backtest complete")
        return portfolio
    
    def debug_trades(self, portfolio: vbt.Portfolio) -> None:
        """
        Debug VectorBT trades object structure.
        """
        print("\n" + "=" * 80)
        print("🔍 DEBUG VECTORBT TRADES OBJECT")
        print("=" * 80)
        
        trades = portfolio.trades
        
        if trades is None:
            print("No trades object")
            return
        
        print(f"\nTrades object type: {type(trades)}")
        print(f"Trades object attributes: {[attr for attr in dir(trades) if not attr.startswith('_')]}")
        
        # Check for common attributes
        if hasattr(trades, '__len__'):
            print(f"Number of trades: {len(trades)}")
        
        if hasattr(trades, 'records'):
            print(f"Has records attribute: {trades.records is not None}")
        
        if hasattr(trades, 'records_readable'):
            print(f"Has records_readable attribute: {trades.records_readable is not None}")
            if trades.records_readable is not None:
                print(f"records_readable shape: {trades.records_readable.shape}")
                print(f"records_readable columns: {list(trades.records_readable.columns)}")
                if not trades.records_readable.empty:
                    print(f"First trade:\n{trades.records_readable.iloc[0]}")
        
        if hasattr(trades, 'pnl'):
            print(f"Has pnl attribute: True")
            try:
                pnl_val = trades.pnl
                if hasattr(pnl_val, 'mean'):
                    print(f"PNL mean: {pnl_val.mean()}")
                else:
                    print(f"PNL type: {type(pnl_val)}, value: {pnl_val}")
            except Exception as e:
                print(f"PNL access error: {e}")
        
        # Check for methods
        methods_to_check = ['win_rate', 'avg_return', 'total_return', 'profit']
        for method in methods_to_check:
            if hasattr(trades, method):
                try:
                    result = getattr(trades, method)()
                    print(f"{method}(): {result}")
                except Exception as e:
                    print(f"{method}() error: {e}")
        
        print("=" * 80)
    
    def analyze_results(
        self,
        portfolio: vbt.Portfolio,
        strategy_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        parameters: Optional[Dict[str, Any]] = None,
        is_walk_forward: bool = False,
        walk_forward_period: Optional[int] = None,
        price_data: Optional[pd.DataFrame] = None
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
                            sharpe_raw = sharpe_series[symbol]
                        else:
                            sharpe_raw = sharpe_series.iloc[0] if len(sharpe_series) > 0 else 0.0
                    else:
                        sharpe_raw = sharpe_series
                    
                    # Convert to float with safety checks
                    try:
                        sharpe = float(sharpe_raw) if sharpe_raw is not None else 0.0
                    except (ValueError, TypeError):
                        sharpe = 0.0
                    
                    # Safety check for invalid Sharpe ratios (NaN, Inf, or extreme values)
                    if np.isnan(sharpe) or np.isinf(sharpe) or abs(sharpe) > 1000:
                        logger.debug(f"Invalid Sharpe ratio for {symbol}: {sharpe}, setting to 0.0")
                        sharpe = 0.0
                    
                    if isinstance(max_dd_series, pd.Series):
                        if symbol in max_dd_series.index:
                            max_dd = float(max_dd_series[symbol]) * 100
                        else:
                            max_dd = float(max_dd_series.iloc[0]) * 100 if len(max_dd_series) > 0 else 0.0
                    else:
                        max_dd = float(max_dd_series) * 100
                    
                    # Get trades count first - if 0 trades, Sharpe should be 0.0
                    trades = portfolio.trades
                    total_trades = 0
                    win_rate = 0.0
                    avg_trade_return = 0.0
                    has_trades = False
                    
                    # Quick check for trades count before validating Sharpe
                    if trades is not None:
                        if hasattr(trades, '__len__'):
                            has_trades = len(trades) > 0
                        elif hasattr(trades, 'records_readable') and trades.records_readable is not None:
                            has_trades = len(trades.records_readable) > 0
                        elif hasattr(trades, 'records') and trades.records is not None:
                            has_trades = len(trades.records) > 0
                    
                    # If no trades, set Sharpe to 0.0 (VectorBT might return invalid values)
                    if not has_trades:
                        logger.debug(f"No trades for {symbol}, setting Sharpe to 0.0")
                        sharpe = 0.0
                    
                    if trades is not None and hasattr(trades, '__len__'):
                        try:
                            if hasattr(trades, 'records') and trades.records is not None:
                                # VectorBT stores trades in records
                                trades_df = trades.records_readable
                                if trades_df is not None and 'Column' in trades_df.columns:
                                    # Get column index for this symbol
                                    try:
                                        # Get price_data from portfolio wrapper if not provided
                                        if price_data is None and hasattr(portfolio, 'wrapper') and hasattr(portfolio.wrapper, 'columns'):
                                            price_data_cols = portfolio.wrapper.columns
                                        elif price_data is not None:
                                            price_data_cols = price_data.columns
                                        else:
                                            price_data_cols = None
                                        
                                        if price_data_cols is not None and symbol in price_data_cols:
                                            col_idx = price_data_cols.get_loc(symbol)
                                            symbol_trades = trades_df[trades_df['Column'] == col_idx]
                                        else:
                                            # Symbol not found, use all trades as fallback
                                            col_idx = None
                                            symbol_trades = None
                                        
                                        if symbol_trades is not None and len(symbol_trades) > 0:
                                            total_trades = len(symbol_trades)
                                            
                                            # Calculate win rate
                                            winning_trades = (symbol_trades['PnL'] > 0).sum() if 'PnL' in symbol_trades.columns else 0
                                            win_rate = (winning_trades / total_trades) * 100
                                            # Get average return from PnL or Return column
                                            if 'Return' in symbol_trades.columns:
                                                avg_trade_return = float(symbol_trades['Return'].mean()) * 100
                                            elif 'PnL' in symbol_trades.columns and 'Size' in symbol_trades.columns:
                                                # Calculate return from PnL / Size
                                                avg_trade_return = float((symbol_trades['PnL'] / symbol_trades['Size']).mean()) * 100 if (symbol_trades['Size'] != 0).any() else 0.0
                                            else:
                                                avg_trade_return = 0.0
                                        else:
                                            # No trades for this symbol, use all trades as fallback
                                            total_trades = len(trades_df) if trades_df is not None else 0
                                            if total_trades > 0:
                                                try:
                                                    win_rate_val = trades.win_rate()
                                                    if isinstance(win_rate_val, pd.Series):
                                                        win_rate = float(win_rate_val.iloc[0]) * 100 if len(win_rate_val) > 0 else 0.0
                                                    else:
                                                        win_rate = float(win_rate_val) * 100
                                                except (ValueError, TypeError, AttributeError):
                                                    win_rate = 0.0
                                                # Calculate avg return from trades_df
                                                if 'Return' in trades_df.columns:
                                                    avg_trade_return = float(trades_df['Return'].mean()) * 100
                                                elif 'PnL' in trades_df.columns and 'Size' in trades_df.columns:
                                                    avg_trade_return = float((trades_df['PnL'] / trades_df['Size']).mean()) * 100 if (trades_df['Size'] != 0).any() else 0.0
                                                else:
                                                    avg_trade_return = 0.0
                                    except (KeyError, ValueError) as e:
                                        # Symbol not found in columns, use all trades
                                        logger.debug(f"Symbol {symbol} not found in price_data columns, using all trades: {e}")
                                        total_trades = len(trades_df) if trades_df is not None else 0
                                        if total_trades > 0:
                                            win_rate = float(trades.win_rate()) * 100
                                            # Calculate avg return from trades_df
                                            if 'Return' in trades_df.columns:
                                                avg_trade_return = float(trades_df['Return'].mean()) * 100
                                            elif 'PnL' in trades_df.columns and 'Size' in trades_df.columns:
                                                avg_trade_return = float((trades_df['PnL'] / trades_df['Size']).mean()) * 100 if (trades_df['Size'] != 0).any() else 0.0
                                            else:
                                                avg_trade_return = 0.0
                                else:
                                    # No records_readable or Column column, use trades directly
                                    total_trades = len(trades) if trades is not None else 0
                                    if total_trades > 0:
                                        try:
                                            win_rate_val = trades.win_rate()
                                            if isinstance(win_rate_val, pd.Series):
                                                win_rate = float(win_rate_val.iloc[0]) * 100 if len(win_rate_val) > 0 else 0.0
                                            else:
                                                win_rate = float(win_rate_val) * 100
                                        except (ValueError, TypeError, AttributeError):
                                            win_rate = 0.0
                                        # Try to get avg return from records_readable if available
                                        if hasattr(trades, 'records_readable') and trades.records_readable is not None:
                                            trades_df = trades.records_readable
                                            if 'Return' in trades_df.columns:
                                                avg_trade_return = float(trades_df['Return'].mean()) * 100
                                            elif 'PnL' in trades_df.columns and 'Size' in trades_df.columns:
                                                avg_trade_return = float((trades_df['PnL'] / trades_df['Size']).mean()) * 100 if (trades_df['Size'] != 0).any() else 0.0
                                            else:
                                                avg_trade_return = 0.0
                                        else:
                                            avg_trade_return = 0.0
                            else:
                                # Fallback - use trades directly
                                total_trades = len(trades) if trades is not None else 0
                                if total_trades > 0:
                                    win_rate = float(trades.win_rate()) * 100
                                    # Try to get avg return from records_readable if available
                                    if hasattr(trades, 'records_readable') and trades.records_readable is not None:
                                        trades_df = trades.records_readable
                                        if 'Return' in trades_df.columns:
                                            avg_trade_return = float(trades_df['Return'].mean()) * 100
                                        elif 'PnL' in trades_df.columns and 'Size' in trades_df.columns:
                                            avg_trade_return = float((trades_df['PnL'] / trades_df['Size']).mean()) * 100 if (trades_df['Size'] != 0).any() else 0.0
                                        else:
                                            avg_trade_return = 0.0
                                    else:
                                        avg_trade_return = 0.0
                        except Exception as e:
                            logger.debug(f"Error filtering trades by symbol {symbol}: {e}")
                            # Fallback
                            total_trades = len(trades) if trades is not None else 0
                            if total_trades > 0:
                                win_rate = float(trades.win_rate()) * 100
                                # Try to get avg return from records_readable if available
                                if hasattr(trades, 'records_readable') and trades.records_readable is not None:
                                    trades_df = trades.records_readable
                                    if 'Return' in trades_df.columns:
                                        avg_trade_return = float(trades_df['Return'].mean()) * 100
                                    elif 'PnL' in trades_df.columns and 'Size' in trades_df.columns:
                                        avg_trade_return = float((trades_df['PnL'] / trades_df['Size']).mean()) * 100 if (trades_df['Size'] != 0).any() else 0.0
                                    else:
                                        avg_trade_return = 0.0
                                else:
                                    avg_trade_return = 0.0
                    
                    # DEBUG: Print trade count
                    logger.debug(f"Portfolio has {total_trades} trades for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"Error extracting per-symbol metrics for {symbol}: {e}. Using portfolio-level.")
                    # Fallback to portfolio-level
                    total_return_val = portfolio.total_return()
                    if isinstance(total_return_val, pd.Series):
                        total_return = float(total_return_val.iloc[0]) * 100 if len(total_return_val) > 0 else 0.0
                    else:
                        try:
                            total_return = float(total_return_val) * 100
                        except (ValueError, TypeError):
                            total_return = 0.0
                    sharpe_raw = portfolio.sharpe_ratio(risk_free=daily_rf)
                    try:
                        sharpe = float(sharpe_raw) if sharpe_raw is not None and hasattr(sharpe_raw, '__float__') else 0.0
                    except (ValueError, TypeError):
                        sharpe = 0.0
                    
                    # Safety check for invalid Sharpe ratios (NaN, Inf, or extreme values)
                    if np.isnan(sharpe) or np.isinf(sharpe) or abs(sharpe) > 1000:
                        logger.debug(f"Invalid Sharpe ratio for {symbol} (fallback): {sharpe}, setting to 0.0")
                        sharpe = 0.0
                    
                    max_dd_val = portfolio.max_drawdown()
                    if isinstance(max_dd_val, pd.Series):
                        max_dd = float(max_dd_val.iloc[0]) * 100 if len(max_dd_val) > 0 else 0.0
                    else:
                        try:
                            max_dd = float(max_dd_val) * 100
                        except (ValueError, TypeError):
                            max_dd = 0.0
                    
                    trades = portfolio.trades
                    total_trades = 0
                    win_rate = 0.0
                    avg_trade_return = 0.0
                    
                    if trades is not None:
                        try:
                            total_trades = len(trades) if hasattr(trades, '__len__') else 0
                        except:
                            total_trades = 0
                        
                        if total_trades > 0:
                            try:
                                win_rate_val = trades.win_rate()
                                if isinstance(win_rate_val, pd.Series):
                                    win_rate = float(win_rate_val.iloc[0]) * 100 if len(win_rate_val) > 0 else 0.0
                                else:
                                    win_rate = float(win_rate_val) * 100
                            except (ValueError, TypeError, AttributeError):
                                win_rate = 0.0
                            
                            # Calculate avg return from records_readable
                            if hasattr(trades, 'records_readable') and trades.records_readable is not None:
                                trades_df = trades.records_readable
                                if 'Return' in trades_df.columns:
                                    avg_trade_return = float(trades_df['Return'].mean()) * 100
                                elif 'PnL' in trades_df.columns and 'Size' in trades_df.columns:
                                    avg_trade_return = float((trades_df['PnL'] / trades_df['Size']).mean()) * 100 if (trades_df['Size'] != 0).any() else 0.0
                                else:
                                    avg_trade_return = 0.0
            else:
                # Portfolio-level analysis (combined across all symbols)
                total_return = float(portfolio.total_return()) * 100 if not isinstance(portfolio.total_return(), pd.Series) else float(portfolio.total_return().mean()) * 100
                sharpe_val = portfolio.sharpe_ratio(risk_free=daily_rf)
                if isinstance(sharpe_val, pd.Series):
                    sharpe = float(sharpe_val.mean()) if len(sharpe_val) > 0 else 0.0
                else:
                    try:
                        sharpe = float(sharpe_val) if sharpe_val is not None else 0.0
                    except (ValueError, TypeError):
                        sharpe = 0.0
                
                # Safety check for invalid Sharpe ratios (NaN, Inf, or extreme values)
                if np.isnan(sharpe) or np.isinf(sharpe) or abs(sharpe) > 1000:
                    logger.debug(f"Invalid Sharpe ratio (portfolio-level): {sharpe}, setting to 0.0")
                    sharpe = 0.0
                
                max_dd_val = portfolio.max_drawdown()
                max_dd = float(max_dd_val) * 100 if not isinstance(max_dd_val, pd.Series) else float(max_dd_val.mean()) * 100
                
                # Calculate win rate
                trades = portfolio.trades
                if trades is not None:
                    try:
                        total_trades = len(trades) if hasattr(trades, '__len__') else 0
                    except:
                        total_trades = 0
                    
                    if total_trades > 0:
                        try:
                            win_rate_val = trades.win_rate()
                            if isinstance(win_rate_val, pd.Series):
                                win_rate = float(win_rate_val.iloc[0]) * 100 if len(win_rate_val) > 0 else 0.0
                            else:
                                win_rate = float(win_rate_val) * 100
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.debug(f"Could not get win_rate: {e}")
                            win_rate = 0.0
                        
                        # Calculate avg return from records_readable
                        if hasattr(trades, 'records_readable') and trades.records_readable is not None:
                            trades_df = trades.records_readable
                            if 'Return' in trades_df.columns:
                                avg_trade_return = float(trades_df['Return'].mean()) * 100
                            elif 'PnL' in trades_df.columns and 'Size' in trades_df.columns:
                                avg_trade_return = float((trades_df['PnL'] / trades_df['Size']).mean()) * 100 if (trades_df['Size'] != 0).any() else 0.0
                            else:
                                avg_trade_return = 0.0
                        else:
                            avg_trade_return = 0.0
                    else:
                        win_rate = 0.0
                        avg_trade_return = 0.0
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
            
            # DEBUG: Print results
            logger.info(
                f"Results for {strategy_name} - {symbol}: "
                f"Return={total_return:.2f}%, "
                f"Sharpe={sharpe:.2f}, "
                f"MaxDD={max_dd:.2f}%, "
                f"WinRate={win_rate:.2f}%, "
                f"Trades={total_trades}, "
                f"Passed={'✅' if passed else '❌'}"
            )
            
            return results
            
        except Exception as e:
            logger.exception(f"Error analyzing backtest results: {e}")
            # Return minimal results to prevent crash
            return {
                'strategy_name': strategy_name,
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'avg_trade_return': 0.0,
                'passed': False,
                'min_sharpe': self.min_sharpe,
                'max_drawdown_threshold': self.max_drawdown_threshold * 100,
                'min_win_rate': self.min_win_rate * 100,
                'parameters': parameters,
                'risk_free_rate': self.risk_free_rate,
                'initial_cash': self.initial_cash,
                'commission': self.commission,
                'is_walk_forward': is_walk_forward,
                'walk_forward_period': walk_forward_period,
                'error': str(e)
            }
    
    def plot_results(self, portfolio: vbt.Portfolio, symbol: str, strategy_name: str, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            portfolio: VectorBT Portfolio object
            symbol: Stock symbol
            strategy_name: Strategy name
            save_path: Optional path to save the plot (if None, shows plot)
        
        Returns:
            matplotlib figure object
        """
        try:
            import matplotlib.pyplot as plt
            
            # VectorBT has built-in plotting
            fig = portfolio.plot(subplots=['orders', 'trade_pnl', 'cum_returns'])
            fig.suptitle(f"{strategy_name} - {symbol}", fontsize=16)
            plt.tight_layout()
            
            if save_path:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"📊 Plot saved to {save_path}")
            else:
                plt.show()
            
            return fig
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")
            return None
    
    def analyze_trade_quality(self, portfolio: vbt.Portfolio, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze trade quality metrics.
        
        Args:
            portfolio: VectorBT Portfolio object
            symbol: Optional symbol to filter trades (for multi-symbol portfolios)
        
        Returns:
            Dictionary with trade quality metrics
        """
        trades = portfolio.trades
        
        if trades is None:
            return {"error": "No trades object"}
        
        try:
            # Get readable trades
            if hasattr(trades, 'records_readable') and trades.records_readable is not None:
                trades_df = trades.records_readable
                
                # Filter by symbol if provided
                if symbol and 'Column' in trades_df.columns:
                    try:
                        if hasattr(portfolio, 'wrapper') and hasattr(portfolio.wrapper, 'columns'):
                            if symbol in portfolio.wrapper.columns:
                                col_idx = portfolio.wrapper.columns.get_loc(symbol)
                                trades_df = trades_df[trades_df['Column'] == col_idx]
                    except Exception:
                        pass  # Use all trades if filtering fails
                
                if len(trades_df) == 0:
                    return {"error": "No trades found"}
                
                # Calculate metrics
                if 'PnL' in trades_df.columns:
                    winning_trades = trades_df[trades_df['PnL'] > 0]
                    losing_trades = trades_df[trades_df['PnL'] < 0]
                    
                    avg_win = float(winning_trades['PnL'].mean()) if len(winning_trades) > 0 else 0.0
                    avg_loss = float(losing_trades['PnL'].mean()) if len(losing_trades) > 0 else 0.0
                    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
                    
                    # Calculate expectancy
                    win_rate = len(winning_trades) / len(trades_df)
                    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
                else:
                    avg_win = 0.0
                    avg_loss = 0.0
                    win_loss_ratio = 0.0
                    expectancy = 0.0
                    win_rate = 0.0
                
                # Try to get profit factor if available
                profit_factor = 0.0
                try:
                    if hasattr(trades, 'profit_factor'):
                        pf = trades.profit_factor()
                        if hasattr(pf, '__float__'):
                            profit_factor = float(pf)
                        elif hasattr(pf, 'iloc'):
                            profit_factor = float(pf.iloc[0]) if len(pf) > 0 else 0.0
                except Exception:
                    pass
                
                return {
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'win_loss_ratio': win_loss_ratio,
                    'expectancy': expectancy,
                    'win_rate': win_rate * 100,  # Convert to percentage
                    'profit_factor': profit_factor,
                    'total_trades': len(trades_df),
                    'winning_trades': len(winning_trades) if 'PnL' in trades_df.columns else 0,
                    'losing_trades': len(losing_trades) if 'PnL' in trades_df.columns else 0
                }
            else:
                return {"error": "No readable trades data"}
        except Exception as e:
            logger.warning(f"Error analyzing trade quality: {e}")
            return {"error": str(e)}
    
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

