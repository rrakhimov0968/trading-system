"""Position Manager - Hard brakes to prevent holding losing positions.

This service monitors open positions and automatically exits when stop-loss
conditions are met. Runs every minute during market hours.
"""
import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from agents.execution_agent import ExecutionAgent
from agents.data_agent import DataAgent
from core.strategies.indicators import calculate_atr, calculate_rsi
from models.market_data import MarketData
from models.enums import OrderSide
from utils.exceptions import ExecutionError
from utils.database import DatabaseManager
from config.settings import AppConfig

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages open positions with automatic stop-loss enforcement.
    
    This is a "hard brake" system that prevents holding losing positions.
    It runs independently from the main trading pipeline and checks positions
    every minute during market hours.
    
    Stop-Loss Rules:
    1. Hard Stop Loss: Max 5% loss from entry price
    2. ATR-Based Dynamic Stop: Entry Price - (2 * ATR) for initial stop
    3. Trailing Stop (future): Current Price - (3 * ATR) for trailing
    """
    
    def __init__(
        self,
        config: AppConfig,
        execution_agent: ExecutionAgent,
        data_agent: DataAgent,
        database_manager: Optional[DatabaseManager] = None
    ):
        """
        Initialize PositionManager.
        
        Args:
            config: Application configuration
            execution_agent: ExecutionAgent for getting positions and executing orders
            data_agent: DataAgent for fetching market data and calculating ATR
            database_manager: DatabaseManager for querying strategy info (optional, will create if None)
        """
        self.config = config
        self.execution_agent = execution_agent
        self.data_agent = data_agent
        self.db_manager = database_manager or DatabaseManager(config)
        
        # Stop-loss configuration (defaults - strategy-specific overrides)
        self.hard_stop_loss_pct = float(
            os.getenv("POSITION_MANAGER_HARD_STOP_LOSS_PCT", "0.05")  # 5% max loss
        )
        self.atr_period = int(
            os.getenv("POSITION_MANAGER_ATR_PERIOD", "14")  # 14-day ATR
        )
        self.min_bars_for_atr = 30  # Minimum bars needed for ATR calculation
        
        # Strategy-specific ATR multipliers
        # Trend following: wider stops (3x ATR) to ride volatility
        # Mean reversion: tight stops (1.5x ATR) - if it doesn't revert quickly, trade is wrong
        self.strategy_atr_multipliers = {
            "TrendFollowing": float(os.getenv("POSITION_MANAGER_ATR_TREND", "3.0")),  # 3x ATR
            "MeanReversion": float(os.getenv("POSITION_MANAGER_ATR_MEAN_REVERSION", "1.5")),  # 1.5x ATR
            "Breakout": float(os.getenv("POSITION_MANAGER_ATR_BREAKOUT", "2.0")),  # 2x ATR
            "VolatilityBreakout": float(os.getenv("POSITION_MANAGER_ATR_VOL_BREAKOUT", "2.5")),  # 2.5x ATR
            "MomentumRotation": float(os.getenv("POSITION_MANAGER_ATR_MOMENTUM", "2.5")),  # 2.5x ATR
            "RelativeStrength": float(os.getenv("POSITION_MANAGER_ATR_RELATIVE", "2.0")),  # 2x ATR
            "SectorRotation": float(os.getenv("POSITION_MANAGER_ATR_SECTOR", "2.5")),  # 2.5x ATR
            "DualMomentum": float(os.getenv("POSITION_MANAGER_ATR_DUAL", "2.5")),  # 2.5x ATR
            "MovingAverageEnvelope": float(os.getenv("POSITION_MANAGER_ATR_ENVELOPE", "2.0")),  # 2x ATR
            "BollingerBands": float(os.getenv("POSITION_MANAGER_ATR_BOLLINGER", "1.5")),  # 1.5x ATR (mean reversion)
        }
        # Default for unknown strategies
        self.default_atr_multiplier = float(os.getenv("POSITION_MANAGER_ATR_DEFAULT", "2.0"))
        
        logger.info(
            f"PositionManager initialized: "
            f"Hard Stop={self.hard_stop_loss_pct*100}%, "
            f"ATR Multipliers (strategy-specific): {min(self.strategy_atr_multipliers.values())}x-{max(self.strategy_atr_multipliers.values())}x "
            f"(default: {self.default_atr_multiplier}x), "
            f"ATR Period={self.atr_period}"
        )
    
    def check_stops(self) -> Dict[str, Any]:
        """
        Check all open positions and execute stop-loss orders if needed.
        
        This runs every minute during market hours. It's the main entry point
        for position management.
        
        Returns:
            Dictionary with check results:
            {
                "positions_checked": int,
                "stop_orders_placed": int,
                "symbols_exited": List[str],
                "errors": List[str]
            }
        """
        results = {
            "positions_checked": 0,
            "stop_orders_placed": 0,
            "symbols_exited": [],
            "errors": []
        }
        
        try:
            # Get all open positions from Alpaca
            alpaca_positions = self.execution_agent.get_positions()
            results["positions_checked"] = len(alpaca_positions)
            
            if not alpaca_positions:
                logger.debug("No open positions to check")
                return results
            
            # Get strategy info from database for each position
            db_positions = self.db_manager.get_open_positions_with_strategy()
            strategy_map = {pos['symbol']: pos for pos in db_positions}
            
            logger.info(f"Checking {len(alpaca_positions)} open position(s) for stop-loss triggers...")
            
            # Check each position
            for alpaca_pos in alpaca_positions:
                try:
                    symbol = alpaca_pos.symbol
                    entry_price = float(alpaca_pos.avg_entry_price)
                    current_price = float(alpaca_pos.current_price)
                    quantity = int(float(alpaca_pos.qty))
                    
                    # Get strategy context from database
                    strategy_info = strategy_map.get(symbol, {})
                    strategy_name = strategy_info.get('strategy_name', 'Unknown')
                    
                    logger.debug(
                        f"Position {symbol} ({strategy_name}): Entry=${entry_price:.2f}, "
                        f"Current=${current_price:.2f}, Qty={quantity}"
                    )
                    
                    # Check strategy-specific exit conditions first (smart exits)
                    strategy_exit_triggered = self._check_strategy_exit(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        entry_price=entry_price,
                        current_price=current_price
                    )
                    
                    if strategy_exit_triggered:
                        reason = strategy_exit_triggered.get('reason', 'Strategy exit condition')
                        logger.warning(
                            f"🎯 STRATEGY EXIT for {symbol} ({strategy_name}): {reason} - "
                            f"Current ${current_price:.2f} (Entry: ${entry_price:.2f})"
                        )
                        
                        exit_success = self._execute_stop_loss(symbol, quantity, current_price, reason=reason)
                        
                        if exit_success:
                            results["stop_orders_placed"] += 1
                            results["symbols_exited"].append(symbol)
                            logger.info(f"✅ Strategy exit order placed for {symbol}")
                        else:
                            results["errors"].append(f"{symbol}: Failed to execute strategy exit order")
                        continue
                    
                    # Calculate stop-loss price (hard safety stop)
                    stop_price = self._calculate_stop_price(
                        symbol=symbol,
                        entry_price=entry_price,
                        current_price=current_price,
                        strategy_name=strategy_name
                    )
                    
                    if stop_price is None:
                        logger.warning(f"Could not calculate stop price for {symbol}, skipping")
                        results["errors"].append(f"{symbol}: Could not calculate stop price")
                        continue
                    
                    # Check if hard stop-loss triggered
                    if current_price <= stop_price:
                        logger.warning(
                            f"🛑 HARD STOP LOSS TRIGGERED for {symbol} ({strategy_name}): "
                            f"Current ${current_price:.2f} <= Stop ${stop_price:.2f} "
                            f"(Entry: ${entry_price:.2f}, Loss: {((current_price - entry_price) / entry_price * 100):.2f}%)"
                        )
                        
                        # Execute stop-loss order
                        exit_success = self._execute_stop_loss(
                            symbol, quantity, current_price,
                            reason=f"Hard stop loss ({strategy_name})"
                        )
                        
                        if exit_success:
                            results["stop_orders_placed"] += 1
                            results["symbols_exited"].append(symbol)
                            logger.info(f"✅ Stop-loss order placed for {symbol}")
                        else:
                            results["errors"].append(f"{symbol}: Failed to execute stop-loss order")
                    else:
                        # Position is still safe
                        distance_to_stop = ((current_price - stop_price) / current_price) * 100
                        logger.debug(
                            f"Position {symbol} ({strategy_name}) OK: Current ${current_price:.2f} > "
                            f"Stop ${stop_price:.2f} ({distance_to_stop:.2f}% above stop)"
                        )
                        
                except Exception as e:
                    error_msg = f"Error checking position {alpaca_pos.symbol if hasattr(alpaca_pos, 'symbol') else 'unknown'}: {str(e)}"
                    logger.exception(error_msg)
                    results["errors"].append(error_msg)
            
            if results["stop_orders_placed"] > 0:
                logger.warning(
                    f"PositionManager: {results['stop_orders_placed']} position(s) exited "
                    f"due to stop-loss: {', '.join(results['symbols_exited'])}"
                )
            
            return results
            
        except Exception as e:
            error_msg = f"Error in PositionManager.check_stops: {str(e)}"
            logger.exception(error_msg)
            results["errors"].append(error_msg)
            return results
    
    def _check_strategy_exit(
        self,
        symbol: str,
        strategy_name: str,
        entry_price: float,
        current_price: float
    ) -> Optional[Dict[str, str]]:
        """
        Check strategy-specific exit conditions.
        
        Different strategies have different exit logic:
        - TrendFollowing: Exit if price closes below MA200 (trend broken)
        - MeanReversion: Exit if RSI > 70 (target reached) or new low (failure)
        - Breakout: Exit if price falls back into range
        
        Args:
            symbol: Stock symbol
            strategy_name: Strategy name from database
            entry_price: Entry price
            current_price: Current price
            
        Returns:
            Dict with 'reason' if exit triggered, None otherwise
        """
        try:
            # Fetch market data for strategy-specific calculations
            market_data = self.data_agent.fetch_data(
                symbol=symbol,
                timeframe="1Day",
                limit=self.min_bars_for_atr + 10
            )
            
            if not market_data or not market_data.bars or len(market_data.bars) < 30:
                logger.warning(f"Insufficient data for strategy exit check for {symbol}")
                return None
            
            df = market_data.to_dataframe()
            required_cols = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return None
            
            strategy_name_normalized = strategy_name.replace(" ", "").replace("_", "")
            
            # TrendFollowing: Exit if price < MA200
            if strategy_name_normalized in ["TrendFollowing", "MovingAverageCrossover"]:
                if len(df) >= 200:
                    df['MA200'] = df['close'].rolling(window=200).mean()
                    ma200 = df['MA200'].iloc[-1]
                    if pd.notna(ma200) and current_price < ma200:
                        return {"reason": f"Price ${current_price:.2f} below MA200 ${ma200:.2f} - Trend broken"}
            
            # MeanReversion: Exit if RSI > 70 (profit target) or new low (failure)
            elif strategy_name_normalized in ["MeanReversion", "RSIOversoldOverbought", "BollingerBands"]:
                if len(df) >= 30:
                    rsi_series = calculate_rsi(df['close'], period=14)
                    current_rsi = rsi_series.iloc[-1]
                    if pd.notna(current_rsi):
                        if current_rsi > 70:
                            return {"reason": f"RSI {current_rsi:.2f} > 70 - Target reached (mean reversion complete)"}
                        
                        # Check for new low (failure case)
                        recent_low = df['low'].iloc[-20:].min()  # Last 20 days
                        entry_low = df[df.index <= df.index[df['close'] <= entry_price].max() if len(df[df['close'] <= entry_price]) > 0 else -1]['low'].min()
                        if entry_low and recent_low < entry_low * 0.98:  # 2% below entry low
                            return {"reason": f"New low ${recent_low:.2f} - Mean reversion failing"}
            
            # Breakout: Exit if price falls back into Donchian channel
            elif strategy_name_normalized in ["Breakout", "VolatilityBreakout", "ConsolidationBreakout"]:
                if len(df) >= 20:
                    upper_channel = df['high'].rolling(window=20).max()
                    lower_channel = df['low'].rolling(window=20).min()
                    if current_price < upper_channel.iloc[-1] and current_price > lower_channel.iloc[-1]:
                        return {"reason": f"Price ${current_price:.2f} back in range - Breakout failed"}
            
            # Add more strategy-specific exits as needed
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking strategy exit for {symbol} ({strategy_name}): {str(e)}")
            return None
    
    def _calculate_stop_price(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        strategy_name: str = "Unknown"
    ) -> Optional[float]:
        """
        Calculate stop-loss price for a position using strategy-aware logic.
        
        Uses strategy-specific ATR multipliers:
        - Trend Following: 3x ATR (wide stops to ride volatility)
        - Mean Reversion: 1.5x ATR (tight stops - if it doesn't revert quickly, trade is wrong)
        - Other strategies: Configurable defaults
        
        Uses two methods and takes the less aggressive (higher) stop:
        1. Hard stop: entry_price * (1 - hard_stop_loss_pct)
        2. ATR-based stop: entry_price - (strategy_atr_multiplier * ATR)
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price of the position
            current_price: Current market price
            strategy_name: Strategy name for context-aware stop calculation
            
        Returns:
            Stop-loss price or None if calculation fails
        """
        try:
            # Method 1: Hard stop loss (percentage-based)
            hard_stop_price = entry_price * (1 - self.hard_stop_loss_pct)
            
            # Method 2: ATR-based dynamic stop (strategy-aware)
            atr_stop_price = None
            try:
                # Get strategy-specific ATR multiplier
                strategy_name_normalized = strategy_name.replace(" ", "").replace("_", "")
                atr_multiplier = self.strategy_atr_multipliers.get(
                    strategy_name_normalized,
                    self.default_atr_multiplier
                )
                
                # Fetch recent data to calculate ATR
                market_data = self.data_agent.fetch_data(
                    symbol=symbol,
                    timeframe="1Day",
                    limit=self.min_bars_for_atr + 10  # Extra bars for safety
                )
                
                if market_data and market_data.bars and len(market_data.bars) >= self.min_bars_for_atr:
                    df = market_data.to_dataframe()
                    
                    # Ensure we have required columns
                    required_cols = ['high', 'low', 'close']
                    if not all(col in df.columns for col in required_cols):
                        logger.warning(f"Missing required columns for ATR: {df.columns.tolist()}")
                        df = None
                    
                    if df is not None and len(df) >= self.min_bars_for_atr:
                        # Calculate ATR
                        atr_series = calculate_atr(
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            period=self.atr_period
                        )
                        
                        latest_atr = atr_series.iloc[-1]
                        
                        if not pd.isna(latest_atr) and latest_atr > 0:
                            atr_stop_price = entry_price - (atr_multiplier * latest_atr)
                            logger.debug(
                                f"{symbol} ({strategy_name}) ATR stop: Entry ${entry_price:.2f} - "
                                f"({atr_multiplier}x * ${latest_atr:.2f} ATR) = ${atr_stop_price:.2f}"
                            )
                        else:
                            logger.warning(f"Invalid ATR value for {symbol}: {latest_atr}")
                    else:
                        logger.warning(f"Insufficient data for ATR calculation: {len(df)} bars")
                else:
                    logger.warning(f"No market data available for {symbol} ATR calculation")
                    
            except Exception as e:
                logger.warning(f"Could not calculate ATR-based stop for {symbol}: {str(e)}")
            
            # Use the less aggressive (higher) stop price
            # This protects more of the capital
            stop_prices = [hard_stop_price]
            if atr_stop_price is not None:
                stop_prices.append(atr_stop_price)
            
            stop_price = max(stop_prices)
            
            logger.debug(
                f"{symbol} ({strategy_name}) Stop Price Calculation: "
                f"Hard Stop=${hard_stop_price:.2f}, "
                f"ATR Stop=${atr_stop_price:.2f if atr_stop_price else 'N/A'}, "
                f"Final Stop=${stop_price:.2f}"
            )
            
            return stop_price
            
        except Exception as e:
            logger.exception(f"Error calculating stop price for {symbol}: {str(e)}")
            return None
    
    def _execute_stop_loss(self, symbol: str, quantity: int, current_price: float, reason: str = "Stop loss") -> bool:
        """
        Execute a stop-loss order (SELL market order).
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares to sell
            current_price: Current market price (for logging)
            reason: Reason for exit (e.g., "Hard stop loss", "Strategy exit condition")
            
        Returns:
            True if order was successfully placed, False otherwise
        """
        try:
            logger.info(
                f"Executing {reason} SELL order: {symbol} x {quantity} shares @ market price"
            )
            
            # Create order request
            order_request = {
                "symbol": symbol,
                "quantity": quantity,
                "side": OrderSide.SELL.value,
                "order_type": "market"
            }
            
            # Execute via ExecutionAgent
            result = self.execution_agent.process(order_request)
            
            if result and result.get("order_id"):
                logger.info(
                    f"✅ {reason} order executed: {symbol} x {quantity} shares, "
                    f"Order ID: {result.get('order_id')}"
                )
                return True
            else:
                logger.error(f"❌ {reason} order failed for {symbol}: No order ID returned")
                return False
                
        except ExecutionError as e:
            logger.error(f"Execution error placing stop-loss order for {symbol}: {str(e)}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error executing stop-loss for {symbol}: {str(e)}")
            return False

