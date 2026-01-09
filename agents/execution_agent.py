"""Execution agent for trade execution via Alpaca API."""
from typing import List, Optional, Dict, Any
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce, TimeInForce as AlpacaTimeInForce
from alpaca.common.exceptions import APIError as AlpacaAPIError

from agents.base import BaseAgent
from models.enums import OrderSide
from models.validation import (
    validate_symbol,
    validate_quantity,
    validate_order_side
)
from utils.exceptions import ExecutionError, APIError, ValidationError
from utils.retry import retry_with_backoff, RetryConfig
from utils.rate_limiter import get_rate_limiter


class ExecutionAgent(BaseAgent):
    """
    Agent responsible for executing trades via broker API.
    
    This is a code-only agent with no LLM involvement in the execution path.
    Focus: Reliability, speed, error handling, logging.
    """
    
    def __init__(self, config=None):
        """
        Initialize the execution agent.
        
        Args:
            config: Application configuration. If None, loads from environment.
        """
        super().__init__(config)
        
        # Fractional shares support flags
        self._fractional_supported = None
        self._fractional_test_done = False
        
        # Initialize rate limiter for API calls
        # Use same limit as DataAgent: 200 requests/minute (shared limit for account)
        # Note: Alpaca's limit is per account, so we need to be conservative
        self.rate_limiter = get_rate_limiter(
            name="alpaca_trading",  # Shared with other trading operations
            max_requests=200,  # Same as data API - shared account limit
            window_seconds=60
        )
        self.log_info(f"Rate limiter configured: 200 requests/minute")
        
        # Initialize Alpaca client
        try:
            self.client = TradingClient(
                api_key=self.config.alpaca.api_key,
                secret_key=self.config.alpaca.secret_key,
                paper=self.config.alpaca.paper,
                url_override=self.config.alpaca.base_url
            )
            self.log_info(
                f"ExecutionAgent initialized (mode: {'PAPER' if self.config.alpaca.paper else 'LIVE'})"
            )
        except Exception as e:
            error = self.handle_error(
                e,
                context={"component": "TradingClient", "config": self.config.alpaca.paper}
            )
            raise ExecutionError(
                "Failed to initialize Alpaca client",
                correlation_id=self._correlation_id,
                details={"original_error": str(e)}
            ) from error
    
    def process(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an order request.
        
        This is the main entry point for the agent.
        
        Args:
            order_request: Dictionary with order details:
                - symbol: str
                - quantity: int
                - side: str ("buy" or "sell")
                - order_type: str (default: "market")
                - limit_price: Optional[float] (for limit orders)
        
        Returns:
            Dictionary with execution result
        
        Raises:
            ExecutionError: If execution fails
            ValidationError: If order request is invalid
        """
        self.generate_correlation_id()
        self.log_info("Processing order request", order_request=order_request)
        
        try:
            # Validate inputs
            symbol = validate_symbol(order_request["symbol"])
            qty = validate_quantity(order_request["quantity"])
            side = validate_order_side(order_request["side"])
            
            # Execute order
            if order_request.get("order_type", "market") == "market":
                order = self.place_market_order(symbol, qty, side)
            else:
                raise ExecutionError(
                    f"Unsupported order type: {order_request['order_type']}",
                    correlation_id=self._correlation_id
                )
            
            # Get fill price if available (for filled orders)
            fill_price = None
            if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
                fill_price = float(order.filled_avg_price)
            elif hasattr(order, 'filled_price') and order.filled_price:
                fill_price = float(order.filled_price)
            
            result = {
                "order_id": order.id,
                "symbol": order.symbol,
                "quantity": order.qty,
                "side": order.side.value,
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "fill_price": fill_price,  # Include fill_price if available
                "correlation_id": self._correlation_id
            }
            
            self.log_info("Order executed successfully", result=result)
            return result
            
        except ValidationError as e:
            raise
        except Exception as e:
            error = self.handle_error(e, context={"order_request": order_request})
            raise ExecutionError(
                f"Failed to process order: {str(e)}",
                correlation_id=self._correlation_id,
                details={"original_error": str(e)}
            ) from error
    
    @retry_with_backoff(
        config=RetryConfig(max_attempts=3, initial_delay=1.0),
    )
    def get_account(self):
        """
        Get account information with retry logic.
        
        Returns:
            Account object from Alpaca API
        
        Raises:
            APIError: If API call fails after retries
        """
        try:
            # Acquire rate limiter before making API call
            if not self.rate_limiter.acquire(blocking=True, timeout=60.0):
                raise APIError(
                    "Rate limiter timeout - too many requests",
                    status_code=429,
                    correlation_id=self._correlation_id
                )
            
            self.log_debug("Fetching account information")
            account = self.client.get_account()
            self.log_info(
                "Account information retrieved",
                cash=account.cash,
                buying_power=account.buying_power,
                equity=account.equity
            )
            return account
        except AlpacaAPIError as e:
            self.log_exception("Failed to get account info", e)
            raise APIError(
                f"Alpaca API error: {str(e)}",
                status_code=getattr(e, 'status_code', None),
                correlation_id=self._correlation_id
            ) from e
        except Exception as e:
            self.log_exception("Unexpected error getting account info", e)
            raise APIError(
                f"Unexpected error: {str(e)}",
                correlation_id=self._correlation_id
            ) from e
    
    def place_market_order(
        self, 
        symbol: str, 
        qty: int, 
        side: OrderSide
    ) -> Any:
        """
        Place a market order with validation and error handling.
        
        Args:
            symbol: Stock symbol (validated)
            qty: Quantity (validated)
            side: Order side enum
        
        Returns:
            Order object from Alpaca API
        
        Raises:
            ExecutionError: If order placement fails
        """
        self.log_info(
            f"Placing market order: {side.value} {qty} shares of {symbol}",
            symbol=symbol,
            quantity=qty,
            side=side.value
        )
        
        try:
            # Acquire rate limiter before making API call
            if not self.rate_limiter.acquire(blocking=True, timeout=60.0):
                raise ExecutionError(
                    "Rate limiter timeout - too many requests",
                    correlation_id=self._correlation_id
                )
            
            # Convert our OrderSide enum to Alpaca's enum
            alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
            
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self._submit_order_with_retry(order_data)
            
            self.log_info(
                f"Order placed successfully: {order.id}",
                order_id=order.id,
                symbol=symbol,
                quantity=qty,
                side=side.value
            )
            
            return order
            
        except AlpacaAPIError as e:
            self.log_exception(f"Alpaca API error placing order for {symbol}", e)
            raise ExecutionError(
                f"Failed to place order: {str(e)}",
                correlation_id=self._correlation_id,
                details={
                    "symbol": symbol,
                    "quantity": qty,
                    "side": side.value,
                    "api_status_code": getattr(e, 'status_code', None)
                }
            ) from e
        except Exception as e:
            self.log_exception(f"Unexpected error placing order for {symbol}", e)
            raise ExecutionError(
                f"Unexpected error placing order: {str(e)}",
                correlation_id=self._correlation_id,
                details={"symbol": symbol, "quantity": qty, "side": side.value}
            ) from e
    
    @retry_with_backoff(
        config=RetryConfig(max_attempts=3, initial_delay=1.0),
    )
    def _submit_order_with_retry(self, order_data: MarketOrderRequest) -> Any:
        """Submit order with retry logic (internal method)."""
        return self.client.submit_order(order_data)
    
    def get_positions(self) -> List[Any]:
        """
        Get all open positions with error handling.
        
        Returns:
            List of position objects from Alpaca API
        
        Raises:
            APIError: If API call fails
        """
        try:
            # Acquire rate limiter before making API call
            if not self.rate_limiter.acquire(blocking=True, timeout=60.0):
                raise APIError(
                    "Rate limiter timeout - too many requests",
                    status_code=429,
                    correlation_id=self._correlation_id
                )
            
            self.log_debug("Fetching open positions")
            positions = self.client.get_all_positions()
            self.log_info(f"Retrieved {len(positions)} open positions")
            return positions
        except AlpacaAPIError as e:
            self.log_exception("Failed to get positions", e)
            raise APIError(
                f"Alpaca API error: {str(e)}",
                status_code=getattr(e, 'status_code', None),
                correlation_id=self._correlation_id
            ) from e
        except Exception as e:
            self.log_exception("Unexpected error getting positions", e)
            raise APIError(
                f"Unexpected error: {str(e)}",
                correlation_id=self._correlation_id
            ) from e
    
    def validate_fractional_support(self) -> bool:
        """
        Validate that fractional shares are actually supported.
        
        Config saying "yes" doesn't mean broker allows it!
        This performs an actual test at startup.
        
        Returns:
            True if fractional shares work, False otherwise
        """
        if self._fractional_test_done:
            return self._fractional_supported
        
        self.log_info("🧪 Testing fractional shares support...")
        
        try:
            # Check account properties first
            account = self.get_account()
            
            # Alpaca-specific check
            if hasattr(account, 'fractional_trading'):
                if not account.fractional_trading:
                    self.log_error(
                        "🚫 Fractional shares NOT supported: "
                        "Account property 'fractional_trading' = False"
                    )
                    self._fractional_supported = False
                    self._fractional_test_done = True
                    return False
            
            # Perform actual test order (only in paper mode)
            if self.config.alpaca.paper:
                try:
                    from alpaca.trading.requests import MarketOrderRequest
                    test_order = self.client.submit_order(
                        MarketOrderRequest(
                            symbol='SPY',
                            notional=10.0,  # $10 test order
                            side=AlpacaOrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                    )
                    
                    # Cancel immediately
                    self.client.cancel_order(test_order.id)
                    
                    self.log_info("✅ Fractional shares supported and tested successfully")
                    self._fractional_supported = True
                    self._fractional_test_done = True
                    return True
                    
                except Exception as order_error:
                    error_msg = str(order_error).lower()
                    
                    # Check for specific fractional-not-supported errors
                    if 'fractional' in error_msg or 'notional' in error_msg:
                        self.log_error(
                            f"🚫 Fractional shares NOT supported: {order_error}"
                        )
                        self._fractional_supported = False
                        self._fractional_test_done = True
                        return False
                    else:
                        # Some other error - assume fractional works
                        self.log_warning(
                            f"⚠️  Fractional test inconclusive: {order_error}. "
                            f"Assuming supported."
                        )
                        self._fractional_supported = True
                        self._fractional_test_done = True
                        return True
            else:
                # Live mode - don't test with real order, check account property
                self.log_info("✅ Fractional shares assumed supported (live mode)")
                self._fractional_supported = True
                self._fractional_test_done = True
                return True
        
        except Exception as e:
            self.log_error(f"🚫 Fractional support validation failed: {e}")
            self._fractional_supported = False
            self._fractional_test_done = True
            return False
    
    def check_symbol_liquidity(
        self,
        symbol: str,
        proposed_notional: float
    ) -> tuple[bool, str]:
        """
        Check if symbol has adequate liquidity for fractional trading.
        
        Avoid fractional shares on:
        - Low volume stocks (< 1M avg daily volume)
        - Very small notionals (< 1% of share price)
        
        Args:
            symbol: Stock symbol
            proposed_notional: Dollar amount of proposed order
        
        Returns:
            (approved: bool, reason: str)
        """
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            
            if hist.empty:
                return False, "No recent price data"
            
            # Check average volume
            avg_volume = hist['Volume'].mean()
            if avg_volume < 1_000_000:
                reason = (
                    f"Low liquidity: {symbol} avg volume = {avg_volume:,.0f} "
                    f"(< 1M). Fractional execution may be poor."
                )
                self.log_warning(f"⚠️  {reason}")
                return False, reason
            
            # Check if notional is reasonable relative to share price
            current_price = hist['Close'].iloc[-1]
            min_notional = current_price * 0.01  # At least 1% of share price
            
            if proposed_notional < min_notional:
                reason = (
                    f"Notional too small: ${proposed_notional:.2f} "
                    f"for {symbol} @ ${current_price:.2f} "
                    f"(< 1% of share price)"
                )
                self.log_warning(f"⚠️  {reason}")
                return False, reason
            
            self.log_debug(f"✅ {symbol} liquidity OK for fractional")
            return True, "Liquidity adequate"
            
        except Exception as e:
            self.log_warning(f"⚠️  Liquidity check failed for {symbol}: {e}")
            # Default to allowing if check fails
            return True, f"Check failed: {e}"
    
    def place_order_with_fallback(
        self,
        symbol: str,
        qty: float,  # Can be fractional
        side: OrderSide,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        time_in_force: str = 'day'
    ) -> Optional[Any]:
        """
        Place order with automatic fallback from fractional to whole shares.
        
        Execution strategy:
        1. Try fractional (notional-based) if qty is fractional
        2. If fractional fails, fallback to whole shares (floor)
        3. If whole shares = 0, reject order
        
        Args:
            symbol: Stock symbol
            qty: Number of shares (can be fractional)
            side: OrderSide enum
            order_type: 'market' or 'limit'
            limit_price: Limit price (if order_type='limit')
            time_in_force: 'day', 'gtc', etc.
        
        Returns:
            Order object if successful, None if failed
        """
        is_fractional = (qty != int(qty))
        
        # Strategy 1: Try fractional if applicable
        if is_fractional:
            self.log_info(
                f"📊 Attempting fractional order: {symbol} {qty:.4f} shares ({side.value})"
            )
            
            # Check if fractional is supported
            if not self.validate_fractional_support():
                self.log_warning(
                    f"⚠️  Fractional not supported, will try whole shares"
                )
            else:
                # Get current price for notional calculation
                try:
                    # Try to get latest quote
                    quote = self.client.get_latest_quote(symbol)
                    if quote and hasattr(quote, 'ask_price'):
                        current_price = float(quote.ask_price)
                    else:
                        # Fallback: use order price if available, or estimate
                        current_price = limit_price if limit_price else qty * 100  # Rough estimate
                        self.log_warning(f"⚠️  Using estimated price for {symbol}")
                except Exception as e:
                    self.log_warning(f"⚠️  Failed to get price for {symbol}: {e}")
                    current_price = limit_price if limit_price else 100.0
                
                notional = qty * current_price
                
                liquidity_ok, reason = self.check_symbol_liquidity(
                    symbol, notional
                )
                
                if not liquidity_ok:
                    self.log_warning(
                        f"⚠️  Skipping fractional due to liquidity: {reason}"
                    )
                else:
                    # Try fractional order
                    try:
                        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
                        from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
                        
                        alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
                        tif_map = {'day': AlpacaTimeInForce.DAY, 'gtc': AlpacaTimeInForce.GTC}
                        tif = tif_map.get(time_in_force.lower(), AlpacaTimeInForce.DAY)
                        
                        if order_type == 'market':
                            order_data = MarketOrderRequest(
                                symbol=symbol,
                                notional=notional,
                                side=alpaca_side,
                                time_in_force=tif
                            )
                        else:
                            order_data = LimitOrderRequest(
                                symbol=symbol,
                                notional=notional,
                                side=alpaca_side,
                                limit_price=limit_price or current_price,
                                time_in_force=tif
                            )
                        
                        order = self.client.submit_order(order_data)
                        
                        self.log_info(
                            f"✅ Fractional order placed: {symbol} "
                            f"${notional:.2f} ({qty:.4f} shares)"
                        )
                        return order
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        self.log_warning(
                            f"⚠️  Fractional order failed: {e}. "
                            f"Attempting whole share fallback..."
                        )
        
        # Strategy 2: Fallback to whole shares
        whole_shares = int(qty)
        
        if whole_shares == 0:
            self.log_error(
                f"🚫 Cannot place order: {symbol} qty too small "
                f"({qty:.4f} → {whole_shares} whole shares)"
            )
            return None
        
        # Use existing place_market_order for whole shares
        try:
            order = self.place_market_order(symbol, whole_shares, side)
            
            if is_fractional:
                self.log_info(
                    f"✅ Whole share fallback successful: {symbol} "
                    f"{whole_shares} shares (wanted {qty:.4f})"
                )
            
            return order
            
        except Exception as e:
            self.log_error(f"🚫 Order failed: {symbol} {side.value} {whole_shares} shares - {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health by attempting to get account info."""
        health = super().health_check()
        try:
            account = self.get_account()
            health.update({
                "account_accessible": True,
                "trading_mode": "paper" if self.config.alpaca.paper else "live",
                "buying_power": str(account.buying_power),
                "fractional_supported": self._fractional_supported,
                "fractional_tested": self._fractional_test_done
            })
        except Exception as e:
            health.update({
                "status": "unhealthy",
                "account_accessible": False,
                "error": str(e)
            })
        return health