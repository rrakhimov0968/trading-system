"""Execution agent for trade execution via Alpaca API."""
from typing import List, Optional, Dict, Any
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce
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
            
            result = {
                "order_id": order.id,
                "symbol": order.symbol,
                "quantity": order.qty,
                "side": order.side.value,
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
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
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health by attempting to get account info."""
        health = super().health_check()
        try:
            account = self.get_account()
            health.update({
                "account_accessible": True,
                "trading_mode": "paper" if self.config.alpaca.paper else "live",
                "buying_power": str(account.buying_power)
            })
        except Exception as e:
            health.update({
                "status": "unhealthy",
                "account_accessible": False,
                "error": str(e)
            })
        return health