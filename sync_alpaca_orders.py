"""
Sync actual Alpaca orders to database.

This script fetches all orders from Alpaca and ensures they're logged in the database.
Use this to backfill missing trades or fix discrepancies.

Usage:
    python sync_alpaca_orders.py
    python sync_alpaca_orders.py --days 7
    python sync_alpaca_orders.py --status filled
"""
import argparse
import sys
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from config.settings import get_config
from utils.database import DatabaseManager, TradeHistory
from agents.execution_agent import ExecutionAgent
from models.signal import SignalAction, TradingSignal
from sqlalchemy.orm import Session
from sqlalchemy import and_

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sync_alpaca_orders_to_db(
    execution_agent: ExecutionAgent,
    db_manager: DatabaseManager,
    days: int = 7,
    status_filter: Optional[str] = None
) -> dict:
    """
    Sync Alpaca orders to database.
    
    Args:
        execution_agent: ExecutionAgent instance (has TradingClient)
        db_manager: DatabaseManager instance
        days: Number of days to look back
        status_filter: Optional status filter (e.g., 'filled', 'filled,partially_filled')
    
    Returns:
        Dictionary with sync statistics
    """
    stats = {
        'total_orders': 0,
        'new_trades': 0,
        'updated_trades': 0,
        'skipped': 0,
        'errors': 0
    }
    
    try:
        # Get orders from Alpaca
        from alpaca.trading.enums import QueryOrderStatus
        
        start_time = datetime.now() - timedelta(days=days)
        
        logger.info(f"Fetching orders from Alpaca since {start_time}...")
        
        # Access the TradingClient from ExecutionAgent
        trading_client = execution_agent.client
        
        # Convert status filter to QueryOrderStatus if provided
        status_list = None
        if status_filter:
            status_list = [QueryOrderStatus(status.strip().upper()) for status in status_filter.split(',')]
        
        # Fetch orders from Alpaca
        # Alpaca Python SDK get_orders() accepts: status, after (datetime or ISO string), until
        try:
            # Try using 'after' parameter for date filtering (if supported)
            try:
                if status_list:
                    orders = trading_client.get_orders(
                        status=status_list,
                        after=start_time
                    )
                else:
                    orders = trading_client.get_orders(after=start_time)
            except TypeError:
                # 'after' parameter might not be supported or needs different format
                # Fall back to fetching all orders and filtering in code
                logger.info("'after' parameter not supported, fetching all orders and filtering in code")
                if status_list:
                    orders = trading_client.get_orders(status=status_list)
                else:
                    orders = trading_client.get_orders()
        except Exception as e:
            logger.error(f"Failed to fetch orders from Alpaca: {e}")
            logger.exception("Full error details:")
            return stats
        
        logger.info(f"Found {len(orders)} orders from Alpaca")
        
        # Alpaca API filters by date using 'after' parameter, so we should already have filtered orders
        # But let's double-check and filter in code as well for safety
        filtered_orders = []
        for order in orders:
            # Check order creation/submission date
            order_date = None
            if hasattr(order, 'created_at') and order.created_at:
                order_date = order.created_at
            elif hasattr(order, 'submitted_at') and order.submitted_at:
                order_date = order.submitted_at
            elif hasattr(order, 'updated_at') and order.updated_at:
                order_date = order.updated_at
            
            # Convert to datetime if it's a string
            if isinstance(order_date, str):
                try:
                    from dateutil.parser import parse
                    order_date = parse(order_date)
                except:
                    order_date = None
            
            # Filter by date
            if order_date:
                # Compare dates (handle timezone-aware and naive datetimes)
                if hasattr(order_date, 'replace'):
                    # Make both timezone-naive for comparison
                    if hasattr(order_date, 'tzinfo') and order_date.tzinfo:
                        order_date_naive = order_date.replace(tzinfo=None)
                    else:
                        order_date_naive = order_date
                    
                    if hasattr(start_time, 'tzinfo') and start_time.tzinfo:
                        start_time_naive = start_time.replace(tzinfo=None)
                    else:
                        start_time_naive = start_time
                    
                    if order_date_naive >= start_time_naive:
                        filtered_orders.append(order)
                else:
                    # If we can't compare, include the order (shouldn't happen)
                    filtered_orders.append(order)
            else:
                # Include orders without dates (shouldn't happen, but be safe)
                logger.warning(f"Order {order.id if hasattr(order, 'id') else 'unknown'} has no date field - including anyway")
                filtered_orders.append(order)
        
        logger.info(f"Filtered to {len(filtered_orders)} orders within date range")
        stats['total_orders'] = len(filtered_orders)
        
        # Process each order
        for order in filtered_orders:
            try:
                # Check if trade already exists in DB
                # Convert UUID to string for comparison (SQLite doesn't support UUID type)
                order_id_str = str(order.id) if order.id else None
                
                with db_manager.session() as session:
                    existing_trade = session.query(TradeHistory).filter(
                        TradeHistory.order_id == order_id_str
                    ).first()
                    
                    if existing_trade:
                        # Trade exists - check if we need to update it
                        needs_update = False
                        
                        # Update fill_price if we have it and it's missing
                        if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
                            fill_price = float(order.filled_avg_price)
                            if not existing_trade.fill_price or existing_trade.fill_price != fill_price:
                                existing_trade.fill_price = fill_price
                                needs_update = True
                        
                        # Update executed status if order is filled
                        order_status = order.status.value if hasattr(order.status, 'value') else str(order.status)
                        if order_status in ['filled', 'partially_filled'] and not existing_trade.executed:
                            existing_trade.executed = True
                            needs_update = True
                        
                        if needs_update:
                            session.commit()
                            stats['updated_trades'] += 1
                            logger.info(f"Updated trade for order {order.id} ({order.symbol})")
                        else:
                            stats['skipped'] += 1
                        
                        session.expunge_all()
                        continue
                    
                    # Trade doesn't exist - create it
                    # We need to reconstruct the TradingSignal from the order
                    # Since we don't have the original signal, we'll use default values
                    
                    # Convert Alpaca order side to our SignalAction
                    order_side = order.side.value if hasattr(order.side, 'value') else str(order.side)
                    if order_side.upper() == 'BUY':
                        action = SignalAction.BUY
                    elif order_side.upper() == 'SELL':
                        action = SignalAction.SELL
                    else:
                        action = SignalAction.HOLD
                    
                    # Get fill price
                    fill_price = None
                    if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
                        fill_price = float(order.filled_avg_price)
                    elif hasattr(order, 'limit_price') and order.limit_price:
                        fill_price = float(order.limit_price)
                    
                    # Determine if executed
                    order_status = order.status.value if hasattr(order.status, 'value') else str(order.status)
                    executed = order_status in ['filled', 'partially_filled']
                    
                    # Create a minimal signal for logging
                    # We'll use "SYNCED" as strategy name to indicate this was synced from Alpaca
                    signal = TradingSignal(
                        symbol=order.symbol,
                        action=action,
                        strategy_name="SYNCED_FROM_ALPACA",  # Mark as synced
                        confidence=0.5,  # Default confidence
                        price=fill_price or float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else 0.0,
                        qty=int(order.qty),
                        timestamp=order.created_at if hasattr(order, 'created_at') else datetime.now()
                    )
                    
                    # Create execution result
                    from models.audit import ExecutionResult
                    # Convert UUID to string if needed (SQLite doesn't support UUID type)
                    order_id_str = str(order.id) if order.id else None
                    
                    exec_result = ExecutionResult(
                        signal=signal,
                        order_id=order_id_str,
                        executed=executed,
                        execution_time=order.filled_at if hasattr(order, 'filled_at') and order.filled_at else datetime.now(),
                        fill_price=fill_price
                    )
                    
                    # Log to database
                    db_manager.log_trade(
                        signal=signal,
                        execution_result=exec_result,
                        correlation_id=f"SYNC_{order_id_str}"
                    )
                    
                    stats['new_trades'] += 1
                    logger.info(f"Synced new trade: {order.symbol} {action.value} {order.qty} @ {fill_price or 'N/A'} (Order ID: {order.id})")
                    
            except Exception as e:
                logger.exception(f"Error processing order {order.id if hasattr(order, 'id') else 'unknown'}: {e}")
                stats['errors'] += 1
                continue
        
        return stats
        
    except Exception as e:
        logger.exception(f"Error syncing orders: {e}")
        stats['errors'] += 1
        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sync Alpaca orders to database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync all orders from last 7 days (default)
  python sync_alpaca_orders.py
  
  # Sync orders from last 30 days
  python sync_alpaca_orders.py --days 30
  
  # Sync only filled orders
  python sync_alpaca_orders.py --status filled
  
  # Sync filled and partially filled orders
  python sync_alpaca_orders.py --status filled,partially_filled
        """
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to look back (default: 7)'
    )
    parser.add_argument(
        '--status',
        type=str,
        default=None,
        help='Filter by order status (e.g., "filled", "filled,partially_filled")'
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = get_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Initialize agents
    execution_agent = ExecutionAgent(config=config)
    db_manager = DatabaseManager(config=config)
    
    # Sync orders
    print("\n" + "=" * 80)
    print("🔄 SYNCING ALPACA ORDERS TO DATABASE")
    print("=" * 80 + "\n")
    
    stats = sync_alpaca_orders_to_db(
        execution_agent=execution_agent,
        db_manager=db_manager,
        days=args.days,
        status_filter=args.status
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SYNC SUMMARY")
    print("=" * 80)
    print(f"Total Orders from Alpaca: {stats['total_orders']}")
    print(f"New Trades Added: {stats['new_trades']}")
    print(f"Existing Trades Updated: {stats['updated_trades']}")
    print(f"Skipped (no changes needed): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 80 + "\n")
    
    if stats['new_trades'] > 0 or stats['updated_trades'] > 0:
        print("✅ Database synced successfully!")
        print("\n💡 Tip: Run 'python query_recent_trades.py' to see the synced trades.\n")
    else:
        print("ℹ️  No changes needed - database is up to date.\n")


if __name__ == "__main__":
    main()

