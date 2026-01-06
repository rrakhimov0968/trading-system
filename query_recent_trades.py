"""
Query recent trades from the database with strategy information.

Usage:
    python query_recent_trades.py
    python query_recent_trades.py --days 7
    python query_recent_trades.py --symbol AAPL
"""
import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional, List

from config.settings import get_config
from utils.database import DatabaseManager, TradeHistory
from sqlalchemy.orm import Session
from sqlalchemy import desc


def query_recent_trades(
    db_manager: DatabaseManager,
    days: int = 7,
    symbol: Optional[str] = None,
    limit: int = 100
) -> List[TradeHistory]:
    """
    Query recent trades from the database.
    
    Args:
        db_manager: DatabaseManager instance
        days: Number of days to look back
        symbol: Optional symbol filter
        limit: Maximum number of results
    
    Returns:
        List of TradeHistory records
    """
    try:
        with db_manager.session() as session:
            # Calculate start date
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Build query
            query = session.query(TradeHistory).filter(
                TradeHistory.timestamp >= start_date
            )
            
            # Filter by symbol if provided
            if symbol:
                query = query.filter(TradeHistory.symbol == symbol.upper())
            
            # Order by timestamp (most recent first)
            query = query.order_by(desc(TradeHistory.timestamp)).limit(limit)
            
            # Execute query
            trades = query.all()
            
            # Detach from session
            session.expunge_all()
            
            return trades
            
    except Exception as e:
        print(f"❌ Error querying trades: {e}")
        return []


def print_trades(trades: List[TradeHistory]) -> None:
    """Print trades in a formatted table."""
    if not trades:
        print("\n📊 No trades found in the specified time period.\n")
        return
    
    print("\n" + "=" * 120)
    print(f"📊 RECENT TRADES ({len(trades)} trades)")
    print("=" * 120)
    print(
        f"{'Timestamp':<20} "
        f"{'Symbol':<8} "
        f"{'Action':<6} "
        f"{'Strategy':<25} "
        f"{'Qty':<8} "
        f"{'Price':<12} "
        f"{'Fill Price':<12} "
        f"{'Confidence':<10} "
        f"{'Status'}"
    )
    print("-" * 120)
    
    for trade in trades:
        timestamp_str = trade.timestamp.strftime("%Y-%m-%d %H:%M:%S") if trade.timestamp else "N/A"
        symbol = trade.symbol or "N/A"
        action = trade.action.value if trade.action else "N/A"
        strategy = trade.strategy_name or "N/A"
        qty = trade.qty or 0
        price = f"${trade.price:.2f}" if trade.price else "N/A"
        fill_price = f"${trade.fill_price:.2f}" if trade.fill_price else "N/A"
        confidence = f"{trade.confidence:.2f}" if trade.confidence else "N/A"
        status = "✅ EXECUTED" if trade.executed else "❌ FAILED"
        if trade.error:
            status = f"❌ ERROR: {trade.error[:30]}"
        
        print(
            f"{timestamp_str:<20} "
            f"{symbol:<8} "
            f"{action:<6} "
            f"{strategy:<25} "
            f"{qty:<8} "
            f"{price:<12} "
            f"{fill_price:<12} "
            f"{confidence:<10} "
            f"{status}"
        )
    
    print("=" * 120)
    
    # Summary statistics
    executed = [t for t in trades if t.executed]
    buy_trades = [t for t in executed if t.action and t.action.value == "BUY"]
    sell_trades = [t for t in executed if t.action and t.action.value == "SELL"]
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Executed: {len(executed)}")
    print(f"  BUY Orders: {len(buy_trades)}")
    print(f"  SELL Orders: {len(sell_trades)}")
    
    # Strategy breakdown
    strategy_counts = {}
    for trade in executed:
        strategy = trade.strategy_name or "UNKNOWN"
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    if strategy_counts:
        print(f"\n📊 STRATEGY BREAKDOWN:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy}: {count} trade(s)")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Query recent trades from the database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show trades from last 7 days (default)
  python query_recent_trades.py
  
  # Show trades from last 30 days
  python query_recent_trades.py --days 30
  
  # Show trades for a specific symbol
  python query_recent_trades.py --symbol AAPL
  
  # Show last 50 trades
  python query_recent_trades.py --limit 50
        """
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to look back (default: 7)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help='Filter by symbol (e.g., AAPL)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of trades to show (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = get_config()
    except Exception as e:
        print(f"⚠️  Warning: Could not load config: {e}")
        config = None
    
    # Initialize database manager
    db_manager = DatabaseManager(config=config)
    
    # Query trades
    trades = query_recent_trades(
        db_manager=db_manager,
        days=args.days,
        symbol=args.symbol,
        limit=args.limit
    )
    
    # Print results
    print_trades(trades)


if __name__ == "__main__":
    main()

