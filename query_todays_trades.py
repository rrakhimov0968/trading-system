#!/usr/bin/env python3
"""
Query today's trades from the database.

Usage:
    python query_todays_trades.py              # Show all today's trades
    python query_todays_trades.py --buys       # Show only BUY orders
    python query_todays_trades.py --sells      # Show only SELL orders
    python query_todays_trades.py --symbol AAPL  # Filter by symbol
"""
import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_config
from utils.database import DatabaseManager, TradeHistory
from models.signal import SignalAction


def query_todays_trades(
    db_manager: DatabaseManager,
    action: Optional[SignalAction] = None,
    symbol: Optional[str] = None,
    executed_only: bool = True
) -> List[TradeHistory]:
    """
    Query today's trades from the database.
    
    Args:
        db_manager: DatabaseManager instance
        action: Filter by action (BUY/SELL) or None for all
        symbol: Optional symbol filter
        executed_only: Only show executed trades (default: True)
    
    Returns:
        List of TradeHistory records
    """
    try:
        with db_manager.session() as session:
            # Get today's date range
            # Try local time first (most common)
            local_now = datetime.now()
            local_today_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Also try UTC (in case timestamps are stored in UTC)
            from datetime import timezone
            utc_now = datetime.now(timezone.utc)
            utc_today_start = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Build query - check if timestamp is today (either local or UTC)
            from sqlalchemy import or_
            query = session.query(TradeHistory).filter(
                or_(
                    # Local timezone range
                    (
                        (TradeHistory.timestamp >= local_today_start) & 
                        (TradeHistory.timestamp <= local_now)
                    ),
                    # UTC timezone range  
                    (
                        (TradeHistory.timestamp >= utc_today_start) & 
                        (TradeHistory.timestamp <= utc_now)
                    )
                )
            )
            
            # Filter by action if provided
            if action:
                query = query.filter(TradeHistory.action == action)
            
            # Filter by symbol if provided
            if symbol:
                query = query.filter(TradeHistory.symbol == symbol.upper())
            
            # Filter executed trades if requested
            if executed_only:
                query = query.filter(TradeHistory.executed == True)
            
            # Order by timestamp (most recent first)
            trades = query.order_by(TradeHistory.timestamp.desc()).all()
            
            # Detach from session
            session.expunge_all()
            
            return trades
            
    except Exception as e:
        print(f"❌ Error querying trades: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return []


def format_trade(trade: TradeHistory) -> str:
    """Format a single trade for display."""
    timestamp_str = trade.timestamp.strftime("%Y-%m-%d %H:%M:%S") if trade.timestamp else "N/A"
    action_emoji = "📈" if trade.action.value == "BUY" else "📉" if trade.action.value == "SELL" else "⏸️"
    status_emoji = "✅" if trade.executed else "❌"
    
    lines = [
        f"{action_emoji} {status_emoji} {trade.symbol} {trade.action.value}",
        f"   Strategy: {trade.strategy_name or 'N/A'}",
        f"   Quantity: {trade.qty or 0} shares",
        f"   Price: ${trade.price:.2f}" if trade.price else "   Price: N/A",
    ]
    
    if trade.fill_price:
        lines.append(f"   Fill Price: ${trade.fill_price:.2f}")
    
    if trade.confidence:
        lines.append(f"   Confidence: {trade.confidence:.2%}")
    
    lines.append(f"   Time: {timestamp_str}")
    
    if trade.order_id:
        lines.append(f"   Order ID: {trade.order_id}")
    
    if trade.error:
        lines.append(f"   Error: {trade.error}")
    
    return "\n".join(lines)


def print_trades_table(trades: List[TradeHistory]) -> None:
    """Print trades in a formatted table."""
    if not trades:
        print("\n📊 No trades found for today.\n")
        return
    
    print("\n" + "=" * 140)
    print(f"📊 TODAY'S TRADES ({len(trades)} total)")
    print("=" * 140)
    print(
        f"{'Time':<20} "
        f"{'Action':<6} "
        f"{'Symbol':<8} "
        f"{'Strategy':<25} "
        f"{'Qty':<8} "
        f"{'Price':<12} "
        f"{'Fill Price':<12} "
        f"{'Amount':<15} "
        f"{'Status':<12}"
    )
    print("-" * 140)
    
    total_buy_amount = 0
    total_sell_amount = 0
    
    for trade in trades:
        timestamp_str = trade.timestamp.strftime("%Y-%m-%d %H:%M:%S") if trade.timestamp else "N/A"
        action = trade.action.value if trade.action else "N/A"
        action_display = f"📈 {action}" if action == "BUY" else f"📉 {action}" if action == "SELL" else action
        symbol = trade.symbol or "N/A"
        strategy = (trade.strategy_name or "N/A")[:23]
        qty = trade.qty or 0
        price = f"${trade.price:.2f}" if trade.price else "N/A"
        fill_price = f"${trade.fill_price:.2f}" if trade.fill_price else "N/A"
        
        # Calculate amount
        if trade.executed and trade.fill_price and trade.qty:
            amount = trade.fill_price * trade.qty
            amount_str = f"${amount:,.2f}"
            if action == "BUY":
                total_buy_amount += amount
            else:
                total_sell_amount += amount
        else:
            amount_str = "N/A"
        
        status = "✅ EXECUTED" if trade.executed else "❌ FAILED"
        if trade.error:
            status = f"❌ {trade.error[:10]}"
        
        print(
            f"{timestamp_str:<20} "
            f"{action_display:<8} "
            f"{symbol:<8} "
            f"{strategy:<25} "
            f"{qty:<8} "
            f"{price:<12} "
            f"{fill_price:<12} "
            f"{amount_str:<15} "
            f"{status:<12}"
        )
    
    print("=" * 140)
    
    # Summary
    executed = [t for t in trades if t.executed]
    buy_trades = [t for t in executed if t.action and t.action.value == "BUY"]
    sell_trades = [t for t in executed if t.action and t.action.value == "SELL"]
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Executed: {len(executed)}")
    print(f"  Failed: {len(trades) - len(executed)}")
    print(f"  BUY Orders: {len(buy_trades)}")
    print(f"  SELL Orders: {len(sell_trades)}")
    
    if total_buy_amount > 0:
        print(f"  Total BUY Amount: ${total_buy_amount:,.2f}")
    if total_sell_amount > 0:
        print(f"  Total SELL Amount: ${total_sell_amount:,.2f}")
    
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
        description='Query today\'s trades from the database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all today's trades
  python query_todays_trades.py
  
  # Show only today's BUY orders
  python query_todays_trades.py --buys
  
  # Show only today's SELL orders
  python query_todays_trades.py --sells
  
  # Show trades for a specific symbol
  python query_todays_trades.py --symbol AAPL
  
  # Include failed trades
  python query_todays_trades.py --include-failed
        """
    )
    
    parser.add_argument(
        '--buys',
        action='store_true',
        help='Show only BUY orders'
    )
    parser.add_argument(
        '--sells',
        action='store_true',
        help='Show only SELL orders'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help='Filter by symbol (e.g., AAPL)'
    )
    parser.add_argument(
        '--include-failed',
        action='store_true',
        help='Include failed/non-executed trades'
    )
    
    args = parser.parse_args()
    
    # Determine action filter
    action = None
    if args.buys:
        action = SignalAction.BUY
    elif args.sells:
        action = SignalAction.SELL
    
    # Load config
    try:
        config = get_config()
    except Exception as e:
        print(f"⚠️  Warning: Could not load config: {e}")
        config = None
    
    # Initialize database manager
    db_manager = DatabaseManager(config=config)
    
    # Query trades
    trades = query_todays_trades(
        db_manager=db_manager,
        action=action,
        symbol=args.symbol,
        executed_only=not args.include_failed
    )
    
    # If no trades found, check if there are any recent trades at all
    if not trades:
        print(f"\n⚠️  No trades found for today.")
        print(f"   Checking for recent trades in the database...")
        
        # Query last 7 days as a fallback
        recent_trades = []
        try:
            # Try to import and use the query_recent_trades function
            import importlib.util
            spec = importlib.util.spec_from_file_location("query_recent_trades", 
                                                         os.path.join(os.path.dirname(__file__), "query_recent_trades.py"))
            query_recent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(query_recent_module)
            
            recent_trades = query_recent_module.query_recent_trades(
                db_manager=db_manager,
                days=7,
                symbol=args.symbol,
                limit=10
            )
        except Exception as e:
            print(f"   (Could not check recent trades: {e})")
            print(f"   💡 Try running: python query_recent_trades.py")
        
        if recent_trades:
            print(f"\n   ✅ Found {len(recent_trades)} recent trade(s) in the last 7 days:")
            print(f"   (Most recent first)\n")
            for trade in recent_trades[:5]:  # Show first 5
                ts = trade.timestamp.strftime("%Y-%m-%d %H:%M:%S") if trade.timestamp else "N/A"
                action_str = trade.action.value if trade.action else "N/A"
                print(f"   • {ts} | {trade.symbol} | {action_str} | {trade.strategy_name or 'N/A'}")
            print(f"\n   💡 Tip: Use 'python query_recent_trades.py' to see all recent trades")
        else:
            print(f"   ❌ No trades found in the database at all.")
            print(f"   💡 This might mean:")
            print(f"      - No trades have been executed yet")
            print(f"      - Trades are still being filtered by RiskAgent")
            print(f"      - The trading system hasn't run today")
        return
    
    # Print results
    if args.buys:
        print(f"\n🛒 TODAY'S PURCHASES (BUY ORDERS)")
    elif args.sells:
        print(f"\n💰 TODAY'S SALES (SELL ORDERS)")
    else:
        print(f"\n📊 TODAY'S TRADES")
    
    print_trades_table(trades)


if __name__ == "__main__":
    main()

