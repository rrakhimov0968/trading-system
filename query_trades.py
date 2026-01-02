#!/usr/bin/env python3
"""Script to query trade history from the database."""
import sys
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.database import TradeHistory, Base
from config.settings import get_config


def query_trades(symbol=None, start_date=None, end_date=None, limit=100):
    """Query trades from the database."""
    config = get_config()
    
    # Get database URL (defaults to SQLite)
    db_url = config.database.url if config.database else "sqlite:///trading_system.db"
    
    # Create engine and session
    engine = create_engine(db_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Build query
        query = session.query(TradeHistory)
        
        if symbol:
            query = query.filter(TradeHistory.symbol == symbol.upper())
        
        if start_date:
            query = query.filter(TradeHistory.timestamp >= start_date)
        
        if end_date:
            query = query.filter(TradeHistory.timestamp <= end_date)
        
        # Order by timestamp descending (most recent first)
        query = query.order_by(TradeHistory.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        trades = query.all()
        
        if not trades:
            return []
        
        return trades
        
    finally:
        session.close()


def format_trade(trade):
    """Format a trade for display."""
    status = "✅ EXECUTED" if trade.executed else "❌ FAILED"
    action_emoji = "📈" if trade.action.value == "BUY" else "📉" if trade.action.value == "SELL" else "⏸️"
    
    print(f"\n{action_emoji} {trade.symbol} - {trade.action.value}")
    print(f"   Status: {status}")
    print(f"   Strategy: {trade.strategy_name}")
    print(f"   Quantity: {trade.qty}")
    print(f"   Price: ${trade.price:.2f}")
    if trade.fill_price:
        print(f"   Fill Price: ${trade.fill_price:.2f}")
    if trade.order_id:
        print(f"   Order ID: {trade.order_id}")
    print(f"   Confidence: {trade.confidence:.2%}" if trade.confidence else "   Confidence: N/A")
    print(f"   Timestamp: {trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    if trade.execution_time:
        print(f"   Executed: {trade.execution_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if trade.error:
        print(f"   Error: {trade.error}")
    print("-" * 60)


def main():
    """Main function to query and display trades."""
    print("=" * 60)
    print("TRADE HISTORY QUERY")
    print("=" * 60)
    
    # Query all recent trades (last 7 days)
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    print(f"\n📅 Querying all trades from the last 7 days...")
    
    # Query all recent trades
    all_trades = query_trades(start_date=seven_days_ago, limit=1000)
    
    if not all_trades:
        print("\n❌ No trades found in the database.")
        print("\n💡 This could mean:")
        print("   1. No trades have been executed yet")
        print("   2. The system hasn't run with executed trades")
        print("   3. Trades are still being filtered by RiskAgent (low confidence)")
        return
    
    print(f"\n✅ Found {len(all_trades)} trade(s) in the last 7 days:\n")
    
    # Filter for AAPL and GOOGL
    aapl_trades = [t for t in all_trades if t.symbol == "AAPL"]
    googl_trades = [t for t in all_trades if t.symbol == "GOOGL"]
    
    if aapl_trades:
        print(f"\n🍎 APPLE (AAPL) - {len(aapl_trades)} trade(s):")
        for trade in aapl_trades:
            format_trade(trade)
    else:
        print("\n🍎 APPLE (AAPL) - No trades found")
    
    if googl_trades:
        print(f"\n🔍 GOOGLE (GOOGL) - {len(googl_trades)} trade(s):")
        for trade in googl_trades:
            format_trade(trade)
    else:
        print("\n🔍 GOOGLE (GOOGL) - No trades found")
    
    # Show all trades
    print(f"\n📊 ALL RECENT TRADES ({len(all_trades)} total):")
    for trade in all_trades:
        format_trade(trade)
    
    # Summary statistics
    executed_count = sum(1 for t in all_trades if t.executed)
    buy_count = sum(1 for t in all_trades if t.action.value == "BUY" and t.executed)
    sell_count = sum(1 for t in all_trades if t.action.value == "SELL" and t.executed)
    
    print(f"\n📈 SUMMARY:")
    print(f"   Total trades: {len(all_trades)}")
    print(f"   Executed: {executed_count}")
    print(f"   BUY orders: {buy_count}")
    print(f"   SELL orders: {sell_count}")
    print(f"   Failed: {len(all_trades) - executed_count}")
    
    # Check log files
    print(f"\n📝 Note: You can also check log files in the 'logs/' directory for detailed execution logs.")


if __name__ == "__main__":
    main()
