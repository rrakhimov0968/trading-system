#!/usr/bin/env python3
"""
Sync current Alpaca positions to database as trades.

This script fetches current open positions from Alpaca and logs them as executed BUY trades
in the database. This is useful when orders aren't syncing properly.

Usage:
    python sync_alpaca_positions.py
    python sync_alpaca_positions.py --dry-run  # Preview without saving
"""
import argparse
import sys
import os
from datetime import datetime
from typing import List, Optional
import logging

from config.settings import get_config
from utils.database import DatabaseManager
from agents.execution_agent import ExecutionAgent
from models.signal import SignalAction, TradingSignal
from models.audit import ExecutionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sync_positions_to_trades(
    execution_agent: ExecutionAgent,
    db_manager: DatabaseManager,
    dry_run: bool = False
) -> dict:
    """
    Sync current Alpaca positions to database as trades.
    
    Args:
        execution_agent: ExecutionAgent instance (has TradingClient)
        db_manager: DatabaseManager instance
        dry_run: If True, don't save to database, just print
    
    Returns:
        Dictionary with sync statistics
    """
    stats = {
        'positions_found': 0,
        'trades_created': 0,
        'trades_updated': 0,
        'skipped': 0,
        'errors': 0
    }
    
    try:
        # Get current positions from Alpaca
        logger.info("Fetching current positions from Alpaca...")
        positions = execution_agent.get_positions()
        
        if not positions:
            logger.info("No open positions found in Alpaca")
            return stats
        
        logger.info(f"Found {len(positions)} open position(s)")
        stats['positions_found'] = len(positions)
        
        # Process each position
        for position in positions:
            try:
                symbol = position.symbol
                qty = int(float(position.qty))
                
                # Get position details
                avg_entry_price = float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') and position.avg_entry_price else None
                current_price = float(position.current_price) if hasattr(position, 'current_price') and position.current_price else None
                market_value = float(position.market_value) if hasattr(position, 'market_value') and position.market_value else None
                
                # Use avg_entry_price as fill_price if available, otherwise current_price
                fill_price = avg_entry_price or current_price or 0.0
                
                logger.info(f"\nPosition: {symbol}")
                logger.info(f"  Quantity: {qty} shares")
                logger.info(f"  Avg Entry Price: ${avg_entry_price:.2f}" if avg_entry_price else "  Avg Entry Price: N/A")
                logger.info(f"  Current Price: ${current_price:.2f}" if current_price else "  Current Price: N/A")
                logger.info(f"  Market Value: ${market_value:.2f}" if market_value else "  Market Value: N/A")
                
                if dry_run:
                    logger.info(f"  [DRY RUN] Would create BUY trade: {qty} {symbol} @ ${fill_price:.2f}")
                    stats['trades_created'] += 1
                    continue
                
                # Check if we already have a recent BUY trade for this position
                # Look for executed BUY orders for this symbol
                with db_manager.session() as session:
                    from utils.database import TradeHistory
                    from sqlalchemy import desc
                    
                    existing_buy = session.query(TradeHistory).filter(
                        TradeHistory.symbol == symbol,
                        TradeHistory.action == SignalAction.BUY,
                        TradeHistory.executed == True
                    ).order_by(desc(TradeHistory.timestamp)).first()
                    
                    if existing_buy:
                        # Update existing trade if needed
                        needs_update = False
                        
                        if not existing_buy.fill_price or existing_buy.fill_price != fill_price:
                            existing_buy.fill_price = fill_price
                            existing_buy.price = fill_price
                            needs_update = True
                        
                        if not existing_buy.executed:
                            existing_buy.executed = True
                            needs_update = True
                        
                        if existing_buy.qty != qty:
                            # Quantity changed - might be multiple positions combined
                            logger.info(f"  Existing trade has qty={existing_buy.qty}, position has qty={qty}")
                            # Update to match current position
                            existing_buy.qty = qty
                            needs_update = True
                        
                        if needs_update:
                            session.commit()
                            stats['trades_updated'] += 1
                            logger.info(f"  ✅ Updated existing trade in database")
                        else:
                            stats['skipped'] += 1
                            logger.info(f"  ⏭️  Trade already exists and is up to date")
                        
                        session.expunge_all()
                        continue
                
                # Create new trade entry
                # Use a timestamp from today (or position creation time if available)
                position_timestamp = datetime.now()
                if hasattr(position, 'created_at') and position.created_at:
                    try:
                        position_timestamp = position.created_at
                        if isinstance(position_timestamp, str):
                            from dateutil.parser import parse
                            position_timestamp = parse(position_timestamp)
                    except:
                        position_timestamp = datetime.now()
                
                # Create a TradingSignal for this position
                signal = TradingSignal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    strategy_name="SYNCED_FROM_ALPACA_POSITION",  # Mark as synced from position
                    confidence=1.0,  # Position exists, so 100% confidence it was executed
                    price=fill_price,
                    qty=qty,
                    timestamp=position_timestamp
                )
                
                # Create execution result
                exec_result = ExecutionResult(
                    signal=signal,
                    order_id=f"POSITION_{symbol}_{position_timestamp.isoformat()}",
                    executed=True,  # Position exists = trade was executed
                    execution_time=position_timestamp,
                    fill_price=fill_price
                )
                
                # Log to database
                correlation_id = f"SYNC_POS_{symbol}_{datetime.now().isoformat()}"
                db_manager.log_trade(
                    signal=signal,
                    execution_result=exec_result,
                    correlation_id=correlation_id
                )
                
                stats['trades_created'] += 1
                logger.info(f"  ✅ Created new trade in database: {qty} {symbol} @ ${fill_price:.2f}")
                
            except Exception as e:
                logger.exception(f"Error processing position {position.symbol if hasattr(position, 'symbol') else 'unknown'}: {e}")
                stats['errors'] += 1
                continue
        
        return stats
        
    except Exception as e:
        logger.exception(f"Error syncing positions: {e}")
        stats['errors'] += 1
        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sync current Alpaca positions to database as trades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync positions to database
  python sync_alpaca_positions.py
  
  # Preview what would be synced (dry run)
  python sync_alpaca_positions.py --dry-run
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without saving to database'
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
    
    # Sync positions
    print("\n" + "=" * 80)
    if args.dry_run:
        print("🔍 DRY RUN: PREVIEW SYNCING ALPACA POSITIONS TO DATABASE")
    else:
        print("🔄 SYNCING ALPACA POSITIONS TO DATABASE")
    print("=" * 80 + "\n")
    
    stats = sync_positions_to_trades(
        execution_agent=execution_agent,
        db_manager=db_manager,
        dry_run=args.dry_run
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SYNC SUMMARY")
    print("=" * 80)
    print(f"Positions Found: {stats['positions_found']}")
    if args.dry_run:
        print(f"Trades That Would Be Created: {stats['trades_created']}")
    else:
        print(f"New Trades Created: {stats['trades_created']}")
        print(f"Existing Trades Updated: {stats['trades_updated']}")
        print(f"Skipped (no changes needed): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print("=" * 80 + "\n")
    
    if args.dry_run:
        print("💡 This was a dry run. Run without --dry-run to actually save to database.\n")
    elif stats['trades_created'] > 0 or stats['trades_updated'] > 0:
        print("✅ Positions synced successfully!")
        print("\n💡 Tip: Run 'python query_todays_trades.py --buys' to see the synced trades.\n")
    else:
        print("ℹ️  No changes needed - positions are already synced.\n")


if __name__ == "__main__":
    main()

