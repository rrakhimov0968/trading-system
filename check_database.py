"""
Check database setup and connectivity.

Usage:
    python check_database.py
"""
import sys
from datetime import datetime

from config.settings import get_config
from utils.database import DatabaseManager, Base, TradeHistory
from sqlalchemy import inspect, text


def check_database_setup():
    """Check if database is set up correctly."""
    print("\n" + "=" * 80)
    print("🔍 DATABASE SETUP CHECK")
    print("=" * 80 + "\n")
    
    try:
        # Load config
        config = get_config()
        print(f"✅ Config loaded successfully")
        print(f"   Database URL: {config.database.url}")
        print()
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(config=config)
        print(f"✅ DatabaseManager initialized")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize DatabaseManager: {e}")
        return False
    
    # Check database connection
    try:
        health = db_manager.health_check()
        if health.get('status') == 'healthy':
            print(f"✅ Database connection: {health['status']}")
            print(f"   Database: {health.get('database', 'N/A')}")
        else:
            print(f"❌ Database connection: {health.get('status', 'unknown')}")
            if 'error' in health:
                print(f"   Error: {health['error']}")
            return False
        print()
    except Exception as e:
        print(f"❌ Database health check failed: {e}")
        return False
    
    # Check if tables exist
    try:
        with db_manager.session() as session:
            inspector = inspect(db_manager.engine)
            tables = inspector.get_table_names()
            
            print(f"📊 Database Tables:")
            required_tables = [
                'trade_history',
                'equity_curve',
                'risk_metrics',
                'iteration_log',
                'audit_report_log',
                'backtest_results'
            ]
            
            for table in required_tables:
                if table in tables:
                    print(f"   ✅ {table}")
                else:
                    print(f"   ❌ {table} - MISSING!")
            
            print()
            
            # Check if all required tables exist
            missing_tables = [t for t in required_tables if t not in tables]
            if missing_tables:
                print(f"⚠️  Missing tables: {', '.join(missing_tables)}")
                print(f"   Creating missing tables...")
                
                # Create all tables
                Base.metadata.create_all(db_manager.engine)
                print(f"   ✅ Tables created")
                print()
            
            # Check TradeHistory table structure
            if 'trade_history' in tables:
                columns = inspector.get_columns('trade_history')
                required_columns = [
                    'id', 'symbol', 'action', 'strategy_name', 'qty', 'price',
                    'fill_price', 'executed', 'order_id', 'timestamp'
                ]
                
                column_names = [col['name'] for col in columns]
                print(f"📋 TradeHistory Table Columns:")
                for col in required_columns:
                    if col in column_names:
                        print(f"   ✅ {col}")
                    else:
                        print(f"   ❌ {col} - MISSING!")
                
                missing_cols = [c for c in required_columns if c not in column_names]
                if missing_cols:
                    print(f"\n⚠️  Missing columns: {', '.join(missing_cols)}")
                    print(f"   You may need to run a migration or recreate the database")
                print()
            
            session.expunge_all()
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test write operation
    try:
        print(f"🧪 Testing write operation...")
        from models.signal import TradingSignal, SignalAction
        
        test_signal = TradingSignal(
            symbol="TEST",
            action=SignalAction.BUY,
            strategy_name="TEST_STRATEGY",
            confidence=0.5,
            timestamp=datetime.now(),
            price=100.0,
            qty=1
        )
        
        from models.audit import ExecutionResult
        test_exec_result = ExecutionResult(
            signal=test_signal,
            order_id="TEST_ORDER_123",
            executed=True,
            execution_time=datetime.now(),
            fill_price=100.0
        )
        
        # Try to log a test trade
        test_id = db_manager.log_trade(
            signal=test_signal,
            execution_result=test_exec_result,
            correlation_id="TEST_CHECK"
        )
        
        print(f"   ✅ Write test successful (ID: {test_id})")
        
        # Clean up test trade
        with db_manager.session() as session:
            test_trade = session.query(TradeHistory).filter(
                TradeHistory.id == "TEST_CHECK"
            ).first()
            if test_trade:
                session.delete(test_trade)
                session.commit()
                print(f"   ✅ Test trade cleaned up")
        
        print()
    except Exception as e:
        print(f"   ❌ Write test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check existing trade count
    try:
        trades = db_manager.get_trade_history(limit=1000)
        print(f"📈 Existing Data:")
        print(f"   Total trades in database: {len(trades)}")
        
        if trades:
            executed = [t for t in trades if t.executed]
            print(f"   Executed trades: {len(executed)}")
            print(f"   Failed trades: {len(trades) - len(executed)}")
            
            # Check for recent trades
            from datetime import timedelta
            recent_cutoff = datetime.now() - timedelta(days=1)
            recent_trades = [t for t in trades if t.timestamp >= recent_cutoff]
            print(f"   Trades in last 24 hours: {len(recent_trades)}")
        print()
    except Exception as e:
        print(f"⚠️  Could not query existing trades: {e}")
        print()
    
    print("=" * 80)
    print("✅ Database setup check complete!")
    print("=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    success = check_database_setup()
    sys.exit(0 if success else 1)

