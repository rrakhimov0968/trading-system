#!/usr/bin/env python3
"""
Quick test script to demonstrate trailing stop-loss functionality.
Run this to see the impact of trailing stops on backtest results.
"""
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_config
from tests.backtest.backtest_engine import BacktestEngine
from tests.backtest.strategy_backtester import StrategyBacktester
from utils.database import DatabaseManager

def test_trailing_stops():
    """Test trailing stop functionality with MeanReversion strategy."""
    
    print("=" * 80)
    print("🔬 TESTING TRAILING STOP-LOSS FUNCTIONALITY")
    print("=" * 80)
    print("\nThis test compares backtest results with trailing stops enabled.")
    print("Expected improvements:")
    print("  - Lower max drawdown (trailing stops exit earlier)")
    print("  - Better Sharpe ratio (fewer large reversals)")
    print("  - Protected gains (winners locked in)")
    print("\n" + "=" * 80)
    
    try:
        config = get_config()
    except Exception as e:
        print(f"⚠️  Could not load config: {e}. Using defaults.")
        config = None
    
    # Initialize components
    db_manager = DatabaseManager(config=config)
    
    engine = BacktestEngine(
        config=config,
        initial_cash=10000,
        commission=0.001,
        risk_free_rate=0.04,
        database_manager=db_manager,
        enable_risk_management=True,  # Enable risk management
        default_stop_loss_pct=0.15,  # 15% fixed stop-loss
        default_take_profit_pct=0.25,  # 25% take-profit
        # Trailing stop is enabled by default at 8% (default_trailing_stop_pct)
    )
    
    backtester = StrategyBacktester(
        strategy_name="MeanReversion",
        engine=engine,
        strategy_config={}
    )
    
    # Run backtest on a shorter period for quick results
    symbols = ["SPY"]
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    print(f"\n📊 Running backtest:")
    print(f"  Strategy: MeanReversion")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Risk Management: ENABLED")
    print(f"    - Fixed Stop-Loss: 15%")
    print(f"    - Trailing Stop: 8% (from peak)")
    print(f"    - Take-Profit: 25%")
    print()
    
    try:
        results = backtester.backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        print("\n" + "=" * 80)
        print("📈 BACKTEST RESULTS")
        print("=" * 80)
        
        if results:
            for symbol, result in results.items():
                print(f"\n{symbol}:")
                print(f"  Total Return:     {result.get('total_return', 0):.2f}%")
                print(f"  Sharpe Ratio:     {result.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown:     {result.get('max_drawdown', 0):.2f}%")
                print(f"  Win Rate:         {result.get('win_rate', 0):.2f}%")
                print(f"  Total Trades:     {result.get('total_trades', 0)}")
                print(f"  Status:           {'✅ PASS' if result.get('passed', False) else '❌ FAIL'}")
                
                # Show validation thresholds
                print(f"\n  Validation Thresholds:")
                print(f"    Min Sharpe:     {result.get('min_sharpe', 0):.2f}")
                print(f"    Max DD Limit:   {result.get('max_drawdown_threshold', 0):.2f}%")
        else:
            print("❌ No results returned")
            
        print("\n" + "=" * 80)
        print("✅ Test complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_trailing_stops()

