#!/usr/bin/env python3
"""Comprehensive strategy analysis script with risk management insights."""
import sys
import os
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.backtest.backtest_engine import BacktestEngine
from tests.backtest.strategy_backtester import StrategyBacktester
from config.settings import AppConfig
from utils.database import DatabaseManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_all_strategies(
    symbols: list = None,
    start_date: str = '2021-01-01',
    end_date: str = None,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    risk_free_rate: float = 0.04
):
    """
    Analyze all strategies with risk management insights.
    
    Args:
        symbols: List of symbols to test (default: ['NVDA'])
        start_date: Start date for backtest
        end_date: End date for backtest (default: today)
        initial_cash: Initial capital
        commission: Commission rate
        risk_free_rate: Annual risk-free rate
    """
    if symbols is None:
        symbols = ['NVDA']
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    config = AppConfig()
    db_manager = DatabaseManager(config) if config.database else None
    
    engine = BacktestEngine(
        config=config,
        initial_cash=initial_cash,
        commission=commission,
        risk_free_rate=risk_free_rate,
        database_manager=db_manager
    )
    
    strategies_to_test = [
        'BollingerBands',
        'MeanReversion',
        'TrendFollowing',
        'MovingAverageEnvelope'
    ]
    
    results = []
    
    for strategy_name in strategies_to_test:
        print(f"\n{'='*80}")
        print(f"Analyzing {strategy_name}")
        print(f"{'='*80}")
        
        try:
            backtester = StrategyBacktester(strategy_name, engine)
            
            # Fetch data
            price_data, ohlc_data = engine.fetch_data(symbols, start_date, end_date)
            
            if price_data.empty:
                print(f"  ❌ No data fetched for {strategy_name}")
                continue
            
            # Generate signals
            entries, exits = engine.generate_signals_from_strategy(
                backtester.strategy_class,
                price_data,
                ohlc_data
            )
            
            # Run backtest
            portfolio = engine.run_backtest(price_data, entries, exits)
            
            # Analyze results for each symbol
            for symbol in symbols:
                if symbol not in price_data.columns:
                    continue
                
                print(f"\n  📊 {symbol}:")
                
                # Get basic results
                result = engine.analyze_results(
                    portfolio,
                    strategy_name,
                    symbol,
                    pd.to_datetime(start_date),
                    pd.to_datetime(end_date),
                    price_data=price_data
                )
                
                # Get trade quality
                trade_quality = engine.analyze_trade_quality(portfolio, symbol=symbol)
                
                # Combine results
                combined = {**result}
                if 'error' not in trade_quality:
                    combined.update(trade_quality)
                else:
                    logger.warning(f"  ⚠️  Could not analyze trade quality: {trade_quality.get('error')}")
                
                results.append(combined)
                
                # Print summary
                print(f"    Return:      {result['total_return']:>10.2f}%")
                print(f"    Sharpe:      {result['sharpe_ratio']:>10.2f}")
                print(f"    Max DD:      {result['max_drawdown']:>10.2f}%")
                print(f"    Win Rate:    {result['win_rate']:>10.2f}%")
                print(f"    Trades:      {result['total_trades']:>10}")
                
                if 'win_loss_ratio' in trade_quality:
                    print(f"    Win/Loss:    {trade_quality['win_loss_ratio']:>10.2f}")
                if 'expectancy' in trade_quality:
                    print(f"    Expectancy:  ${trade_quality['expectancy']:>9.2f}")
                if 'profit_factor' in trade_quality and trade_quality['profit_factor'] > 0:
                    print(f"    Profit Factor: {trade_quality['profit_factor']:>6.2f}")
                
                status = "✅ PASS" if result['passed'] else "❌ FAIL"
                print(f"    Status:      {status:>10}")
                
        except Exception as e:
            logger.exception(f"  ❌ Error analyzing {strategy_name}: {e}")
            continue
    
    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('total_return', ascending=False)
        
        print(f"\n{'='*80}")
        print("📊 STRATEGY RANKING")
        print(f"{'='*80}\n")
        
        # Display summary table
        summary_cols = ['strategy_name', 'symbol', 'total_return', 'sharpe_ratio', 
                       'max_drawdown', 'win_rate', 'total_trades', 'passed']
        available_cols = [col for col in summary_cols if col in df.columns]
        
        print(df[available_cols].to_string(index=False))
        
        # Save to CSV if results directory exists
        try:
            os.makedirs('tests/results', exist_ok=True)
            csv_path = f"tests/results/strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n💾 Results saved to {csv_path}")
        except Exception as e:
            logger.warning(f"Could not save results to CSV: {e}")
        
        return df
    else:
        print("\n❌ No results to display")
        return pd.DataFrame()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze all strategies with risk management insights',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        default='NVDA',
        help='Comma-separated list of symbols (default: NVDA)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2021-01-01',
        help='Start date YYYY-MM-DD (default: 2021-01-01)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date YYYY-MM-DD (default: today)'
    )
    parser.add_argument(
        '--initial-cash',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000.0)'
    )
    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (default: 0.001 = 0.1%%)'
    )
    parser.add_argument(
        '--rf-rate',
        type=float,
        default=0.04,
        dest='risk_free_rate',
        help='Annual risk-free rate (default: 0.04 = 4%%)'
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    analyze_all_strategies(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.initial_cash,
        commission=args.commission,
        risk_free_rate=args.risk_free_rate
    )


if __name__ == "__main__":
    main()

