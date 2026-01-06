"""
Backtesting framework for validating trading strategies.

Usage:
    python tests/backtest_strategies.py --strategy TrendFollowing --symbols AAPL,SPY
    python tests/backtest_strategies.py --all --symbols AAPL,GOOGL,MSFT,SPY
    python tests/backtest_strategies.py --strategy MeanReversion --walk-forward
"""
import argparse
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
from core.strategies import STRATEGY_REGISTRY
from tests.backtest.backtest_engine import BacktestEngine
from tests.backtest.strategy_backtester import StrategyBacktester
from tests.backtest.walk_forward import WalkForwardAnalyzer
from utils.database import DatabaseManager

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "SPY", "QQQ"]
DEFAULT_START_DATE = '2021-01-01'  # 3 years of data
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")


def print_header():
    """Print backtest header."""
    print("=" * 80)
    print("🔬 STRATEGY BACKTESTING FRAMEWORK")
    print("=" * 80)


def run_single_strategy(
    strategy_name: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    config,
    risk_free_rate: float,
    initial_cash: float,
    commission: float,
    walk_forward: bool = False,
    monte_carlo: bool = False
) -> Dict[str, Any]:
    """
    Run backtest for a single strategy.
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Testing Strategy: {strategy_name}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_cash:,.2f}")
    print(f"Commission: {commission:.3%}")
    print(f"Risk-Free Rate: {risk_free_rate:.2%} (annual)")
    print(f"{'='*80}\n")
    
    # Initialize database manager
    db_manager = DatabaseManager(config=config)
    
    # Initialize backtest engine
    engine = BacktestEngine(
        config=config,
        initial_cash=initial_cash,
        commission=commission,
        risk_free_rate=risk_free_rate,
        database_manager=db_manager
    )
    
    # Initialize strategy backtester
    backtester = StrategyBacktester(
        strategy_name=strategy_name,
        engine=engine,
        strategy_config={}  # Use default config for now
    )
    
    # Run backtest
    try:
        results = backtester.backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            per_symbol=True
        )
        
        # Print results
        print(f"\n{'='*80}")
        print(f"📊 RESULTS SUMMARY: {strategy_name}")
        print(f"{'='*80}")
        print(f"\n{'Symbol':<10} {'Return':<12} {'Sharpe':<10} {'Max DD':<12} {'Win Rate':<12} {'Trades':<10} {'Status'}")
        print("-" * 80)
        
        passed_count = 0
        total_count = 0
        
        for symbol, result in results.items():
            total_count += 1
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            if result['passed']:
                passed_count += 1
            
            print(
                f"{symbol:<10} "
                f"{result['total_return']:>10.2f}%  "
                f"{result['sharpe_ratio']:>8.2f}  "
                f"{result['max_drawdown']:>10.2f}%  "
                f"{result['win_rate']:>10.2f}%  "
                f"{result['total_trades']:>8d}  "
                f"{status}"
            )
        
        print(f"\n{'='*80}")
        print(f"📊 VERDICT: {passed_count}/{total_count} symbols passed validation")
        
        if passed_count == 0:
            print("⚠️  WARNING: Strategy failed on ALL symbols - DO NOT TRADE")
        elif passed_count < total_count // 2:
            print("⚠️  CAUTION: Strategy failed on majority of symbols")
        else:
            print("✅ Strategy shows promise on multiple symbols")
        
        print(f"{'='*80}\n")
        
        # Walk-forward analysis if requested
        if walk_forward:
            print(f"\n{'='*80}")
            print("🔍 WALK-FORWARD ANALYSIS")
            print(f"{'='*80}\n")
            
            # Fetch full data for walk-forward
            price_data = engine.fetch_data(symbols, start_date, end_date)
            
            walk_forward_analyzer = WalkForwardAnalyzer(engine, n_splits=4)
            wf_results = walk_forward_analyzer.analyze(
                STRATEGY_REGISTRY[strategy_name],
                price_data,
                strategy_config={}
            )
            
            if wf_results.get('periods_tested', 0) > 0:
                print(f"\nWalk-Forward Summary:")
                print(f"  Periods Tested: {wf_results['periods_tested']}")
                print(f"  Periods Passed: {wf_results['periods_passed']}")
                print(f"  Avg Sharpe: {wf_results['avg_sharpe']:.2f}")
                print(f"  Sharpe Range: {wf_results['sharpe_range']:.2f}")
                print(f"  Consistency: {'✅ Consistent' if wf_results['is_consistent'] else '❌ Variable (possible overfitting)'}")
        
        return results
        
    except Exception as e:
        logger.exception(f"Error running backtest for {strategy_name}: {e}")
        print(f"\n❌ Error: {str(e)}\n")
        return {}


def run_all_strategies(
    symbols: List[str],
    start_date: str,
    end_date: str,
    config,
    risk_free_rate: float,
    initial_cash: float,
    commission: float
) -> Dict[str, Dict[str, Any]]:
    """
    Run backtests for all strategies and compare.
    
    Returns:
        Dictionary mapping strategy names to results
    """
    print(f"\n{'='*80}")
    print("🔬 COMPREHENSIVE STRATEGY COMPARISON")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    # Get all strategy names (excluding legacy mappings)
    strategy_names = [
        name for name in STRATEGY_REGISTRY.keys()
        if name not in ["MovingAverageCrossover", "Momentum", "RSI_OversoldOverbought",
                       "VolumeProfile", "SupportResistance", "ConsolidationBreakout"]
    ]
    
    for strategy_name in strategy_names:
        try:
            results = run_single_strategy(
                strategy_name=strategy_name,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                config=config,
                risk_free_rate=risk_free_rate,
                initial_cash=initial_cash,
                commission=commission,
                walk_forward=False  # Skip walk-forward for comprehensive comparison
            )
            all_results[strategy_name] = results
        except Exception as e:
            logger.exception(f"Error testing {strategy_name}: {e}")
            continue
    
    # Comparative summary
    print(f"\n{'='*80}")
    print("📊 COMPARATIVE SUMMARY")
    print(f"{'='*80}\n")
    
    for strategy_name, results in all_results.items():
        if not results:
            continue
        
        # Calculate averages across symbols
        returns = [r['total_return'] for r in results.values()]
        sharpes = [r['sharpe_ratio'] for r in results.values()]
        passed = sum(1 for r in results.values() if r['passed'])
        
        avg_return = sum(returns) / len(returns) if returns else 0.0
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
        
        print(f"{strategy_name}:")
        print(f"  Avg Return:  {avg_return:>10.2f}%")
        print(f"  Avg Sharpe:  {avg_sharpe:>10.2f}")
        print(f"  Pass Rate:   {passed}/{len(results)}")
        print()
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Backtest trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single strategy
  python tests/backtest_strategies.py --strategy TrendFollowing --symbols AAPL,SPY
  
  # Test all strategies
  python tests/backtest_strategies.py --all --symbols AAPL,GOOGL,MSFT,SPY,QQQ
  
  # Test with walk-forward analysis
  python tests/backtest_strategies.py --strategy MeanReversion --walk-forward
  
  # Custom risk-free rate and commission
  python tests/backtest_strategies.py --strategy TrendFollowing --rf-rate 0.05 --commission 0.002
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        help='Strategy name to test (use --all for all strategies)',
        choices=list(STRATEGY_REGISTRY.keys())
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all strategies'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default=','.join(DEFAULT_SYMBOLS),
        help=f'Comma-separated list of symbols (default: {",".join(DEFAULT_SYMBOLS)})'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=DEFAULT_START_DATE,
        help=f'Start date YYYY-MM-DD (default: {DEFAULT_START_DATE})'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help=f'End date YYYY-MM-DD (default: today)'
    )
    parser.add_argument(
        '--initial-cash',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000)'
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
        help='Annual risk-free rate (default: 0.04 = 4%%)'
    )
    parser.add_argument(
        '--walk-forward',
        action='store_true',
        help='Run walk-forward analysis (prevents overfitting detection)'
    )
    parser.add_argument(
        '--monte-carlo',
        action='store_true',
        help='Run Monte Carlo simulation (robustness testing)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.strategy and not args.all:
        parser.error("Must specify either --strategy or --all")
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Parse dates
    start_date = args.start_date
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    
    # Load config
    try:
        config = get_config()
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        config = None
    
    # Print header
    print_header()
    
    # Run backtests
    try:
        if args.all:
            run_all_strategies(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                config=config,
                risk_free_rate=args.rf_rate,
                initial_cash=args.initial_cash,
                commission=args.commission
            )
        else:
            run_single_strategy(
                strategy_name=args.strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                config=config,
                risk_free_rate=args.rf_rate,
                initial_cash=args.initial_cash,
                commission=args.commission,
                walk_forward=args.walk_forward,
                monte_carlo=args.monte_carlo
            )
        
        print("\n✅ Backtesting complete!\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtesting interrupted by user\n")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during backtesting")
        print(f"\n❌ Fatal error: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

