#!/usr/bin/env python3
"""
Backtest ALL strategies on ALL S&P 500 stocks overnight.

This script will:
1. Iterate through all available strategies
2. For each strategy, backtest all S&P 500 symbols one by one
3. Save results to separate CSV files (one per strategy)
4. Save progress periodically so it can resume if interrupted

Usage:
    python scripts/backtest_all_strategies_sp500.py
    python scripts/backtest_all_strategies_sp500.py --start-date 2021-01-01 --max-symbols 50  # For testing
"""
import argparse
import sys
import os
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import warnings
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
from core.strategies import STRATEGY_REGISTRY
from tests.backtest.backtest_engine import BacktestEngine
from tests.backtest.strategy_backtester import StrategyBacktester
from utils.database import DatabaseManager

warnings.filterwarnings('ignore')

# Setup logging to both console and file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"backtest_all_sp500_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Progress tracking file
PROGRESS_FILE = "backtest_progress.json"


def get_sp500_symbols() -> List[str]:
    """Fetch S&P 500 symbols list."""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        
        symbols = []
        if table:
            rows = table.find_all('tr')[1:]  # Skip header
            for row in rows:
                cells = row.find_all('td')
                if cells:
                    symbol = cells[0].text.strip()
                    symbol = symbol.replace('.', '-')  # BRK.B -> BRK-B
                    symbol = symbol.split('\n')[0].strip()
                    if symbol:
                        symbols.append(symbol)
        
        if symbols:
            logger.info(f"✅ Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
            return symbols
            
    except ImportError:
        logger.warning("beautifulsoup4 not installed, using static list")
    except Exception as e:
        logger.warning(f"Failed to fetch from Wikipedia: {e}")
    
    # Fallback: Comprehensive static list
    logger.info("Using static list of S&P 500 stocks")
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'WMT', 'XOM', 'JPM', 'PG', 'MA', 'CVX', 'LLY', 'AVGO', 'COST',
        'MRK', 'PEP', 'AMD', 'ADBE', 'TMO', 'MCD', 'CSCO', 'ABBV', 'ACN', 'NFLX',
        'CMCSA', 'CRM', 'DIS', 'WFC', 'VZ', 'PM', 'NKE', 'TXN', 'RTX', 'AMGN',
        'LIN', 'QCOM', 'INTC', 'AMAT', 'COP', 'INTU', 'HON', 'UNP', 'CAT', 'LOW',
        'ISRG', 'IBM', 'SBUX', 'SPGI', 'GS', 'AXP', 'DE', 'AMT', 'GE', 'BKNG',
        'BLK', 'ADP', 'ELV', 'TJX', 'C', 'MO', 'MDT', 'SYK', 'GILD', 'ZTS',
        'REGN', 'MMC', 'CI', 'EQIX', 'LMT', 'ADI', 'ETN', 'WM', 'SHW', 'APD',
        'NXPI', 'AON', 'CDNS', 'ITW', 'KLAC', 'SNPS', 'TMUS', 'DG', 'CME', 'CTAS',
        'MCK', 'FERG', 'PH', 'PAYX', 'MCHP', 'FTNT', 'MSI', 'CTSH', 'TRV', 'FAST',
        'ANSS', 'APH', 'CDW', 'DOV', 'EMR', 'FTV', 'GPN', 'HUM', 'ICE', 'IQV',
        'KEYS', 'LDOS', 'LHX', 'MPWR', 'ODFL', 'ON', 'OTIS', 'POOL', 'ROP', 'RMD',
        'ROST', 'RSG', 'TDG', 'TFX', 'TTC', 'VRSK', 'WAB', 'WAT', 'A', 'AAL',
        'ABT', 'ACGL', 'ADM', 'AEE', 'AES', 'AFL', 'AIG', 'AKAM', 'ALB', 'ALGN',
        'ALK', 'ALL', 'AME', 'AMED', 'AMP', 'ANET', 'ANTM', 'AOS', 'APA', 'APTV',
        'ARE', 'ATO', 'ATVI', 'AVB', 'AVY', 'AXON', 'AZO', 'BA', 'BALL', 'BANR',
        'BAX', 'BBWI', 'BBY', 'BEN', 'BF-B', 'BIO', 'BK', 'BKR', 'BLDR', 'BMY',
        'BR', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CALM',
        'CAR', 'CARR', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CE', 'CF',
        'CFG', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLH', 'CLX', 'CMA', 'CMG',
        'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COIN', 'COO', 'COR', 'CPAY', 'CPB',
        'CPRT', 'CPT', 'CRL', 'CRWD', 'CSGP', 'CSL', 'CSX', 'CTLT', 'CTRA', 'CTVA',
        'CTXS', 'CUBE', 'CURI', 'CVBF', 'CVE', 'CVNA', 'CVS', 'CZR', 'D', 'DAL',
        'DAR', 'DAY', 'DBX', 'DD', 'DDOG', 'DFS', 'DGX', 'DHI', 'DHR', 'DIN',
        'DISH', 'DKNG', 'DLR', 'DLTR', 'DOCN', 'DOCU', 'DOW', 'DPZ', 'DQ', 'DRE',
        'DRI', 'DT', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED',
        'EFX', 'EGP', 'EIX', 'EL', 'ELAN', 'EMN', 'ENPH', 'ENTG', 'ENV', 'EOG',
        'EPAM', 'EQH', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS', 'ESTC', 'ETR', 'ETSY',
        'EVRG', 'EW', 'EXAS', 'EXC', 'EXPD', 'EXPE', 'EXPO', 'F', 'FANG', 'FDS',
        'FDX', 'FE', 'FFIV', 'FICO', 'FIS', 'FITB', 'FIVE', 'FIVN', 'FLT', 'FMC',
        'FOX', 'FOXA', 'FRC', 'FRT', 'FSLR', 'FTS', 'FWRD', 'G', 'GDDY', 'GEHC',
        'GEN', 'GFS', 'GIB', 'GIS', 'GL', 'GLW', 'GM', 'GMAB', 'GNRC', 'GPC',
        'GRMN', 'GRUB', 'GSHD', 'GTLS', 'HAL', 'HAS', 'HAYW', 'HBAN', 'HCA', 'HCP',
        'HD', 'HEI', 'HES', 'HIG', 'HIW', 'HLT', 'HOLX', 'HOPE', 'HPE', 'HPQ',
        'HRL', 'HSIC', 'HST', 'HSY', 'HWM', 'HZN', 'IBP', 'ICUI', 'IDXX', 'IEX',
        'IFF', 'IGT', 'ILMN', 'INCY', 'INDI', 'INFO', 'INVH', 'IONS', 'IP', 'IPG',
        'IR', 'IRDM', 'IRM', 'IT', 'IVZ', 'J', 'JBHT', 'JCI', 'JD', 'JKHY',
        'JLL', 'JNPR', 'JOE', 'K', 'KBR', 'KDP', 'KEX', 'KEY', 'KHC', 'KIM',
        'KKR', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LAMR', 'LBRDK',
        'LEN', 'LEVI', 'LFX', 'LI', 'LKQ', 'LNC', 'LNT', 'LNW', 'LOGI', 'LPLA',
        'LRCX', 'LSCC', 'LSTR', 'LTBR', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYFT',
        'LYV', 'MAA', 'MANH', 'MAR', 'MAS', 'MAT', 'MCO', 'MDLZ', 'MELI', 'MGM',
        'MHK', 'MKC', 'MKTX', 'MLI', 'MMM', 'MNST', 'MODG', 'MOH', 'MOS', 'MPC',
        'MPW', 'MRNA', 'MRO', 'MRVL', 'MS', 'MSCI', 'MTB', 'MTCH', 'MTD', 'MU',
        'MUR', 'NDAQ', 'NEE', 'NEM', 'NI', 'NOC', 'NOV', 'NOW', 'NRG', 'NSC',
        'NTAP', 'NTRS', 'NUE', 'NVAX', 'NVR', 'NWS', 'NWSA', 'O', 'OGN', 'OKE',
        'OKTA', 'OMC', 'ONON', 'ORCL', 'ORLY', 'OSCR', 'OSK', 'OVV', 'OXY', 'PAAS',
        'PAYC', 'PB', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEN', 'PFE', 'PFG', 'PGR',
        'PHM', 'PKG', 'PKI', 'PLD', 'PLTR', 'PNC', 'PNR', 'PNW', 'PODD', 'POR',
        'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PTC', 'PTEN', 'PWR', 'PXD',
        'PYPL', 'QRVO', 'RBLX', 'RCL', 'RE', 'REG', 'RELX', 'RF', 'RGA', 'RHI',
        'RJF', 'RL', 'RMBS', 'ROK', 'ROL', 'RPM', 'RPT', 'RVTY', 'RYAAY', 'RZG',
        'S', 'SAIA', 'SBAC', 'SCHW', 'SEDG', 'SEE', 'SIRI', 'SJM', 'SLB', 'SMCI',
        'SNA', 'SNOW', 'SO', 'SOLV', 'SON', 'SPLK', 'SQM', 'SQSP', 'SRCL', 'SRE',
        'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWAV', 'SWK', 'SWKS', 'SYF', 'SYY',
        'T', 'TAP', 'TDY', 'TEL', 'TER', 'TFC', 'TGT', 'TKO', 'TPG', 'TROW',
        'TRU', 'TSCO', 'TSN', 'TT', 'TTD', 'TTWO', 'TXT', 'TYL', 'U', 'UAL',
        'UBER', 'UDR', 'UHS', 'ULTA', 'UPS', 'URI', 'USB', 'USFD', 'UTHR', 'VALE',
        'VEEV', 'VFC', 'VICI', 'VMC', 'VNO', 'VRSN', 'VRTX', 'VSAT', 'VST', 'VTR',
        'VTRS', 'W', 'WAL', 'WBD', 'WBS', 'WCC', 'WDAY', 'WDC', 'WEC', 'WELL',
        'WHR', 'WMB', 'WRB', 'WRK', 'WST', 'WTW', 'WU', 'WWD', 'WY', 'WYNN',
        'XEL', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZD', 'ZEN', 'ZION'
    ]


def load_progress() -> Dict[str, Any]:
    """Load progress from file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")
    return {}


def save_progress(progress: Dict[str, Any]):
    """Save progress to file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save progress: {e}")


def backtest_symbol(
    symbol: str,
    strategy_name: str,
    engine: BacktestEngine,
    start_date: str,
    end_date: str,
    timeframe: str = "1Day"
) -> Optional[Dict[str, Any]]:
    """Backtest a single symbol."""
    try:
        backtester = StrategyBacktester(
            strategy_name=strategy_name,
            engine=engine,
            strategy_config={}
        )
        
        results = backtester.backtest(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            per_symbol=True,
            timeframe=timeframe
        )
        
        if results and symbol in results:
            result = results[symbol]
            result['symbol'] = symbol
            result['strategy_name'] = strategy_name
            return result
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error backtesting {symbol} with {strategy_name}: {e}")
        return None


def save_results_to_csv(results: List[Dict[str, Any]], output_file):
    """
    Save results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output file (str or Path object)
    """
    # Convert Path to string if needed
    output_path = str(output_file)
    
    if not results:
        logger.warning(f"⚠️  No results to save for {output_path}")
        print(f"  ⚠️  Warning: No results to save for {os.path.basename(output_path)}")
        return
    
    try:
        # Ensure directory exists
        output_dir_path = os.path.dirname(output_path)
        if output_dir_path:
            os.makedirs(output_dir_path, exist_ok=True)
            logger.debug(f"Ensured directory exists: {output_dir_path}")
        
        all_keys = set()
        valid_results = []
        for result in results:
            if result and isinstance(result, dict):
                all_keys.update(result.keys())
                valid_results.append(result)
        
        if not all_keys:
            logger.warning(f"⚠️  No valid keys found in results for {output_path}")
            print(f"  ⚠️  Warning: No valid data in results for {os.path.basename(output_path)}")
            return
        
        if not valid_results:
            logger.warning(f"⚠️  No valid result dictionaries for {output_path}")
            print(f"  ⚠️  Warning: All results were empty for {os.path.basename(output_path)}")
            return
        
        priority_keys = [
            'symbol', 'strategy_name', 'total_return', 'sharpe_ratio', 'max_drawdown',
            'win_rate', 'total_trades', 'avg_trade_return', 'passed',
            'start_date', 'end_date', 'initial_cash', 'commission', 'risk_free_rate'
        ]
        
        ordered_keys = [k for k in priority_keys if k in all_keys]
        ordered_keys.extend(sorted([k for k in all_keys if k not in priority_keys]))
        
        # Write CSV file
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
            writer.writeheader()
            rows_written = 0
            for result in valid_results:
                try:
                    writer.writerow(result)
                    rows_written += 1
                except Exception as row_error:
                    logger.error(f"Error writing row for {output_path}: {row_error}")
                    logger.debug(f"Problematic row: {result}")
        
        logger.info(f"✅ Saved {rows_written} results to {output_path}")
        print(f"  💾 Saved {rows_written} rows to {os.path.basename(output_path)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results to {output_path}: {e}", exc_info=True)
        print(f"  ❌ ERROR: Failed to save {os.path.basename(output_path)}: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Backtest ALL strategies on ALL S&P 500 stocks'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2021-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--initial-cash',
        type=float,
        default=10000.0,
        help='Initial capital'
    )
    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate (0.001 = 0.1%%)'
    )
    parser.add_argument(
        '--rf-rate',
        type=float,
        default=0.04,
        help='Risk-free rate (annual, default 0.04 = 4%%)'
    )
    parser.add_argument(
        '--max-symbols',
        type=int,
        default=None,
        help='Maximum number of symbols to test (for testing)'
    )
    parser.add_argument(
        '--skip-strategies',
        type=str,
        default='',
        help='Comma-separated list of strategies to skip'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='backtest_results',
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of stocks to process per batch (default: 5)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1Day',
        choices=['1Day', '1Hour', '15Min', '5Min', '1Min'],
        help='Bar timeframe for backtesting. Options: 1Day (default), 1Hour (intraday), 15Min, 5Min, 1Min. Use 1Hour for better exit timing.'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        # Test write access
        test_file = output_dir / '.write_test'
        test_file.write_text('test')
        test_file.unlink()
        logger.info(f"✅ Output directory is writable: {output_dir.absolute()}")
    except Exception as e:
        logger.error(f"❌ Cannot create/write to output directory {output_dir}: {e}")
        print(f"❌ ERROR: Cannot create output directory: {e}")
        sys.exit(1)
    
    # Load config
    try:
        config = get_config()
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        config = None
    
    # Initialize components
    db_manager = DatabaseManager(config=config)
    
    engine = BacktestEngine(
        config=config,
        initial_cash=args.initial_cash,
        commission=args.commission,
        risk_free_rate=args.rf_rate,
        database_manager=db_manager,
        enable_risk_management=True
    )
    
    # Get strategies
    skip_strategies = [s.strip() for s in args.skip_strategies.split(',') if s.strip()]
    strategies = [s for s in STRATEGY_REGISTRY.keys() if s not in skip_strategies]
    
    # Get S&P 500 symbols
    print("=" * 80)
    print("🌙 OVERNIGHT S&P 500 BACKTEST - ALL STRATEGIES")
    print("=" * 80)
    print(f"Strategies: {len(strategies)} ({', '.join(strategies)})")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Timeframe: {args.timeframe}")
    if args.timeframe == "1Hour":
        print("  ℹ️  Using hourly bars for intraday trading and better exit timing")
    elif args.timeframe == "1Day":
        print("  ℹ️  Using daily bars (end-of-day prices)")
    else:
        print(f"  ℹ️  Using {args.timeframe} bars for high-frequency analysis")
    print(f"Output Directory: {output_dir}")
    print(f"Log File: {log_file}")
    print("=" * 80)
    print("\nFetching S&P 500 symbols...")
    
    symbols = get_sp500_symbols()
    if args.max_symbols:
        symbols = symbols[:args.max_symbols]
        print(f"⚠️  TESTING MODE: Limiting to first {args.max_symbols} symbols")
    
    print(f"Found {len(symbols)} symbols to backtest\n")
    
    # Load progress
    progress = load_progress()
    progress_key = f"all_strategies_{args.start_date}_{args.end_date}"
    
    # Track completed stocks (stocks that have all strategies run)
    if progress_key in progress:
        completed_stocks = set(progress[progress_key].get('completed_stocks', []))
        logger.info(f"Resuming: {len(completed_stocks)} stocks already completed")
    else:
        completed_stocks = set()
    
    # Organize results by strategy (for CSV output per strategy)
    strategy_results = {strategy: [] for strategy in strategies}
    
    # Split symbols into batches
    batch_size = args.batch_size
    symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    
    total_combinations = len(strategies) * len(symbols)
    completed = 0
    successful = 0
    failed = 0
    
    start_time = datetime.now()
    
    print(f"\n📦 Processing {len(symbols)} stocks in {len(symbol_batches)} batches of {batch_size}")
    print(f"   Each stock will run {len(strategies)} strategies\n")
    
    # Process stocks in batches
    for batch_idx, batch_symbols in enumerate(symbol_batches, 1):
        print("\n" + "=" * 80)
        print(f"BATCH {batch_idx}/{len(symbol_batches)}: {', '.join(batch_symbols)}")
        print("=" * 80)
        
        batch_start_time = datetime.now()
        
        # Process each stock in this batch
        for symbol_idx, symbol in enumerate(batch_symbols, 1):
            # Skip if already completed
            if symbol in completed_stocks:
                logger.info(f"[Batch {batch_idx}/{len(symbol_batches)}] {symbol} - Already completed, skipping")
                continue
            
            print(f"\n[{symbol_idx}/{len(batch_symbols)}] 📈 {symbol}")
            print("-" * 80)
            
            stock_results = []
            
            # Run all strategies for this stock
            for strategy_idx, strategy_name in enumerate(strategies, 1):
                print(f"  [{strategy_idx}/{len(strategies)}] {strategy_name}...", end=' ', flush=True)
                
                try:
                    result = backtest_symbol(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        engine=engine,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        timeframe=args.timeframe
                    )
                    
                    if result:
                        stock_results.append(result)
                        if strategy_name not in strategy_results:
                            strategy_results[strategy_name] = []
                        strategy_results[strategy_name].append(result)
                        successful += 1
                        print(f"✅ {result.get('total_return', 0):.2f}% | "
                              f"Sharpe: {result.get('sharpe_ratio', 0):.2f} | "
                              f"Trades: {result.get('total_trades', 0)}")
                        logger.debug(f"Added result for {symbol} {strategy_name}: {len(strategy_results[strategy_name])} total results for this strategy")
                    else:
                        failed += 1
                        error_result = {
                            'symbol': symbol,
                            'strategy_name': strategy_name,
                            'error': 'Failed'
                        }
                        stock_results.append(error_result)
                        if strategy_name not in strategy_results:
                            strategy_results[strategy_name] = []
                        strategy_results[strategy_name].append(error_result)
                        print(f"❌ Failed")
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"Error running {strategy_name} on {symbol}: {e}")
                    error_result = {
                        'symbol': symbol,
                        'strategy_name': strategy_name,
                        'error': str(e)
                    }
                    stock_results.append(error_result)
                    if strategy_name not in strategy_results:
                        strategy_results[strategy_name] = []
                    strategy_results[strategy_name].append(error_result)
                    print(f"❌ Error: {str(e)[:50]}")
                
                completed += 1
            
            # Mark stock as completed if all strategies ran (even if some failed)
            completed_stocks.add(symbol)
            
            # Save progress after each stock
            progress[progress_key] = {
                'completed_stocks': list(completed_stocks),
                'batch': batch_idx,
                'last_symbol': symbol,
                'total_completed': completed,
                'last_update': datetime.now().isoformat()
            }
            save_progress(progress)
            
            # Save results to CSV files after each stock (one file per strategy)
            print(f"  💾 Saving CSV files...")
            for strategy_name in strategies:
                output_file = output_dir / f"{strategy_name}_results.csv"
                result_count = len(strategy_results.get(strategy_name, []))
                
                if result_count == 0:
                    logger.debug(f"No results yet for {strategy_name}, skipping save")
                    continue
                
                try:
                    logger.info(f"Saving {result_count} results for {strategy_name} to {output_file}")
                    save_results_to_csv(strategy_results[strategy_name], output_file)
                except Exception as e:
                    logger.error(f"Failed to save CSV for {strategy_name}: {e}", exc_info=True)
                    print(f"  ❌ ERROR: Failed to save {strategy_name} results: {e}")
            
            # Print stock summary
            valid_results = [r for r in stock_results if r and 'total_return' in r]
            if valid_results:
                returns = [r.get('total_return', 0) for r in valid_results]
                passed = sum(1 for r in valid_results if r.get('passed', False))
                print(f"\n  📊 {symbol} Summary: {passed}/{len(valid_results)} strategies passed")
                print(f"     Avg Return: {sum(returns)/len(returns):.2f}% | "
                      f"Best: {max(returns):.2f}% | "
                      f"Worst: {min(returns):.2f}%")
        
        # Print batch summary
        batch_elapsed = (datetime.now() - batch_start_time).total_seconds() / 60
        print(f"\n✅ Batch {batch_idx} complete in {batch_elapsed:.1f} minutes")
        
        # Verify CSV files were saved
        csv_files_created = []
        for strategy_name in strategies:
            csv_file = output_dir / f"{strategy_name}_results.csv"
            if csv_file.exists():
                file_size = csv_file.stat().st_size
                csv_files_created.append(f"{strategy_name} ({file_size} bytes)")
        
        if csv_files_created:
            print(f"💾 CSV files saved: {', '.join(csv_files_created)}")
        else:
            print(f"⚠️  Warning: No CSV files found in {output_dir}")
        
        # Overall progress
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        remaining_batches = len(symbol_batches) - batch_idx
        if batch_idx > 0:
            avg_batch_time = elapsed / batch_idx
            remaining_time = remaining_batches * avg_batch_time
            print(f"📊 Overall: {completed}/{total_combinations} combinations | "
                  f"{len(completed_stocks)}/{len(symbols)} stocks | "
                  f"~{remaining_time:.1f} min remaining")
        
        print(f"💾 Progress saved. Results in: {output_dir.absolute()}/")
    
    # Final summary
    elapsed_time = (datetime.now() - start_time).total_seconds() / 3600
    print("\n" + "=" * 80)
    print("🌅 FINAL SUMMARY")
    print("=" * 80)
    print(f"Stocks Processed: {len(completed_stocks)}/{len(symbols)}")
    print(f"Strategies per Stock: {len(strategies)}")
    print(f"Total Combinations: {total_combinations}")
    print(f"Completed: {completed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful/completed*100:.1f}%" if completed > 0 else "N/A")
    print(f"Total Time: {elapsed_time:.2f} hours")
    print(f"Results Directory: {output_dir}")
    print(f"Log File: {log_file}")
    print("\n📁 Results saved per strategy:")
    for strategy_name in strategies:
        output_file = output_dir / f"{strategy_name}_results.csv"
        result_count = len(strategy_results.get(strategy_name, []))
        print(f"   • {strategy_name}_results.csv ({result_count} rows)")
    print("=" * 80)
    print("\n✅ All backtests complete! Check the CSV files in the output directory.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user. Progress has been saved.")
        print(f"   Resume by running the script again - it will skip completed symbols.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\n❌ Fatal error: {str(e)}")
        print(f"   Check log file: {log_file}")
        sys.exit(1)

