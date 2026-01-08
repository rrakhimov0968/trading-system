#!/usr/bin/env python3
"""
Backtest all S&P 500 stocks and save results to CSV file.

Usage:
    python scripts/backtest_sp500.py --strategy MeanReversion --output results.csv
    python scripts/backtest_sp500.py --strategy TrendFollowing --start-date 2022-01-01 --output sp500_results.csv
"""
import argparse
import sys
import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
from core.strategies import STRATEGY_REGISTRY
from tests.backtest.backtest_engine import BacktestEngine
from tests.backtest.strategy_backtester import StrategyBacktester
from utils.database import DatabaseManager

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sp500_symbols() -> List[str]:
    """
    Fetch S&P 500 symbols list.
    
    Returns:
        List of S&P 500 stock symbols
    """
    try:
        # Try using yfinance to get S&P 500 list
        import yfinance as yf
        sp500 = yf.Ticker("^GSPC")
        # Alternative: Use Wikipedia scraping (more reliable)
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
                    # Handle symbols with dots (e.g., BRK.B -> BRK-B for yfinance)
                    symbol = symbol.replace('.', '-')
                    symbols.append(symbol)
        
        logger.info(f"Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
        return symbols
        
    except ImportError:
        logger.warning("beautifulsoup4 not installed (pip install beautifulsoup4), using static list")
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 list from web: {e}")
        logger.info("Using static list of S&P 500 stocks")
    
    # Fallback: Return comprehensive static list (500+ symbols)
    # Note: Install beautifulsoup4 for automatic updates: pip install beautifulsoup4
    logger.info("Using static list of S&P 500 stocks")
    
    # Comprehensive list of S&P 500 symbols (as of 2024)
    # This is a fallback - the script will try to fetch live data from Wikipedia first
    sp500_symbols = [
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
        'ALK', 'ALL', 'AME', 'AMED', 'AMP', 'AMT', 'ANET', 'ANTM', 'AOS', 'APA',
        'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVY', 'AXON', 'AZO', 'BA', 'BALL',
        'BANR', 'BAX', 'BBWI', 'BBY', 'BEN', 'BF-B', 'BIO', 'BK', 'BKR', 'BLDR',
        'BMY', 'BR', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH',
        'CALM', 'CAR', 'CARR', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDAY', 'CE',
        'CF', 'CFG', 'CHD', 'CHRW', 'CI', 'CINF', 'CL', 'CLH', 'CLX', 'CMA',
        'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COIN', 'COO', 'COR', 'CPAY',
        'CPB', 'CPRT', 'CPT', 'CRL', 'CRWD', 'CSGP', 'CSL', 'CSX', 'CTLT', 'CTRA',
        'CTVA', 'CTXS', 'CUBE', 'CURI', 'CVBF', 'CVE', 'CVNA', 'CVS', 'CZR', 'D',
        'DAL', 'DAR', 'DAY', 'DBX', 'DD', 'DDOG', 'DFS', 'DGX', 'DHI', 'DHR',
        'DIN', 'DISH', 'DKNG', 'DLR', 'DLTR', 'DOCN', 'DOCU', 'DOW', 'DPZ', 'DQ',
        'DRE', 'DRI', 'DT', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL',
        'ED', 'EFX', 'EGP', 'EIX', 'EL', 'ELAN', 'EMN', 'ENPH', 'ENTG', 'ENV',
        'EOG', 'EPAM', 'EQH', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS', 'ESTC', 'ETR',
        'ETSY', 'EVRG', 'EW', 'EXAS', 'EXC', 'EXPD', 'EXPE', 'EXPO', 'F', 'FANG',
        'FDS', 'FDX', 'FE', 'FFIV', 'FICO', 'FIS', 'FITB', 'FIVE', 'FIVN', 'FLT',
        'FMC', 'FOX', 'FOXA', 'FRC', 'FRT', 'FSLR', 'FTS', 'FWRD', 'G', 'GDDY',
        'GEHC', 'GEN', 'GFS', 'GIB', 'GIS', 'GL', 'GLW', 'GM', 'GMAB', 'GNRC',
        'GPC', 'GRMN', 'GRUB', 'GSHD', 'GTLS', 'HAL', 'HAS', 'HAYW', 'HBAN', 'HCA',
        'HCP', 'HD', 'HEI', 'HES', 'HIG', 'HIW', 'HLT', 'HOLX', 'HOPE', 'HPE',
        'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HWM', 'HZN', 'IBP', 'ICUI', 'IDXX',
        'IEX', 'IFF', 'IGT', 'ILMN', 'INCY', 'INDI', 'INFO', 'INVH', 'IONS', 'IP',
        'IPG', 'IR', 'IRDM', 'IRM', 'IT', 'IVZ', 'J', 'JBHT', 'JCI', 'JD',
        'JKHY', 'JLL', 'JNPR', 'JOE', 'K', 'KBR', 'KDP', 'KEX', 'KEY', 'KHC',
        'KIM', 'KKR', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LAMR',
        'LBRDK', 'LEN', 'LEVI', 'LFX', 'LI', 'LKQ', 'LNC', 'LNT', 'LNW', 'LOGI',
        'LPLA', 'LRCX', 'LSCC', 'LSTR', 'LTBR', 'LULU', 'LUV', 'LVS', 'LW', 'LYB',
        'LYFT', 'LYV', 'MAA', 'MANH', 'MAR', 'MAS', 'MAT', 'MCO', 'MDLZ', 'MELI',
        'MGM', 'MHK', 'MKC', 'MKTX', 'MLI', 'MMM', 'MNST', 'MODG', 'MOH', 'MOS',
        'MPC', 'MPW', 'MRNA', 'MRO', 'MRVL', 'MS', 'MSCI', 'MTB', 'MTCH', 'MTD',
        'MU', 'MUR', 'NDAQ', 'NEE', 'NEM', 'NI', 'NOC', 'NOV', 'NOW', 'NRG',
        'NSC', 'NTAP', 'NTRS', 'NUE', 'NVAX', 'NVR', 'NWS', 'NWSA', 'O', 'OGN',
        'OKE', 'OKTA', 'OMC', 'ONON', 'ORCL', 'ORLY', 'OSCR', 'OSK', 'OVV', 'OXY',
        'PAAS', 'PAYC', 'PB', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEN', 'PFE', 'PFG',
        'PGR', 'PHM', 'PKG', 'PKI', 'PLD', 'PLTR', 'PNC', 'PNR', 'PNW', 'PODD',
        'POR', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PTC', 'PTEN', 'PWR',
        'PXD', 'PYPL', 'QRVO', 'RBLX', 'RCL', 'RE', 'REG', 'RELX', 'RF', 'RGA',
        'RHI', 'RJF', 'RL', 'RMBS', 'ROK', 'ROL', 'RPM', 'RPT', 'RVTY', 'RYAAY',
        'RZG', 'S', 'SAIA', 'SBAC', 'SCHW', 'SEDG', 'SEE', 'SIRI', 'SJM', 'SLB',
        'SMCI', 'SNA', 'SNOW', 'SO', 'SOLV', 'SON', 'SPLK', 'SQM', 'SQSP', 'SRCL',
        'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWAV', 'SWK', 'SWKS', 'SYF',
        'SYY', 'T', 'TAP', 'TDY', 'TEL', 'TER', 'TFC', 'TGT', 'TKO', 'TPG',
        'TROW', 'TRU', 'TSCO', 'TSN', 'TT', 'TTD', 'TTWO', 'TXT', 'TYL', 'U',
        'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UPS', 'URI', 'USB', 'USFD', 'UTHR',
        'VALE', 'VEEV', 'VFC', 'VICI', 'VMC', 'VNO', 'VRSN', 'VRTX', 'VSAT', 'VST',
        'VTR', 'VTRS', 'W', 'WAL', 'WBD', 'WBS', 'WCC', 'WDAY', 'WDC', 'WEC',
        'WELL', 'WHR', 'WMB', 'WRB', 'WRK', 'WST', 'WTW', 'WU', 'WWD', 'WY',
        'WYNN', 'XEL', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZD', 'ZEN', 'ZION'
    ]
    
    return sp500_symbols


def backtest_symbol(
    symbol: str,
    strategy_name: str,
    engine: BacktestEngine,
    start_date: str,
    end_date: str,
    strategy_config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Backtest a single symbol.
    
    Returns:
        Dictionary with results, or None if failed
    """
    try:
        backtester = StrategyBacktester(
            strategy_name=strategy_name,
            engine=engine,
            strategy_config=strategy_config or {}
        )
        
        results = backtester.backtest(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date
        )
        
        if results and symbol in results:
            return results[symbol]
        else:
            logger.warning(f"No results for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"Error backtesting {symbol}: {e}")
        return None


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """
    Save backtest results to CSV file (one line per symbol).
    
    Args:
        results: List of result dictionaries
        output_file: Output CSV file path
    """
    if not results:
        logger.warning("No results to save")
        return
    
    # Get all possible keys from results
    all_keys = set()
    for result in results:
        if result:
            all_keys.update(result.keys())
    
    # Define column order (prioritize important metrics)
    priority_keys = [
        'symbol', 'strategy_name', 'total_return', 'sharpe_ratio', 'max_drawdown',
        'win_rate', 'total_trades', 'avg_trade_return', 'passed',
        'start_date', 'end_date', 'initial_cash', 'commission', 'risk_free_rate'
    ]
    
    # Order columns: priority first, then rest alphabetically
    ordered_keys = [k for k in priority_keys if k in all_keys]
    ordered_keys.extend(sorted([k for k in all_keys if k not in priority_keys]))
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
        writer.writeheader()
        
        for result in results:
            if result:
                writer.writerow(result)
    
    logger.info(f"Saved {len([r for r in results if r])} results to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Backtest all S&P 500 stocks and save results to CSV'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=list(STRATEGY_REGISTRY.keys()),
        help='Strategy name to backtest'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=f'sp500_backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        help='Output CSV file path'
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
        help='Maximum number of symbols to backtest (for testing)'
    )
    parser.add_argument(
        '--skip-symbols',
        type=str,
        default='',
        help='Comma-separated list of symbols to skip'
    )
    
    args = parser.parse_args()
    
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
    
    # Get S&P 500 symbols
    print("=" * 80)
    print("📊 S&P 500 BACKTEST")
    print("=" * 80)
    print(f"Strategy: {args.strategy}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Output: {args.output}")
    print("=" * 80)
    print("\nFetching S&P 500 symbols...")
    
    symbols = get_sp500_symbols()
    
    # Filter symbols
    skip_list = [s.strip().upper() for s in args.skip_symbols.split(',') if s.strip()]
    symbols = [s for s in symbols if s not in skip_list]
    
    if args.max_symbols:
        symbols = symbols[:args.max_symbols]
        print(f"Limiting to first {args.max_symbols} symbols (testing mode)")
    
    print(f"Found {len(symbols)} symbols to backtest\n")
    
    # Backtest each symbol
    results = []
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Backtesting {symbol}...", end=' ', flush=True)
        
        result = backtest_symbol(
            symbol=symbol,
            strategy_name=args.strategy,
            engine=engine,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if result:
            result['symbol'] = symbol  # Ensure symbol is in result
            result['strategy_name'] = args.strategy
            results.append(result)
            successful += 1
            print(f"✅ {result.get('total_return', 0):.2f}% return, "
                  f"Sharpe: {result.get('sharpe_ratio', 0):.2f}, "
                  f"DD: {result.get('max_drawdown', 0):.2f}%")
        else:
            failed += 1
            results.append({'symbol': symbol, 'strategy_name': args.strategy, 'error': 'Failed'})
            print(f"❌ Failed")
        
        # Save progress every 10 symbols
        if i % 10 == 0:
            save_results_to_csv(results, args.output + '.partial')
            print(f"  💾 Progress saved ({i}/{len(symbols)} complete)")
    
    # Save final results
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(symbols)*100:.1f}%")
    
    # Calculate summary statistics
    if successful > 0:
        returns = [r.get('total_return', 0) for r in results if r and 'total_return' in r]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results if r and 'sharpe_ratio' in r]
        max_drawdowns = [r.get('max_drawdown', 0) for r in results if r and 'max_drawdown' in r]
        
        print(f"\nSummary Statistics:")
        print(f"  Average Return: {sum(returns)/len(returns):.2f}%")
        print(f"  Average Sharpe: {sum(sharpe_ratios)/len(sharpe_ratios):.2f}")
        print(f"  Average Max DD: {sum(max_drawdowns)/len(max_drawdowns):.2f}%")
        print(f"  Best Return: {max(returns):.2f}%")
        print(f"  Worst Return: {min(returns):.2f}%")
        print(f"  Passed Validation: {sum(1 for r in results if r and r.get('passed', False))}")
    
    save_results_to_csv(results, args.output)
    print(f"\n✅ Results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

