#!/usr/bin/env python3
"""
Daily Opportunity Scanner
Run at 4:05 PM ET to identify oversold opportunities for next day.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import numpy as np
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    """Scanner configuration"""
    # Tier 1: Index ETFs
    tier1_symbols = ['SPY', 'QQQ', 'DIA', 'IWM']
    
    # Tier 2: Sector ETFs
    tier2_symbols = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY']
    
    # Tier 3: Individual Stocks
    tier3_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        'UNH', 'JPM', 'V', 'MA', 'LLY', 'WMT', 'HD', 'PG',
        'XOM', 'CVX', 'JNJ', 'BAC', 'COST'
    ]
    
    # Parameters
    z_score_threshold = -1.0
    min_volume_ratio = 1.2  # 20% above average
    max_signal_count = 10   # Top N signals
    lookback_days = 20


class DailyScanner:
    def __init__(self, config: ScannerConfig = None):
        self.config = config or ScannerConfig()
        self.all_symbols = (
            self.config.tier1_symbols +
            self.config.tier2_symbols +
            self.config.tier3_symbols
        )
    
    def scan_all_symbols(self) -> Dict:
        """
        Scan all symbols for oversold conditions.
        
        Returns:
            Dict with scan results including prices and scores
        """
        logger.info(f"📊 Scanning {len(self.all_symbols)} symbols...")
        
        results = []
        
        # Download all data in batches
        logger.info("📥 Fetching market data...")
        try:
            # Download in batches to avoid timeouts
            batch_size = 10
            
            for i in range(0, len(self.all_symbols), batch_size):
                batch = self.all_symbols[i:i + batch_size]
                logger.info(f"  Fetching batch {i//batch_size + 1}: {batch}")
                
                try:
                    batch_data = yf.download(
                        batch,
                        period='1mo',
                        interval='1d',
                        group_by='ticker',
                        progress=False,
                        timeout=30
                    )
                    
                    # Process each symbol in batch
                    for symbol in batch:
                        try:
                            if len(batch) == 1:
                                # Single symbol case
                                closes = batch_data['Close'].dropna()
                                volumes = batch_data['Volume'].dropna()
                            else:
                                # Multi-symbol case
                                if isinstance(batch_data.columns, pd.MultiIndex):
                                    if symbol in batch_data.columns.get_level_values(0):
                                        closes = batch_data[symbol]['Close'].dropna()
                                        volumes = batch_data[symbol]['Volume'].dropna()
                                    else:
                                        continue
                                else:
                                    continue
                            
                            if len(closes) >= self.config.lookback_days:
                                # Calculate z-score
                                recent_closes = closes[-self.config.lookback_days:]
                                current_price = closes.iloc[-1]
                                mean_price = recent_closes.mean()
                                std_price = recent_closes.std()
                                
                                if std_price > 0:
                                    z_score = (current_price - mean_price) / std_price
                                else:
                                    z_score = 0
                                
                                # Calculate volume ratio
                                recent_volumes = volumes[-self.config.lookback_days:]
                                current_volume = volumes.iloc[-1]
                                avg_volume = recent_volumes.mean()
                                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                                
                                # Composite score (lower z-score = more oversold)
                                score = z_score  # Negative is good for buys
                                if volume_ratio > self.config.min_volume_ratio:
                                    score -= 0.5  # Bonus for volume spike
                                
                                results.append({
                                    'symbol': symbol,
                                    'price': float(current_price),
                                    'z_score': float(z_score),
                                    'volume_ratio': float(volume_ratio),
                                    'score': float(score),
                                    'mean_price': float(mean_price),
                                    'std_price': float(std_price),
                                    'timestamp': datetime.now().isoformat()
                                })
                                
                        except Exception as e:
                            logger.warning(f"⚠️  Error processing {symbol}: {e}")
                            continue
                
                except Exception as e:
                    logger.warning(f"⚠️  Batch download failed, trying sequential: {e}")
                    # Fallback to sequential
                    results.extend(self._scan_symbols_sequential(batch))
        
        except Exception as e:
            logger.error(f"🚫 Batch download failed: {e}, falling back to sequential")
            results = self._scan_symbols_sequential(self.all_symbols)
        
        # Sort by score (most oversold first)
        results.sort(key=lambda x: x['score'])
        
        # Take top signals
        top_signals = results[:self.config.max_signal_count]
        
        # Always include baseline symbols
        baseline_symbols = ['SPY', 'QQQ']
        baseline_signals = []
        for symbol in baseline_symbols:
            if symbol not in [s['symbol'] for s in top_signals]:
                # Get their data if not already in top signals
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='5d')
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        baseline_signals.append({
                            'symbol': symbol,
                            'price': float(current_price),
                            'z_score': 0,
                            'volume_ratio': 1.0,
                            'score': 0,
                            'mean_price': float(current_price),
                            'std_price': 0,
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.warning(f"⚠️  Error getting baseline {symbol}: {e}")
        
        # Combine and save
        all_signals = top_signals + baseline_signals
        
        output = {
            'scan_timestamp': datetime.now().isoformat(),
            'symbols': [s['symbol'] for s in all_signals],
            'scan_prices': {s['symbol']: s['price'] for s in all_signals},
            'scores': {s['symbol']: s['score'] for s in all_signals},
            'signals': all_signals,
            'metadata': {
                'total_scanned': len(self.all_symbols),
                'signals_found': len(all_signals),
                'z_score_threshold': self.config.z_score_threshold,
                'lookback_days': self.config.lookback_days
            }
        }
        
        return output
    
    def _scan_symbols_sequential(self, symbols: List[str]) -> List[Dict]:
        """Fallback: scan symbols one by one"""
        results = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1mo')
                
                if len(hist) >= self.config.lookback_days:
                    closes = hist['Close'].values
                    volumes = hist['Volume'].values
                    
                    current_price = closes[-1]
                    mean_price = closes[-self.config.lookback_days:].mean()
                    std_price = closes[-self.config.lookback_days:].std()
                    
                    if std_price > 0:
                        z_score = (current_price - mean_price) / std_price
                    else:
                        z_score = 0
                    
                    current_volume = volumes[-1]
                    avg_volume = volumes[-self.config.lookback_days:].mean()
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    score = z_score
                    if volume_ratio > self.config.min_volume_ratio:
                        score -= 0.5
                    
                    results.append({
                        'symbol': symbol,
                        'price': float(current_price),
                        'z_score': float(z_score),
                        'volume_ratio': float(volume_ratio),
                        'score': float(score),
                        'mean_price': float(mean_price),
                        'std_price': float(std_price),
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️  Error scanning {symbol}: {e}")
                continue
        
        return results
    
    def save_results(self, results: Dict, filename: str = "candidates.json"):
        """Save scan results to file"""
        try:
            filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), filename)
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"💾 Saved {len(results['symbols'])} signals to {filepath}")
            
            # Print summary
            print("\n" + "="*80)
            print("📊 DAILY SCANNER RESULTS")
            print("="*80)
            
            print(f"\n🔍 Top {len(results['symbols'])} Signals:")
            print(f"{'Symbol':<8} {'Price':<10} {'Z-Score':<10} {'Volume Ratio':<12}")
            print("-"*45)
            
            for signal in results['signals'][:10]:
                print(f"{signal['symbol']:<8} ${signal['price']:<9.2f} {signal['z_score']:<9.2f} {signal['volume_ratio']:<11.2f}x")
            
            print(f"\n📈 Metadata:")
            print(f"  Total scanned: {results['metadata']['total_scanned']}")
            print(f"  Signals found: {results['metadata']['signals_found']}")
            print(f"  Scan time: {results['scan_timestamp']}")
            print("="*80)
            
        except Exception as e:
            logger.error(f"🚫 Failed to save results: {e}")


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🎯 DAILY OPPORTUNITY SCANNER")
    print("="*80)
    
    # Initialize scanner
    scanner = DailyScanner()
    
    # Run scan
    try:
        results = scanner.scan_all_symbols()
        
        # Save results
        scanner.save_results(results)
        
        # Create backup copy
        date_str = datetime.now().strftime('%Y%m%d')
        backup_file = f"candidates_{date_str}.json"
        scanner.save_results(results, backup_file)
        
        logger.info(f"✅ Scan completed successfully")
        
    except Exception as e:
        logger.error(f"🚫 Scanner failed: {e}")
        raise


if __name__ == "__main__":
    main()
