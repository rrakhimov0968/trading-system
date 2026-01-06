"""Walk-forward analysis for preventing overfitting."""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from tests.backtest.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)


class WalkForwardAnalyzer:
    """
    Walk-forward analysis to test strategy robustness across time periods.
    
    Prevents overfitting by testing on multiple non-overlapping periods.
    If performance varies significantly, the strategy is likely overfit.
    """
    
    def __init__(self, engine: BacktestEngine, n_splits: int = 4):
        """
        Initialize walk-forward analyzer.
        
        Args:
            engine: BacktestEngine instance
            n_splits: Number of time periods to split data into
        """
        self.engine = engine
        self.n_splits = n_splits
    
    def analyze(
        self,
        strategy_class,
        price_data: pd.DataFrame,
        strategy_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis.
        
        Args:
            strategy_class: Strategy class to test
            price_data: Full price data DataFrame
            strategy_config: Optional strategy configuration
        
        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Running walk-forward analysis with {self.n_splits} periods...")
        
        # Split data into periods
        total_days = len(price_data)
        period_length = total_days // self.n_splits
        
        periods = []
        for i in range(self.n_splits):
            start_idx = i * period_length
            end_idx = (i + 1) * period_length if i < self.n_splits - 1 else total_days
            
            start_date = price_data.index[start_idx]
            end_date = price_data.index[end_idx - 1]
            
            periods.append({
                'period': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_date': start_date,
                'end_date': end_date,
                'data': price_data.iloc[start_idx:end_idx]
            })
        
        results = []
        
        for period_info in periods:
            period_data = period_info['data']
            
            if len(period_data) < 50:  # Minimum data needed
                logger.warning(f"Period {period_info['period']} has insufficient data, skipping")
                continue
            
            logger.info(
                f"Testing period {period_info['period']}: "
                f"{period_info['start_date'].strftime('%Y-%m-%d')} to "
                f"{period_info['end_date'].strftime('%Y-%m-%d')} "
                f"({len(period_data)} days)"
            )
            
            try:
                # Generate signals
                entries, exits = self.engine.generate_signals_from_strategy(
                    strategy_class,
                    period_data,
                    strategy_config
                )
                
                # Run backtest
                portfolio = self.engine.run_backtest(period_data, entries, exits)
                
                # Calculate metrics
                total_return = portfolio.total_return() * 100
                sharpe = portfolio.sharpe_ratio(risk_free=self.engine.risk_free_rate / 252)
                max_dd = portfolio.max_drawdown() * 100
                
                period_result = {
                    'period': period_info['period'],
                    'start_date': period_info['start_date'],
                    'end_date': period_info['end_date'],
                    'total_return': float(total_return),
                    'sharpe_ratio': float(sharpe),
                    'max_drawdown': float(max_dd),
                    'passed': (
                        sharpe >= self.engine.min_sharpe and
                        max_dd <= self.engine.max_drawdown_threshold * 100
                    )
                }
                
                results.append(period_result)
                
            except Exception as e:
                logger.exception(f"Error in period {period_info['period']}: {e}")
                continue
        
        # Analyze consistency
        if results:
            sharpes = [r['sharpe_ratio'] for r in results]
            returns = [r['total_return'] for r in results]
            drawdowns = [r['max_drawdown'] for r in results]
            
            sharpe_range = max(sharpes) - min(sharpes)
            return_range = max(returns) - min(returns)
            
            # Check for consistency
            is_consistent = (
                sharpe_range < 1.0 and  # Sharpe doesn't vary by more than 1.0
                return_range < 50.0  # Returns don't vary by more than 50%
            )
            
            summary = {
                'periods_tested': len(results),
                'periods_passed': sum(1 for r in results if r['passed']),
                'avg_sharpe': np.mean(sharpes),
                'avg_return': np.mean(returns),
                'avg_max_drawdown': np.mean(drawdowns),
                'sharpe_range': sharpe_range,
                'return_range': return_range,
                'is_consistent': is_consistent,
                'period_results': results
            }
            
            logger.info(
                f"Walk-forward complete: {summary['periods_passed']}/{summary['periods_tested']} "
                f"periods passed. Consistency: {'✅' if is_consistent else '❌'}"
            )
            
            return summary
        else:
            logger.warning("No valid periods for walk-forward analysis")
            return {'periods_tested': 0, 'periods_passed': 0, 'is_consistent': False}

