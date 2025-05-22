"""
Performance Reporting Module

This module handles reporting functionalities:
- Daily and cumulative performance reports
- Risk metrics calculation (Sharpe ratio, drawdown, etc.)
- Trade statistics and analytics
- Report generation in various formats
"""
import os
import logging
import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class PerformanceReporter:
    """
    Performance tracking and reporting
    """
    
    def __init__(self, report_dir: str = "reports"):
        """
        Initialize performance reporter
        
        Args:
            report_dir: Directory for saving reports
        """
        self.logger = logging.getLogger(__name__)
        self.report_dir = report_dir
        self.trades = []
        self.daily_results = {}
        self.performance_metrics = {}
        
        # Create reports directory if it doesn't exist
        os.makedirs(report_dir, exist_ok=True)
    
    def add_trade(self, trade_data: Dict) -> None:
        """
        Add a completed trade to the history
        
        Args:
            trade_data: Dictionary with trade information
        """
        self.trades.append(trade_data)
        self.logger.debug(f"Trade recorded: {trade_data['symbol']} {trade_data['strategy']}")
    
    def add_daily_result(self, date: datetime.date, 
                        pnl: float, 
                        trades_count: int,
                        ending_cash: float) -> None:
        """
        Add daily trading result
        
        Args:
            date: Trading date
            pnl: Profit/Loss for the day
            trades_count: Number of trades executed
            ending_cash: Ending cash balance
        """
        date_str = date.strftime("%Y-%m-%d")
        self.daily_results[date_str] = {
            'date': date_str,
            'pnl': pnl,
            'trades_count': trades_count,
            'ending_cash': ending_cash
        }
        self.logger.debug(f"Daily result recorded for {date_str}: PnL = ${pnl:.2f}")
        
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics based on recorded trades and daily results
        
        Returns:
            dict: Performance metrics
        """
        if not self.daily_results:
            return {'error': 'No performance data available'}
            
        # Convert daily results to DataFrame
        daily_df = pd.DataFrame.from_dict(self.daily_results, orient='index')
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df.set_index('date', inplace=True)
        daily_df.sort_index(inplace=True)
        
        # Convert trade history to DataFrame
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
        else:
            trades_df = pd.DataFrame()
            
        # Calculate performance metrics
        metrics = {}
        
        # Basic metrics
        metrics['total_days'] = len(daily_df)
        metrics['total_pnl'] = daily_df['pnl'].sum()
        
        if not trades_df.empty:
            metrics['total_trades'] = len(trades_df)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            metrics['winning_trades'] = len(winning_trades)
            metrics['win_rate'] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        else:
            metrics['total_trades'] = 0
            metrics['winning_trades'] = 0
            metrics['win_rate'] = 0
        
        # Returns
        if metrics['total_days'] > 0:
            metrics['avg_daily_pnl'] = metrics['total_pnl'] / metrics['total_days']
        else:
            metrics['avg_daily_pnl'] = 0
            
        # Calculate daily returns
        if len(daily_df) > 1:
            daily_df['return'] = daily_df['ending_cash'].pct_change()
            
            # Risk metrics
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(daily_df['return'])
            metrics['max_drawdown'] = self._calculate_max_drawdown(daily_df['ending_cash'])
            metrics['volatility'] = daily_df['return'].std() * np.sqrt(252)  # Annualized
        else:
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0
            metrics['volatility'] = 0
            
        self.performance_metrics = metrics
        return metrics
    
    def _calculate_sharpe_ratio(self, returns_series: pd.Series, 
                              risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns_series: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            float: Sharpe ratio
        """
        if returns_series.empty or returns_series.std() == 0:
            return 0
            
        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate excess returns
        excess_returns = returns_series - daily_rf
        
        # Calculate Sharpe ratio (annualized)
        sharpe = excess_returns.mean() / returns_series.std() * np.sqrt(252)
        
        return sharpe
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            equity_series: Series of equity values
            
        Returns:
            float: Maximum drawdown (percentage)
        """
        if equity_series.empty:
            return 0
            
        # Calculate running maximum
        running_max = equity_series.cummax()
        
        # Calculate drawdown series
        drawdown = (equity_series - running_max) / running_max
        
        # Get maximum drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def generate_daily_report(self, date: Optional[datetime.date] = None) -> Dict:
        """
        Generate a report for a specific day
        
        Args:
            date: Date to report on (defaults to most recent)
            
        Returns:
            dict: Daily report data
        """
        if not self.daily_results:
            return {'error': 'No performance data available'}
            
        # Use the most recent date if none provided
        if date is None:
            date_str = max(self.daily_results.keys())
        else:
            date_str = date.strftime("%Y-%m-%d")
            
        # Check if we have data for this date
        if date_str not in self.daily_results:
            return {'error': f'No data available for {date_str}'}
            
        # Get daily result
        daily_result = self.daily_results[date_str]
        
        # Find trades for this day
        if self.trades:
            day_trades = [t for t in self.trades if t.get('exit_time', '').startswith(date_str)]
        else:
            day_trades = []
            
        # Compile report
        report = {
            'date': date_str,
            'summary': daily_result,
            'trades': day_trades,
            'trade_count': len(day_trades),
            'winning_trades': len([t for t in day_trades if t.get('pnl', 0) > 0]),
            'metrics': self.performance_metrics if self.performance_metrics else 
                      self.calculate_performance_metrics()
        }
        
        return report
    
    def save_report_json(self, report_data: Dict, filename: str) -> str:
        """
        Save report data as JSON
        
        Args:
            report_data: Report data dictionary
            filename: Base filename (without extension)
            
        Returns:
            str: Path to saved report
        """
        # Ensure file has .json extension
        if not filename.endswith('.json'):
            filename = f"{filename}.json"
            
        # Create full path
        filepath = os.path.join(self.report_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
                
            self.logger.info(f"Report saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving report: {str(e)}")
            return ""
            
    def generate_performance_chart(self, 
                                  chart_type: str = 'equity',
                                  save_path: Optional[str] = None) -> str:
        """
        Generate performance chart
        
        Args:
            chart_type: Chart type ('equity', 'returns', or 'drawdown')
            save_path: Path to save chart or None to auto-generate
            
        Returns:
            str: Path to saved chart
        """
        if not self.daily_results:
            self.logger.warning("No data available for chart generation")
            return ""
            
        # Convert daily results to DataFrame
        daily_df = pd.DataFrame.from_dict(self.daily_results, orient='index')
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df.set_index('date', inplace=True)
        daily_df.sort_index(inplace=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'equity':
            # Plot equity curve
            plt.plot(daily_df.index, daily_df['ending_cash'])
            plt.title('Equity Curve')
            plt.ylabel('Account Value ($)')
            
        elif chart_type == 'returns':
            # Calculate and plot daily returns
            if len(daily_df) > 1:
                returns = daily_df['ending_cash'].pct_change().dropna()
                plt.bar(returns.index, returns * 100)
                plt.title('Daily Returns')
                plt.ylabel('Return (%)')
                
        elif chart_type == 'drawdown':
            # Calculate and plot drawdown
            if len(daily_df) > 1:
                equity = daily_df['ending_cash']
                running_max = equity.cummax()
                drawdown = (equity - running_max) / running_max * 100
                plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
                plt.title('Drawdown')
                plt.ylabel('Drawdown (%)')
                
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            date_str = datetime.datetime.now().strftime('%Y%m%d')
            save_path = os.path.join(self.report_dir, f"{chart_type}_{date_str}.png")
        
        try:
            plt.savefig(save_path)
            plt.close()
            self.logger.info(f"Chart saved to {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error saving chart: {str(e)}")
            plt.close()
            return ""