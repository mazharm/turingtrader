#!/usr/bin/env python3
"""Generate performance reports for TuringTrader."""

import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import datetime
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """Generate performance reports for TuringTrader."""
    
    def __init__(self, data_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """Initialize the performance reporter.
        
        Args:
            data_dir (str, optional): Directory containing trading data
            output_dir (str, optional): Directory for saving reports
        """
        # Set up data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path.home() / ".turingtrader" / "data"
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path.home() / ".turingtrader" / "reports"
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_trading_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load trading data from JSON files.
        
        Args:
            start_date (str, optional): Start date (YYYY-MM-DD)
            end_date (str, optional): End date (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Trading data
        """
        # Convert string dates to datetime objects if provided
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.datetime.now()
        
        # Find all summary files
        summary_files = list(self.data_dir.glob("summary_*.json"))
        if not summary_files:
            logger.error(f"No summary files found in {self.data_dir}")
            return pd.DataFrame()
        
        # Load data from each file
        data_list = []
        for file_path in summary_files:
            # Extract date from filename
            date_str = file_path.stem.replace("summary_", "")
            try:
                file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Could not parse date from filename: {file_path.name}")
                continue
                
            # Skip if outside date range
            if (start_dt and file_date < start_dt) or (end_dt and file_date > end_dt):
                continue
                
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    data_list.append(data)
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {e}")
        
        if not data_list:
            logger.warning("No data loaded within the specified date range")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        logger.info(f"Loaded {len(df)} days of trading data")
        return df
    
    def load_backtest_results(self, results_file: str) -> Dict[str, Any]:
        """Load backtest results from a JSON file.
        
        Args:
            results_file (str): Path to backtest results file
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            logger.info(f"Loaded backtest results from {results_file}")
            return results
        except Exception as e:
            logger.error(f"Error loading backtest results from {results_file}: {e}")
            return {}
    
    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics from trading data.
        
        Args:
            data (pd.DataFrame): Trading data
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if data.empty:
            return {}
        
        # Calculate daily returns
        data['daily_return'] = data['final_balance'].pct_change().fillna(0)
        
        # Calculate cumulative returns
        initial_balance = data['final_balance'].iloc[0]
        data['cumulative_return'] = (data['final_balance'] / initial_balance) - 1
        
        # Calculate total return
        total_days = len(data)
        total_return = data['cumulative_return'].iloc[-1]
        
        # Calculate annualized return (trading days)
        annual_return = ((1 + total_return) ** (252 / total_days)) - 1
        
        # Calculate volatility
        daily_std = data['daily_return'].std()
        annual_volatility = daily_std * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate max drawdown
        data['peak'] = data['final_balance'].cummax()
        data['drawdown'] = (data['final_balance'] - data['peak']) / data['peak']
        max_drawdown = data['drawdown'].min()
        
        # Calculate win rate
        win_days = len(data[data['daily_return'] > 0])
        win_rate = win_days / total_days if total_days > 0 else 0
        
        # Calculate profit factor
        gross_profit = data.loc[data['daily_return'] > 0, 'daily_return'].sum()
        gross_loss = abs(data.loc[data['daily_return'] < 0, 'daily_return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_days": total_days,
            "initial_balance": initial_balance,
            "final_balance": data['final_balance'].iloc[-1],
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": data['trades'].sum(),
        }
    
    def generate_trading_report(
        self,
        data: pd.DataFrame,
        output_file: Optional[str] = None,
        show_plot: bool = False
    ) -> None:
        """Generate a performance report from trading data.
        
        Args:
            data (pd.DataFrame): Trading data
            output_file (str, optional): Output file for the report
            show_plot (bool): Whether to display the plot
        """
        if data.empty:
            logger.error("No data to generate report from")
            return
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(data)
        if not metrics:
            logger.error("Failed to calculate performance metrics")
            return
        
        # Set up plotting
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 18))
        
        # Plot account balance over time
        plt.subplot(3, 1, 1)
        plt.plot(data.index, data['final_balance'], label="Account Balance")
        plt.title("Account Balance Over Time")
        plt.xlabel("Date")
        plt.ylabel("Account Balance ($)")
        plt.grid(True)
        
        # Plot daily P&L
        plt.subplot(3, 1, 2)
        plt.bar(data.index, data['pnl'], color=np.where(data['pnl'] >= 0, 'green', 'red'))
        plt.title("Daily P&L")
        plt.xlabel("Date")
        plt.ylabel("P&L ($)")
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(3, 1, 3)
        plt.fill_between(data.index, data['drawdown'] * 100, 0, color='red', alpha=0.3)
        plt.title("Drawdown Over Time")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        
        # Add performance metrics as text
        plt.figtext(
            0.1, 0.01, 
            f"Performance Metrics:\n"
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annual Return: {metrics['annual_return']:.2%}\n"
            f"Annual Volatility: {metrics['annual_volatility']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Win Rate: {metrics['win_rate']:.2%}\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Total Trades: {metrics['total_trades']}\n"
            f"Trading Days: {metrics['total_days']}",
            fontsize=10
        )
        
        # Save or show the plot
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if output_file:
            output_path = self.output_dir / output_file
            plt.savefig(output_path)
            logger.info(f"Trading report saved to {output_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def generate_risk_comparison_report(
        self,
        risk_levels: List[str] = ["LOW", "MEDIUM", "HIGH", "AGGRESSIVE"],
        backtest_dir: Optional[str] = None,
        output_file: Optional[str] = None,
        show_plot: bool = False
    ) -> None:
        """Generate a report comparing different risk levels.
        
        Args:
            risk_levels (List[str]): Risk levels to compare
            backtest_dir (str, optional): Directory containing backtest results
            output_file (str, optional): Output file for the report
            show_plot (bool): Whether to display the plot
        """
        if backtest_dir:
            backtest_path = Path(backtest_dir)
        else:
            backtest_path = Path.home() / ".turingtrader" / "backtest"
        
        # Find backtest results for each risk level
        results = {}
        for risk in risk_levels:
            risk_files = list(backtest_path.glob(f"*{risk.lower()}*.json"))
            if not risk_files:
                logger.warning(f"No backtest results found for risk level {risk}")
                continue
                
            # Use the most recent file
            latest_file = max(risk_files, key=lambda p: p.stat().st_mtime)
            risk_results = self.load_backtest_results(str(latest_file))
            
            if risk_results:
                results[risk] = risk_results
        
        if not results:
            logger.error("No backtest results found")
            return
        
        # Set up plotting
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 10))
        
        # Extract metrics for comparison
        metrics = ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown", "win_rate"]
        metric_names = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown", "Win Rate"]
        
        strategies = list(next(iter(results.values())).keys())  # Get strategy names from first result
        
        for strategy_idx, strategy in enumerate(strategies):
            plt.subplot(len(strategies), 1, strategy_idx + 1)
            
            metric_values = {}
            for metric in metrics:
                metric_values[metric] = [
                    results[risk][strategy].get(metric, 0) for risk in results.keys()
                ]
            
            # Create a grouped bar chart
            x = np.arange(len(metrics))
            bar_width = 0.15
            
            for i, risk in enumerate(results.keys()):
                plt.bar(
                    x + i * bar_width,
                    [results[risk][strategy].get(m, 0) for m in metrics],
                    bar_width,
                    label=risk
                )
            
            plt.title(f"{strategy} Strategy - Risk Level Comparison")
            plt.xticks(x + bar_width * (len(results) - 1) / 2, metric_names)
            plt.legend()
            plt.grid(True)
        
        # Save or show the plot
        plt.tight_layout()
        
        if output_file:
            output_path = self.output_dir / output_file
            plt.savefig(output_path)
            logger.info(f"Risk comparison report saved to {output_path}")
            
        if show_plot:
            plt.show()
        else:
            plt.close()


def main() -> None:
    """Main entry point for the report generator script."""
    parser = argparse.ArgumentParser(description="Generate performance reports for TuringTrader")
    parser.add_argument(
        "--data-dir",
        help="Directory containing trading data"
    )
    parser.add_argument(
        "--backtest-dir",
        help="Directory containing backtest results"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for saving reports"
    )
    parser.add_argument(
        "--start-date",
        help="Start date for report (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        help="End date for report (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--trading-report",
        action="store_true",
        help="Generate trading performance report"
    )
    parser.add_argument(
        "--risk-report",
        action="store_true",
        help="Generate risk comparison report"
    )
    parser.add_argument(
        "--trading-report-file",
        default="trading_report.png",
        help="Output file for trading report"
    )
    parser.add_argument(
        "--risk-report-file",
        default="risk_comparison.png",
        help="Output file for risk comparison report"
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display plots"
    )
    args = parser.parse_args()
    
    # Create reporter
    reporter = PerformanceReporter(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Generate reports
    if args.trading_report:
        # Load trading data
        data = reporter.load_trading_data(args.start_date, args.end_date)
        if not data.empty:
            reporter.generate_trading_report(
                data,
                args.trading_report_file,
                args.show_plot
            )
    
    if args.risk_report:
        reporter.generate_risk_comparison_report(
            backtest_dir=args.backtest_dir,
            output_file=args.risk_report_file,
            show_plot=args.show_plot
        )


if __name__ == "__main__":
    main()