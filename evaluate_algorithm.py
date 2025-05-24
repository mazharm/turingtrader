#!/usr/bin/env python3
"""
TuringTrader Algorithm Evaluation Script

This script evaluates the TuringTrader algorithm using historical data and generates comprehensive reports
on its performance across different risk levels.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logging_util import setup_logging
from ibkr_trader.config import Config
from backtesting.backtest_engine import BacktestEngine
from utils.reporting import generate_report, generate_multi_risk_report, export_multi_risk_summary
from historical_data.data_fetcher import HistoricalDataFetcher
from backtesting.performance_analyzer import PerformanceAnalyzer
# Import mock data for testing
from backtesting.mock_data import MockDataFetcher


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TuringTrader Algorithm Evaluation')
    
    # Initial investment
    parser.add_argument(
        '--initial-investment',
        type=float,
        default=100000.0,
        help='Initial investment amount (default: $100,000)'
    )
    
    # Data period
    parser.add_argument(
        '--period',
        type=int,
        default=365,
        help='Number of days for historical evaluation (default: 365)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_reports',
        help='Output directory for reports and data'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    # Testing mode
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run with mock data for testing purposes'
    )
    
    # Force refreshing data
    parser.add_argument(
        '--refresh-data',
        action='store_true',
        help='Force refresh of historical data (ignore cache)'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level'
    )
    
    return parser.parse_args()


def generate_investment_growth_chart(results_by_risk, initial_investment, output_dir):
    """
    Generate a chart showing the growth of investment over time across different risk levels.
    
    Args:
        results_by_risk: Dictionary mapping risk levels to results
        initial_investment: Initial investment amount
        output_dir: Output directory for the chart
        
    Returns:
        Path to the saved chart
    """
    plt.figure(figsize=(16, 10))
    
    # Create dataframes for each risk level
    for risk_level, results in sorted(results_by_risk.items()):
        if 'daily_values' not in results:
            logging.warning(f"No daily values found for risk level {risk_level}")
            continue
            
        df = pd.DataFrame(results['daily_values'])
        if df.empty:
            continue
            
        # Convert dates if they're strings
        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])
            
        plt.plot(df['date'], df['balance'], label=f"Risk Level {risk_level}")
    
    plt.title(f"Growth of ${initial_investment:,.0f} Investment Over Time", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Add annotations for final values
    last_date = None
    for risk_level, results in sorted(results_by_risk.items()):
        if 'daily_values' not in results or not results['daily_values']:
            continue
            
        daily_values = results['daily_values']
        final_balance = daily_values[-1]['balance']
        
        if last_date is None:
            last_date = pd.to_datetime(daily_values[-1]['date']) if isinstance(daily_values[-1]['date'], str) else daily_values[-1]['date']
            
        plt.annotate(
            f"${final_balance:,.0f}",
            xy=(last_date, final_balance),
            xytext=(10, 0),
            textcoords='offset points',
            fontsize=9
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the chart
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = f"{output_dir}/investment_growth_{timestamp}.png"
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Investment growth chart saved to {file_path}")
    return file_path


def evaluate_algorithm(args):
    """
    Evaluate the TuringTrader algorithm using historical data.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = Config(args.config)
    
    # Define date range (past year by default)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.period)).strftime('%Y-%m-%d')
    
    logging.info(f"Evaluating algorithm from {start_date} to {end_date}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data fetcher - use MockDataFetcher in test mode, or HistoricalDataFetcher otherwise
    if args.test_mode:
        data_fetcher = MockDataFetcher()
        logging.warning("Using MOCK data for backtesting. Results will NOT reflect real market performance.")
    else:
        data_dir = os.path.join(os.getcwd(), 'data')
        logging.info(f"Using REAL historical data from Yahoo Finance. Data will be cached in {data_dir}")
        # Clear cache if refresh data is requested
        if args.refresh_data:
            logging.info("Refreshing historical data - ignoring cache")
            # Create a data fetcher and clear its cache
            temp_fetcher = HistoricalDataFetcher(data_dir=data_dir)
            temp_fetcher.clear_cache()
        
        # Create the data fetcher with the appropriate cache setting
        data_fetcher = HistoricalDataFetcher(data_dir=data_dir)
    
    # Create backtest engine
    engine = BacktestEngine(
        config=config,
        initial_balance=args.initial_investment,
        data_fetcher=data_fetcher
    )
    
    # Define risk levels (1-10)
    risk_levels = range(1, 11)
    results_by_risk = {}
    
    # Run backtest for each risk level
    for risk_level in risk_levels:
        logging.info(f"Running backtest for risk level {risk_level}...")
        
        results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            risk_level=risk_level,
            use_cache=not args.refresh_data
        )
        
        # Store results
        results_by_risk[risk_level] = results
        
        # Generate individual report
        report_path = generate_report(
            results=results,
            output_dir=args.output_dir,
            risk_level=risk_level
        )
        logging.info(f"Report for risk level {risk_level} generated: {report_path}")
    
    # Generate comparison report
    comparison_path = generate_multi_risk_report(
        results_by_risk=results_by_risk,
        output_dir=args.output_dir
    )
    logging.info(f"Risk comparison report generated: {comparison_path}")
    
    # Export summary table
    summary_path = export_multi_risk_summary(
        results_by_risk=results_by_risk,
        output_dir=args.output_dir
    )
    logging.info(f"Risk level summary exported: {summary_path}")
    
    # Generate investment growth chart
    growth_chart_path = generate_investment_growth_chart(
        results_by_risk=results_by_risk,
        initial_investment=args.initial_investment,
        output_dir=args.output_dir
    )
    logging.info(f"Investment growth chart generated: {growth_chart_path}")
    
    # Print summary
    analyzer = PerformanceAnalyzer()
    comparison = analyzer.compare_risk_levels(results_by_risk)
    
    print("\n=== ALGORITHM EVALUATION SUMMARY ===")
    print(f"Evaluation Period: {start_date} to {end_date}")
    print(f"Initial Investment: ${args.initial_investment:,.2f}")
    print("\nBest Performance by Risk Level:")
    
    if 'optimal_risk_level' in comparison:
        print(f"- Best Sharpe Ratio: Risk Level {comparison['optimal_risk_level']} "
              f"(Sharpe: {comparison['sharpe_ratios'][comparison['risk_levels'].index(comparison['optimal_risk_level'])]:,.2f})")
    
    if 'best_return_risk_level' in comparison:
        print(f"- Best Absolute Return: Risk Level {comparison['best_return_risk_level']} "
              f"(Return: {comparison['total_returns'][comparison['risk_levels'].index(comparison['best_return_risk_level'])]:,.2f}%)")
    
    print(f"\nResults saved to: {args.output_dir}")
    
    return 0


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    try:
        # Run the evaluation
        return evaluate_algorithm(args)
    
    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user")
        return 0
    
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())