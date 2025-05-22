#!/usr/bin/env python3
"""
Backtesting script for the TuringTrader algorithm.
"""

import os
import sys
import argparse
import logging
import pickle
from datetime import datetime, timedelta

from utils.logging_util import setup_logging
from ibkr_trader.config import Config
from backtesting.backtest_engine import BacktestEngine
from utils.reporting import generate_report, generate_multi_risk_report, export_multi_risk_summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TuringTrader Backtester')
    
    # Backtest period
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD), defaults to 1 year ago'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD), defaults to today'
    )
    
    # Risk level
    parser.add_argument(
        '--risk-level',
        type=int,
        help='Risk level (1-10, where 10 is highest risk)'
    )
    
    parser.add_argument(
        '--all-risk-levels',
        action='store_true',
        help='Run backtest for all risk levels (1-10)'
    )
    
    # Initial balance
    parser.add_argument(
        '--initial-balance',
        type=float,
        default=100000.0,
        help='Initial account balance'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./reports',
        help='Output directory for reports and data'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results to pickle file'
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


def run_backtest(args):
    """
    Run backtests based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = Config(args.config)
    
    # Create backtest engine
    engine = BacktestEngine(
        config=config,
        initial_balance=args.initial_balance
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which risk levels to test
    if args.all_risk_levels:
        risk_levels = range(1, 11)
    elif args.risk_level:
        risk_levels = [args.risk_level]
    else:
        # Default to middle risk
        risk_levels = [5]
    
    results_by_risk = {}
    
    # Run backtest for each risk level
    for risk_level in risk_levels:
        logging.info(f"Running backtest for risk level {risk_level}...")
        
        results = engine.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            risk_level=risk_level
        )
        
        # Store results
        results_by_risk[risk_level] = results
        
        # Generate individual report if multiple risk levels
        if len(risk_levels) > 1:
            report_path = generate_report(
                results=results,
                output_dir=args.output_dir,
                risk_level=risk_level
            )
            logging.info(f"Report for risk level {risk_level} generated: {report_path}")
    
    # Generate comparison report if multiple risk levels
    if len(risk_levels) > 1:
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
    
    # Save results to pickle if requested
    if args.save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if len(risk_levels) > 1:
            pickle_path = f"{args.output_dir}/backtest_all_risks_{timestamp}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(results_by_risk, f)
        else:
            risk_level = list(results_by_risk.keys())[0]
            pickle_path = f"{args.output_dir}/backtest_risk{risk_level}_{timestamp}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(results_by_risk[risk_level], f)
                
        logging.info(f"Results saved to {pickle_path}")
    
    return 0


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    try:
        # Run the backtest
        return run_backtest(args)
    
    except KeyboardInterrupt:
        logging.info("Backtesting interrupted by user")
        return 0
    
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())