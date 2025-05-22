#!/usr/bin/env python3
"""
Main entry point for the TuringTrader algorithm.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from utils.logging_util import setup_logging
from ibkr_trader.trader import TuringTrader
from ibkr_trader.config import Config, create_default_config_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TuringTrader - AI driven algorithmic trader')
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'paper', 'backtest', 'test'],
        default='paper',
        help='Trading mode: live, paper, backtest, or test connection'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a default configuration file'
    )
    
    # Risk level
    parser.add_argument(
        '--risk-level',
        type=int,
        choices=range(1, 11),
        help='Risk level (1-10, where 10 is highest risk)'
    )
    
    # Trading parameters
    parser.add_argument(
        '--max-cycles',
        type=int,
        default=-1,
        help='Maximum number of trading cycles to run (-1 for unlimited)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Seconds between trading cycles'
    )
    
    # Backtest parameters
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD)'
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


def run_trader(args):
    """
    Run the trader based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    # Create default config if requested
    if args.create_config:
        config_path = args.config or 'config.ini'
        create_default_config_file(config_path)
        print(f"Created default configuration file: {config_path}")
        return 0
    
    # Initialize the trader
    trader = TuringTrader(args.config)
    
    # Set risk level if provided
    if args.risk_level is not None:
        trader.config.risk.adjust_for_risk_level(args.risk_level)
        logging.info(f"Risk level set to {args.risk_level}")
    
    # Run in the selected mode
    if args.mode == 'test':
        # Just test connection
        if trader.connect():
            logging.info("Successfully connected to Interactive Brokers")
            trader.disconnect()
            return 0
        else:
            logging.error("Failed to connect to Interactive Brokers")
            return 1
    
    elif args.mode == 'backtest':
        # Run backtest
        from datetime import datetime
        
        # Validate dates
        if not args.start_date or not args.end_date:
            logging.error("Start and end dates are required for backtest mode")
            return 1
            
        # Run backtest
        results = trader.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            risk_level=args.risk_level or 5
        )
        
        # Generate report
        from utils.reporting import generate_report
        report_path = generate_report(results, risk_level=args.risk_level)
        logging.info(f"Backtest report generated: {report_path}")
        
        return 0
        
    elif args.mode in ['live', 'paper']:
        # Run trading loop
        trader.run_trading_loop(
            max_cycles=args.max_cycles, 
            interval_seconds=args.interval
        )
        return 0
    
    return 0


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    try:
        # Run the trader
        return run_trader(args)
    
    except KeyboardInterrupt:
        logging.info("Trading interrupted by user")
        return 0
    
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())