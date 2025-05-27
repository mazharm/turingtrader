#!/usr/bin/env python3
"""
Enhanced backtesting script for the TuringTrader algorithm.
This script runs backtest with optimized parameters and real market-like data.
"""

import sys
import os
import numpy as np
import datetime
from backtesting.realistic_mock_data import RealisticMockDataFetcher
from backtesting.backtest_engine import BacktestEngine
from ibkr_trader.config import Config
from utils.logging_util import setup_logging

def run_enhanced_backtest():
    """Run an enhanced backtest with more realistic market data."""
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Initialize configuration
    config = Config()
    
    # Load optimized configuration from config.ini
    config.load_config("config.ini")
    
    # Create output directory
    output_dir = "./optimized_backtest_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up date range - 180 days
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=180)
    
    # Create enhanced mock data fetcher with more realistic IV patterns
    data_fetcher = RealisticMockDataFetcher()
    
    # Override the IV calculation to ensure more volatility harvesting opportunities
    data_fetcher.iv_hv_ratio_min = 1.2  # Minimum IV/HV ratio 
    data_fetcher.iv_hv_ratio_max = 2.5  # Maximum IV/HV ratio
    data_fetcher.base_vix_level = 25.0  # Higher base VIX level
    
    # Run backtest for each risk level
    results = {}
    for risk_level in range(1, 11):
        print(f"Running backtest for risk level {risk_level}...")
        
        # Update risk level in config
        config.risk.risk_level = risk_level
        config.risk.adjust_for_risk_level(risk_level)
        
        # Initialize backtest engine with our custom data fetcher
        engine = BacktestEngine(config)
        engine.data_fetcher = data_fetcher
        
        # Run backtest
        result = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_balance=100000.0
        )
        
        # Store results
        results[risk_level] = result
        
        # Save individual risk level results
        result_file = os.path.join(output_dir, f"risk_level_{risk_level}_results.csv")
        if hasattr(result, 'daily_values') and result.daily_values is not None:
            result.daily_values.to_csv(result_file)
        
        # Print summary
        print(f"Risk Level {risk_level} Results:")
        print(f"  Final Balance: ${result.final_balance:.2f}")
        print(f"  Total Return: {result.total_return_pct:.2f}%")
        print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Win Rate: {result.win_rate_pct:.2f}%")
        print(f"  Trades: {result.total_trades}")
        print("")
    
    # Find best performing risk level
    best_risk_level = max(results.keys(), key=lambda x: results[x].sharpe_ratio)
    best_result = results[best_risk_level]
    
    # Print overall summary
    print("\n===== BACKTEST SUMMARY =====")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Best Risk Level: {best_risk_level}")
    print(f"Best Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
    print(f"Best Return: {best_result.total_return_pct:.2f}%")
    print(f"Initial Investment: $100,000.00")
    print(f"Final Balance: ${best_result.final_balance:.2f}")
    print("============================")
    
    # Create summary file
    summary_file = os.path.join(output_dir, "risk_level_summary.csv")
    with open(summary_file, 'w') as f:
        f.write("risk_level,total_return_pct,annualized_return_pct,volatility_pct,sharpe_ratio,max_drawdown_pct,win_rate_pct,trades\n")
        for level, result in results.items():
            f.write(f"{level},{result.total_return_pct:.2f},{result.annualized_return_pct:.2f},")
            f.write(f"{result.annualized_volatility_pct:.2f},{result.sharpe_ratio:.2f},")
            f.write(f"{result.max_drawdown_pct:.2f},{result.win_rate_pct:.2f},{result.total_trades}\n")
    
    print(f"Results saved to {output_dir}")
    return results

if __name__ == "__main__":
    run_enhanced_backtest()