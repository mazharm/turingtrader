#!/usr/bin/env python3
"""
Minimal backtesting script for the Enhanced Adaptive Volatility-Harvesting System.
Tests the strategy with different risk settings and generates comparison reports.
Uses mock data with a reduced test set for quick execution.
"""

import os
import sys
import logging
import json
import random
from datetime import datetime, timedelta
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ibkr_trader.config import Config
from backtesting.backtest_engine import BacktestEngine
from ibkr_trader.volatility_analyzer import VolatilityAnalyzer
from ibkr_trader.options_strategy import OptionsStrategy
from ibkr_trader.risk_manager import RiskManager
from backtesting.mock_utils import MockDataFetcher # Added import


# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')


def run_backtest(config, start_date, end_date, risk_level=5, days_limit=None):
    """Run a backtest with the given configuration."""
    # Create components
    volatility_analyzer = VolatilityAnalyzer(config)
    risk_manager = RiskManager(config)
    options_strategy = OptionsStrategy(config, volatility_analyzer, risk_manager)
    
    # Create backtest engine
    engine = BacktestEngine(
        config=config,
        volatility_analyzer=volatility_analyzer,
        risk_manager=risk_manager,
        options_strategy=options_strategy,
        initial_balance=100000.0,
        data_fetcher=MockDataFetcher() # Inject MockDataFetcher
    )
    
    # Modify data fetcher calls if days_limit is used by MockDataFetcher
    # This part depends on how MockDataFetcher is modified to accept days_limit
    # For now, assuming MockDataFetcher handles it internally or via its methods
    
    # Run backtest
    results = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        risk_level=risk_level
    )
    
    return results


def run_risk_level_tests(output_dir, days_limit_mock_data=30):
    """Run tests for different risk levels and generate a report."""
    start_date = '2022-01-01'
    # End date is effectively determined by start_date + days_limit_mock_data in MockDataFetcher
    end_date = (pd.to_datetime(start_date) + timedelta(days=days_limit_mock_data)).strftime('%Y-%m-%d') 

    results = {}
    risk_levels = [1, 3, 5, 7, 10]  # Reduced test set
    
    for risk_level in risk_levels:
        logging.info(f"Testing risk level {risk_level}")
        
        # Create config
        config = Config()
        config.risk.adjust_for_risk_level(risk_level)
        
        # Run backtest - MockDataFetcher will use its internal days_limit logic if set
        # Or, pass days_limit to run_backtest if that function is adapted
        result = run_backtest(config, start_date, end_date, risk_level, days_limit=days_limit_mock_data)
        result['risk_level'] = risk_level
        results[risk_level] = result
    
    # Extract metrics
    data = []
    for risk_level, result in results.items():
        data.append({
            'Risk Level': risk_level,
            'Total Return (%)': result.get('total_return_pct', 0),
            'Sharpe Ratio': result.get('sharpe_ratio', 0),
            'Max Drawdown (%)': result.get('max_drawdown_pct', 0),
            'Win Rate (%)': result.get('win_rate', 0)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate visualization
    plt.figure(figsize=(12, 10))
    
    # Plot returns
    plt.subplot(2, 2, 1)
    plt.bar(df['Risk Level'], df['Total Return (%)'])
    plt.title('Total Return by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Total Return (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot Sharpe ratios
    plt.subplot(2, 2, 2)
    plt.bar(df['Risk Level'], df['Sharpe Ratio'], color='green')
    plt.title('Sharpe Ratio by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Plot max drawdowns
    plt.subplot(2, 2, 3)
    plt.bar(df['Risk Level'], df['Max Drawdown (%)'], color='red')
    plt.title('Maximum Drawdown by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Max Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot win rates
    plt.subplot(2, 2, 4)
    plt.bar(df['Risk Level'], df['Win Rate (%)'], color='purple')
    plt.title('Win Rate by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Win Rate (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Risk Level Comparison for Volatility Harvesting Strategy (Mini)', fontsize=16, y=1.02)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'risk_level_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Risk level comparison saved to {plot_path}")
    
    # Save data
    csv_path = os.path.join(output_dir, 'risk_level_results.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Risk level data saved to {csv_path}")
    
    return results


def run_parameter_tests(output_dir, days_limit_mock_data=30):
    """Run tests for different strategy parameters and generate a report."""
    start_date = '2022-01-01'
    # End date is effectively determined by start_date + days_limit_mock_data
    end_date = (pd.to_datetime(start_date) + timedelta(days=days_limit_mock_data)).strftime('%Y-%m-%d') 

    # Base configuration
    config = Config()
    
    # Test parameters
    iv_hv_ratios = [1.1, 1.3, 1.5]  # Reduced test set
    min_iv_thresholds = [15.0, 25.0, 35.0]  # Reduced test set
    strike_widths = [3.0, 5.0, 7.0]  # Reduced test set
    
    results = {}
    all_data = []
    
    # Test IV/HV ratios
    for ratio in iv_hv_ratios:
        logging.info(f"Testing IV/HV ratio {ratio}")
        test_config = Config()  # Fresh config
        test_config.vol_harvesting.iv_hv_ratio_threshold = ratio
        result = run_backtest(test_config, start_date, end_date, days_limit=days_limit_mock_data)
        
        results[f'iv_hv_{ratio}'] = result
        all_data.append({
            'Parameter': 'IV/HV Ratio',
            'Value': ratio,
            'Total Return (%)': result.get('total_return_pct', 0),
            'Sharpe Ratio': result.get('sharpe_ratio', 0),
            'Max Drawdown (%)': result.get('max_drawdown_pct', 0),
            'Win Rate (%)': result.get('win_rate', 0)
        })
    
    # Test min IV thresholds
    for threshold in min_iv_thresholds:
        logging.info(f"Testing min IV threshold {threshold}")
        test_config = Config()  # Fresh config
        test_config.vol_harvesting.min_iv_threshold = threshold
        result = run_backtest(test_config, start_date, end_date, days_limit=days_limit_mock_data)
        
        results[f'min_iv_{threshold}'] = result
        all_data.append({
            'Parameter': 'Min IV Threshold',
            'Value': threshold,
            'Total Return (%)': result.get('total_return_pct', 0),
            'Sharpe Ratio': result.get('sharpe_ratio', 0),
            'Max Drawdown (%)': result.get('max_drawdown_pct', 0),
            'Win Rate (%)': result.get('win_rate', 0)
        })
    
    # Test strike widths
    for width in strike_widths:
        logging.info(f"Testing strike width {width}")
        test_config = Config()  # Fresh config
        test_config.vol_harvesting.strike_width_pct = width
        result = run_backtest(test_config, start_date, end_date, days_limit=days_limit_mock_data)
        
        results[f'width_{width}'] = result
        all_data.append({
            'Parameter': 'Strike Width',
            'Value': width,
            'Total Return (%)': result.get('total_return_pct', 0),
            'Sharpe Ratio': result.get('sharpe_ratio', 0),
            'Max Drawdown (%)': result.get('max_drawdown_pct', 0),
            'Win Rate (%)': result.get('win_rate', 0)
        })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Generate visualization
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    
    # Plot each parameter group
    for i, param in enumerate(['IV/HV Ratio', 'Min IV Threshold', 'Strike Width']):
        param_data = df[df['Parameter'] == param]
        
        # Returns
        axs[i, 0].plot(param_data['Value'], param_data['Total Return (%)'], marker='o', linewidth=2)
        axs[i, 0].set_title(f'Total Return vs {param}')
        axs[i, 0].set_xlabel(param)
        axs[i, 0].set_ylabel('Total Return (%)')
        axs[i, 0].grid(True)
        
        # Sharpe ratio
        axs[i, 1].plot(param_data['Value'], param_data['Sharpe Ratio'], marker='o', linewidth=2, color='green')
        axs[i, 1].set_title(f'Sharpe Ratio vs {param}')
        axs[i, 1].set_xlabel(param)
        axs[i, 1].set_ylabel('Sharpe Ratio')
        axs[i, 1].grid(True)
    
    plt.tight_layout()
    plt.suptitle('Parameter Sensitivity Analysis for Volatility Harvesting Strategy (Mini)', fontsize=16, y=0.98)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'parameter_sensitivity.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Parameter sensitivity analysis saved to {plot_path}")
    
    # Save data
    csv_path = os.path.join(output_dir, 'parameter_results.csv')
    df.to_csv(csv_path, index=False)
    logging.info(f"Parameter test data saved to {csv_path}")
    
    return results


def main(quick_run: bool = True):
    """Main function."""
    output_dir_base = './reports/volatility_harvesting_mock'
    # Use a timestamped subdirectory for each run to avoid overwriting results
    output_dir = os.path.join(output_dir_base, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)

    days_for_mock = 30 if quick_run else 90 # 30 days for mini/quick, 90 for a bit longer
    title_suffix = "(Quick)" if quick_run else "(Standard Mock)"

    try:
        logging.info(f"Running risk level tests {title_suffix}...")
        run_risk_level_tests(output_dir, days_limit_mock_data=days_for_mock)
        
        logging.info(f"Running parameter sensitivity tests {title_suffix}...")
        run_parameter_tests(output_dir, days_limit_mock_data=days_for_mock)
        
        logging.info(f"Mock testing completed successfully! Reports in {output_dir}")
        return 0
        
    except Exception as e:
        logging.error(f"Error during mock testing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # sys.exit(main(quick_run=True)) # For a quick run (like original mini)
    sys.exit(main(quick_run=False)) # For a slightly longer mock run