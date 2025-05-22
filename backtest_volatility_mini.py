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


# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')


class MockDataFetcher:
    """Mock data fetcher to avoid database dependencies."""
    
    def __init__(self):
        """Initialize mock data fetcher."""
        self.logger = logging.getLogger(__name__)
    
    def fetch_data(self, symbol, start_date, end_date):
        """Generate mock historical price data for backtesting."""
        # Convert dates to datetime objects
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate shorter date range - just 30 days for quick testing
        dates = pd.date_range(start=start_date, end=start_date + timedelta(days=30), freq='B')
        
        # Initial price
        if symbol == 'VIX':
            initial_price = 20.0
            volatility = 0.15  # Higher volatility for VIX
        else:  # Assume SPY or other index
            initial_price = 400.0
            volatility = 0.01
        
        # Generate price series with random walk
        np.random.seed(42)  # For reproducibility
        
        prices = [initial_price]
        for i in range(1, len(dates)):
            if symbol == 'VIX':
                # Mean-reverting with occasional spikes
                mean_reversion = 0.05 * (20.0 - prices[-1])
                spike = 0.0
                if random.random() < 0.05:  # 5% chance of spike
                    spike = random.uniform(3.0, 10.0)
                change = mean_reversion + spike + np.random.normal(0, volatility * prices[-1])
            else:
                # Trend with random noise
                trend = 0.0002  # Slight upward bias
                change = trend * prices[-1] + np.random.normal(0, volatility * prices[-1])
            
            # Ensure price doesn't go negative
            new_price = max(0.1, prices[-1] + change)
            prices.append(new_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + volatility * random.uniform(0, 1)) for p in prices],
            'low': [p * (1 - volatility * random.uniform(0, 1)) for p in prices],
            'close': prices,
            'volume': [int(1e6 * random.uniform(0.5, 1.5)) for _ in prices]
        }, index=dates)
        
        return df
        
    def fetch_vix_data(self, start_date, end_date):
        """Fetch mock VIX data."""
        return self.fetch_data('VIX', start_date, end_date)
    
    def fetch_sp500_data(self, start_date, end_date):
        """Fetch mock S&P500 data."""
        return self.fetch_data('SPY', start_date, end_date)


# Patch the BacktestEngine to use our mock data fetcher
original_init = BacktestEngine.__init__

def patched_init(self, config=None, volatility_analyzer=None, risk_manager=None, 
               options_strategy=None, initial_balance=100000.0):
    original_init(self, config, volatility_analyzer, risk_manager, options_strategy, initial_balance)
    self.data_fetcher = MockDataFetcher()

BacktestEngine.__init__ = patched_init


def run_backtest(config, start_date, end_date, risk_level=5):
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
        initial_balance=100000.0
    )
    
    # Run backtest
    results = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        risk_level=risk_level
    )
    
    return results


def run_risk_level_tests(output_dir):
    """Run tests for different risk levels and generate a report."""
    start_date = '2022-01-01'
    end_date = '2022-02-01'  # Short timeframe for quick testing
    
    results = {}
    risk_levels = [1, 3, 5, 7, 10]  # Reduced test set
    
    for risk_level in risk_levels:
        logging.info(f"Testing risk level {risk_level}")
        
        # Create config
        config = Config()
        config.risk.adjust_for_risk_level(risk_level)
        
        # Run backtest
        result = run_backtest(config, start_date, end_date, risk_level)
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
    plt.suptitle('Risk Level Comparison for Volatility Harvesting Strategy', fontsize=16, y=1.02)
    
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


def run_parameter_tests(output_dir):
    """Run tests for different strategy parameters and generate a report."""
    start_date = '2022-01-01'
    end_date = '2022-02-01'  # Short timeframe for quick testing
    
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
        result = run_backtest(test_config, start_date, end_date)
        
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
        result = run_backtest(test_config, start_date, end_date)
        
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
        result = run_backtest(test_config, start_date, end_date)
        
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
    plt.suptitle('Parameter Sensitivity Analysis for Volatility Harvesting Strategy', fontsize=16, y=0.98)
    
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


def main():
    """Main function."""
    output_dir = './reports/volatility_harvesting'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        logging.info("Running risk level tests...")
        run_risk_level_tests(output_dir)
        
        logging.info("Running parameter sensitivity tests...")
        run_parameter_tests(output_dir)
        
        logging.info("Testing completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Error during testing: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())