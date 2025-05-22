#!/usr/bin/env python3
"""
Simplified backtesting script for the Enhanced Adaptive Volatility-Harvesting System.
Tests the strategy with different risk settings and generates comparison reports.
Uses mock data to avoid database dependencies.
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
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Initial price
        if symbol == 'VIX':
            initial_price = 20.0
            volatility = 0.15  # Higher volatility for VIX
        else:  # Assume SPY or other index
            initial_price = 400.0
            volatility = 0.01
        
        # Generate price series with random walk
        np.random.seed(42)  # For reproducibility
        
        # Add some realistic market behavior
        # For SPY - generally upward trend with periodic drawdowns
        # For VIX - mean reverting with spikes
        
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


class BacktestRunner:
    """Runner for volatility harvesting strategy backtests."""
    
    def __init__(self, output_dir: str = './reports/volatility_harvesting'):
        """
        Initialize the backtest runner.
        
        Args:
            output_dir: Directory for storing output
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Default risk parameter values for testing
        self.default_params = {
            'iv_hv_ratio_threshold': 1.2,
            'min_iv_threshold': 20.0,
            'strike_width_pct': 5.0,
            'target_short_delta': 0.30
        }
    
    def run_with_parameters(self, params: Dict, start_date: str, end_date: str) -> Dict:
        """
        Run a backtest with specific parameters.
        
        Args:
            params: Parameter dictionary
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with backtest results
        """
        # Create a fresh config
        config = Config()
        
        # Apply parameters
        for param, value in params.items():
            if hasattr(config.vol_harvesting, param):
                setattr(config.vol_harvesting, param, value)
        
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
            risk_level=config.risk.risk_level
        )
        
        # Add parameters to results
        for param, value in params.items():
            results[f"param_{param}"] = value
        
        return results
        
    def run_parameter_tests(self, start_date: str, end_date: str) -> Dict[str, Dict]:
        """
        Run tests with different parameter values.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        # Test different IV/HV ratio thresholds
        for ratio in [1.1, 1.2, 1.3, 1.4, 1.5]:
            params = self.default_params.copy()
            params['iv_hv_ratio_threshold'] = ratio
            test_id = f"iv_hv_ratio_{ratio}"
            logging.info(f"Running test {test_id}")
            results[test_id] = self.run_with_parameters(params, start_date, end_date)
        
        # Test different min IV thresholds
        for threshold in [15.0, 20.0, 25.0, 30.0]:
            params = self.default_params.copy()
            params['min_iv_threshold'] = threshold
            test_id = f"min_iv_{threshold}"
            logging.info(f"Running test {test_id}")
            results[test_id] = self.run_with_parameters(params, start_date, end_date)
        
        # Test different strike widths
        for width in [2.0, 3.0, 5.0, 7.0, 10.0]:
            params = self.default_params.copy()
            params['strike_width_pct'] = width
            test_id = f"strike_width_{width}"
            logging.info(f"Running test {test_id}")
            results[test_id] = self.run_with_parameters(params, start_date, end_date)
        
        # Test different target deltas
        for delta in [0.20, 0.25, 0.30, 0.35, 0.40]:
            params = self.default_params.copy()
            params['target_short_delta'] = delta
            test_id = f"delta_{delta}"
            logging.info(f"Running test {test_id}")
            results[test_id] = self.run_with_parameters(params, start_date, end_date)
        
        return results
    
    def run_risk_level_tests(self, start_date: str, end_date: str) -> Dict[int, Dict]:
        """
        Run tests with different risk levels.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of risk level test results
        """
        results_by_risk = {}
        
        for risk_level in range(1, 11):
            logging.info(f"Running test for risk level {risk_level}")
            
            # Create config with this risk level
            config = Config()
            config.risk.adjust_for_risk_level(risk_level)
            
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
            
            # Add risk level to results
            results['risk_level'] = risk_level
            results_by_risk[risk_level] = results
        
        return results_by_risk
    
    def generate_parameter_report(self, results: Dict[str, Dict]) -> None:
        """
        Generate a report for parameter test results.
        
        Args:
            results: Dictionary of test results
        """
        # Extract metrics by parameter type
        params_data = {
            'iv_hv_ratio': [],
            'min_iv': [],
            'strike_width': [],
            'delta': []
        }
        
        for test_id, result in results.items():
            if test_id.startswith('iv_hv_ratio_'):
                value = float(test_id.split('_')[-1])
                params_data['iv_hv_ratio'].append({
                    'value': value,
                    'return': result.get('total_return_pct', 0),
                    'sharpe': result.get('sharpe_ratio', 0),
                    'max_dd': result.get('max_drawdown_pct', 0),
                    'win_rate': result.get('win_rate', 0)
                })
            elif test_id.startswith('min_iv_'):
                value = float(test_id.split('_')[-1])
                params_data['min_iv'].append({
                    'value': value,
                    'return': result.get('total_return_pct', 0),
                    'sharpe': result.get('sharpe_ratio', 0),
                    'max_dd': result.get('max_drawdown_pct', 0),
                    'win_rate': result.get('win_rate', 0)
                })
            elif test_id.startswith('strike_width_'):
                value = float(test_id.split('_')[-1])
                params_data['strike_width'].append({
                    'value': value,
                    'return': result.get('total_return_pct', 0),
                    'sharpe': result.get('sharpe_ratio', 0),
                    'max_dd': result.get('max_drawdown_pct', 0),
                    'win_rate': result.get('win_rate', 0)
                })
            elif test_id.startswith('delta_'):
                value = float(test_id.split('_')[-1])
                params_data['delta'].append({
                    'value': value,
                    'return': result.get('total_return_pct', 0),
                    'sharpe': result.get('sharpe_ratio', 0),
                    'max_dd': result.get('max_drawdown_pct', 0),
                    'win_rate': result.get('win_rate', 0)
                })
        
        # Convert to DataFrames
        dfs = {}
        for param, data in params_data.items():
            if data:
                dfs[param] = pd.DataFrame(data).sort_values('value')
        
        # Create plot
        fig, axs = plt.subplots(4, 2, figsize=(16, 20))
        
        # Parameter names for display
        param_names = {
            'iv_hv_ratio': 'IV/HV Ratio Threshold',
            'min_iv': 'Minimum IV Threshold',
            'strike_width': 'Strike Width Percentage',
            'delta': 'Target Short Delta'
        }
        
        # Plot returns and sharpe ratios
        for i, (param, df) in enumerate(dfs.items()):
            if df.empty:
                continue
            
            # Returns
            axs[i, 0].plot(df['value'], df['return'], marker='o', linewidth=2)
            axs[i, 0].set_title(f"Total Return vs {param_names[param]}")
            axs[i, 0].set_xlabel(param_names[param])
            axs[i, 0].set_ylabel('Total Return (%)')
            axs[i, 0].grid(True)
            
            # Sharpe ratio
            axs[i, 1].plot(df['value'], df['sharpe'], marker='o', linewidth=2, color='green')
            axs[i, 1].set_title(f"Sharpe Ratio vs {param_names[param]}")
            axs[i, 1].set_xlabel(param_names[param])
            axs[i, 1].set_ylabel('Sharpe Ratio')
            axs[i, 1].grid(True)
        
        plt.tight_layout()
        plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, y=1.02)
        
        # Save plot
        report_path = os.path.join(self.output_dir, 'parameter_sensitivity.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        logging.info(f"Parameter report saved to {report_path}")
        
        # Create summary table
        summary_rows = []
        
        for param, df in dfs.items():
            if df.empty:
                continue
                
            # Find best return and sharpe
            best_return_idx = df['return'].idxmax()
            best_sharpe_idx = df['sharpe'].idxmax()
            
            summary_rows.append({
                'Parameter': param_names[param],
                'Best Return Value': df.loc[best_return_idx, 'value'],
                'Best Return (%)': df.loc[best_return_idx, 'return'],
                'Best Sharpe Value': df.loc[best_sharpe_idx, 'value'],
                'Best Sharpe Ratio': df.loc[best_sharpe_idx, 'sharpe']
            })
        
        # Save summary table
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(self.output_dir, 'parameter_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Parameter summary saved to {summary_path}")
    
    def generate_risk_level_report(self, results: Dict[int, Dict]) -> None:
        """
        Generate a report for risk level test results.
        
        Args:
            results: Dictionary of risk level test results
        """
        # Extract metrics by risk level
        risk_levels = []
        returns = []
        sharpe_ratios = []
        max_drawdowns = []
        win_rates = []
        volatilities = []
        
        for risk_level in sorted(results.keys()):
            result = results[risk_level]
            risk_levels.append(risk_level)
            returns.append(result.get('total_return_pct', 0))
            sharpe_ratios.append(result.get('sharpe_ratio', 0))
            max_drawdowns.append(result.get('max_drawdown_pct', 0))
            win_rates.append(result.get('win_rate', 0))
            volatilities.append(result.get('annualized_volatility_pct', 0))
        
        # Create plot
        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot returns by risk level
        axs[0, 0].bar(risk_levels, returns)
        axs[0, 0].set_title('Total Return by Risk Level')
        axs[0, 0].set_xlabel('Risk Level')
        axs[0, 0].set_ylabel('Total Return (%)')
        axs[0, 0].set_xticks(risk_levels)
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot Sharpe ratio by risk level
        axs[0, 1].bar(risk_levels, sharpe_ratios, color='green')
        axs[0, 1].set_title('Sharpe Ratio by Risk Level')
        axs[0, 1].set_xlabel('Risk Level')
        axs[0, 1].set_ylabel('Sharpe Ratio')
        axs[0, 1].set_xticks(risk_levels)
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot max drawdown by risk level
        axs[1, 0].bar(risk_levels, max_drawdowns, color='red')
        axs[1, 0].set_title('Maximum Drawdown by Risk Level')
        axs[1, 0].set_xlabel('Risk Level')
        axs[1, 0].set_ylabel('Maximum Drawdown (%)')
        axs[1, 0].set_xticks(risk_levels)
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot win rate by risk level
        axs[1, 1].bar(risk_levels, win_rates, color='purple')
        axs[1, 1].set_title('Win Rate by Risk Level')
        axs[1, 1].set_xlabel('Risk Level')
        axs[1, 1].set_ylabel('Win Rate (%)')
        axs[1, 1].set_xticks(risk_levels)
        axs[1, 1].grid(True, alpha=0.3)
        
        # Plot risk-return scatter
        axs[2, 0].scatter(volatilities, returns, s=100)
        axs[2, 0].set_title('Risk-Return Profile')
        axs[2, 0].set_xlabel('Volatility (%)')
        axs[2, 0].set_ylabel('Return (%)')
        axs[2, 0].grid(True)
        
        # Add labels for each point
        for i, risk in enumerate(risk_levels):
            axs[2, 0].annotate(
                f"Risk {risk}", 
                (volatilities[i], returns[i]),
                xytext=(5, 5), 
                textcoords='offset points'
            )
        
        # Find best risk levels
        best_return_idx = returns.index(max(returns))
        best_return_risk = risk_levels[best_return_idx]
        
        best_sharpe_idx = sharpe_ratios.index(max(sharpe_ratios))
        best_sharpe_risk = risk_levels[best_sharpe_idx]
        
        # Plot summary text
        summary_text = (
            f"Best Return: Risk Level {best_return_risk} ({returns[best_return_idx]:.2f}%)\n"
            f"Best Sharpe Ratio: Risk Level {best_sharpe_risk} ({sharpe_ratios[best_sharpe_idx]:.2f})\n\n"
            f"Risk Level Comparison:\n"
            f"- Lower Risk (1-3): More conservative, fewer trades\n"
            f"- Medium Risk (4-7): Balanced approach, moderate position sizing\n"
            f"- Higher Risk (8-10): More aggressive, larger positions\n\n"
            f"Recommendation: Consider Risk Level {best_sharpe_risk} for optimal risk-adjusted returns."
        )
        
        axs[2, 1].text(0.5, 0.5, summary_text, 
                      ha='center', va='center', 
                      bbox=dict(facecolor='white', alpha=0.8),
                      fontsize=12)
        axs[2, 1].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Risk Level Analysis for Volatility Harvesting Strategy', fontsize=16, y=0.98)
        
        # Save plot
        report_path = os.path.join(self.output_dir, 'risk_level_analysis.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        logging.info(f"Risk level report saved to {report_path}")
        
        # Save summary data
        summary_data = []
        for i, risk_level in enumerate(risk_levels):
            summary_data.append({
                'Risk Level': risk_level,
                'Total Return (%)': returns[i],
                'Annualized Return (%)': results[risk_level].get('annualized_return_pct', 0),
                'Sharpe Ratio': sharpe_ratios[i],
                'Max Drawdown (%)': max_drawdowns[i],
                'Volatility (%)': volatilities[i],
                'Win Rate (%)': win_rates[i],
                'Profit Factor': results[risk_level].get('profit_factor', 0),
                'Trades': results[risk_level].get('trades', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, 'risk_level_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Risk level summary saved to {summary_path}")


def main():
    """Main function."""
    try:
        # Set up output directory
        output_dir = './reports/volatility_harvesting'
        os.makedirs(output_dir, exist_ok=True)
        
        # Set test period (1 year)
        start_date = '2022-01-01'
        end_date = '2023-01-01'
        
        # Create runner
        runner = BacktestRunner(output_dir)
        
        # Run parameter tests
        logging.info("Running parameter sensitivity tests...")
        param_results = runner.run_parameter_tests(start_date, end_date)
        runner.generate_parameter_report(param_results)
        
        # Run risk level tests
        logging.info("Running risk level tests...")
        risk_results = runner.run_risk_level_tests(start_date, end_date)
        runner.generate_risk_level_report(risk_results)
        
        logging.info("Backtesting completed successfully")
        return 0
    
    except Exception as e:
        logging.error(f"Error in backtesting: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())