#!/usr/bin/env python3
"""
Backtesting script specifically for the Enhanced Adaptive Volatility-Harvesting System.
Tests the strategy with different risk settings and generates comparison reports.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logging_util import setup_logging
from ibkr_trader.config import Config
from backtesting.backtest_engine import BacktestEngine
from ibkr_trader.volatility_analyzer import VolatilityAnalyzer
from ibkr_trader.options_strategy import OptionsStrategy
from ibkr_trader.risk_manager import RiskManager
from utils.reporting import generate_report, generate_multi_risk_report, export_multi_risk_summary


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Volatility Harvesting Strategy Backtester'
    )
    
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
    
    # Risk settings
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
    
    parser.add_argument(
        '--parameter-sweep',
        action='store_true',
        help='Run parameter sweep across key volatility harvesting parameters'
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
        default='./reports/volatility_harvesting',
        help='Output directory for reports and data'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results to JSON/CSV files'
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


def define_parameter_sets() -> Dict[str, List[Any]]:
    """
    Define parameter sets for testing the volatility harvesting strategy.
    
    Returns:
        Dictionary of parameter names to list of values to test
    """
    return {
        # IV/HV ratio threshold (values above this trigger trades)
        'iv_hv_ratio_threshold': [1.1, 1.2, 1.3, 1.4, 1.5],
        
        # Minimum implied volatility required (percentage)
        'min_iv_threshold': [15.0, 20.0, 25.0, 30.0, 35.0],
        
        # Strike width percentage for iron condors
        'strike_width_pct': [2.0, 3.0, 5.0, 7.0, 10.0],
        
        # Target delta for short legs
        'target_short_delta': [0.20, 0.25, 0.30, 0.35, 0.40]
    }


def run_volatility_harvesting_backtest(
    config: Config,
    start_date: str,
    end_date: str,
    initial_balance: float = 100000.0,
    parameter_set: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Run a backtest for the volatility harvesting strategy with specific parameters.
    
    Args:
        config: Configuration object
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_balance: Initial account balance
        parameter_set: Dictionary of parameters to override
        
    Returns:
        Dictionary with backtest results
    """
    # Apply parameter overrides if provided
    if parameter_set:
        # Update volatility harvesting parameters
        for param, value in parameter_set.items():
            if hasattr(config.vol_harvesting, param):
                setattr(config.vol_harvesting, param, value)
                logging.info(f"Overriding {param} = {value}")
    
    # Create component instances with updated config
    volatility_analyzer = VolatilityAnalyzer(config)
    risk_manager = RiskManager(config)
    options_strategy = OptionsStrategy(config, volatility_analyzer, risk_manager)
    
    # Create backtest engine
    engine = BacktestEngine(
        config=config,
        volatility_analyzer=volatility_analyzer,
        risk_manager=risk_manager,
        options_strategy=options_strategy,
        initial_balance=initial_balance
    )
    
    # Run the backtest
    results = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        risk_level=config.risk.risk_level
    )
    
    # Add parameter info to results
    if parameter_set:
        for param, value in parameter_set.items():
            results[f"param_{param}"] = value
    
    return results


def run_parameter_sweep(
    base_config: Config,
    start_date: str,
    end_date: str,
    initial_balance: float,
    output_dir: str,
    save_results: bool = False
) -> Dict[str, Dict]:
    """
    Run a parameter sweep across key volatility harvesting parameters.
    
    Args:
        base_config: Base configuration object
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_balance: Initial account balance
        output_dir: Output directory for reports
        save_results: Whether to save detailed results
        
    Returns:
        Dictionary mapping parameter set IDs to results
    """
    logging.info("Running parameter sweep for volatility harvesting strategy")
    
    # Define parameter sets
    param_sets = define_parameter_sets()
    
    # Create output directory for parameter sweep
    sweep_dir = os.path.join(output_dir, 'parameter_sweep')
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Tracking best parameters
    best_results = {
        'best_return': {'value': -float('inf'), 'params': None, 'results': None},
        'best_sharpe': {'value': -float('inf'), 'params': None, 'results': None},
        'best_win_rate': {'value': -float('inf'), 'params': None, 'results': None}
    }
    
    # Storage for all results
    all_results = {}
    summary_data = []
    
    # Test each parameter individually (one-factor-at-a-time)
    for param_name, param_values in param_sets.items():
        param_results = {}
        
        logging.info(f"Testing parameter: {param_name}")
        
        for value in param_values:
            # Create config copy with this parameter value
            config_copy = Config()  # Creates a fresh config
            if hasattr(config_copy.vol_harvesting, param_name):
                setattr(config_copy.vol_harvesting, param_name, value)
                
            # Parameter set ID
            param_set_id = f"{param_name}_{value}"
            
            # Run backtest
            logging.info(f"  Running backtest with {param_name} = {value}")
            param_set = {param_name: value}
            results = run_volatility_harvesting_backtest(
                config=config_copy,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance,
                parameter_set=param_set
            )
            
            # Store results
            param_results[value] = results
            all_results[param_set_id] = results
            
            # Add to summary data
            summary_row = {
                'parameter': param_name,
                'value': value,
                'total_return_pct': results.get('total_return_pct', 0),
                'annualized_return_pct': results.get('annualized_return_pct', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown_pct': results.get('max_drawdown_pct', 0),
                'win_rate': results.get('win_rate', 0),
                'profit_factor': results.get('profit_factor', 0),
                'trades': results.get('trades', 0)
            }
            summary_data.append(summary_row)
            
            # Check if this is the best so far
            if results.get('total_return_pct', 0) > best_results['best_return']['value']:
                best_results['best_return'] = {
                    'value': results.get('total_return_pct', 0),
                    'params': param_set,
                    'results': results
                }
                
            if results.get('sharpe_ratio', 0) > best_results['best_sharpe']['value']:
                best_results['best_sharpe'] = {
                    'value': results.get('sharpe_ratio', 0),
                    'params': param_set,
                    'results': results
                }
                
            if results.get('win_rate', 0) > best_results['best_win_rate']['value']:
                best_results['best_win_rate'] = {
                    'value': results.get('win_rate', 0),
                    'params': param_set,
                    'results': results
                }
        
        # Generate comparison plot for this parameter
        plot_parameter_comparison(param_name, param_values, param_results, sweep_dir)
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(sweep_dir, 'parameter_sweep_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Parameter sweep summary saved to {summary_path}")
    
    # Save best parameters
    best_params_path = os.path.join(sweep_dir, 'best_parameters.json')
    with open(best_params_path, 'w') as f:
        # Convert to serializable format
        serializable_best = {
            k: {
                'value': v['value'],
                'params': v['params']
            } for k, v in best_results.items()
        }
        json.dump(serializable_best, f, indent=4)
    
    logging.info(f"Best parameters saved to {best_params_path}")
    
    # Generate comprehensive report
    generate_parameter_sweep_report(summary_df, best_results, sweep_dir)
    
    return all_results


def plot_parameter_comparison(
    param_name: str,
    param_values: List[Any],
    results: Dict[Any, Dict],
    output_dir: str
) -> str:
    """
    Generate a comparison plot for a parameter sweep.
    
    Args:
        param_name: Parameter name
        param_values: List of parameter values tested
        results: Dictionary mapping parameter values to backtest results
        output_dir: Output directory for plots
        
    Returns:
        Path to the generated plot file
    """
    # Extract metrics for each parameter value
    total_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    win_rates = []
    
    for value in param_values:
        result = results.get(value, {})
        total_returns.append(result.get('total_return_pct', 0))
        sharpe_ratios.append(result.get('sharpe_ratio', 0))
        max_drawdowns.append(result.get('max_drawdown_pct', 0))
        win_rates.append(result.get('win_rate', 0))
    
    # Create plot with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Format parameters for display
    param_display = param_name.replace('_', ' ').title()
    str_values = [str(val) for val in param_values]
    
    # Plot total return
    axs[0, 0].bar(str_values, total_returns)
    axs[0, 0].set_title(f'Total Return vs {param_display}')
    axs[0, 0].set_ylabel('Total Return (%)')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot Sharpe ratio
    axs[0, 1].bar(str_values, sharpe_ratios, color='green')
    axs[0, 1].set_title(f'Sharpe Ratio vs {param_display}')
    axs[0, 1].set_ylabel('Sharpe Ratio')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot max drawdown
    axs[1, 0].bar(str_values, max_drawdowns, color='red')
    axs[1, 0].set_title(f'Max Drawdown vs {param_display}')
    axs[1, 0].set_ylabel('Max Drawdown (%)')
    axs[1, 0].invert_yaxis()  # Invert so smaller drawdowns are at top
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot win rate
    axs[1, 1].bar(str_values, win_rates, color='purple')
    axs[1, 1].set_title(f'Win Rate vs {param_display}')
    axs[1, 1].set_ylabel('Win Rate (%)')
    axs[1, 1].grid(True, alpha=0.3)
    
    # Add main title
    plt.suptitle(f'Parameter Sweep for {param_display}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'param_sweep_{param_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path


def generate_parameter_sweep_report(
    summary_df: pd.DataFrame,
    best_results: Dict,
    output_dir: str
) -> str:
    """
    Generate a comprehensive report for parameter sweep results.
    
    Args:
        summary_df: DataFrame with summary data for all parameter tests
        best_results: Dictionary with best parameter sets for different metrics
        output_dir: Output directory
        
    Returns:
        Path to the generated report file
    """
    # Create figure
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 2)
    
    # 1. Top parameters for total return
    ax1 = fig.add_subplot(gs[0, :])
    top_return = summary_df.sort_values('total_return_pct', ascending=False).head(10)
    sns.barplot(data=top_return, x='value', y='total_return_pct', hue='parameter', ax=ax1)
    ax1.set_title('Top 10 Parameters by Total Return')
    ax1.set_ylabel('Total Return (%)')
    ax1.set_xlabel('Parameter Value')
    
    # 2. Top parameters for Sharpe ratio
    ax2 = fig.add_subplot(gs[1, :])
    top_sharpe = summary_df.sort_values('sharpe_ratio', ascending=False).head(10)
    sns.barplot(data=top_sharpe, x='value', y='sharpe_ratio', hue='parameter', ax=ax2)
    ax2.set_title('Top 10 Parameters by Sharpe Ratio')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_xlabel('Parameter Value')
    
    # 3. Effects of IV/HV ratio threshold
    ax3 = fig.add_subplot(gs[2, 0])
    iv_hv_data = summary_df[summary_df['parameter'] == 'iv_hv_ratio_threshold']
    if not iv_hv_data.empty:
        sns.lineplot(data=iv_hv_data, x='value', y='total_return_pct', marker='o', ax=ax3)
        ax3.set_title('Effect of IV/HV Ratio Threshold')
        ax3.set_xlabel('IV/HV Ratio Threshold')
        ax3.set_ylabel('Total Return (%)')
        ax3.grid(True)
    
    # 4. Effects of min IV threshold
    ax4 = fig.add_subplot(gs[2, 1])
    min_iv_data = summary_df[summary_df['parameter'] == 'min_iv_threshold']
    if not min_iv_data.empty:
        sns.lineplot(data=min_iv_data, x='value', y='total_return_pct', marker='o', ax=ax4)
        ax4.set_title('Effect of Min IV Threshold')
        ax4.set_xlabel('Min IV Threshold (%)')
        ax4.set_ylabel('Total Return (%)')
        ax4.grid(True)
    
    # 5. Effects of strike width
    ax5 = fig.add_subplot(gs[3, 0])
    width_data = summary_df[summary_df['parameter'] == 'strike_width_pct']
    if not width_data.empty:
        sns.lineplot(data=width_data, x='value', y='total_return_pct', marker='o', ax=ax5)
        ax5.set_title('Effect of Strike Width')
        ax5.set_xlabel('Strike Width (%)')
        ax5.set_ylabel('Total Return (%)')
        ax5.grid(True)
    
    # 6. Effects of target delta
    ax6 = fig.add_subplot(gs[3, 1])
    delta_data = summary_df[summary_df['parameter'] == 'target_short_delta']
    if not delta_data.empty:
        sns.lineplot(data=delta_data, x='value', y='total_return_pct', marker='o', ax=ax6)
        ax6.set_title('Effect of Target Short Delta')
        ax6.set_xlabel('Target Short Delta')
        ax6.set_ylabel('Total Return (%)')
        ax6.grid(True)
    
    # 7. Parameter correlations with return
    ax7 = fig.add_subplot(gs[4, :])
    
    # Pivot the data to create a parameter x metric matrix
    pivot_df = summary_df.pivot_table(
        index='parameter',
        columns='value',
        values='total_return_pct'
    )
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0, ax=ax7)
    ax7.set_title('Parameter Value vs Total Return (%)')
    
    # 8. Best parameters summary table
    ax8 = fig.add_subplot(gs[5, :])
    ax8.axis('tight')
    ax8.axis('off')
    
    # Create table data
    table_data = [
        ["Metric", "Best Value", "Parameter", "Value"],
        ["Total Return", f"{best_results['best_return']['value']:.2f}%", 
         list(best_results['best_return']['params'].keys())[0], 
         list(best_results['best_return']['params'].values())[0]],
        ["Sharpe Ratio", f"{best_results['best_sharpe']['value']:.2f}", 
         list(best_results['best_sharpe']['params'].keys())[0], 
         list(best_results['best_sharpe']['params'].values())[0]],
        ["Win Rate", f"{best_results['best_win_rate']['value']:.2f}%", 
         list(best_results['best_win_rate']['params'].keys())[0], 
         list(best_results['best_win_rate']['params'].values())[0]]
    ]
    
    # Create the table
    table = ax8.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Improve formatting
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[i, j]
            if i == 0:  # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')
            elif j == 0:  # First column
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#D9E1F2')
            elif i % 2 == 1:  # Alternating rows
                cell.set_facecolor('#E9EDF4')
    
    ax8.set_title('Best Parameters Summary', pad=20)
    
    # Add overall title
    plt.suptitle('Volatility Harvesting Parameter Sweep Analysis', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    report_path = os.path.join(output_dir, 'parameter_sweep_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return report_path


def run_risk_level_comparison(
    config: Config,
    start_date: str,
    end_date: str,
    initial_balance: float,
    output_dir: str,
    save_results: bool = False
) -> Dict[int, Dict]:
    """
    Run backtest across all risk levels and generate comparison reports.
    
    Args:
        config: Configuration object
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_balance: Initial account balance
        output_dir: Output directory for reports
        save_results: Whether to save detailed results
        
    Returns:
        Dictionary mapping risk levels to backtest results
    """
    logging.info("Running backtests across all risk levels")
    
    results_by_risk = {}
    
    # Run backtest for each risk level (1-10)
    for risk_level in range(1, 11):
        logging.info(f"Running backtest for risk level {risk_level}...")
        
        # Create a copy of the config with this risk level
        config_copy = Config()  # Creates a fresh config
        config_copy.risk.adjust_for_risk_level(risk_level)
        
        # Run backtest
        results = run_volatility_harvesting_backtest(
            config=config_copy,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance
        )
        
        # Add risk level to results
        results['risk_level'] = risk_level
        results_by_risk[risk_level] = results
        
        # Generate individual report
        report_path = generate_report(
            results=results,
            output_dir=output_dir,
            risk_level=risk_level
        )
        logging.info(f"Report for risk level {risk_level} generated: {report_path}")
    
    # Generate comparison report
    comparison_path = generate_multi_risk_report(
        results_by_risk=results_by_risk,
        output_dir=output_dir
    )
    logging.info(f"Risk comparison report generated: {comparison_path}")
    
    # Export summary table
    summary_path = export_multi_risk_summary(
        results_by_risk=results_by_risk,
        output_dir=output_dir
    )
    logging.info(f"Risk level summary exported: {summary_path}")
    
    # Save results JSON if requested
    if save_results:
        # Convert results to a serializable format
        serializable_results = {}
        for risk_level, results in results_by_risk.items():
            # Remove non-serializable objects and large arrays
            serializable_results[risk_level] = {
                k: v for k, v in results.items()
                if k not in ['daily_values', 'monthly_returns']
            }
        
        # Save as JSON
        results_path = os.path.join(output_dir, 'risk_level_results.json')
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logging.info(f"Results saved to {results_path}")
    
    return results_by_risk


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load configuration
        config = Config(args.config)
        
        if args.parameter_sweep:
            # Run parameter sweep
            run_parameter_sweep(
                base_config=config,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_balance=args.initial_balance,
                output_dir=args.output_dir,
                save_results=args.save_results
            )
            
        elif args.all_risk_levels:
            # Run comparison across all risk levels
            run_risk_level_comparison(
                config=config,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_balance=args.initial_balance,
                output_dir=args.output_dir,
                save_results=args.save_results
            )
            
        else:
            # Set risk level if provided
            if args.risk_level:
                config.risk.adjust_for_risk_level(args.risk_level)
                risk_level = args.risk_level
            else:
                risk_level = config.risk.risk_level  # Use default risk level
                
            # Run single backtest
            logging.info(f"Running backtest with risk level {risk_level}...")
            
            results = run_volatility_harvesting_backtest(
                config=config,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_balance=args.initial_balance
            )
            
            # Generate report
            report_path = generate_report(
                results=results,
                output_dir=args.output_dir,
                risk_level=risk_level
            )
            logging.info(f"Report generated: {report_path}")
            
            # Save results if requested
            if args.save_results:
                results_path = os.path.join(args.output_dir, f'backtest_results_risk{risk_level}.json')
                
                # Remove non-serializable or large objects
                serializable_results = {
                    k: v for k, v in results.items()
                    if k not in ['daily_values', 'monthly_returns']
                }
                
                with open(results_path, 'w') as f:
                    json.dump(serializable_results, f, indent=4)
                    
                logging.info(f"Results saved to {results_path}")
        
        logging.info("Backtesting completed successfully")
        return 0
    
    except KeyboardInterrupt:
        logging.info("Backtesting interrupted by user")
        return 0
    
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())