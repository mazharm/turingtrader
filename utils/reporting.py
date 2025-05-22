"""
Reporting utilities for the TuringTrader algorithm.
"""

import os
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from backtesting.performance_analyzer import PerformanceAnalyzer


def generate_report(results: Dict, 
                   output_dir: str = './reports', 
                   risk_level: Optional[int] = None,
                   filename: Optional[str] = None) -> str:
    """
    Generate a performance report for a backtest.
    
    Args:
        results: Backtest results dictionary
        output_dir: Output directory for the report
        risk_level: Risk level of the backtest
        filename: Custom filename for the report
        
    Returns:
        Path to the generated report file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer(output_dir)
    
    # Generate report filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        risk_str = f"_risk{risk_level}" if risk_level is not None else ""
        filename = f"{output_dir}/performance_report{risk_str}_{timestamp}.png"
    else:
        filename = os.path.join(output_dir, filename)
    
    # Generate the report
    report_path = analyzer.generate_report(results, filename)
    
    return report_path


def generate_multi_risk_report(results_by_risk: Dict[int, Dict],
                              output_dir: str = './reports',
                              filename: Optional[str] = None) -> str:
    """
    Generate a report comparing multiple risk levels.
    
    Args:
        results_by_risk: Dictionary mapping risk levels to results
        output_dir: Output directory for the report
        filename: Custom filename for the report
        
    Returns:
        Path to the generated report file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer(output_dir)
    
    # Compare risk levels
    comparison = analyzer.compare_risk_levels(results_by_risk)
    
    # Generate report filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/risk_comparison_{timestamp}.png"
    else:
        filename = os.path.join(output_dir, filename)
    
    # Generate the comparison report
    report_path = analyzer.generate_comparison_report(comparison, results_by_risk, filename)
    
    return report_path


def export_results_to_csv(results: Dict, 
                         output_dir: str = './reports', 
                         risk_level: Optional[int] = None,
                         filename: Optional[str] = None) -> str:
    """
    Export backtest results to CSV file.
    
    Args:
        results: Backtest results dictionary
        output_dir: Output directory for the CSV file
        risk_level: Risk level of the backtest
        filename: Custom filename for the CSV file
        
    Returns:
        Path to the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract relevant data
    if 'daily_values' not in results:
        logging.error("No daily values data found in results")
        return ""
    
    # Create DataFrame from daily values
    df = pd.DataFrame(results['daily_values'])
    
    # Generate CSV filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        risk_str = f"_risk{risk_level}" if risk_level is not None else ""
        filename = f"{output_dir}/backtest_results{risk_str}_{timestamp}.csv"
    else:
        filename = os.path.join(output_dir, filename)
    
    # Export to CSV
    df.to_csv(filename, index=False)
    
    logging.info(f"Results exported to {filename}")
    return filename


def export_multi_risk_summary(results_by_risk: Dict[int, Dict],
                             output_dir: str = './reports',
                             filename: Optional[str] = None) -> str:
    """
    Export a summary table of results across multiple risk levels.
    
    Args:
        results_by_risk: Dictionary mapping risk levels to results
        output_dir: Output directory for the CSV file
        filename: Custom filename for the CSV file
        
    Returns:
        Path to the CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary DataFrame
    summary_data = []
    
    for risk_level, results in sorted(results_by_risk.items()):
        row = {
            'risk_level': risk_level,
            'total_return_pct': results.get('total_return_pct', 0),
            'annualized_return_pct': results.get('annualized_return_pct', 0),
            'annualized_volatility_pct': results.get('annualized_volatility_pct', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'max_drawdown_pct': results.get('max_drawdown_pct', 0),
            'win_rate_pct': results.get('win_rate', 0),
            'profit_factor': results.get('profit_factor', 0),
            'trades': results.get('trades', 0),
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Generate CSV filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{output_dir}/risk_level_summary_{timestamp}.csv"
    else:
        filename = os.path.join(output_dir, filename)
    
    # Export to CSV
    df.to_csv(filename, index=False)
    
    logging.info(f"Risk level summary exported to {filename}")
    return filename


def main():
    """Main function for CLI interface."""
    parser = argparse.ArgumentParser(description='Generate performance reports for TuringTrader backtest results')
    parser.add_argument('--input', type=str, help='Input results file (pickle format)')
    parser.add_argument('--output-dir', type=str, default='./reports', help='Output directory for reports')
    parser.add_argument('--risk-level', type=int, help='Risk level for single-risk report')
    parser.add_argument('--multi-risk', action='store_true', help='Generate multi-risk comparison report')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load results
    if args.input:
        try:
            results = pd.read_pickle(args.input)
            
            if args.multi_risk:
                # Assume the pickle contains a dictionary mapping risk levels to results
                generate_multi_risk_report(results, args.output_dir)
            else:
                # Assume the pickle contains a single result set
                generate_report(results, args.output_dir, args.risk_level)
                
        except Exception as e:
            logging.error(f"Error loading or processing results: {e}")
            return 1
    else:
        logging.error("No input file specified")
        return 1
        
    return 0


if __name__ == "__main__":
    main()