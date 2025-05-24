#!/usr/bin/env python3
"""
Example script showing how to use the TuringTrader algorithm evaluation.

This script demonstrates how to evaluate the trading algorithm with
different parameters and shows how to access and utilize the results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from evaluate_algorithm import evaluate_algorithm, generate_investment_growth_chart
from utils.logging_util import setup_logging


def main():
    """Example evaluation of the TuringTrader algorithm."""
    # Setup logging
    setup_logging(level='INFO')
    
    # Example 1: Basic evaluation with default parameters
    print("\n=== EXAMPLE 1: Basic Evaluation ===")
    args1 = type('Args', (), {
        'initial_investment': 100000.0,
        'period': 365,
        'output_dir': './example_evaluations/basic',
        'config': None,
        'log_level': 'INFO'
    })
    
    evaluate_algorithm(args1)
    
    # Example 2: Evaluation with different parameters
    print("\n=== EXAMPLE 2: Custom Evaluation ===")
    args2 = type('Args', (), {
        'initial_investment': 250000.0,
        'period': 180,
        'output_dir': './example_evaluations/custom',
        'config': None,
        'log_level': 'INFO'
    })
    
    evaluate_algorithm(args2)
    
    # Example 3: Comparing evaluations from different periods
    print("\n=== EXAMPLE 3: Comparison Across Time Periods ===")
    os.makedirs('./example_evaluations/comparison', exist_ok=True)
    
    # Set up time periods
    end_date = datetime.now()
    
    # Define periods
    periods = {
        '3m': 90,
        '6m': 180,
        '1y': 365
    }
    
    # Run evaluation for each period
    for period_name, days in periods.items():
        print(f"\nEvaluating {period_name} period...")
        
        start_date = (end_date - timedelta(days=days)).strftime('%Y-%m-%d')
        period_args = type('Args', (), {
            'initial_investment': 100000.0,
            'period': days,
            'output_dir': f'./example_evaluations/comparison/{period_name}',
            'config': None,
            'log_level': 'INFO'
        })
        
        evaluate_algorithm(period_args)
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()