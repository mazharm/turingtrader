#!/usr/bin/env python3
"""
Visualization script for the Enhanced Adaptive Volatility-Harvesting System.
Generates risk level and parameter analysis charts using simulated data.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
output_dir = './reports/volatility_harvesting'
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate simulated risk level results
def generate_risk_level_data():
    """Generate simulated risk level test results."""
    risk_levels = list(range(1, 11))  # Risk levels 1-10
    
    # Generate realistic metrics with increasing risk correlation
    # Higher risk typically means higher returns but also higher drawdowns
    base_returns = [random.uniform(5, 8) for _ in range(3)]  # Base returns for low risk
    base_returns.extend([random.uniform(8, 15) for _ in range(4)])  # Medium risk
    base_returns.extend([random.uniform(15, 25) for _ in range(3)])  # High risk
    
    # Add some randomness
    returns = [r * (1 + random.uniform(-0.2, 0.2)) for r in base_returns]
    
    # Sharpe ratios typically peak in the middle-high risk range then decline
    sharpe_pattern = [0.6, 0.8, 1.0, 1.2, 1.5, 1.7, 1.6, 1.4, 1.2, 1.0]
    sharpe_ratios = [s * (1 + random.uniform(-0.1, 0.1)) for s in sharpe_pattern]
    
    # Max drawdowns increase with risk
    drawdowns = []
    for i in range(10):
        base_drawdown = -5 - (i * 2)  # Increasing drawdowns with risk
        drawdowns.append(base_drawdown * (1 + random.uniform(-0.2, 0.2)))
    
    # Win rates typically highest in middle risk range
    win_rates = []
    for i in range(10):
        if i < 3:  # Low risk - conservative trades with moderate win rate
            base_win = random.uniform(55, 65)
        elif i < 7:  # Medium risk - balanced with higher win rate
            base_win = random.uniform(62, 72)
        else:  # High risk - aggressive with more losses
            base_win = random.uniform(50, 60)
        win_rates.append(base_win)
    
    # Annualized volatility increases with risk
    volatilities = [3 + (i * 1.5) + random.uniform(-1, 1) for i in range(10)]
    
    # Create DataFrame
    data = {
        'Risk Level': risk_levels,
        'Total Return (%)': returns,
        'Sharpe Ratio': sharpe_ratios,
        'Max Drawdown (%)': drawdowns,
        'Win Rate (%)': win_rates,
        'Volatility (%)': volatilities
    }
    
    return pd.DataFrame(data)

# Generate simulated parameter test results
def generate_parameter_data():
    """Generate simulated parameter test results."""
    data = []
    
    # IV/HV ratio thresholds (typically lower is more aggressive but less precise)
    iv_hv_ratios = [1.1, 1.2, 1.3, 1.4, 1.5]
    returns = [20, 18, 15, 12, 10]  # Higher returns for more aggressive settings
    sharpes = [1.1, 1.3, 1.5, 1.4, 1.3]  # Peak efficiency in the middle
    drawdowns = [-22, -18, -15, -13, -11]  # Higher drawdowns for more aggressive settings
    
    for i, ratio in enumerate(iv_hv_ratios):
        data.append({
            'Parameter': 'IV/HV Ratio',
            'Value': ratio,
            'Total Return (%)': returns[i] * (1 + random.uniform(-0.1, 0.1)),
            'Sharpe Ratio': sharpes[i] * (1 + random.uniform(-0.1, 0.1)),
            'Max Drawdown (%)': drawdowns[i] * (1 + random.uniform(-0.1, 0.1)),
        })
    
    # Min IV thresholds (lower is more aggressive)
    min_iv_thresholds = [15.0, 20.0, 25.0, 30.0, 35.0]
    returns = [22, 19, 16, 13, 10]  # Higher returns for more aggressive settings
    sharpes = [1.0, 1.2, 1.5, 1.3, 1.1]  # Peak efficiency in the middle
    drawdowns = [-25, -20, -16, -13, -10]  # Higher drawdowns for more aggressive settings
    
    for i, threshold in enumerate(min_iv_thresholds):
        data.append({
            'Parameter': 'Min IV Threshold',
            'Value': threshold,
            'Total Return (%)': returns[i] * (1 + random.uniform(-0.1, 0.1)),
            'Sharpe Ratio': sharpes[i] * (1 + random.uniform(-0.1, 0.1)),
            'Max Drawdown (%)': drawdowns[i] * (1 + random.uniform(-0.1, 0.1)),
        })
    
    # Strike widths (wider is more conservative)
    strike_widths = [2.0, 3.0, 5.0, 7.0, 10.0]
    returns = [24, 20, 16, 13, 10]  # Higher returns for tighter strikes
    sharpes = [1.1, 1.3, 1.5, 1.4, 1.2]  # Peak efficiency in the middle
    drawdowns = [-28, -22, -16, -13, -10]  # Higher drawdowns for tighter strikes
    
    for i, width in enumerate(strike_widths):
        data.append({
            'Parameter': 'Strike Width',
            'Value': width,
            'Total Return (%)': returns[i] * (1 + random.uniform(-0.1, 0.1)),
            'Sharpe Ratio': sharpes[i] * (1 + random.uniform(-0.1, 0.1)),
            'Max Drawdown (%)': drawdowns[i] * (1 + random.uniform(-0.1, 0.1)),
        })
    
    # Target delta (higher is more aggressive)
    target_deltas = [0.20, 0.25, 0.30, 0.35, 0.40]
    returns = [12, 15, 18, 21, 24]  # Higher returns for more aggressive settings
    sharpes = [1.2, 1.4, 1.5, 1.3, 1.0]  # Peak efficiency in the middle
    drawdowns = [-12, -15, -18, -22, -26]  # Higher drawdowns for more aggressive settings
    
    for i, delta in enumerate(target_deltas):
        data.append({
            'Parameter': 'Target Delta',
            'Value': delta,
            'Total Return (%)': returns[i] * (1 + random.uniform(-0.1, 0.1)),
            'Sharpe Ratio': sharpes[i] * (1 + random.uniform(-0.1, 0.1)),
            'Max Drawdown (%)': drawdowns[i] * (1 + random.uniform(-0.1, 0.1)),
        })
    
    return pd.DataFrame(data)

# Generate and save risk level analysis
def generate_risk_level_analysis():
    """Generate and save risk level analysis chart."""
    df = generate_risk_level_data()
    
    # Create plot
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    
    # Plot returns by risk level
    axs[0, 0].bar(df['Risk Level'], df['Total Return (%)'])
    axs[0, 0].set_title('Total Return by Risk Level')
    axs[0, 0].set_xlabel('Risk Level')
    axs[0, 0].set_ylabel('Total Return (%)')
    axs[0, 0].set_xticks(df['Risk Level'])
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot Sharpe ratio by risk level
    axs[0, 1].bar(df['Risk Level'], df['Sharpe Ratio'], color='green')
    axs[0, 1].set_title('Sharpe Ratio by Risk Level')
    axs[0, 1].set_xlabel('Risk Level')
    axs[0, 1].set_ylabel('Sharpe Ratio')
    axs[0, 1].set_xticks(df['Risk Level'])
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot max drawdown by risk level
    axs[1, 0].bar(df['Risk Level'], df['Max Drawdown (%)'], color='red')
    axs[1, 0].set_title('Maximum Drawdown by Risk Level')
    axs[1, 0].set_xlabel('Risk Level')
    axs[1, 0].set_ylabel('Maximum Drawdown (%)')
    axs[1, 0].set_xticks(df['Risk Level'])
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot win rate by risk level
    axs[1, 1].bar(df['Risk Level'], df['Win Rate (%)'], color='purple')
    axs[1, 1].set_title('Win Rate by Risk Level')
    axs[1, 1].set_xlabel('Risk Level')
    axs[1, 1].set_ylabel('Win Rate (%)')
    axs[1, 1].set_xticks(df['Risk Level'])
    axs[1, 1].grid(True, alpha=0.3)
    
    # Plot risk-return scatter
    axs[2, 0].scatter(df['Volatility (%)'], df['Total Return (%)'], s=100)
    axs[2, 0].set_title('Risk-Return Profile')
    axs[2, 0].set_xlabel('Volatility (%)')
    axs[2, 0].set_ylabel('Return (%)')
    axs[2, 0].grid(True)
    
    # Add labels for each point
    for i, risk in enumerate(df['Risk Level']):
        axs[2, 0].annotate(
            f"Risk {risk}", 
            (df['Volatility (%)'].iloc[i], df['Total Return (%)'].iloc[i]),
            xytext=(5, 5), 
            textcoords='offset points'
        )
    
    # Find best risk levels
    best_return_idx = df['Total Return (%)'].idxmax()
    best_return_risk = df['Risk Level'].iloc[best_return_idx]
    
    best_sharpe_idx = df['Sharpe Ratio'].idxmax()
    best_sharpe_risk = df['Risk Level'].iloc[best_sharpe_idx]
    
    # Plot summary text
    summary_text = (
        f"Best Return: Risk Level {best_return_risk} ({df['Total Return (%)'].iloc[best_return_idx]:.2f}%)\n"
        f"Best Sharpe Ratio: Risk Level {best_sharpe_risk} ({df['Sharpe Ratio'].iloc[best_sharpe_idx]:.2f})\n\n"
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
    report_path = os.path.join(output_dir, 'risk_level_analysis.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save data to CSV
    csv_path = os.path.join(output_dir, 'risk_level_summary.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Risk level analysis saved to {report_path}")
    print(f"Risk level data saved to {csv_path}")

# Generate and save parameter sensitivity analysis
def generate_parameter_analysis():
    """Generate and save parameter sensitivity analysis chart."""
    df = generate_parameter_data()
    
    fig, axs = plt.subplots(4, 2, figsize=(16, 20))
    
    # Process each parameter
    param_names = ['IV/HV Ratio', 'Min IV Threshold', 'Strike Width', 'Target Delta']
    
    for i, param in enumerate(param_names):
        param_df = df[df['Parameter'] == param]
        
        # Plot returns vs parameter
        axs[i, 0].plot(param_df['Value'], param_df['Total Return (%)'], marker='o', linewidth=2)
        axs[i, 0].set_title(f'Total Return vs {param}')
        axs[i, 0].set_xlabel(param)
        axs[i, 0].set_ylabel('Total Return (%)')
        axs[i, 0].grid(True)
        
        # Plot Sharpe vs parameter
        axs[i, 1].plot(param_df['Value'], param_df['Sharpe Ratio'], marker='o', linewidth=2, color='green')
        axs[i, 1].set_title(f'Sharpe Ratio vs {param}')
        axs[i, 1].set_xlabel(param)
        axs[i, 1].set_ylabel('Sharpe Ratio')
        axs[i, 1].grid(True)
    
    plt.tight_layout()
    plt.suptitle('Parameter Sensitivity Analysis for Volatility Harvesting Strategy', fontsize=16, y=0.98)
    
    # Save plot
    report_path = os.path.join(output_dir, 'parameter_sensitivity.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save data to CSV
    csv_path = os.path.join(output_dir, 'parameter_summary.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Parameter analysis saved to {report_path}")
    print(f"Parameter data saved to {csv_path}")

# Main execution
if __name__ == "__main__":
    print("Generating risk level analysis...")
    generate_risk_level_analysis()
    
    print("Generating parameter sensitivity analysis...")
    generate_parameter_analysis()
    
    print("Analysis complete!")