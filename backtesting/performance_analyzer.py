"""
Performance analysis tools for the TuringTrader algorithm.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceAnalyzer:
    """Analyze and report on trading performance."""
    
    def __init__(self, output_dir: str = './reports'):
        """
        Initialize the performance analyzer.
        
        Args:
            output_dir: Directory for output reports
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
    def analyze_results(self, results: Dict, risk_level: Optional[int] = None) -> Dict:
        """
        Analyze backtest results.
        
        Args:
            results: Results dictionary from backtest
            risk_level: Risk level of this backtest
            
        Returns:
            Dict with enhanced analysis
        """
        analysis = results.copy()
        
        # Add risk level if provided
        if risk_level is not None:
            analysis['risk_level'] = risk_level
        
        # Calculate additional metrics if data is available
        if 'daily_values' in results:
            df = pd.DataFrame(results['daily_values'])
            
            if len(df) > 0:
                # Convert date strings to datetime if needed
                if isinstance(df['date'].iloc[0], str):
                    df['date'] = pd.to_datetime(df['date'])
                
                # Sort by date
                df = df.sort_values('date')
                
                # Calculate daily returns
                df['daily_return'] = df['balance'].pct_change()
                
                # Calculate rolling metrics
                df['rolling_return_30d'] = df['balance'].pct_change(30) * 100
                df['rolling_volatility_30d'] = df['daily_return'].rolling(30).std() * np.sqrt(252) * 100
                
                # Calculate maximum consecutive winning and losing days
                df['return_positive'] = df['daily_return'] > 0
                
                # Get winning and losing streaks
                streak = (df['return_positive'] != df['return_positive'].shift(1)).cumsum()
                winning_streak = df['return_positive'].groupby(streak).cumsum()
                losing_streak = (~df['return_positive']).groupby(streak).cumsum()
                
                max_winning_streak = winning_streak.max()
                max_losing_streak = losing_streak.max()
                
                # Calculate underwater periods (drawdowns)
                df['peak'] = df['balance'].cummax()
                df['drawdown'] = (df['balance'] - df['peak']) / df['peak'] * 100
                
                # Find underwater periods
                underwater_periods = []
                in_drawdown = False
                drawdown_start = None
                current_drawdown = 0
                
                for i, row in df.iterrows():
                    if row['drawdown'] < -5 and not in_drawdown:  # Start tracking significant drawdowns
                        in_drawdown = True
                        drawdown_start = row['date']
                        current_drawdown = row['drawdown']
                    elif row['drawdown'] < current_drawdown and in_drawdown:
                        current_drawdown = row['drawdown']
                    elif row['drawdown'] > -1 and in_drawdown:  # Recovered from drawdown
                        underwater_periods.append({
                            'start': drawdown_start,
                            'end': row['date'],
                            'duration_days': (row['date'] - drawdown_start).days,
                            'max_drawdown': current_drawdown
                        })
                        in_drawdown = False
                
                # Add results to analysis
                analysis['max_winning_streak'] = int(max_winning_streak)
                analysis['max_losing_streak'] = int(max_losing_streak)
                analysis['underwater_periods'] = underwater_periods
                
                # Calculate percentage of profitable months
                df['month'] = df['date'].dt.to_period('M')
                monthly_returns = df.groupby('month')['balance'].agg(['first', 'last'])
                monthly_returns['return'] = (monthly_returns['last'] / monthly_returns['first'] - 1) * 100
                
                analysis['profitable_months_pct'] = (monthly_returns['return'] > 0).mean() * 100
                
                # Best and worst periods
                analysis['best_day_pct'] = df['daily_return'].max() * 100
                analysis['worst_day_pct'] = df['daily_return'].min() * 100
                analysis['best_month_pct'] = monthly_returns['return'].max()
                analysis['worst_month_pct'] = monthly_returns['return'].min()
        
        return analysis
    
    def compare_risk_levels(self, results_by_risk: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Compare results across different risk levels.
        
        Args:
            results_by_risk: Dictionary mapping risk levels to results
            
        Returns:
            Dict with comparison data
        """
        if not results_by_risk:
            return {'error': 'No results provided'}
            
        # Extract key metrics for comparison
        comparison = {
            'risk_levels': [],
            'total_returns': [],
            'annualized_returns': [],
            'max_drawdowns': [],
            'sharpe_ratios': [],
            'win_rates': [],
            'volatility': []
        }
        
        for risk_level, results in sorted(results_by_risk.items()):
            comparison['risk_levels'].append(risk_level)
            comparison['total_returns'].append(results.get('total_return_pct', 0))
            comparison['annualized_returns'].append(results.get('annualized_return_pct', 0))
            comparison['max_drawdowns'].append(results.get('max_drawdown_pct', 0))
            comparison['sharpe_ratios'].append(results.get('sharpe_ratio', 0))
            comparison['win_rates'].append(results.get('win_rate', 0))
            comparison['volatility'].append(results.get('annualized_volatility_pct', 0))
        
        # Find optimal risk level based on risk-adjusted return (Sharpe ratio)
        best_sharpe_idx = np.argmax(comparison['sharpe_ratios'])
        optimal_risk_level = comparison['risk_levels'][best_sharpe_idx]
        
        # Find best absolute return
        best_return_idx = np.argmax(comparison['total_returns'])
        best_return_risk_level = comparison['risk_levels'][best_return_idx]
        
        comparison['optimal_risk_level'] = optimal_risk_level
        comparison['best_return_risk_level'] = best_return_risk_level
        
        return comparison
    
    def generate_report(self, results: Dict, output_file: Optional[str] = None) -> str:
        """
        Generate a performance report.
        
        Args:
            results: Analysis results
            output_file: Output file path
            
        Returns:
            Path to generated report
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create plots
            fig = plt.figure(figsize=(12, 16))
            gs = fig.add_gridspec(5, 2)
            
            # 1. Balance curve
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_balance_curve(results, ax1)
            
            # 2. Drawdown chart
            ax2 = fig.add_subplot(gs[1, :])
            self._plot_drawdown(results, ax2)
            
            # 3. Monthly returns heatmap
            ax3 = fig.add_subplot(gs[2, 0])
            self._plot_monthly_returns(results, ax3)
            
            # 4. Return distribution
            ax4 = fig.add_subplot(gs[2, 1])
            self._plot_return_distribution(results, ax4)
            
            # 5. Key metrics table
            ax5 = fig.add_subplot(gs[3, :])
            self._plot_metrics_table(results, ax5)
            
            # 6. Trade analysis
            ax6 = fig.add_subplot(gs[4, 0])
            self._plot_trade_analysis(results, ax6)
            
            # 7. Risk profile
            ax7 = fig.add_subplot(gs[4, 1])
            self._plot_risk_profile(results, ax7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Determine output file path
            if output_file is None:
                risk_level = results.get('risk_level', 'all')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"{self.output_dir}/performance_report_risk{risk_level}_{timestamp}.png"
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Performance report saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return ""
    
    def generate_comparison_report(self, 
                                 comparison_data: Dict, 
                                 results_by_risk: Dict[int, Dict],
                                 output_file: Optional[str] = None) -> str:
        """
        Generate a report comparing different risk levels.
        
        Args:
            comparison_data: Comparison data
            results_by_risk: Results by risk level
            output_file: Output file path
            
        Returns:
            Path to generated report
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create plots
            fig = plt.figure(figsize=(12, 15))
            gs = fig.add_gridspec(4, 2)
            
            # 1. Returns by risk level
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_returns_by_risk(comparison_data, ax1)
            
            # 2. Sharpe ratio by risk level
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_sharpe_by_risk(comparison_data, ax2)
            
            # 3. Drawdown by risk level
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_drawdown_by_risk(comparison_data, ax3)
            
            # 4. Win rate by risk level
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_win_rate_by_risk(comparison_data, ax4)
            
            # 5. Balance curves for all risk levels
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_balance_curves_comparison(results_by_risk, ax5)
            
            # 6. Risk-return scatter plot
            ax6 = fig.add_subplot(gs[3, 0])
            self._plot_risk_return_scatter(comparison_data, ax6)
            
            # 7. Summary metrics table
            ax7 = fig.add_subplot(gs[3, 1])
            self._plot_comparison_table(comparison_data, results_by_risk, ax7)
            
            # Adjust layout
            plt.tight_layout()
            
            # Determine output file path
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"{self.output_dir}/risk_comparison_{timestamp}.png"
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"Risk comparison report saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating comparison report: {e}")
            return ""
    
    def _plot_balance_curve(self, results: Dict, ax: plt.Axes) -> None:
        """Plot account balance over time."""
        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Account Balance")
            return
            
        df = pd.DataFrame(results['daily_values'])
        
        if len(df) == 0:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Account Balance")
            return
            
        # Convert date if needed
        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])
            
        # Plot balance
        ax.plot(df['date'], df['balance'], 'b-', linewidth=2)
        
        # Add initial balance reference line
        initial_balance = results.get('initial_balance', df['balance'].iloc[0])
        ax.axhline(y=initial_balance, color='r', linestyle='--', alpha=0.7)
        
        # Format
        ax.set_title("Account Balance Over Time")
        ax.set_ylabel("Balance ($)")
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter('${x:,.0f}')
        
        # Add annotations
        final_balance = df['balance'].iloc[-1]
        total_return = ((final_balance / initial_balance) - 1) * 100
        ax.annotate(f"Final: ${final_balance:,.2f} ({total_return:.1f}%)", 
                   xy=(df['date'].iloc[-1], final_balance),
                   xytext=(10, -20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    def _plot_drawdown(self, results: Dict, ax: plt.Axes) -> None:
        """Plot drawdown over time."""
        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Drawdown")
            return
            
        df = pd.DataFrame(results['daily_values'])
        
        if len(df) == 0:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Drawdown")
            return
            
        # Convert date if needed
        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])
            
        # Calculate drawdown
        df['peak'] = df['balance'].cummax()
        df['drawdown'] = (df['balance'] - df['peak']) / df['peak'] * 100
        
        # Plot drawdown
        ax.fill_between(df['date'], 0, df['drawdown'], facecolor='red', alpha=0.3)
        ax.plot(df['date'], df['drawdown'], 'r-', linewidth=1)
        
        # Format
        ax.set_title("Equity Drawdown")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter('{x:.0f}%')
        
        # Y-axis inverted (negative values at top)
        ax.invert_yaxis()
        
        # Add annotation for max drawdown
        max_dd = df['drawdown'].min()
        max_dd_date = df.loc[df['drawdown'].idxmin(), 'date']
        
        ax.annotate(f"Max DD: {max_dd:.1f}%", 
                   xy=(max_dd_date, max_dd),
                   xytext=(10, 20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    def _plot_monthly_returns(self, results: Dict, ax: plt.Axes) -> None:
        """Plot monthly returns heatmap."""
        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Monthly Returns")
            return
            
        df = pd.DataFrame(results['daily_values'])
        
        if len(df) == 0:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Monthly Returns")
            return
            
        # Convert date if needed
        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])
            
        # Calculate monthly returns
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Calculate returns by month
        monthly_returns = df.groupby(['year', 'month'])['balance'].agg(['first', 'last'])
        monthly_returns['return'] = (monthly_returns['last'] / monthly_returns['first'] - 1) * 100
        
        # Convert to pivoted DataFrame for heatmap
        pivot_data = []
        for (year, month), row in monthly_returns.iterrows():
            pivot_data.append({'year': year, 'month': month, 'return': row['return']})
            
        pivot_df = pd.DataFrame(pivot_data)
        pivot_table = pivot_df.pivot('year', 'month', 'return')
        
        # Plot heatmap
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlGn", center=0, 
                   cbar_kws={'label': 'Return (%)'}, ax=ax)
                   
        # Format
        ax.set_title("Monthly Returns (%)")
        ax.set_ylabel("Year")
        ax.set_xlabel("Month")
        
        # Use month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names)
    
    def _plot_return_distribution(self, results: Dict, ax: plt.Axes) -> None:
        """Plot return distribution."""
        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Return Distribution")
            return
            
        df = pd.DataFrame(results['daily_values'])
        
        if len(df) == 0:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Return Distribution")
            return
            
        # Calculate daily returns
        df['daily_return'] = df['balance'].pct_change() * 100
        
        # Drop NaN values
        df = df.dropna(subset=['daily_return'])
        
        if len(df) == 0:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Return Distribution")
            return
        
        # Plot histogram with kernel density estimate
        sns.histplot(df['daily_return'], kde=True, ax=ax, color='blue', bins=30, alpha=0.6)
        
        # Add normal distribution for comparison
        mean = df['daily_return'].mean()
        std = df['daily_return'].std()
        x = np.linspace(mean - 3*std, mean + 3*std, 100)
        y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-(x-mean)**2/(2*std**2))
        y = y * (len(df) * (df['daily_return'].max() - df['daily_return'].min()) / 30)  # Scale to match histogram
        ax.plot(x, y, 'r--', linewidth=2, label='Normal Distribution')
        
        # Format
        ax.set_title("Daily Return Distribution")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        # Add statistics as text
        stats_text = (
            f"Mean: {mean:.2f}%\n"
            f"Std Dev: {std:.2f}%\n"
            f"Skewness: {df['daily_return'].skew():.2f}\n"
            f"Kurtosis: {df['daily_return'].kurtosis():.2f}"
        )
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_metrics_table(self, results: Dict, ax: plt.Axes) -> None:
        """Plot key metrics table."""
        # Turn off axis
        ax.axis('off')
        ax.set_title("Performance Metrics", fontsize=14, fontweight='bold')
        
        # Create a table with two columns
        metrics = [
            ("Initial Balance", f"${results.get('initial_balance', 0):,.2f}"),
            ("Final Balance", f"${results.get('final_balance', 0):,.2f}"),
            ("Total Return", f"{results.get('total_return_pct', 0):.2f}%"),
            ("Annualized Return", f"{results.get('annualized_return_pct', 0):.2f}%"),
            ("Annualized Volatility", f"{results.get('annualized_volatility_pct', 0):.2f}%"),
            ("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}"),
            ("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.2f}%"),
            ("Win Rate", f"{results.get('win_rate', 0):.2f}%"),
            ("Profit Factor", f"{results.get('profit_factor', 0):.2f}"),
            ("Total Trades", f"{results.get('trades', 0)}"),
            ("Avg Win", f"${results.get('avg_win', 0):,.2f}"),
            ("Avg Loss", f"${results.get('avg_loss', 0):,.2f}")
        ]
        
        if 'risk_level' in results:
            metrics.insert(0, ("Risk Level", str(results['risk_level'])))
            
        cell_text = [[metric, value] for metric, value in metrics]
        
        # Create the table
        tbl = ax.table(cellText=cell_text, loc='center', cellLoc='left',
                      colWidths=[0.3, 0.3], bbox=[0.1, 0.0, 0.8, 0.9])
                      
        # Style the table
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        
        # Make headers bold
        for i, key in enumerate(['Metric', 'Value']):
            cell = tbl[0, i]
            cell.get_text().set_fontweight('bold')
            
        # Make alternating rows different colors
        for i in range(1, len(metrics) + 1):
            for j in range(2):
                cell = tbl[i, j]
                if i % 2 == 0:
                    cell.set_facecolor('#f2f2f2')  # Light gray
    
    def _plot_trade_analysis(self, results: Dict, ax: plt.Axes) -> None:
        """Plot trade analysis."""
        # Check if we have trade history
        if 'trades' not in results or results.get('trades', 0) == 0:
            ax.text(0.5, 0.5, "No trade data available", ha='center', va='center')
            ax.set_title("Trade Analysis")
            return
            
        # Create pie chart for win/loss ratio
        win_rate = results.get('win_rate', 0)
        loss_rate = 100 - win_rate
        
        ax.pie([win_rate, loss_rate], labels=['Win', 'Loss'], autopct='%1.1f%%',
              colors=['#4CAF50', '#F44336'], startangle=90)
              
        ax.set_title("Trade Outcomes")
        
        # Add additional metrics as text
        metrics_text = (
            f"Avg Win: ${results.get('avg_win', 0):,.2f}\n"
            f"Avg Loss: ${results.get('avg_loss', 0):,.2f}\n"
            f"Profit Factor: {results.get('profit_factor', 0):.2f}\n"
            f"Total Trades: {results.get('trades', 0)}"
        )
        
        ax.text(1.1, 0.5, metrics_text, transform=ax.transAxes, 
               verticalalignment='center', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_risk_profile(self, results: Dict, ax: plt.Axes) -> None:
        """Plot risk profile metrics."""
        # Turn off axis
        ax.axis('off')
        ax.set_title("Risk Profile", fontsize=14, fontweight='bold')
        
        # Define risk metrics
        risk_metrics = [
            ("Risk Level", str(results.get('risk_level', 'N/A'))),
            ("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.2f}%"),
            ("Volatility", f"{results.get('annualized_volatility_pct', 0):.2f}%"),
            ("Risk/Reward", f"{results.get('annualized_return_pct', 0) / results.get('annualized_volatility_pct', 1):.2f}"),
            ("Max Win Streak", str(results.get('max_winning_streak', 'N/A'))),
            ("Max Loss Streak", str(results.get('max_losing_streak', 'N/A'))),
        ]
        
        # Create table with risk metrics
        cell_text = [[metric, value] for metric, value in risk_metrics]
        
        # Create the table
        tbl = ax.table(cellText=cell_text, loc='center', cellLoc='left',
                      colWidths=[0.3, 0.3], bbox=[0.1, 0.2, 0.8, 0.6])
                      
        # Style the table
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        
        # Make headers bold
        for i, key in enumerate(['Risk Metric', 'Value']):
            cell = tbl[0, i]
            cell.get_text().set_fontweight('bold')
            
        # Make alternating rows different colors
        for i in range(1, len(risk_metrics) + 1):
            for j in range(2):
                cell = tbl[i, j]
                if i % 2 == 0:
                    cell.set_facecolor('#f2f2f2')  # Light gray
    
    def _plot_returns_by_risk(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot returns by risk level."""
        # Check if we have data
        if not comparison.get('risk_levels'):
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Returns by Risk Level")
            return
            
        # Create bar chart
        ax.bar(comparison['risk_levels'], comparison['total_returns'], color='blue', alpha=0.7)
        
        # Highlight optimal risk level
        optimal_risk = comparison.get('optimal_risk_level')
        if optimal_risk in comparison['risk_levels']:
            idx = comparison['risk_levels'].index(optimal_risk)
            ax.bar([optimal_risk], [comparison['total_returns'][idx]], color='green', alpha=0.7)
        
        # Format
        ax.set_title("Total Return by Risk Level")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Total Return (%)")
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks to integer risk levels
        ax.set_xticks(comparison['risk_levels'])
        
        # Add value labels on bars
        for i, v in enumerate(comparison['total_returns']):
            ax.text(comparison['risk_levels'][i], v + 1, f"{v:.1f}%", 
                   ha='center', va='bottom', fontsize=8)
    
    def _plot_sharpe_by_risk(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot Sharpe ratio by risk level."""
        # Check if we have data
        if not comparison.get('risk_levels'):
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Sharpe Ratio by Risk Level")
            return
            
        # Create bar chart
        ax.bar(comparison['risk_levels'], comparison['sharpe_ratios'], color='purple', alpha=0.7)
        
        # Highlight optimal risk level
        optimal_risk = comparison.get('optimal_risk_level')
        if optimal_risk in comparison['risk_levels']:
            idx = comparison['risk_levels'].index(optimal_risk)
            ax.bar([optimal_risk], [comparison['sharpe_ratios'][idx]], color='green', alpha=0.7)
        
        # Format
        ax.set_title("Sharpe Ratio by Risk Level")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Sharpe Ratio")
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks to integer risk levels
        ax.set_xticks(comparison['risk_levels'])
        
        # Add value labels on bars
        for i, v in enumerate(comparison['sharpe_ratios']):
            ax.text(comparison['risk_levels'][i], v + 0.05, f"{v:.2f}", 
                   ha='center', va='bottom', fontsize=8)
    
    def _plot_drawdown_by_risk(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot maximum drawdown by risk level."""
        # Check if we have data
        if not comparison.get('risk_levels'):
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Max Drawdown by Risk Level")
            return
            
        # Create bar chart
        ax.bar(comparison['risk_levels'], comparison['max_drawdowns'], color='red', alpha=0.7)
        
        # Format
        ax.set_title("Maximum Drawdown by Risk Level")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Max Drawdown (%)")
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to be inverted (negative values at top)
        ax.invert_yaxis()
        
        # Set x-ticks to integer risk levels
        ax.set_xticks(comparison['risk_levels'])
        
        # Add value labels on bars
        for i, v in enumerate(comparison['max_drawdowns']):
            ax.text(comparison['risk_levels'][i], v - 1, f"{v:.1f}%", 
                   ha='center', va='top', fontsize=8)
    
    def _plot_win_rate_by_risk(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot win rate by risk level."""
        # Check if we have data
        if not comparison.get('risk_levels'):
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Win Rate by Risk Level")
            return
            
        # Create bar chart
        ax.bar(comparison['risk_levels'], comparison['win_rates'], color='green', alpha=0.7)
        
        # Format
        ax.set_title("Win Rate by Risk Level")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Win Rate (%)")
        ax.grid(True, alpha=0.3)
        
        # Set x-ticks to integer risk levels
        ax.set_xticks(comparison['risk_levels'])
        
        # Add value labels on bars
        for i, v in enumerate(comparison['win_rates']):
            ax.text(comparison['risk_levels'][i], v + 1, f"{v:.1f}%", 
                   ha='center', va='bottom', fontsize=8)
    
    def _plot_balance_curves_comparison(self, results_by_risk: Dict[int, Dict], ax: plt.Axes) -> None:
        """Plot balance curves for all risk levels."""
        # Check if we have data
        if not results_by_risk:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Balance Curves Comparison")
            return
            
        # Plot each risk level
        for risk_level, results in sorted(results_by_risk.items()):
            if 'daily_values' not in results:
                continue
                
            df = pd.DataFrame(results['daily_values'])
            
            if len(df) == 0:
                continue
                
            # Convert date if needed
            if isinstance(df['date'].iloc[0], str):
                df['date'] = pd.to_datetime(df['date'])
                
            # Plot balance curve
            ax.plot(df['date'], df['balance'], label=f"Risk {risk_level}")
        
        # Add initial balance reference line
        initial_balance = next(iter(results_by_risk.values())).get('initial_balance', 100000)
        ax.axhline(y=initial_balance, color='black', linestyle='--', alpha=0.5, label='Initial Balance')
        
        # Format
        ax.set_title("Account Balance Comparison Across Risk Levels")
        ax.set_xlabel("Date")
        ax.set_ylabel("Balance ($)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter('${x:,.0f}')
    
    def _plot_risk_return_scatter(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot risk-return scatter plot."""
        # Check if we have data
        if not comparison.get('risk_levels'):
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            ax.set_title("Risk-Return Profile")
            return
            
        # Create scatter plot
        sc = ax.scatter(comparison['volatility'], comparison['annualized_returns'], 
                      c=comparison['risk_levels'], cmap='viridis', 
                      s=100, alpha=0.7)
        
        # Add labels for each point
        for i, risk_level in enumerate(comparison['risk_levels']):
            ax.annotate(f"Risk {risk_level}", 
                       (comparison['volatility'][i], comparison['annualized_returns'][i]),
                       xytext=(5, 5), textcoords='offset points')
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Risk Level')
        
        # Format
        ax.set_title("Risk-Return Profile")
        ax.set_xlabel("Volatility (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.grid(True, alpha=0.3)
        
        # Add the Capital Market Line (CML)
        if len(comparison['volatility']) >= 2:
            # Use the risk-free asset as the origin (0, 0)
            x = np.array([0] + comparison['volatility'])
            
            # Simple linear regression
            optimal_idx = comparison['sharpe_ratios'].index(max(comparison['sharpe_ratios']))
            slope = comparison['annualized_returns'][optimal_idx] / comparison['volatility'][optimal_idx]
            y = slope * x
            
            ax.plot(x, y, 'r--', alpha=0.5, label='Capital Market Line')
            ax.legend()
    
    def _plot_comparison_table(self, comparison: Dict, results_by_risk: Dict, ax: plt.Axes) -> None:
        """Plot comparison summary table."""
        # Turn off axis
        ax.axis('off')
        ax.set_title("Risk Level Comparison Summary", fontsize=14, fontweight='bold')
        
        # Get optimal risk levels
        optimal_risk = comparison.get('optimal_risk_level', 'N/A')
        best_return_risk = comparison.get('best_return_risk_level', 'N/A')
        
        # Create a summary table
        summary_text = [
            ["Metric", "Value"],
            ["Optimal Risk Level (Best Sharpe)", str(optimal_risk)],
            ["Best Return Risk Level", str(best_return_risk)],
            ["Risk Levels Tested", f"{min(comparison['risk_levels'])} - {max(comparison['risk_levels'])}"],
            ["Return Range", f"{min(comparison['total_returns']):.1f}% - {max(comparison['total_returns']):.1f}%"],
            ["Volatility Range", f"{min(comparison['volatility']):.1f}% - {max(comparison['volatility']):.1f}%"],
            ["Drawdown Range", f"{min(comparison['max_drawdowns']):.1f}% - {max(comparison['max_drawdowns']):.1f}%"],
            ["Win Rate Range", f"{min(comparison['win_rates']):.1f}% - {max(comparison['win_rates']):.1f}%"]
        ]
        
        # Create the table
        tbl = ax.table(cellText=summary_text, loc='center', cellLoc='left',
                      colWidths=[0.4, 0.3], bbox=[0.1, 0.0, 0.8, 0.9])
                      
        # Style the table
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        
        # Make headers bold
        for i in range(2):
            cell = tbl[0, i]
            cell.get_text().set_fontweight('bold')
            
        # Highlight the optimal risk level row
        for i in range(2):
            cell = tbl[1, i]
            cell.set_facecolor('#d4edda')  # Light green
        
        # Make alternating rows different colors
        for i in range(2, len(summary_text)):
            for j in range(2):
                cell = tbl[i, j]
                if i % 2 == 0:
                    cell.set_facecolor('#f2f2f2')  # Light gray