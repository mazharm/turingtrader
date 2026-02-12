"""
Performance analysis tools for the TuringTrader algorithm.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# Professional dark theme colors
COLORS = {
    'bg': '#1a1a2e',
    'panel': '#16213e',
    'grid': '#2a2a4a',
    'text': '#e0e0e0',
    'text_muted': '#8888aa',
    'accent': '#00d2ff',
    'profit': '#00e676',
    'loss': '#ff5252',
    'warn': '#ffd740',
    'line1': '#00d2ff',
    'line2': '#7c4dff',
    'line3': '#ff6e40',
    'line4': '#64ffda',
    'line5': '#ea80fc',
    'table_header': '#0a3d62',
    'table_row_even': '#1e2d4a',
    'table_row_odd': '#16213e',
}


def _apply_dark_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent dark styling to an axis."""
    ax.set_facecolor(COLORS['panel'])
    ax.tick_params(colors=COLORS['text'], labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.spines['left'].set_color(COLORS['grid'])
    ax.grid(True, alpha=0.15, color=COLORS['grid'], linestyle='--')
    if title:
        ax.set_title(title, color=COLORS['text'], fontsize=12, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS['text_muted'], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS['text_muted'], fontsize=9)


class PerformanceAnalyzer:
    """Analyze and report on trading performance."""

    def __init__(self, output_dir: str = './reports'):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir

    def analyze_results(self, results: Dict, risk_level: Optional[int] = None) -> Dict:
        """Analyze backtest results with enhanced metrics."""
        analysis = results.copy()

        if risk_level is not None:
            analysis['risk_level'] = risk_level

        if 'daily_values' in results:
            df = pd.DataFrame(results['daily_values'])

            if len(df) > 0:
                if isinstance(df['date'].iloc[0], str):
                    df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')

                df['daily_return'] = df['balance'].pct_change()

                # Rolling metrics
                df['rolling_return_30d'] = df['balance'].pct_change(30) * 100
                df['rolling_volatility_30d'] = df['daily_return'].rolling(30).std() * np.sqrt(252) * 100

                # Streaks
                df['return_positive'] = df['daily_return'] > 0
                streak = (df['return_positive'] != df['return_positive'].shift(1)).cumsum()
                winning_streak = df['return_positive'].groupby(streak).cumsum()
                losing_streak = (~df['return_positive']).groupby(streak).cumsum()
                analysis['max_winning_streak'] = int(winning_streak.max())
                analysis['max_losing_streak'] = int(losing_streak.max())

                # Drawdowns
                df['peak'] = df['balance'].cummax()
                df['drawdown'] = (df['balance'] - df['peak']) / df['peak'] * 100

                # Sortino ratio (downside deviation only)
                downside_returns = df['daily_return'].copy()
                downside_returns[downside_returns > 0] = 0
                downside_std = downside_returns.std() * np.sqrt(252)
                mean_return_ann = df['daily_return'].mean() * 252
                analysis['sortino_ratio'] = mean_return_ann / downside_std if downside_std > 0 else 0

                # Calmar ratio (return / max drawdown)
                max_dd = abs(df['drawdown'].min()) / 100
                analysis['calmar_ratio'] = mean_return_ann / max_dd if max_dd > 0 else 0

                # Monthly stats
                df['month'] = df['date'].dt.to_period('M')
                monthly_returns = df.groupby('month')['balance'].agg(['first', 'last'])
                monthly_returns['return'] = (monthly_returns['last'] / monthly_returns['first'] - 1) * 100

                analysis['profitable_months_pct'] = (monthly_returns['return'] > 0).mean() * 100
                analysis['best_day_pct'] = df['daily_return'].max() * 100
                analysis['worst_day_pct'] = df['daily_return'].min() * 100
                analysis['best_month_pct'] = monthly_returns['return'].max()
                analysis['worst_month_pct'] = monthly_returns['return'].min()

        return analysis

    def compare_risk_levels(self, results_by_risk: Dict[int, Dict]) -> Dict[str, Any]:
        """Compare results across different risk levels."""
        if not results_by_risk:
            return {'error': 'No results provided'}

        comparison = {
            'risk_levels': [],
            'total_returns': [],
            'annualized_returns': [],
            'max_drawdowns': [],
            'sharpe_ratios': [],
            'win_rates': [],
            'volatility': [],
            'sortino_ratios': [],
            'calmar_ratios': [],
        }

        for risk_level, results in sorted(results_by_risk.items()):
            comparison['risk_levels'].append(risk_level)
            comparison['total_returns'].append(results.get('total_return_pct', 0))
            comparison['annualized_returns'].append(results.get('annualized_return_pct', 0))
            comparison['max_drawdowns'].append(results.get('max_drawdown_pct', 0))
            comparison['sharpe_ratios'].append(results.get('sharpe_ratio', 0))
            comparison['win_rates'].append(results.get('win_rate', 0))
            comparison['volatility'].append(results.get('annualized_volatility_pct', 0))
            comparison['sortino_ratios'].append(results.get('sortino_ratio', 0))
            comparison['calmar_ratios'].append(results.get('calmar_ratio', 0))

        best_sharpe_idx = np.argmax(comparison['sharpe_ratios'])
        comparison['optimal_risk_level'] = comparison['risk_levels'][best_sharpe_idx]

        best_return_idx = np.argmax(comparison['total_returns'])
        comparison['best_return_risk_level'] = comparison['risk_levels'][best_return_idx]

        return comparison

    def generate_report(self, results: Dict, output_file: Optional[str] = None) -> str:
        """Generate a performance report with professional dark theme."""
        try:
            import matplotlib
            matplotlib.use('Agg')

            fig = plt.figure(figsize=(16, 22), facecolor=COLORS['bg'])
            fig.suptitle('TuringTrader Performance Report',
                        color=COLORS['accent'], fontsize=18, fontweight='bold', y=0.98)

            gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3,
                                 left=0.06, right=0.94, top=0.95, bottom=0.03)

            # 1. Balance curve (full width)
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_balance_curve(results, ax1)

            # 2. Drawdown (full width)
            ax2 = fig.add_subplot(gs[1, :])
            self._plot_drawdown(results, ax2)

            # 3. Monthly returns heatmap
            ax3 = fig.add_subplot(gs[2, 0])
            self._plot_monthly_returns(results, ax3)

            # 4. Return distribution
            ax4 = fig.add_subplot(gs[2, 1])
            self._plot_return_distribution(results, ax4)

            # 5. Key metrics table (full width)
            ax5 = fig.add_subplot(gs[3, :])
            self._plot_metrics_table(results, ax5)

            # 6. Trade analysis
            ax6 = fig.add_subplot(gs[4, 0])
            self._plot_trade_analysis(results, ax6)

            # 7. Rolling Sharpe
            ax7 = fig.add_subplot(gs[4, 1])
            self._plot_rolling_sharpe(results, ax7)

            # 8. Cumulative returns (full width)
            ax8 = fig.add_subplot(gs[5, :])
            self._plot_cumulative_returns(results, ax8)

            if output_file is None:
                risk_level = results.get('risk_level', 'all')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"{self.output_dir}/performance_report_risk{risk_level}_{timestamp}.png"

            plt.savefig(output_file, dpi=150, bbox_inches='tight',
                       facecolor=COLORS['bg'], edgecolor='none')
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
        """Generate a comparison report with professional dark theme."""
        try:
            import matplotlib
            matplotlib.use('Agg')

            fig = plt.figure(figsize=(16, 20), facecolor=COLORS['bg'])
            fig.suptitle('TuringTrader Risk Level Comparison',
                        color=COLORS['accent'], fontsize=18, fontweight='bold', y=0.98)

            gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3,
                                 left=0.06, right=0.94, top=0.95, bottom=0.03)

            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_returns_by_risk(comparison_data, ax1)

            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_sharpe_by_risk(comparison_data, ax2)

            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_drawdown_by_risk(comparison_data, ax3)

            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_win_rate_by_risk(comparison_data, ax4)

            ax5 = fig.add_subplot(gs[2, :])
            self._plot_balance_curves_comparison(results_by_risk, ax5)

            ax6 = fig.add_subplot(gs[3, 0])
            self._plot_risk_return_scatter(comparison_data, ax6)

            ax7 = fig.add_subplot(gs[3, 1])
            self._plot_comparison_table(comparison_data, results_by_risk, ax7)

            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"{self.output_dir}/risk_comparison_{timestamp}.png"

            plt.savefig(output_file, dpi=150, bbox_inches='tight',
                       facecolor=COLORS['bg'], edgecolor='none')
            plt.close(fig)

            self.logger.info(f"Risk comparison report saved to: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error generating comparison report: {e}")
            return ""

    def _plot_balance_curve(self, results: Dict, ax: plt.Axes) -> None:
        """Plot account balance over time with gradient fill."""
        _apply_dark_style(ax, title='Equity Curve', ylabel='Balance ($)')

        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', color=COLORS['text_muted'])
            return

        df = pd.DataFrame(results['daily_values'])
        if len(df) == 0:
            return

        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])

        initial_balance = results.get('initial_balance', df['balance'].iloc[0])

        ax.plot(df['date'], df['balance'], color=COLORS['line1'], linewidth=1.8, zorder=3)
        ax.fill_between(df['date'], initial_balance, df['balance'],
                        where=df['balance'] >= initial_balance,
                        facecolor=COLORS['profit'], alpha=0.12, interpolate=True)
        ax.fill_between(df['date'], initial_balance, df['balance'],
                        where=df['balance'] < initial_balance,
                        facecolor=COLORS['loss'], alpha=0.12, interpolate=True)

        ax.axhline(y=initial_balance, color=COLORS['text_muted'], linestyle='--', alpha=0.5, linewidth=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        final_balance = df['balance'].iloc[-1]
        total_return = ((final_balance / initial_balance) - 1) * 100
        color = COLORS['profit'] if total_return >= 0 else COLORS['loss']
        ax.annotate(f"${final_balance:,.0f} ({total_return:+.1f}%)",
                   xy=(df['date'].iloc[-1], final_balance),
                   xytext=(-120, 15), textcoords='offset points',
                   color=color, fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    def _plot_drawdown(self, results: Dict, ax: plt.Axes) -> None:
        """Plot drawdown over time."""
        _apply_dark_style(ax, title='Drawdown', ylabel='Drawdown (%)')

        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', color=COLORS['text_muted'])
            return

        df = pd.DataFrame(results['daily_values'])
        if len(df) == 0:
            return

        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])

        df['peak'] = df['balance'].cummax()
        df['drawdown'] = (df['balance'] - df['peak']) / df['peak'] * 100

        ax.fill_between(df['date'], 0, df['drawdown'], facecolor=COLORS['loss'], alpha=0.3)
        ax.plot(df['date'], df['drawdown'], color=COLORS['loss'], linewidth=1, alpha=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))

        max_dd = df['drawdown'].min()
        max_dd_date = df.loc[df['drawdown'].idxmin(), 'date']
        ax.annotate(f"Max: {max_dd:.1f}%",
                   xy=(max_dd_date, max_dd),
                   xytext=(10, -20), textcoords='offset points',
                   color=COLORS['loss'], fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=COLORS['loss'], lw=1.2))

    def _plot_monthly_returns(self, results: Dict, ax: plt.Axes) -> None:
        """Plot monthly returns heatmap."""
        ax.set_facecolor(COLORS['panel'])

        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', color=COLORS['text_muted'])
            ax.set_title("Monthly Returns", color=COLORS['text'])
            return

        df = pd.DataFrame(results['daily_values'])
        if len(df) == 0:
            return

        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        monthly_returns = df.groupby(['year', 'month'])['balance'].agg(['first', 'last'])
        monthly_returns['return'] = (monthly_returns['last'] / monthly_returns['first'] - 1) * 100

        pivot_data = []
        for (year, month), row in monthly_returns.iterrows():
            pivot_data.append({'year': year, 'month': month, 'return': row['return']})

        pivot_df = pd.DataFrame(pivot_data)
        if len(pivot_df) == 0:
            return

        pivot_table = pivot_df.pivot(index='year', columns='month', values='return')

        cmap = sns.diverging_palette(10, 150, s=80, l=55, as_cmap=True)
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap=cmap, center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax,
                   annot_kws={'size': 8, 'color': COLORS['text']},
                   linewidths=0.5, linecolor=COLORS['grid'])

        ax.set_title("Monthly Returns (%)", color=COLORS['text'], fontsize=12, fontweight='bold')
        ax.set_ylabel("Year", color=COLORS['text_muted'], fontsize=9)
        ax.set_xlabel("Month", color=COLORS['text_muted'], fontsize=9)
        ax.tick_params(colors=COLORS['text'], labelsize=8)

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names[:len(pivot_table.columns)], rotation=0)

    def _plot_return_distribution(self, results: Dict, ax: plt.Axes) -> None:
        """Plot return distribution."""
        _apply_dark_style(ax, title='Daily Return Distribution', xlabel='Daily Return (%)', ylabel='Frequency')

        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', color=COLORS['text_muted'])
            return

        df = pd.DataFrame(results['daily_values'])
        if len(df) == 0:
            return

        df['daily_return'] = df['balance'].pct_change() * 100
        df = df.dropna(subset=['daily_return'])
        if len(df) == 0:
            return

        returns = df['daily_return'].values
        ax.hist(returns[returns >= 0], bins=25, color=COLORS['profit'], alpha=0.5, edgecolor='none')
        ax.hist(returns[returns < 0], bins=25, color=COLORS['loss'], alpha=0.5, edgecolor='none')

        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(returns)
            x_range = np.linspace(returns.min(), returns.max(), 200)
            kde_vals = kde(x_range)
            bin_width = (returns.max() - returns.min()) / 25
            ax.plot(x_range, kde_vals * len(returns) * bin_width,
                   color=COLORS['accent'], linewidth=1.5, alpha=0.8)
        except Exception:
            pass

        mean = df['daily_return'].mean()
        std = df['daily_return'].std()
        skew = df['daily_return'].skew()
        kurt = df['daily_return'].kurtosis()

        stats_text = f"Mean: {mean:.3f}%\nStd: {std:.3f}%\nSkew: {skew:.2f}\nKurt: {kurt:.2f}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=8, color=COLORS['text'],
               bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['panel'],
                        edgecolor=COLORS['grid'], alpha=0.9))

    def _plot_metrics_table(self, results: Dict, ax: plt.Axes) -> None:
        """Plot key metrics table with professional styling."""
        ax.axis('off')
        ax.set_facecolor(COLORS['bg'])

        metrics_left = [
            ("Initial Balance", f"${results.get('initial_balance', 0):,.2f}"),
            ("Final Balance", f"${results.get('final_balance', 0):,.2f}"),
            ("Total Return", f"{results.get('total_return_pct', 0):+.2f}%"),
            ("Annualized Return", f"{results.get('annualized_return_pct', 0):+.2f}%"),
            ("Ann. Volatility", f"{results.get('annualized_volatility_pct', 0):.2f}%"),
            ("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.2f}%"),
        ]

        metrics_right = [
            ("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}"),
            ("Sortino Ratio", f"{results.get('sortino_ratio', 0):.2f}"),
            ("Calmar Ratio", f"{results.get('calmar_ratio', 0):.2f}"),
            ("Win Rate", f"{results.get('win_rate', 0):.1f}%"),
            ("Profit Factor", f"{results.get('profit_factor', 0):.2f}"),
            ("Total Trades", f"{results.get('trades', 0)}"),
        ]

        all_metrics = []
        for i in range(len(metrics_left)):
            all_metrics.append([
                metrics_left[i][0], metrics_left[i][1],
                metrics_right[i][0], metrics_right[i][1]
            ])

        tbl = ax.table(cellText=all_metrics, loc='center', cellLoc='center',
                      colWidths=[0.2, 0.15, 0.2, 0.15],
                      colLabels=['Metric', 'Value', 'Metric', 'Value'],
                      bbox=[0.05, 0.0, 0.9, 0.95])

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(COLORS['grid'])
            if row == 0:
                cell.set_facecolor(COLORS['table_header'])
                cell.get_text().set_color(COLORS['accent'])
                cell.get_text().set_fontweight('bold')
            elif row % 2 == 0:
                cell.set_facecolor(COLORS['table_row_even'])
                cell.get_text().set_color(COLORS['text'])
            else:
                cell.set_facecolor(COLORS['table_row_odd'])
                cell.get_text().set_color(COLORS['text'])
            cell.set_height(0.12)

        ax.set_title("Performance Metrics", color=COLORS['accent'],
                    fontsize=14, fontweight='bold', pad=10)

    def _plot_trade_analysis(self, results: Dict, ax: plt.Axes) -> None:
        """Plot trade analysis with donut chart."""
        ax.set_facecolor(COLORS['panel'])

        if 'trades' not in results or results.get('trades', 0) == 0:
            ax.text(0.5, 0.5, "No trade data", ha='center', va='center', color=COLORS['text_muted'])
            ax.set_title("Trade Analysis", color=COLORS['text'])
            return

        win_rate = results.get('win_rate', 0)
        loss_rate = 100 - win_rate

        sizes = [win_rate, loss_rate]
        colors = [COLORS['profit'], COLORS['loss']]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=['Win', 'Loss'], autopct='%1.1f%%',
            colors=colors, startangle=90, pctdistance=0.75,
            textprops={'color': COLORS['text'], 'fontsize': 10},
            wedgeprops={'width': 0.4, 'edgecolor': COLORS['panel'], 'linewidth': 2}
        )
        for t in autotexts:
            t.set_fontweight('bold')

        ax.set_title("Trade Outcomes", color=COLORS['text'], fontsize=12, fontweight='bold')

        total_trades = results.get('trades', 0)
        ax.text(0, 0, f"{total_trades}\ntrades", ha='center', va='center',
               fontsize=12, fontweight='bold', color=COLORS['text'])

    def _plot_rolling_sharpe(self, results: Dict, ax: plt.Axes) -> None:
        """Plot rolling 30-day Sharpe ratio."""
        _apply_dark_style(ax, title='Rolling 30-Day Sharpe Ratio', ylabel='Sharpe Ratio')

        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', color=COLORS['text_muted'])
            return

        df = pd.DataFrame(results['daily_values'])
        if len(df) < 31:
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', color=COLORS['text_muted'])
            return

        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])

        df['daily_return'] = df['balance'].pct_change()
        rolling_mean = df['daily_return'].rolling(30).mean() * 252
        rolling_std = df['daily_return'].rolling(30).std() * np.sqrt(252)
        df['rolling_sharpe'] = rolling_mean / rolling_std

        df = df.dropna(subset=['rolling_sharpe'])

        ax.plot(df['date'], df['rolling_sharpe'], color=COLORS['line2'], linewidth=1.2)
        ax.fill_between(df['date'], 0, df['rolling_sharpe'],
                        where=df['rolling_sharpe'] >= 0,
                        facecolor=COLORS['profit'], alpha=0.1)
        ax.fill_between(df['date'], 0, df['rolling_sharpe'],
                        where=df['rolling_sharpe'] < 0,
                        facecolor=COLORS['loss'], alpha=0.1)
        ax.axhline(y=0, color=COLORS['text_muted'], linestyle='--', linewidth=0.5)

    def _plot_cumulative_returns(self, results: Dict, ax: plt.Axes) -> None:
        """Plot cumulative returns percentage over time."""
        _apply_dark_style(ax, title='Cumulative Returns (%)', ylabel='Return (%)')

        if 'daily_values' not in results:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', color=COLORS['text_muted'])
            return

        df = pd.DataFrame(results['daily_values'])
        if len(df) == 0:
            return

        if isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date'])

        initial = results.get('initial_balance', df['balance'].iloc[0])
        df['cum_return'] = (df['balance'] / initial - 1) * 100

        ax.plot(df['date'], df['cum_return'], color=COLORS['line1'], linewidth=1.8, label='Strategy')
        ax.fill_between(df['date'], 0, df['cum_return'],
                        where=df['cum_return'] >= 0,
                        facecolor=COLORS['profit'], alpha=0.08)
        ax.fill_between(df['date'], 0, df['cum_return'],
                        where=df['cum_return'] < 0,
                        facecolor=COLORS['loss'], alpha=0.08)
        ax.axhline(y=0, color=COLORS['text_muted'], linestyle='--', linewidth=0.5)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:+.1f}%'))
        ax.legend(loc='upper left', fontsize=9, facecolor=COLORS['panel'],
                 edgecolor=COLORS['grid'], labelcolor=COLORS['text'])

    def _plot_risk_profile(self, results: Dict, ax: plt.Axes) -> None:
        """Plot risk profile metrics."""
        ax.axis('off')
        ax.set_facecolor(COLORS['bg'])
        ax.set_title("Risk Profile", color=COLORS['accent'], fontsize=14, fontweight='bold')

        vol = results.get('annualized_volatility_pct', 1)
        risk_metrics = [
            ("Risk Level", str(results.get('risk_level', 'N/A'))),
            ("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.2f}%"),
            ("Volatility", f"{vol:.2f}%"),
            ("Sharpe", f"{results.get('sharpe_ratio', 0):.2f}"),
            ("Sortino", f"{results.get('sortino_ratio', 0):.2f}"),
            ("Calmar", f"{results.get('calmar_ratio', 0):.2f}"),
        ]

        cell_text = [[m, v] for m, v in risk_metrics]
        tbl = ax.table(cellText=cell_text, loc='center', cellLoc='left',
                      colWidths=[0.3, 0.25], bbox=[0.1, 0.1, 0.8, 0.75])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(COLORS['grid'])
            if row % 2 == 0:
                cell.set_facecolor(COLORS['table_row_even'])
            else:
                cell.set_facecolor(COLORS['table_row_odd'])
            cell.get_text().set_color(COLORS['text'])

    # --- Comparison report plots ---

    def _plot_returns_by_risk(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot returns by risk level."""
        _apply_dark_style(ax, title='Total Return by Risk Level', xlabel='Risk Level', ylabel='Total Return (%)')

        if not comparison.get('risk_levels'):
            return

        bars = ax.bar(comparison['risk_levels'], comparison['total_returns'],
                     color=COLORS['line1'], alpha=0.8, edgecolor=COLORS['accent'], linewidth=0.5)

        optimal = comparison.get('optimal_risk_level')
        if optimal in comparison['risk_levels']:
            idx = comparison['risk_levels'].index(optimal)
            bars[idx].set_color(COLORS['profit'])
            bars[idx].set_edgecolor(COLORS['profit'])

        ax.set_xticks(comparison['risk_levels'])
        for i, v in enumerate(comparison['total_returns']):
            color = COLORS['profit'] if v >= 0 else COLORS['loss']
            ax.text(comparison['risk_levels'][i], v + 0.5, f"{v:.1f}%",
                   ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

    def _plot_sharpe_by_risk(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot Sharpe ratio by risk level."""
        _apply_dark_style(ax, title='Sharpe Ratio by Risk Level', xlabel='Risk Level', ylabel='Sharpe Ratio')

        if not comparison.get('risk_levels'):
            return

        bars = ax.bar(comparison['risk_levels'], comparison['sharpe_ratios'],
                     color=COLORS['line2'], alpha=0.8, edgecolor=COLORS['line2'], linewidth=0.5)

        optimal = comparison.get('optimal_risk_level')
        if optimal in comparison['risk_levels']:
            idx = comparison['risk_levels'].index(optimal)
            bars[idx].set_color(COLORS['profit'])

        ax.set_xticks(comparison['risk_levels'])
        for i, v in enumerate(comparison['sharpe_ratios']):
            ax.text(comparison['risk_levels'][i], v + 0.02, f"{v:.2f}",
                   ha='center', va='bottom', fontsize=8, color=COLORS['text'])

    def _plot_drawdown_by_risk(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot maximum drawdown by risk level."""
        _apply_dark_style(ax, title='Max Drawdown by Risk Level', xlabel='Risk Level', ylabel='Max Drawdown (%)')

        if not comparison.get('risk_levels'):
            return

        ax.bar(comparison['risk_levels'], comparison['max_drawdowns'],
              color=COLORS['loss'], alpha=0.7, edgecolor=COLORS['loss'], linewidth=0.5)

        ax.set_xticks(comparison['risk_levels'])
        for i, v in enumerate(comparison['max_drawdowns']):
            ax.text(comparison['risk_levels'][i], v - 0.5, f"{v:.1f}%",
                   ha='center', va='top', fontsize=8, color=COLORS['text'])

    def _plot_win_rate_by_risk(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot win rate by risk level."""
        _apply_dark_style(ax, title='Win Rate by Risk Level', xlabel='Risk Level', ylabel='Win Rate (%)')

        if not comparison.get('risk_levels'):
            return

        ax.bar(comparison['risk_levels'], comparison['win_rates'],
              color=COLORS['profit'], alpha=0.7, edgecolor=COLORS['profit'], linewidth=0.5)

        ax.set_xticks(comparison['risk_levels'])
        ax.set_ylim(0, 100)
        for i, v in enumerate(comparison['win_rates']):
            ax.text(comparison['risk_levels'][i], v + 1, f"{v:.1f}%",
                   ha='center', va='bottom', fontsize=8, color=COLORS['text'])

    def _plot_balance_curves_comparison(self, results_by_risk: Dict[int, Dict], ax: plt.Axes) -> None:
        """Plot balance curves for all risk levels."""
        _apply_dark_style(ax, title='Equity Curves by Risk Level', ylabel='Balance ($)')

        if not results_by_risk:
            return

        palette = [COLORS['line1'], COLORS['line2'], COLORS['line3'],
                  COLORS['line4'], COLORS['line5'], COLORS['accent'],
                  COLORS['profit'], COLORS['warn'], COLORS['loss'], '#ffffff']

        for i, (risk_level, results) in enumerate(sorted(results_by_risk.items())):
            if 'daily_values' not in results:
                continue

            df = pd.DataFrame(results['daily_values'])
            if len(df) == 0:
                continue

            if isinstance(df['date'].iloc[0], str):
                df['date'] = pd.to_datetime(df['date'])

            color = palette[i % len(palette)]
            ax.plot(df['date'], df['balance'], label=f"Risk {risk_level}",
                   color=color, linewidth=1.2, alpha=0.85)

        initial = next(iter(results_by_risk.values())).get('initial_balance', 100000)
        ax.axhline(y=initial, color=COLORS['text_muted'], linestyle='--', alpha=0.4, linewidth=0.8)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.legend(loc='upper left', fontsize=8, ncol=2,
                 facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
                 labelcolor=COLORS['text'])

    def _plot_risk_return_scatter(self, comparison: Dict, ax: plt.Axes) -> None:
        """Plot risk-return scatter plot."""
        _apply_dark_style(ax, title='Risk-Return Profile', xlabel='Volatility (%)', ylabel='Ann. Return (%)')

        if not comparison.get('risk_levels'):
            return

        sc = ax.scatter(comparison['volatility'], comparison['annualized_returns'],
                       c=comparison['risk_levels'], cmap='cool',
                       s=120, alpha=0.85, edgecolors=COLORS['text'], linewidth=0.5, zorder=3)

        for i, rl in enumerate(comparison['risk_levels']):
            ax.annotate(f"R{rl}",
                       (comparison['volatility'][i], comparison['annualized_returns'][i]),
                       xytext=(6, 6), textcoords='offset points',
                       color=COLORS['text'], fontsize=8, fontweight='bold')

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Risk Level', color=COLORS['text_muted'], fontsize=9)
        cbar.ax.tick_params(colors=COLORS['text'], labelsize=8)

    def _plot_comparison_table(self, comparison: Dict, results_by_risk: Dict, ax: plt.Axes) -> None:
        """Plot comparison summary table."""
        ax.axis('off')
        ax.set_facecolor(COLORS['bg'])

        optimal = comparison.get('optimal_risk_level', 'N/A')
        best_ret = comparison.get('best_return_risk_level', 'N/A')

        rows = [
            ["Best Sharpe (Risk Level)", str(optimal)],
            ["Best Return (Risk Level)", str(best_ret)],
            ["Return Range", f"{min(comparison['total_returns']):.1f}% to {max(comparison['total_returns']):.1f}%"],
            ["Volatility Range", f"{min(comparison['volatility']):.1f}% to {max(comparison['volatility']):.1f}%"],
            ["Drawdown Range", f"{min(comparison['max_drawdowns']):.1f}% to {max(comparison['max_drawdowns']):.1f}%"],
            ["Win Rate Range", f"{min(comparison['win_rates']):.1f}% to {max(comparison['win_rates']):.1f}%"],
        ]

        tbl = ax.table(cellText=rows, loc='center', cellLoc='left',
                      colWidths=[0.35, 0.3],
                      colLabels=['Metric', 'Value'],
                      bbox=[0.05, 0.05, 0.9, 0.85])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(COLORS['grid'])
            if row == 0:
                cell.set_facecolor(COLORS['table_header'])
                cell.get_text().set_color(COLORS['accent'])
                cell.get_text().set_fontweight('bold')
            elif row == 1:
                cell.set_facecolor('#0d3b2e')
                cell.get_text().set_color(COLORS['profit'])
                cell.get_text().set_fontweight('bold')
            elif row % 2 == 0:
                cell.set_facecolor(COLORS['table_row_even'])
                cell.get_text().set_color(COLORS['text'])
            else:
                cell.set_facecolor(COLORS['table_row_odd'])
                cell.get_text().set_color(COLORS['text'])

        ax.set_title("Summary", color=COLORS['accent'], fontsize=14, fontweight='bold', pad=10)
