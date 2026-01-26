"""
Multi-scenario backtesting framework.
Runs backtests across multiple market scenarios and risk levels,
generating comparison reports and analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from dataclasses import dataclass, asdict

from backtesting.backtest_engine import BacktestEngine
from backtesting.market_scenarios import (
    MarketScenario,
    ScenarioDataFetcher,
    PREDEFINED_SCENARIOS,
    list_scenarios,
    get_scenario,
)
from ibkr_trader.config import Config


@dataclass
class ScenarioResult:
    """Results from a single scenario backtest."""
    scenario_name: str
    risk_level: int
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    final_balance: float
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class MultiScenarioBacktest:
    """
    Run backtests across multiple market scenarios and risk levels.
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        start_date: str = "2023-01-01",
        end_date: str = "2023-06-30",
        seed: Optional[int] = 42
    ):
        """
        Initialize the multi-scenario backtest runner.

        Args:
            initial_balance: Initial account balance
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            seed: Random seed for reproducibility
        """
        self.initial_balance = initial_balance
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed
        self.logger = logging.getLogger(__name__)

        # Cache for results
        self._results_cache: Dict[Tuple[str, int], ScenarioResult] = {}

    def run_scenario(
        self,
        scenario: MarketScenario,
        risk_level: int
    ) -> ScenarioResult:
        """
        Run a backtest for a single scenario and risk level.

        Args:
            scenario: MarketScenario to test
            risk_level: Risk level (1-10)

        Returns:
            ScenarioResult with backtest metrics
        """
        cache_key = (scenario.name, risk_level)
        if cache_key in self._results_cache:
            return self._results_cache[cache_key]

        self.logger.info(f"Running backtest: {scenario.name} @ risk level {risk_level}")

        try:
            # Create scenario-based data fetcher
            data_fetcher = ScenarioDataFetcher(
                scenario=scenario,
                seed=self.seed
            )

            # Create config with risk level
            config = Config()
            config.risk.adjust_for_risk_level(risk_level)

            # Create and run backtest
            engine = BacktestEngine(
                config=config,
                initial_balance=self.initial_balance,
                data_fetcher=data_fetcher
            )

            results = engine.run_backtest(
                start_date=self.start_date,
                end_date=self.end_date,
                risk_level=risk_level,
                use_cache=False
            )

            if 'error' in results:
                result = ScenarioResult(
                    scenario_name=scenario.name,
                    risk_level=risk_level,
                    total_return_pct=0,
                    annualized_return_pct=0,
                    annualized_volatility_pct=0,
                    sharpe_ratio=0,
                    max_drawdown_pct=0,
                    win_rate=0,
                    profit_factor=0,
                    total_trades=0,
                    final_balance=self.initial_balance,
                    error=results['error']
                )
            else:
                result = ScenarioResult(
                    scenario_name=scenario.name,
                    risk_level=risk_level,
                    total_return_pct=results.get('total_return_pct', 0),
                    annualized_return_pct=results.get('annualized_return_pct', 0),
                    annualized_volatility_pct=results.get('annualized_volatility_pct', 0),
                    sharpe_ratio=results.get('sharpe_ratio', 0),
                    max_drawdown_pct=results.get('max_drawdown_pct', 0),
                    win_rate=results.get('win_rate', 0),
                    profit_factor=results.get('profit_factor', 0),
                    total_trades=results.get('trades', 0),
                    final_balance=results.get('final_balance', self.initial_balance)
                )

            self._results_cache[cache_key] = result
            return result

        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            return ScenarioResult(
                scenario_name=scenario.name,
                risk_level=risk_level,
                total_return_pct=0,
                annualized_return_pct=0,
                annualized_volatility_pct=0,
                sharpe_ratio=0,
                max_drawdown_pct=0,
                win_rate=0,
                profit_factor=0,
                total_trades=0,
                final_balance=self.initial_balance,
                error=str(e)
            )

    def run_all_scenarios(self, risk_level: int = 5) -> Dict[str, ScenarioResult]:
        """
        Run backtest across all predefined scenarios at a given risk level.

        Args:
            risk_level: Risk level to test (1-10)

        Returns:
            Dict mapping scenario names to ScenarioResult
        """
        results = {}

        for scenario_name in list_scenarios():
            scenario = get_scenario(scenario_name)
            result = self.run_scenario(scenario, risk_level)
            results[scenario_name] = result

        return results

    def run_scenario_risk_matrix(
        self,
        scenarios: Optional[List[str]] = None,
        risk_levels: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Run backtests for a matrix of scenarios x risk levels.

        Args:
            scenarios: List of scenario names (None for all)
            risk_levels: List of risk levels (None for 1-10)

        Returns:
            DataFrame with results matrix
        """
        if scenarios is None:
            scenarios = list_scenarios()
        if risk_levels is None:
            risk_levels = list(range(1, 11))

        results = []

        total_runs = len(scenarios) * len(risk_levels)
        current_run = 0

        for scenario_name in scenarios:
            scenario = get_scenario(scenario_name)

            for risk_level in risk_levels:
                current_run += 1
                self.logger.info(f"Progress: {current_run}/{total_runs}")

                result = self.run_scenario(scenario, risk_level)
                results.append(result.to_dict())

        df = pd.DataFrame(results)
        return df

    def generate_comparison_report(
        self,
        results: Optional[Dict[str, ScenarioResult]] = None,
        risk_level: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a comparison report across scenarios.

        Args:
            results: Pre-computed results (None to run fresh)
            risk_level: Risk level if running fresh

        Returns:
            Dict with comparison metrics and analysis
        """
        if results is None:
            results = self.run_all_scenarios(risk_level)

        # Convert to DataFrame for analysis
        df = pd.DataFrame([r.to_dict() for r in results.values()])

        # Calculate summary statistics
        report = {
            'risk_level': risk_level,
            'num_scenarios': len(results),
            'summary': {
                'best_return_scenario': df.loc[df['total_return_pct'].idxmax(), 'scenario_name'],
                'worst_return_scenario': df.loc[df['total_return_pct'].idxmin(), 'scenario_name'],
                'best_sharpe_scenario': df.loc[df['sharpe_ratio'].idxmax(), 'scenario_name'],
                'lowest_drawdown_scenario': df.loc[df['max_drawdown_pct'].idxmin(), 'scenario_name'],
                'avg_return': df['total_return_pct'].mean(),
                'avg_sharpe': df['sharpe_ratio'].mean(),
                'avg_drawdown': df['max_drawdown_pct'].mean(),
                'avg_win_rate': df['win_rate'].mean(),
            },
            'by_scenario': {
                row['scenario_name']: {
                    'return': row['total_return_pct'],
                    'sharpe': row['sharpe_ratio'],
                    'drawdown': row['max_drawdown_pct'],
                    'win_rate': row['win_rate'],
                    'trades': row['total_trades'],
                }
                for _, row in df.iterrows()
            },
            'rankings': {
                'by_return': df.sort_values('total_return_pct', ascending=False)['scenario_name'].tolist(),
                'by_sharpe': df.sort_values('sharpe_ratio', ascending=False)['scenario_name'].tolist(),
                'by_drawdown': df.sort_values('max_drawdown_pct', ascending=True)['scenario_name'].tolist(),
            }
        }

        return report

    def find_optimal_risk_level(
        self,
        scenario_name: str,
        objective: str = 'sharpe_ratio'
    ) -> Tuple[int, ScenarioResult]:
        """
        Find the optimal risk level for a given scenario.

        Args:
            scenario_name: Name of scenario to optimize
            objective: Metric to optimize ('sharpe_ratio', 'total_return_pct', etc.)

        Returns:
            Tuple of (optimal_risk_level, result)
        """
        scenario = get_scenario(scenario_name)

        best_level = 1
        best_result = None
        best_value = float('-inf')

        for risk_level in range(1, 11):
            result = self.run_scenario(scenario, risk_level)

            if result.error:
                continue

            value = getattr(result, objective, 0)

            # For drawdown, lower is better
            if objective == 'max_drawdown_pct':
                value = -value

            if value > best_value:
                best_value = value
                best_level = risk_level
                best_result = result

        return best_level, best_result

    def generate_risk_analysis(self, scenario_name: str) -> Dict[str, Any]:
        """
        Generate detailed risk analysis for a scenario across all risk levels.

        Args:
            scenario_name: Scenario to analyze

        Returns:
            Dict with risk analysis
        """
        scenario = get_scenario(scenario_name)

        results_by_level = {}
        for risk_level in range(1, 11):
            result = self.run_scenario(scenario, risk_level)
            results_by_level[risk_level] = result

        # Find optimal levels for different objectives
        optimal_sharpe = max(results_by_level.items(),
                           key=lambda x: x[1].sharpe_ratio if not x[1].error else float('-inf'))
        optimal_return = max(results_by_level.items(),
                            key=lambda x: x[1].total_return_pct if not x[1].error else float('-inf'))
        optimal_drawdown = min(results_by_level.items(),
                              key=lambda x: x[1].max_drawdown_pct if not x[1].error else float('inf'))

        return {
            'scenario': scenario_name,
            'optimal_levels': {
                'sharpe_ratio': optimal_sharpe[0],
                'return': optimal_return[0],
                'drawdown': optimal_drawdown[0],
            },
            'risk_return_tradeoff': {
                level: {
                    'return': r.total_return_pct,
                    'volatility': r.annualized_volatility_pct,
                    'sharpe': r.sharpe_ratio,
                    'drawdown': r.max_drawdown_pct,
                }
                for level, r in results_by_level.items() if not r.error
            },
            'recommendation': self._generate_recommendation(results_by_level)
        }

    def _generate_recommendation(
        self,
        results_by_level: Dict[int, ScenarioResult]
    ) -> str:
        """Generate a risk level recommendation based on results."""
        valid_results = {k: v for k, v in results_by_level.items() if not v.error}

        if not valid_results:
            return "Unable to generate recommendation - no valid results"

        # Find level with best risk-adjusted return
        sharpe_by_level = {k: v.sharpe_ratio for k, v in valid_results.items()}
        best_sharpe_level = max(sharpe_by_level, key=sharpe_by_level.get)

        # Find level with acceptable drawdown
        acceptable_dd_levels = [k for k, v in valid_results.items()
                               if v.max_drawdown_pct <= 15]

        if acceptable_dd_levels:
            recommended = max(acceptable_dd_levels,
                            key=lambda k: valid_results[k].sharpe_ratio)
        else:
            recommended = best_sharpe_level

        return (
            f"Recommended risk level: {recommended}. "
            f"Best Sharpe at level {best_sharpe_level} "
            f"({sharpe_by_level[best_sharpe_level]:.2f}). "
            f"Levels with <15% drawdown: {acceptable_dd_levels or 'None'}"
        )

    def export_results(self, filepath: str, format: str = 'json') -> None:
        """
        Export cached results to file.

        Args:
            filepath: Output file path
            format: 'json' or 'csv'
        """
        results_list = [r.to_dict() for r in self._results_cache.values()]

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(results_list, f, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(results_list)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        self.logger.info(f"Exported {len(results_list)} results to {filepath}")

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()


def run_quick_comparison(risk_level: int = 5) -> Dict[str, Any]:
    """
    Convenience function to run a quick comparison across all scenarios.

    Args:
        risk_level: Risk level to test

    Returns:
        Comparison report
    """
    runner = MultiScenarioBacktest()
    return runner.generate_comparison_report(risk_level=risk_level)


def run_full_matrix() -> pd.DataFrame:
    """
    Run full scenario x risk level matrix.

    Returns:
        DataFrame with all results
    """
    runner = MultiScenarioBacktest()
    return runner.run_scenario_risk_matrix()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run example comparison
    runner = MultiScenarioBacktest()

    print("\n=== Running all scenarios at risk level 5 ===")
    results = runner.run_all_scenarios(risk_level=5)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Return: {result.total_return_pct:.2f}%")
        print(f"  Sharpe: {result.sharpe_ratio:.2f}")
        print(f"  Max DD: {result.max_drawdown_pct:.2f}%")
        print(f"  Trades: {result.total_trades}")

    # Generate comparison report
    report = runner.generate_comparison_report(results)
    print("\n=== Comparison Report ===")
    print(f"Best return: {report['summary']['best_return_scenario']}")
    print(f"Best Sharpe: {report['summary']['best_sharpe_scenario']}")
    print(f"Lowest DD: {report['summary']['lowest_drawdown_scenario']}")
