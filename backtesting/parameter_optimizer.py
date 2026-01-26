"""
Parameter optimization for the TuringTrader algorithm.
Provides grid search, single parameter sweeps, and optimization utilities
to find the best parameter configurations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from itertools import product
from dataclasses import dataclass, field
import json

from backtesting.backtest_engine import BacktestEngine
from backtesting.realistic_mock_data import RealisticMockDataFetcher
from ibkr_trader.config import Config, RiskParameters, VolatilityHarvestingConfig


@dataclass
class ParameterRange:
    """Defines the range for a parameter to optimize."""
    name: str
    min_value: float
    max_value: float
    step: float
    config_path: str  # e.g., "vol_harvesting.iv_hv_ratio_threshold"

    def get_values(self) -> List[float]:
        """Generate list of values to test."""
        values = []
        current = self.min_value
        while current <= self.max_value + 0.0001:  # Small epsilon for float comparison
            values.append(round(current, 4))
            current += self.step
        return values


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""
    best_params: Dict[str, float]
    best_value: float
    objective: str
    all_results: List[Dict[str, Any]]
    param_importance: Dict[str, float] = field(default_factory=dict)


# Default parameter ranges for optimization
DEFAULT_PARAMETER_RANGES = {
    'iv_hv_ratio_threshold': ParameterRange(
        name='iv_hv_ratio_threshold',
        min_value=1.2,
        max_value=2.0,
        step=0.1,
        config_path='vol_harvesting.iv_hv_ratio_threshold'
    ),
    'min_iv_threshold': ParameterRange(
        name='min_iv_threshold',
        min_value=25.0,
        max_value=40.0,
        step=2.5,
        config_path='vol_harvesting.min_iv_threshold'
    ),
    'strike_width_pct': ParameterRange(
        name='strike_width_pct',
        min_value=5.0,
        max_value=12.0,
        step=1.0,
        config_path='vol_harvesting.strike_width_pct'
    ),
    'target_short_delta': ParameterRange(
        name='target_short_delta',
        min_value=0.15,
        max_value=0.30,
        step=0.02,
        config_path='vol_harvesting.target_short_delta'
    ),
    'min_dte': ParameterRange(
        name='min_dte',
        min_value=14,
        max_value=28,
        step=2,
        config_path='vol_harvesting.min_dte'
    ),
    'max_dte': ParameterRange(
        name='max_dte',
        min_value=28,
        max_value=45,
        step=3,
        config_path='vol_harvesting.max_dte'
    ),
}


class ParameterOptimizer:
    """
    Optimizes trading strategy parameters using grid search
    and sensitivity analysis.
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        start_date: str = "2023-01-01",
        end_date: str = "2023-06-30",
        risk_level: int = 5,
        seed: Optional[int] = 42
    ):
        """
        Initialize the parameter optimizer.

        Args:
            initial_balance: Initial account balance
            start_date: Backtest start date
            end_date: Backtest end date
            risk_level: Base risk level for optimization
            seed: Random seed for reproducibility
        """
        self.initial_balance = initial_balance
        self.start_date = start_date
        self.end_date = end_date
        self.risk_level = risk_level
        self.seed = seed
        self.logger = logging.getLogger(__name__)

        # Results cache
        self._results_cache: Dict[str, Dict[str, Any]] = {}

    def _set_param_value(self, config: Config, param_path: str, value: float) -> None:
        """
        Set a parameter value on the config object.

        Args:
            config: Config object to modify
            param_path: Dot-separated path (e.g., "vol_harvesting.min_iv_threshold")
            value: Value to set
        """
        parts = param_path.split('.')
        obj = config

        # Navigate to the parent object
        for part in parts[:-1]:
            obj = getattr(obj, part)

        # Set the final attribute
        setattr(obj, parts[-1], value)

    def _get_cache_key(self, params: Dict[str, float]) -> str:
        """Generate cache key from parameters."""
        sorted_items = sorted(params.items())
        return json.dumps(sorted_items)

    def _run_backtest_with_params(
        self,
        params: Dict[str, float],
        param_ranges: Dict[str, ParameterRange]
    ) -> Dict[str, Any]:
        """
        Run a backtest with specific parameter values.

        Args:
            params: Parameter name -> value mapping
            param_ranges: Parameter range definitions

        Returns:
            Backtest results dict
        """
        cache_key = self._get_cache_key(params)
        if cache_key in self._results_cache:
            return self._results_cache[cache_key]

        # Create fresh config and set parameters
        config = Config()
        config.risk.adjust_for_risk_level(self.risk_level)

        for param_name, value in params.items():
            if param_name in param_ranges:
                param_path = param_ranges[param_name].config_path
                self._set_param_value(config, param_path, value)

        # Create data fetcher
        data_fetcher = RealisticMockDataFetcher()
        if self.seed is not None:
            np.random.seed(self.seed)

        # Run backtest
        engine = BacktestEngine(
            config=config,
            initial_balance=self.initial_balance,
            data_fetcher=data_fetcher
        )

        result = engine.run_backtest(
            start_date=self.start_date,
            end_date=self.end_date,
            risk_level=self.risk_level,
            use_cache=False
        )

        # Add params to result
        result['params'] = params

        self._results_cache[cache_key] = result
        return result

    def grid_search(
        self,
        param_ranges: Optional[Dict[str, ParameterRange]] = None,
        objective: str = 'sharpe_ratio',
        params_to_optimize: Optional[List[str]] = None
    ) -> OptimizationResult:
        """
        Perform exhaustive grid search over parameter combinations.

        Args:
            param_ranges: Parameter ranges to search (None for defaults)
            objective: Metric to optimize
            params_to_optimize: List of param names to optimize (None for all)

        Returns:
            OptimizationResult with best parameters
        """
        if param_ranges is None:
            param_ranges = DEFAULT_PARAMETER_RANGES

        if params_to_optimize is None:
            params_to_optimize = list(param_ranges.keys())

        # Filter to only requested parameters
        ranges_to_use = {k: v for k, v in param_ranges.items()
                        if k in params_to_optimize}

        # Generate all combinations
        param_names = list(ranges_to_use.keys())
        param_values = [ranges_to_use[name].get_values() for name in param_names]

        total_combinations = np.prod([len(v) for v in param_values])
        self.logger.info(f"Grid search: {total_combinations} combinations for {param_names}")

        all_results = []
        best_value = float('-inf')
        best_params = {}

        for i, values in enumerate(product(*param_values)):
            params = dict(zip(param_names, values))

            if (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {i + 1}/{total_combinations}")

            result = self._run_backtest_with_params(params, ranges_to_use)

            if 'error' in result:
                continue

            objective_value = result.get(objective, 0)

            result_record = {
                **params,
                'objective_value': objective_value,
                'total_return_pct': result.get('total_return_pct', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                'win_rate': result.get('win_rate', 0),
            }
            all_results.append(result_record)

            if objective_value > best_value:
                best_value = objective_value
                best_params = params.copy()

        # Calculate parameter importance
        param_importance = self._calculate_param_importance(
            all_results, param_names, objective
        )

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            objective=objective,
            all_results=all_results,
            param_importance=param_importance
        )

    def single_param_sweep(
        self,
        param_name: str,
        param_range: Optional[ParameterRange] = None,
        base_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Sweep a single parameter while holding others constant.

        Args:
            param_name: Name of parameter to sweep
            param_range: Range for the parameter (None for default)
            base_params: Base parameter values (None for defaults)

        Returns:
            Dict with sweep results
        """
        if param_range is None:
            param_range = DEFAULT_PARAMETER_RANGES.get(param_name)
            if param_range is None:
                raise ValueError(f"Unknown parameter: {param_name}")

        if base_params is None:
            # Use default values from config
            config = Config()
            base_params = {}

        results = []
        values = param_range.get_values()

        for value in values:
            params = base_params.copy()
            params[param_name] = value

            result = self._run_backtest_with_params(params, {param_name: param_range})

            if 'error' not in result:
                results.append({
                    'value': value,
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'total_return_pct': result.get('total_return_pct', 0),
                    'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                    'win_rate': result.get('win_rate', 0),
                })

        # Find optimal value for each metric
        df = pd.DataFrame(results)

        if df.empty:
            return {'param_name': param_name, 'error': 'No valid results'}

        return {
            'param_name': param_name,
            'values': values,
            'results': results,
            'optimal': {
                'sharpe_ratio': {
                    'value': df.loc[df['sharpe_ratio'].idxmax(), 'value'],
                    'metric': df['sharpe_ratio'].max()
                },
                'total_return': {
                    'value': df.loc[df['total_return_pct'].idxmax(), 'value'],
                    'metric': df['total_return_pct'].max()
                },
                'min_drawdown': {
                    'value': df.loc[df['max_drawdown_pct'].idxmin(), 'value'],
                    'metric': df['max_drawdown_pct'].min()
                }
            },
            'sensitivity': {
                'sharpe_std': df['sharpe_ratio'].std(),
                'return_std': df['total_return_pct'].std(),
                'drawdown_std': df['max_drawdown_pct'].std(),
            }
        }

    def find_optimal_params(
        self,
        params_to_optimize: Optional[List[str]] = None,
        objective: str = 'sharpe_ratio'
    ) -> Dict[str, float]:
        """
        Find optimal parameter configuration using grid search.

        Args:
            params_to_optimize: Parameters to optimize (None for key params)
            objective: Optimization objective

        Returns:
            Dict of optimal parameter values
        """
        if params_to_optimize is None:
            # Default to most impactful parameters
            params_to_optimize = [
                'iv_hv_ratio_threshold',
                'min_iv_threshold',
                'strike_width_pct',
                'target_short_delta'
            ]

        result = self.grid_search(
            params_to_optimize=params_to_optimize,
            objective=objective
        )

        self.logger.info(f"Optimal params (objective={objective}): {result.best_params}")
        self.logger.info(f"Best {objective}: {result.best_value:.4f}")

        return result.best_params

    def _calculate_param_importance(
        self,
        results: List[Dict],
        param_names: List[str],
        objective: str
    ) -> Dict[str, float]:
        """
        Calculate importance of each parameter based on results variance.

        Args:
            results: List of result dictionaries
            param_names: Parameter names
            objective: Objective metric name

        Returns:
            Dict mapping param name to importance score
        """
        if not results:
            return {}

        df = pd.DataFrame(results)
        importance = {}

        for param_name in param_names:
            # Group by parameter value and calculate variance explained
            grouped = df.groupby(param_name)['objective_value'].mean()
            param_variance = grouped.var()

            total_variance = df['objective_value'].var()

            if total_variance > 0:
                importance[param_name] = param_variance / total_variance
            else:
                importance[param_name] = 0.0

        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        return importance

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get all cached results as a DataFrame.

        Returns:
            DataFrame with all optimization results
        """
        records = []
        for result in self._results_cache.values():
            if 'error' not in result:
                record = {
                    **result.get('params', {}),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'total_return_pct': result.get('total_return_pct', 0),
                    'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                    'win_rate': result.get('win_rate', 0),
                    'trades': result.get('trades', 0),
                }
                records.append(record)

        return pd.DataFrame(records)

    def export_results(self, filepath: str) -> None:
        """Export optimization results to CSV."""
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False)
        self.logger.info(f"Exported {len(df)} results to {filepath}")

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()


def quick_optimize(
    objective: str = 'sharpe_ratio',
    risk_level: int = 5
) -> Dict[str, float]:
    """
    Quick optimization of key parameters.

    Args:
        objective: Metric to optimize
        risk_level: Risk level for backtest

    Returns:
        Optimal parameter values
    """
    optimizer = ParameterOptimizer(risk_level=risk_level)
    return optimizer.find_optimal_params(objective=objective)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run parameter optimization
    optimizer = ParameterOptimizer()

    # Single parameter sweeps
    print("\n=== Single Parameter Sweeps ===")
    for param_name in ['iv_hv_ratio_threshold', 'min_iv_threshold']:
        result = optimizer.single_param_sweep(param_name)
        print(f"\n{param_name}:")
        print(f"  Optimal for Sharpe: {result['optimal']['sharpe_ratio']['value']}")
        print(f"  Optimal for Return: {result['optimal']['total_return']['value']}")
        print(f"  Sensitivity (Sharpe std): {result['sensitivity']['sharpe_std']:.4f}")

    # Grid search
    print("\n=== Grid Search ===")
    opt_result = optimizer.grid_search(
        params_to_optimize=['iv_hv_ratio_threshold', 'min_iv_threshold'],
        objective='sharpe_ratio'
    )
    print(f"Best params: {opt_result.best_params}")
    print(f"Best Sharpe: {opt_result.best_value:.4f}")
    print(f"Parameter importance: {opt_result.param_importance}")
