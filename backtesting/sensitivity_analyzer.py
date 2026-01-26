"""
Sensitivity analysis for trading strategy parameters.
Provides tools for understanding parameter impact, generating heatmaps,
and calculating robustness scores.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from backtesting.parameter_optimizer import (
    ParameterOptimizer,
    ParameterRange,
    DEFAULT_PARAMETER_RANGES,
)
from backtesting.backtest_engine import BacktestEngine
from backtesting.realistic_mock_data import RealisticMockDataFetcher
from ibkr_trader.config import Config


@dataclass
class SensitivityReport:
    """Report from sensitivity analysis."""
    param_name: str
    values_tested: List[float]
    metric_values: Dict[str, List[float]]
    optimal_value: float
    sensitivity_score: float
    robustness_zone: Tuple[float, float]
    recommendations: List[str]


@dataclass
class HeatmapData:
    """Data for a 2D parameter interaction heatmap."""
    param1_name: str
    param2_name: str
    param1_values: List[float]
    param2_values: List[float]
    metric_values: List[List[float]]
    optimal_params: Dict[str, float]
    interaction_strength: float


class SensitivityAnalyzer:
    """
    Analyzes parameter sensitivity and interactions for the trading strategy.
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
        Initialize the sensitivity analyzer.

        Args:
            initial_balance: Initial account balance
            start_date: Backtest start date
            end_date: Backtest end date
            risk_level: Base risk level
            seed: Random seed for reproducibility
        """
        self.initial_balance = initial_balance
        self.start_date = start_date
        self.end_date = end_date
        self.risk_level = risk_level
        self.seed = seed
        self.logger = logging.getLogger(__name__)

        # Initialize optimizer for running backtests
        self.optimizer = ParameterOptimizer(
            initial_balance=initial_balance,
            start_date=start_date,
            end_date=end_date,
            risk_level=risk_level,
            seed=seed
        )

    def generate_sensitivity_report(
        self,
        param_name: str,
        param_range: Optional[ParameterRange] = None,
        metrics: Optional[List[str]] = None
    ) -> SensitivityReport:
        """
        Generate a detailed sensitivity report for a single parameter.

        Args:
            param_name: Name of parameter to analyze
            param_range: Range for parameter (None for default)
            metrics: Metrics to analyze (None for defaults)

        Returns:
            SensitivityReport with analysis results
        """
        if param_range is None:
            param_range = DEFAULT_PARAMETER_RANGES.get(param_name)
            if param_range is None:
                raise ValueError(f"Unknown parameter: {param_name}")

        if metrics is None:
            metrics = ['sharpe_ratio', 'total_return_pct', 'max_drawdown_pct', 'win_rate']

        # Run parameter sweep
        sweep_result = self.optimizer.single_param_sweep(param_name, param_range)

        if 'error' in sweep_result:
            raise ValueError(f"Sweep failed: {sweep_result['error']}")

        values = sweep_result['values']
        results = sweep_result['results']

        # Extract metric values
        metric_values = {metric: [] for metric in metrics}
        for result in results:
            for metric in metrics:
                metric_values[metric].append(result.get(metric, 0))

        # Calculate sensitivity score (normalized variance)
        sharpe_values = metric_values.get('sharpe_ratio', [])
        if sharpe_values and np.std(sharpe_values) > 0:
            sensitivity_score = np.std(sharpe_values) / (np.mean(sharpe_values) + 0.001)
        else:
            sensitivity_score = 0.0

        # Find optimal value (for Sharpe ratio)
        if sharpe_values:
            optimal_idx = np.argmax(sharpe_values)
            optimal_value = values[optimal_idx]
        else:
            optimal_value = values[len(values) // 2]

        # Calculate robustness zone (values within 90% of optimal)
        robustness_zone = self._calculate_robustness_zone(
            values, sharpe_values, optimal_value, threshold=0.9
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            param_name, optimal_value, sensitivity_score, robustness_zone, sweep_result
        )

        return SensitivityReport(
            param_name=param_name,
            values_tested=values,
            metric_values=metric_values,
            optimal_value=optimal_value,
            sensitivity_score=sensitivity_score,
            robustness_zone=robustness_zone,
            recommendations=recommendations
        )

    def _calculate_robustness_zone(
        self,
        values: List[float],
        metric_values: List[float],
        optimal_value: float,
        threshold: float = 0.9
    ) -> Tuple[float, float]:
        """
        Calculate the range of values that achieve near-optimal performance.

        Args:
            values: Parameter values tested
            metric_values: Corresponding metric values
            optimal_value: The optimal parameter value
            threshold: Fraction of optimal performance to consider "robust"

        Returns:
            Tuple of (min_value, max_value) for robust zone
        """
        if not metric_values:
            return (optimal_value, optimal_value)

        max_metric = max(metric_values)
        threshold_metric = max_metric * threshold

        robust_values = [v for v, m in zip(values, metric_values)
                        if m >= threshold_metric]

        if not robust_values:
            return (optimal_value, optimal_value)

        return (min(robust_values), max(robust_values))

    def _generate_recommendations(
        self,
        param_name: str,
        optimal_value: float,
        sensitivity_score: float,
        robustness_zone: Tuple[float, float],
        sweep_result: Dict
    ) -> List[str]:
        """Generate actionable recommendations based on sensitivity analysis."""
        recommendations = []

        # Based on sensitivity score
        if sensitivity_score > 0.5:
            recommendations.append(
                f"HIGH SENSITIVITY: {param_name} has high impact on performance. "
                f"Fine-tuning this parameter is important."
            )
        elif sensitivity_score < 0.1:
            recommendations.append(
                f"LOW SENSITIVITY: {param_name} has minimal impact on performance. "
                f"Default values are likely acceptable."
            )

        # Based on robustness zone
        zone_width = robustness_zone[1] - robustness_zone[0]
        param_range = sweep_result['values'][-1] - sweep_result['values'][0]

        if zone_width / param_range > 0.5:
            recommendations.append(
                f"ROBUST: Performance is stable across a wide range "
                f"({robustness_zone[0]:.3f} to {robustness_zone[1]:.3f})."
            )
        else:
            recommendations.append(
                f"FRAGILE: Optimal zone is narrow. Use {optimal_value:.3f} "
                f"but monitor performance carefully."
            )

        # Based on optimal position
        if optimal_value <= param_range[0]:
            recommendations.append(
                f"BOUNDARY: Optimal value at lower boundary. "
                f"Consider testing lower values."
            )
        elif optimal_value >= param_range[-1]:
            recommendations.append(
                f"BOUNDARY: Optimal value at upper boundary. "
                f"Consider testing higher values."
            )

        return recommendations

    def generate_heatmap(
        self,
        param1_name: str,
        param2_name: str,
        param1_range: Optional[ParameterRange] = None,
        param2_range: Optional[ParameterRange] = None,
        metric: str = 'sharpe_ratio',
        resolution: int = 10
    ) -> HeatmapData:
        """
        Generate a 2D heatmap showing parameter interactions.

        Args:
            param1_name: First parameter name
            param2_name: Second parameter name
            param1_range: Range for first parameter
            param2_range: Range for second parameter
            metric: Metric to analyze
            resolution: Number of points per dimension

        Returns:
            HeatmapData for visualization
        """
        if param1_range is None:
            param1_range = DEFAULT_PARAMETER_RANGES.get(param1_name)
        if param2_range is None:
            param2_range = DEFAULT_PARAMETER_RANGES.get(param2_name)

        if param1_range is None or param2_range is None:
            raise ValueError("Unknown parameter(s)")

        # Generate reduced resolution ranges
        param1_values = np.linspace(
            param1_range.min_value,
            param1_range.max_value,
            resolution
        ).tolist()
        param2_values = np.linspace(
            param2_range.min_value,
            param2_range.max_value,
            resolution
        ).tolist()

        # Create custom ranges with specific values
        custom_range1 = ParameterRange(
            name=param1_name,
            min_value=param1_values[0],
            max_value=param1_values[-1],
            step=(param1_values[-1] - param1_values[0]) / (resolution - 1),
            config_path=param1_range.config_path
        )
        custom_range2 = ParameterRange(
            name=param2_name,
            min_value=param2_values[0],
            max_value=param2_values[-1],
            step=(param2_values[-1] - param2_values[0]) / (resolution - 1),
            config_path=param2_range.config_path
        )

        # Run grid search
        result = self.optimizer.grid_search(
            param_ranges={param1_name: custom_range1, param2_name: custom_range2},
            objective=metric,
            params_to_optimize=[param1_name, param2_name]
        )

        # Build metric matrix
        metric_matrix = [[0.0] * len(param2_values) for _ in range(len(param1_values))]

        for r in result.all_results:
            v1 = r[param1_name]
            v2 = r[param2_name]

            # Find closest indices
            i1 = min(range(len(param1_values)),
                    key=lambda i: abs(param1_values[i] - v1))
            i2 = min(range(len(param2_values)),
                    key=lambda i: abs(param2_values[i] - v2))

            metric_matrix[i1][i2] = r['objective_value']

        # Calculate interaction strength
        interaction_strength = self._calculate_interaction_strength(
            metric_matrix, param1_values, param2_values
        )

        return HeatmapData(
            param1_name=param1_name,
            param2_name=param2_name,
            param1_values=param1_values,
            param2_values=param2_values,
            metric_values=metric_matrix,
            optimal_params=result.best_params,
            interaction_strength=interaction_strength
        )

    def _calculate_interaction_strength(
        self,
        metric_matrix: List[List[float]],
        param1_values: List[float],
        param2_values: List[float]
    ) -> float:
        """
        Calculate the strength of interaction between two parameters.

        Interaction strength measures how much the effect of one parameter
        depends on the value of the other.

        Returns:
            Float between 0 (no interaction) and 1 (strong interaction)
        """
        matrix = np.array(metric_matrix)

        # Calculate row and column effects
        row_means = matrix.mean(axis=1)
        col_means = matrix.mean(axis=0)
        grand_mean = matrix.mean()

        # Calculate expected values under no interaction
        expected = np.outer(row_means - grand_mean + 1, col_means - grand_mean + 1) * grand_mean

        # Calculate residual (interaction effect)
        residual = matrix - expected

        # Interaction strength = variance of residuals / total variance
        total_var = matrix.var()
        if total_var > 0:
            interaction_var = residual.var()
            return min(1.0, interaction_var / total_var)

        return 0.0

    def calculate_robustness_score(
        self,
        params: Dict[str, float],
        perturbation_pct: float = 0.1
    ) -> Dict[str, Any]:
        """
        Calculate a robustness score for a parameter configuration.

        Tests how stable performance is when parameters are perturbed.

        Args:
            params: Parameter configuration to test
            perturbation_pct: Percentage to perturb each parameter

        Returns:
            Dict with robustness metrics
        """
        # Get baseline performance
        baseline_result = self.optimizer._run_backtest_with_params(
            params, DEFAULT_PARAMETER_RANGES
        )

        if 'error' in baseline_result:
            return {'error': baseline_result['error']}

        baseline_sharpe = baseline_result.get('sharpe_ratio', 0)

        # Test perturbations
        perturbed_sharpes = []

        for param_name, value in params.items():
            if param_name not in DEFAULT_PARAMETER_RANGES:
                continue

            # Perturb up
            perturbed_params = params.copy()
            perturbed_params[param_name] = value * (1 + perturbation_pct)
            result = self.optimizer._run_backtest_with_params(
                perturbed_params, DEFAULT_PARAMETER_RANGES
            )
            if 'error' not in result:
                perturbed_sharpes.append(result.get('sharpe_ratio', 0))

            # Perturb down
            perturbed_params = params.copy()
            perturbed_params[param_name] = value * (1 - perturbation_pct)
            result = self.optimizer._run_backtest_with_params(
                perturbed_params, DEFAULT_PARAMETER_RANGES
            )
            if 'error' not in result:
                perturbed_sharpes.append(result.get('sharpe_ratio', 0))

        if not perturbed_sharpes:
            return {'error': 'No valid perturbed results'}

        # Calculate robustness metrics
        sharpe_std = np.std(perturbed_sharpes)
        sharpe_min = min(perturbed_sharpes)
        sharpe_mean = np.mean(perturbed_sharpes)

        # Robustness score: 1 - (coefficient of variation)
        if sharpe_mean != 0:
            cv = sharpe_std / abs(sharpe_mean)
            robustness_score = max(0, 1 - cv)
        else:
            robustness_score = 0.0

        # Degradation from baseline
        worst_degradation = (baseline_sharpe - sharpe_min) / (abs(baseline_sharpe) + 0.001)

        return {
            'baseline_sharpe': baseline_sharpe,
            'perturbed_mean_sharpe': sharpe_mean,
            'perturbed_std_sharpe': sharpe_std,
            'perturbed_min_sharpe': sharpe_min,
            'robustness_score': robustness_score,
            'worst_degradation_pct': worst_degradation * 100,
            'is_robust': robustness_score > 0.7 and worst_degradation < 0.2,
            'recommendation': (
                'Configuration is robust' if robustness_score > 0.7
                else 'Consider parameter re-optimization - configuration may be fragile'
            )
        }

    def full_sensitivity_analysis(
        self,
        params_to_analyze: Optional[List[str]] = None
    ) -> Dict[str, SensitivityReport]:
        """
        Run full sensitivity analysis on multiple parameters.

        Args:
            params_to_analyze: Parameters to analyze (None for all defaults)

        Returns:
            Dict mapping parameter names to SensitivityReport
        """
        if params_to_analyze is None:
            params_to_analyze = list(DEFAULT_PARAMETER_RANGES.keys())

        reports = {}

        for param_name in params_to_analyze:
            self.logger.info(f"Analyzing sensitivity: {param_name}")
            try:
                report = self.generate_sensitivity_report(param_name)
                reports[param_name] = report
            except Exception as e:
                self.logger.error(f"Error analyzing {param_name}: {e}")

        return reports

    def export_analysis(
        self,
        reports: Dict[str, SensitivityReport],
        filepath: str
    ) -> None:
        """Export sensitivity analysis to JSON file."""
        export_data = {}

        for param_name, report in reports.items():
            export_data[param_name] = {
                'values_tested': report.values_tested,
                'optimal_value': report.optimal_value,
                'sensitivity_score': report.sensitivity_score,
                'robustness_zone': list(report.robustness_zone),
                'recommendations': report.recommendations,
                'metric_values': report.metric_values,
            }

        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Exported analysis to {filepath}")


def quick_sensitivity_check(param_name: str) -> SensitivityReport:
    """
    Quick sensitivity check for a single parameter.

    Args:
        param_name: Parameter to analyze

    Returns:
        SensitivityReport
    """
    analyzer = SensitivityAnalyzer()
    return analyzer.generate_sensitivity_report(param_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    analyzer = SensitivityAnalyzer()

    # Single parameter sensitivity
    print("\n=== Sensitivity Analysis: iv_hv_ratio_threshold ===")
    report = analyzer.generate_sensitivity_report('iv_hv_ratio_threshold')
    print(f"Optimal value: {report.optimal_value:.3f}")
    print(f"Sensitivity score: {report.sensitivity_score:.3f}")
    print(f"Robustness zone: {report.robustness_zone}")
    print(f"Recommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")

    # Parameter interaction heatmap
    print("\n=== Parameter Interaction Heatmap ===")
    heatmap = analyzer.generate_heatmap(
        'iv_hv_ratio_threshold',
        'min_iv_threshold',
        resolution=5
    )
    print(f"Optimal params: {heatmap.optimal_params}")
    print(f"Interaction strength: {heatmap.interaction_strength:.3f}")

    # Robustness score
    print("\n=== Robustness Score ===")
    robustness = analyzer.calculate_robustness_score({
        'iv_hv_ratio_threshold': 1.6,
        'min_iv_threshold': 32.0
    })
    print(f"Robustness score: {robustness.get('robustness_score', 0):.3f}")
    print(f"Is robust: {robustness.get('is_robust', False)}")
    print(f"Recommendation: {robustness.get('recommendation', 'N/A')}")
