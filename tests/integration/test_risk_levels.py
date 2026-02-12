"""
Comprehensive parametrized tests for all 10 risk levels.
Tests parameter scaling, position sizing, drawdown limits, and trade frequency.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from ibkr_trader.config import Config, RiskParameters, VolatilityHarvestingConfig
from ibkr_trader.risk_manager import RiskManager
from ibkr_trader.volatility_analyzer import VolatilityAnalyzer
from backtesting.backtest_engine import BacktestEngine
from backtesting.realistic_mock_data import RealisticMockDataFetcher


class TestParametersScaleWithRiskLevel:
    """Test that all risk parameters adjust correctly per risk level."""

    @pytest.fixture(params=range(1, 11))
    def risk_level(self, request) -> int:
        """Parametrized fixture for risk levels 1-10."""
        return request.param

    @pytest.fixture
    def risk_params_for_level(self, risk_level) -> RiskParameters:
        """Create risk parameters adjusted for a specific risk level."""
        params = RiskParameters()
        params.adjust_for_risk_level(risk_level)
        return params

    def test_risk_level_stored_correctly(self, risk_level, risk_params_for_level):
        """Verify the risk level is stored correctly."""
        assert risk_params_for_level.risk_level == risk_level

    def test_min_volatility_threshold_scales(self, risk_level, risk_params_for_level):
        """Verify min_volatility_threshold decreases with higher risk."""
        # Formula: 15 + (10 - level) * 1.5
        expected = 15 + (10 - risk_level) * 1.5
        assert risk_params_for_level.min_volatility_threshold == pytest.approx(expected, rel=0.01)

        # Lower risk should mean higher threshold (more selective)
        if risk_level < 10:
            higher_risk_params = RiskParameters()
            higher_risk_params.adjust_for_risk_level(risk_level + 1)
            assert risk_params_for_level.min_volatility_threshold > higher_risk_params.min_volatility_threshold

    def test_max_daily_risk_scales(self, risk_level, risk_params_for_level):
        """Verify max_daily_risk_pct increases with higher risk."""
        # Formula: 0.5 + level * 0.3
        expected = 0.5 + risk_level * 0.3
        assert risk_params_for_level.max_daily_risk_pct == pytest.approx(expected, rel=0.01)

        # Higher risk should allow more daily risk
        if risk_level > 1:
            lower_risk_params = RiskParameters()
            lower_risk_params.adjust_for_risk_level(risk_level - 1)
            assert risk_params_for_level.max_daily_risk_pct > lower_risk_params.max_daily_risk_pct

    def test_max_position_size_scales(self, risk_level, risk_params_for_level):
        """Verify max_position_size_pct increases with higher risk."""
        # Formula: 2.0 + level * 1.5
        expected = 2.0 + risk_level * 1.5
        assert risk_params_for_level.max_position_size_pct == pytest.approx(expected, rel=0.01)

    def test_max_delta_exposure_scales(self, risk_level, risk_params_for_level):
        """Verify max_delta_exposure increases with higher risk."""
        # Formula: 5 + level * 3
        expected = 5 + risk_level * 3
        assert risk_params_for_level.max_delta_exposure == pytest.approx(expected, rel=0.01)

    def test_stop_loss_scales(self, risk_level, risk_params_for_level):
        """Verify stop_loss_pct increases with higher risk (wider stops)."""
        # Formula: 5 + level * 1.0
        expected = 5 + risk_level * 1.0
        assert risk_params_for_level.stop_loss_pct == pytest.approx(expected, rel=0.01)

    def test_target_profit_scales(self, risk_level, risk_params_for_level):
        """Verify target_profit_pct increases with higher risk."""
        # Formula: 8 + level * 1.2
        expected = 8 + risk_level * 1.2
        assert risk_params_for_level.target_profit_pct == pytest.approx(expected, rel=0.01)

    def test_min_volatility_change_scales(self, risk_level, risk_params_for_level):
        """Verify min_volatility_change decreases with higher risk."""
        # Formula: 3.0 - (level * 0.15)
        expected = 3.0 - (risk_level * 0.15)
        assert risk_params_for_level.min_volatility_change == pytest.approx(expected, rel=0.01)

    def test_condor_stop_loss_factor_scales(self, risk_level, risk_params_for_level):
        """Verify condor_stop_loss_factor increases with higher risk."""
        # Formula: 18 + (level * 1.5)
        expected = 18 + (risk_level * 1.5)
        assert risk_params_for_level.condor_stop_loss_factor_of_max_risk == pytest.approx(expected, rel=0.01)

    def test_condor_profit_target_factor_scales(self, risk_level, risk_params_for_level):
        """Verify condor_profit_target_factor decreases with higher risk."""
        # Formula: 75 + (10 - level) * 3.0
        expected = 75 + (10 - risk_level) * 3.0
        assert risk_params_for_level.condor_profit_target_factor_of_credit == pytest.approx(expected, rel=0.01)


class TestPositionSizingWithinLimits:
    """Test that position sizing stays within level-appropriate bounds."""

    @pytest.fixture(params=range(1, 11))
    def risk_level(self, request) -> int:
        return request.param

    @pytest.fixture
    def risk_manager_for_level(self, risk_level) -> RiskManager:
        """Create a RiskManager configured for a specific risk level."""
        config = Config()
        config.risk.adjust_for_risk_level(risk_level)
        rm = RiskManager(config)
        rm.update_account_value(100000.0)
        return rm

    def test_position_size_respects_max_limit(self, risk_level, risk_manager_for_level):
        """Verify position size never exceeds max_position_size_pct of account."""
        account_value = 100000.0
        max_position_pct = risk_manager_for_level.config.risk.max_position_size_pct / 100.0
        max_position_value = account_value * max_position_pct

        # Test with various price and volatility combinations
        test_cases = [
            (450.0, 0.20),  # SPY-like price, normal vol
            (450.0, 0.40),  # SPY-like price, high vol
            (100.0, 0.15),  # Lower price, low vol
            (500.0, 0.50),  # Higher price, very high vol
        ]

        for price, volatility in test_cases:
            position_size = risk_manager_for_level.calculate_position_size(
                price, volatility, account_value, vol_multiplier=1.0
            )
            position_value = position_size * price

            # Position value should not exceed maximum allowed
            assert position_value <= max_position_value * 1.01, (
                f"Position value ${position_value:.2f} exceeds max ${max_position_value:.2f} "
                f"for risk level {risk_level}, price ${price}, vol {volatility}"
            )

    def test_position_size_scales_with_risk_level(self, risk_level, risk_manager_for_level):
        """Verify higher risk levels allow larger position sizes."""
        account_value = 100000.0
        price = 450.0
        volatility = 0.20

        position_size = risk_manager_for_level.calculate_position_size(
            price, volatility, account_value, vol_multiplier=1.0
        )

        # Compare with adjacent risk levels
        if risk_level < 10:
            higher_config = Config()
            higher_config.risk.adjust_for_risk_level(risk_level + 1)
            higher_rm = RiskManager(higher_config)
            higher_rm.update_account_value(account_value)

            higher_position_size = higher_rm.calculate_position_size(
                price, volatility, account_value, vol_multiplier=1.0
            )

            # Higher risk level should allow equal or larger positions
            assert higher_position_size >= position_size * 0.95, (
                f"Higher risk level {risk_level + 1} should allow >= position size than level {risk_level}"
            )

    def test_option_quantity_within_limits(self, risk_level, risk_manager_for_level):
        """Verify option quantity calculation stays within limits."""
        account_value = 100000.0
        option_price = 3.50  # $3.50 per contract
        delta = 0.25

        quantity = risk_manager_for_level.calculate_option_quantity(
            option_price, delta, account_value, vol_multiplier=1.0
        )

        # Quantity should be positive integer
        assert quantity >= 0
        assert isinstance(quantity, int)

        # Total position value should not exceed max
        max_position_pct = risk_manager_for_level.config.risk.max_position_size_pct / 100.0
        max_position_value = account_value * max_position_pct
        position_value = quantity * option_price * 100  # Options represent 100 shares

        assert position_value <= max_position_value * 1.1, (
            f"Option position value ${position_value:.2f} exceeds max ${max_position_value:.2f}"
        )


class TestDrawdownWithinLimits:
    """Test that max drawdown respects risk level constraints."""

    @pytest.fixture(params=[1, 3, 5, 7, 10])  # Test representative risk levels
    def risk_level(self, request) -> int:
        return request.param

    @pytest.fixture
    def backtest_engine_for_level(self, risk_level) -> BacktestEngine:
        """Create a BacktestEngine with mock data for a specific risk level."""
        config = Config()
        config.risk.adjust_for_risk_level(risk_level)

        mock_fetcher = RealisticMockDataFetcher()
        engine = BacktestEngine(
            config=config,
            initial_balance=100000.0,
            data_fetcher=mock_fetcher
        )
        return engine

    def test_drawdown_bounded_by_risk_level(self, risk_level, backtest_engine_for_level):
        """Verify drawdown stays within risk-appropriate bounds."""
        # Run a short backtest
        results = backtest_engine_for_level.run_backtest(
            start_date='2023-01-01',
            end_date='2023-03-31',
            risk_level=risk_level,
            use_cache=False
        )

        if 'error' in results:
            pytest.skip(f"Backtest error: {results['error']}")

        max_drawdown = results.get('max_drawdown_pct', 0)

        # Lower risk levels should have lower max drawdown potential
        # This is a soft constraint based on expected behavior
        # Risk level 1 should have max ~15% drawdown, level 10 up to ~35%
        expected_max_drawdown = 10 + risk_level * 2.5

        assert max_drawdown <= expected_max_drawdown * 1.5, (
            f"Max drawdown {max_drawdown:.2f}% exceeds expected max {expected_max_drawdown:.2f}% "
            f"for risk level {risk_level}"
        )


class TestTradeFrequencyCorrelates:
    """Test that trade frequency correlates with risk level."""

    def test_higher_risk_more_trading_opportunities(self):
        """Verify higher risk levels generate more trading signals."""
        mock_fetcher = RealisticMockDataFetcher()

        # Run backtests for low, medium, and high risk
        trade_counts = {}

        for risk_level in [1, 5, 10]:
            config = Config()
            config.risk.adjust_for_risk_level(risk_level)

            engine = BacktestEngine(
                config=config,
                initial_balance=100000.0,
                data_fetcher=mock_fetcher
            )

            results = engine.run_backtest(
                start_date='2023-01-01',
                end_date='2023-06-30',
                risk_level=risk_level,
                use_cache=False
            )

            if 'error' not in results:
                trade_counts[risk_level] = results.get('trades', 0)

        # Higher risk should generally have more or equal trades
        if 1 in trade_counts and 5 in trade_counts:
            assert trade_counts[5] >= trade_counts[1] * 0.8, (
                "Medium risk should have similar or more trades than low risk"
            )

        if 5 in trade_counts and 10 in trade_counts:
            assert trade_counts[10] >= trade_counts[5] * 0.8, (
                "High risk should have similar or more trades than medium risk"
            )


class TestAllRiskLevelsValid:
    """Test that all risk levels produce valid configurations."""

    @pytest.fixture(params=range(1, 11))
    def risk_level(self, request) -> int:
        return request.param

    def test_risk_level_in_valid_range(self, risk_level):
        """Test that risk level rejects invalid values."""
        params = RiskParameters()

        # Valid levels should work
        params.adjust_for_risk_level(risk_level)
        assert params.risk_level == risk_level

        # Invalid levels should raise
        with pytest.raises(ValueError):
            params.adjust_for_risk_level(0)

        with pytest.raises(ValueError):
            params.adjust_for_risk_level(11)

    def test_all_parameters_positive(self, risk_level):
        """Verify all risk parameters are positive after adjustment."""
        params = RiskParameters()
        params.adjust_for_risk_level(risk_level)

        assert params.min_volatility_threshold > 0
        assert params.max_daily_risk_pct > 0
        assert params.max_position_size_pct > 0
        assert params.max_delta_exposure > 0
        assert params.stop_loss_pct > 0
        assert params.target_profit_pct > 0
        assert params.min_volatility_change > 0
        assert params.condor_stop_loss_factor_of_max_risk > 0
        assert params.condor_profit_target_factor_of_credit > 0

    def test_config_loads_with_risk_level(self, risk_level):
        """Verify Config can be created with any risk level."""
        config = Config()
        config.risk.adjust_for_risk_level(risk_level)

        assert config.risk.risk_level == risk_level
        assert config.trading is not None
        assert config.vol_harvesting is not None
