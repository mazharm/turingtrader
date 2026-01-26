"""
Shared pytest fixtures for TuringTrader tests.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from typing import Dict, Generator

# Import modules to test
from ibkr_trader.config import Config, RiskParameters, TradingConfig
from ibkr_trader.risk_manager import RiskManager
from ibkr_trader.volatility_analyzer import VolatilityAnalyzer
from ibkr_trader.options_strategy import OptionsStrategy
from backtesting.backtest_engine import BacktestEngine
from backtesting.realistic_mock_data import RealisticMockDataFetcher


@pytest.fixture
def config():
    """Create a Config instance with default settings."""
    return Config()


@pytest.fixture
def risk_config():
    """Create a RiskParameters instance with default settings."""
    return RiskParameters()


@pytest.fixture
def trading_config():
    """Create a TradingConfig instance with default settings."""
    return TradingConfig()


@pytest.fixture
def risk_manager(config):
    """Create a RiskManager instance."""
    rm = RiskManager(config)
    rm.update_account_value(100000.0)
    return rm


@pytest.fixture
def volatility_analyzer(config):
    """Create a VolatilityAnalyzer instance."""
    return VolatilityAnalyzer(config)


@pytest.fixture
def options_strategy(config, volatility_analyzer, risk_manager):
    """Create an OptionsStrategy instance."""
    return OptionsStrategy(config, volatility_analyzer, risk_manager)


@pytest.fixture
def sample_vix_data():
    """Sample VIX data for testing."""
    return [
        {'date': '2022-01-01', 'close': 17.0},
        {'date': '2022-01-02', 'close': 17.5},
        {'date': '2022-01-03', 'close': 18.0},
        {'date': '2022-01-04', 'close': 19.0},
        {'date': '2022-01-05', 'close': 20.0},
        {'date': '2022-01-06', 'close': 22.0}
    ]


@pytest.fixture
def sample_vix_analysis():
    """Sample VIX analysis result for testing."""
    return {
        'current_vix': 22.0,
        'vix_change_1d': 2.0,
        'vix_change_5d': 5.0,
        'volatility_state': 'normal',
        'signal': 'buy',
        'vix_trend': 'up',
        'vix_regime': 'normal'
    }


@pytest.fixture
def sample_option_chain():
    """Sample option chain data for testing."""
    return {
        '20221216': {
            'days_to_expiry': 28,
            'calls': {
                380: {'bid': 25.0, 'ask': 25.5, 'iv': 0.25, 'delta': 0.85, 'volume': 500, 'open_interest': 2000},
                390: {'bid': 18.0, 'ask': 18.5, 'iv': 0.27, 'delta': 0.72, 'volume': 800, 'open_interest': 3000},
                400: {'bid': 12.0, 'ask': 12.5, 'iv': 0.28, 'delta': 0.55, 'volume': 1200, 'open_interest': 5000},
                410: {'bid': 7.0, 'ask': 7.5, 'iv': 0.30, 'delta': 0.35, 'volume': 1500, 'open_interest': 6000},
                420: {'bid': 4.0, 'ask': 4.5, 'iv': 0.32, 'delta': 0.22, 'volume': 1000, 'open_interest': 4000},
                430: {'bid': 2.0, 'ask': 2.5, 'iv': 0.35, 'delta': 0.12, 'volume': 600, 'open_interest': 2500},
                440: {'bid': 0.8, 'ask': 1.2, 'iv': 0.38, 'delta': 0.06, 'volume': 300, 'open_interest': 1000},
            },
            'puts': {
                380: {'bid': 2.0, 'ask': 2.5, 'iv': 0.35, 'delta': -0.15, 'volume': 600, 'open_interest': 2500},
                390: {'bid': 4.0, 'ask': 4.5, 'iv': 0.32, 'delta': -0.22, 'volume': 1000, 'open_interest': 4000},
                400: {'bid': 7.0, 'ask': 7.5, 'iv': 0.30, 'delta': -0.35, 'volume': 1500, 'open_interest': 6000},
                410: {'bid': 12.0, 'ask': 12.5, 'iv': 0.28, 'delta': -0.55, 'volume': 1200, 'open_interest': 5000},
                420: {'bid': 18.0, 'ask': 18.5, 'iv': 0.27, 'delta': -0.72, 'volume': 800, 'open_interest': 3000},
                430: {'bid': 25.0, 'ask': 25.5, 'iv': 0.25, 'delta': -0.85, 'volume': 500, 'open_interest': 2000},
            },
            'strikes': [380, 390, 400, 410, 420, 430, 440]
        }
    }


@pytest.fixture
def mock_ib_client():
    """Create a mock IB client for testing."""
    mock = MagicMock()
    mock.isConnected.return_value = True
    mock.reqAccountSummary.return_value = []
    return mock


# ============================================================================
# Risk Level Fixtures
# ============================================================================

@pytest.fixture(params=range(1, 11))
def all_risk_levels(request) -> int:
    """
    Parametrized fixture that provides all risk levels 1-10.
    Use this to run a test across all risk levels.
    """
    return request.param


@pytest.fixture
def risk_level_config():
    """
    Factory fixture that creates a Config configured for a specific risk level.
    Usage: config = risk_level_config(5)  # Get config for risk level 5
    """
    def _create_config(level: int) -> Config:
        if not 1 <= level <= 10:
            raise ValueError("Risk level must be between 1 and 10")
        config = Config()
        config.risk.adjust_for_risk_level(level)
        return config
    return _create_config


@pytest.fixture(scope="session")
def all_risk_results() -> Dict[int, Dict]:
    """
    Session-scoped fixture that pre-computes backtest results for all risk levels.
    This is expensive to compute, so it's shared across all tests in a session.

    Returns:
        Dict mapping risk levels (1-10) to their backtest results
    """
    mock_fetcher = RealisticMockDataFetcher()
    results = {}

    for risk_level in range(1, 11):
        config = Config()
        config.risk.adjust_for_risk_level(risk_level)

        engine = BacktestEngine(
            config=config,
            initial_balance=100000.0,
            data_fetcher=mock_fetcher
        )

        result = engine.run_backtest(
            start_date='2023-01-01',
            end_date='2023-06-30',
            risk_level=risk_level,
            use_cache=False
        )

        if 'error' not in result:
            results[risk_level] = result

    return results


@pytest.fixture
def mock_data_fetcher() -> RealisticMockDataFetcher:
    """Create a RealisticMockDataFetcher instance for testing."""
    return RealisticMockDataFetcher()


@pytest.fixture
def backtest_engine_factory(mock_data_fetcher):
    """
    Factory fixture that creates a BacktestEngine for a specific risk level.
    Usage: engine = backtest_engine_factory(5)  # Get engine for risk level 5
    """
    def _create_engine(risk_level: int, initial_balance: float = 100000.0) -> BacktestEngine:
        config = Config()
        config.risk.adjust_for_risk_level(risk_level)

        return BacktestEngine(
            config=config,
            initial_balance=initial_balance,
            data_fetcher=mock_data_fetcher
        )
    return _create_engine


@pytest.fixture
def risk_parameters_for_level():
    """
    Factory fixture that creates RiskParameters adjusted for a specific risk level.
    Usage: params = risk_parameters_for_level(5)  # Get params for risk level 5
    """
    def _create_params(level: int) -> RiskParameters:
        params = RiskParameters()
        params.adjust_for_risk_level(level)
        return params
    return _create_params
