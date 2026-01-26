"""
Market scenario definitions and scenario-based data fetcher for backtesting.
Provides predefined market scenarios (bull, bear, crisis, etc.) and a data fetcher
that generates synthetic data matching those scenarios.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os

from backtesting.realistic_mock_data import RealisticMockDataFetcher


@dataclass
class MarketScenario:
    """
    Defines a market scenario with specific characteristics.

    Attributes:
        name: Scenario identifier
        spy_drift: Daily drift rate (e.g., 0.0005 for bull market)
        spy_volatility: Daily volatility (e.g., 0.008 for low vol)
        vix_base: Base VIX level for this scenario
        vix_spike_prob: Probability of VIX spikes (0-1)
        duration_days: Default scenario duration in trading days
        description: Human-readable description
    """
    name: str
    spy_drift: float
    spy_volatility: float
    vix_base: float
    vix_spike_prob: float
    duration_days: int = 126  # ~6 months of trading days
    description: str = ""

    def __post_init__(self):
        """Validate scenario parameters."""
        if not -0.01 <= self.spy_drift <= 0.01:
            # Allow extreme scenarios but warn
            pass
        if not 0 <= self.vix_spike_prob <= 1:
            raise ValueError("vix_spike_prob must be between 0 and 1")
        if self.vix_base < 10:
            raise ValueError("vix_base must be at least 10")


# Predefined market scenarios
BULL_MARKET = MarketScenario(
    name="bull_market",
    spy_drift=0.0005,      # +0.05% daily drift (~12.6% annual)
    spy_volatility=0.008,  # 0.8% daily vol (~12.7% annual)
    vix_base=15.0,
    vix_spike_prob=0.02,
    description="Strong uptrend with low volatility"
)

BEAR_MARKET = MarketScenario(
    name="bear_market",
    spy_drift=-0.0003,     # -0.03% daily drift (~-7.5% annual)
    spy_volatility=0.015,  # 1.5% daily vol (~23.8% annual)
    vix_base=28.0,
    vix_spike_prob=0.05,
    description="Downtrend with elevated volatility"
)

HIGH_VOLATILITY = MarketScenario(
    name="high_volatility",
    spy_drift=0.0001,      # +0.01% daily drift (~2.5% annual)
    spy_volatility=0.020,  # 2.0% daily vol (~31.7% annual)
    vix_base=35.0,
    vix_spike_prob=0.08,
    description="Choppy market with high volatility"
)

LOW_VOLATILITY = MarketScenario(
    name="low_volatility",
    spy_drift=0.0002,      # +0.02% daily drift (~5% annual)
    spy_volatility=0.005,  # 0.5% daily vol (~7.9% annual)
    vix_base=12.0,
    vix_spike_prob=0.01,
    description="Grinding upward with compressed volatility"
)

CRISIS = MarketScenario(
    name="crisis",
    spy_drift=-0.001,      # -0.1% daily drift (~-25% annual)
    spy_volatility=0.035,  # 3.5% daily vol (~55.5% annual)
    vix_base=50.0,
    vix_spike_prob=0.15,
    description="Sharp decline with extreme volatility"
)

RECOVERY = MarketScenario(
    name="recovery",
    spy_drift=0.0008,      # +0.08% daily drift (~20% annual)
    spy_volatility=0.012,  # 1.2% daily vol (~19% annual)
    vix_base=22.0,
    vix_spike_prob=0.04,
    description="Post-crisis bounce with declining volatility"
)

# Dictionary of all predefined scenarios
PREDEFINED_SCENARIOS: Dict[str, MarketScenario] = {
    "bull_market": BULL_MARKET,
    "bear_market": BEAR_MARKET,
    "high_volatility": HIGH_VOLATILITY,
    "low_volatility": LOW_VOLATILITY,
    "crisis": CRISIS,
    "recovery": RECOVERY,
}


def get_scenario(name: str) -> MarketScenario:
    """
    Get a predefined scenario by name.

    Args:
        name: Scenario name

    Returns:
        MarketScenario instance

    Raises:
        KeyError: If scenario name not found
    """
    if name not in PREDEFINED_SCENARIOS:
        available = ", ".join(PREDEFINED_SCENARIOS.keys())
        raise KeyError(f"Unknown scenario '{name}'. Available: {available}")
    return PREDEFINED_SCENARIOS[name]


def list_scenarios() -> List[str]:
    """Return list of available scenario names."""
    return list(PREDEFINED_SCENARIOS.keys())


class ScenarioDataFetcher(RealisticMockDataFetcher):
    """
    Data fetcher that generates synthetic data based on a market scenario.
    Extends RealisticMockDataFetcher with scenario-specific data generation.
    """

    def __init__(
        self,
        scenario: MarketScenario,
        seed: Optional[int] = None,
        data_dir: str = './data'
    ):
        """
        Initialize the scenario data fetcher.

        Args:
            scenario: MarketScenario defining the market conditions
            seed: Random seed for reproducibility (None for random)
            data_dir: Directory for caching data
        """
        super().__init__(data_dir=data_dir)
        self.scenario = scenario
        self.seed = seed
        self.logger = logging.getLogger(__name__)

        if seed is not None:
            np.random.seed(seed)

        # Override base class properties based on scenario
        self.base_vix_level = scenario.vix_base
        self.iv_hv_ratio_min = 1.1 + (scenario.vix_base - 15) / 50  # Higher VIX = higher ratio
        self.iv_hv_ratio_max = self.iv_hv_ratio_min + 0.8

    def fetch_data(
        self,
        symbol: str,
        start_date,
        end_date,
        interval: str = '1d',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch scenario-based market data.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            interval: Data interval (ignored)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with scenario-adjusted market data
        """
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # Generate scenario-specific data
        if symbol.upper() in ['SPY', '^SPX', 'SPX']:
            return self._generate_scenario_spy_data(start_date, end_date)
        elif symbol.upper() in ['VIX', '^VIX']:
            return self._generate_scenario_vix_data(start_date, end_date)
        else:
            return super().fetch_data(symbol, start_date, end_date, interval, use_cache)

    def _generate_scenario_spy_data(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate SPY data based on the scenario.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with scenario-based SPY data
        """
        # Generate business day range
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        df = pd.DataFrame(index=date_range)

        num_days = len(date_range)
        start_price = 450.0

        # Generate daily returns based on scenario
        daily_returns = np.random.normal(
            self.scenario.spy_drift,
            self.scenario.spy_volatility,
            num_days
        )

        # Add occasional jumps based on scenario volatility
        jump_freq = int(max(10, 50 / (self.scenario.spy_volatility / 0.01)))
        for i in range(num_days // jump_freq):
            jump_idx = np.random.randint(0, num_days)
            jump_direction = 1 if np.random.random() < 0.5 else -1
            # In bear/crisis, jumps are more likely negative
            if self.scenario.spy_drift < 0:
                jump_direction = -1 if np.random.random() < 0.7 else 1
            daily_returns[jump_idx] *= 2.5 * jump_direction

        # Calculate cumulative prices
        cumulative_returns = np.cumprod(1 + daily_returns)
        prices = start_price * cumulative_returns

        # Create OHLC data
        df['open'] = prices * (1 + np.random.normal(0, 0.002, num_days))
        df['close'] = prices
        df['high'] = np.maximum(df['open'], df['close']) * (
            1 + np.abs(np.random.normal(0.001, 0.003, num_days))
        )
        df['low'] = np.minimum(df['open'], df['close']) * (
            1 - np.abs(np.random.normal(0.001, 0.003, num_days))
        )

        # Volume increases with volatility and price moves
        base_volume = 50000000
        volume_factor = 1 + 5 * np.abs(daily_returns)
        df['volume'] = (base_volume * volume_factor).astype(int)
        df['adj close'] = df['close']

        return df

    def _generate_scenario_vix_data(self, start_date, end_date) -> pd.DataFrame:
        """
        Generate VIX data based on the scenario.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with scenario-based VIX data
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        df = pd.DataFrame(index=date_range)

        num_days = len(date_range)

        # VIX parameters from scenario
        mean_level = self.scenario.vix_base
        reversion_strength = 0.05
        daily_volatility = 0.04 + self.scenario.spy_volatility * 2

        # Generate VIX with mean reversion
        vix_levels = np.zeros(num_days)
        vix_levels[0] = mean_level

        for i in range(1, num_days):
            # Mean reversion
            mean_reversion = reversion_strength * (mean_level - vix_levels[i - 1])

            # Random component
            random_shock = np.random.normal(0, daily_volatility * vix_levels[i - 1])

            vix_levels[i] = vix_levels[i - 1] + mean_reversion + random_shock
            vix_levels[i] = max(10.0, vix_levels[i])

        # Add spikes based on scenario spike probability
        spike_days = int(num_days * self.scenario.vix_spike_prob)
        for _ in range(spike_days):
            spike_idx = np.random.randint(0, num_days)
            spike_magnitude = np.random.uniform(1.3, 2.0)
            vix_levels[spike_idx] *= spike_magnitude

            # Decay after spike
            decay_period = min(15, num_days - spike_idx - 1)
            decay_factor = np.linspace(0.7, 0.1, decay_period)
            for j in range(decay_period):
                if spike_idx + j + 1 < num_days:
                    extra = vix_levels[spike_idx] - mean_level
                    vix_levels[spike_idx + j + 1] += extra * decay_factor[j]

        # Create OHLC
        df['close'] = vix_levels
        df['open'] = np.roll(vix_levels, 1)
        df['open'].iloc[0] = vix_levels[0]

        daily_range = 0.03 + self.scenario.spy_volatility
        df['high'] = df['close'] * (1 + np.random.uniform(0, daily_range, num_days))
        df['low'] = df['close'] * (1 - np.random.uniform(0, daily_range, num_days))
        df['volume'] = np.random.randint(1000000, 5000000, num_days)
        df['adj close'] = df['close']

        return df

    def get_scenario_info(self) -> Dict:
        """
        Get information about the current scenario.

        Returns:
            Dict with scenario details
        """
        return {
            'name': self.scenario.name,
            'description': self.scenario.description,
            'spy_drift': self.scenario.spy_drift,
            'spy_volatility': self.scenario.spy_volatility,
            'vix_base': self.scenario.vix_base,
            'vix_spike_prob': self.scenario.vix_spike_prob,
            'duration_days': self.scenario.duration_days,
            'annualized_drift': self.scenario.spy_drift * 252,
            'annualized_volatility': self.scenario.spy_volatility * np.sqrt(252),
        }


def create_custom_scenario(
    name: str,
    spy_annual_return: float,
    spy_annual_volatility: float,
    vix_base: float,
    vix_spike_frequency: str = "normal"
) -> MarketScenario:
    """
    Create a custom market scenario from annual metrics.

    Args:
        name: Scenario name
        spy_annual_return: Expected annual return (e.g., 0.10 for 10%)
        spy_annual_volatility: Annual volatility (e.g., 0.15 for 15%)
        vix_base: Base VIX level
        vix_spike_frequency: "low", "normal", "high", "extreme"

    Returns:
        MarketScenario instance
    """
    # Convert annual to daily
    spy_drift = spy_annual_return / 252
    spy_volatility = spy_annual_volatility / np.sqrt(252)

    # Map spike frequency to probability
    spike_probs = {
        "low": 0.01,
        "normal": 0.03,
        "high": 0.06,
        "extreme": 0.12
    }
    vix_spike_prob = spike_probs.get(vix_spike_frequency, 0.03)

    return MarketScenario(
        name=name,
        spy_drift=spy_drift,
        spy_volatility=spy_volatility,
        vix_base=vix_base,
        vix_spike_prob=vix_spike_prob,
        description=f"Custom scenario: {spy_annual_return*100:.1f}% return, "
                    f"{spy_annual_volatility*100:.1f}% vol, VIX ~{vix_base}"
    )


def combine_scenarios(
    scenarios: List[Tuple[MarketScenario, int]],
    name: str = "combined"
) -> List[Tuple[MarketScenario, int]]:
    """
    Create a sequence of scenarios for multi-phase backtesting.

    Args:
        scenarios: List of (MarketScenario, duration_days) tuples
        name: Name for the combined scenario

    Returns:
        List of scenario segments ready for sequential backtesting

    Example:
        segments = combine_scenarios([
            (BULL_MARKET, 63),   # Q1: Bull
            (CRISIS, 21),        # Mid-Q2: Crisis
            (RECOVERY, 63),      # Rest of year: Recovery
        ])
    """
    return scenarios
