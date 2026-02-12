# TuringTrader Configuration Guide

This guide explains the configuration options available for TuringTrader and how to customize the trading system to suit your needs.

## Risk Management Configuration

TuringTrader supports different risk levels that affect position sizing, stop loss percentages, and options selection criteria.

### Available Risk Levels

| Risk Level | Description | Position Size | Max Daily Loss | Stop Loss |
|------------|-------------|---------------|----------------|-----------|
| LOW | Conservative approach | 2% of account | 1% of account | 10% |
| MEDIUM | Balanced approach | 5% of account | 2% of account | 15% |
| HIGH | Aggressive approach | 10% of account | 4% of account | 25% |
| AGGRESSIVE | Maximum risk | 15% of account | 6% of account | 35% |

### Modifying Risk Parameters

You can modify risk parameters by editing the `risk_manager.py` file in the source code:

```python
# src/turingtrader/risk_manager.py
def _configure_risk_profile(self, risk_level: RiskLevel) -> None:
    # Customize risk parameters here
    if risk_level == RiskLevel.MEDIUM:
        self.max_position_size_pct = 0.05  # 5% of account
        self.max_daily_loss_pct = 0.02     # 2% of account
        self.stop_loss_pct = 0.15          # 15% stop loss
        # ...
```

## Volatility Assessment Configuration

The volatility assessment module determines when market conditions are favorable for trading.

### Adjustable Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| lookback_period | Days of historical data to use | 20 |
| volatility_threshold | Historical volatility threshold | 0.15 |
| vix_threshold | VIX threshold for high volatility | 20.0 |

### Modifying Volatility Parameters

You can adjust these parameters when initializing the `VolatilityAnalyzer` class:

```python
from src.turingtrader.volatility import VolatilityAnalyzer

# Custom volatility thresholds
analyzer = VolatilityAnalyzer(
    lookback_period=30,       # 30-day lookback
    volatility_threshold=0.18, # 18% annualized volatility
    vix_threshold=25.0        # VIX above 25
)
```

Or modify the defaults in the source code:

```python
# src/turingtrader/volatility.py
def __init__(
    self,
    lookback_period: int = 20,              # Adjust this
    volatility_threshold: float = 0.15,     # Adjust this
    vix_threshold: float = 20.0,            # Adjust this
):
    # ...
```

## Options Trading Configuration

TuringTrader includes two main options strategies: iron condors and vertical spreads.

### Strategy Selection Logic

The system selects strategies based on market conditions:

- **High Volatility**: Iron condors (profit from volatility contraction)
- **Low Volatility**: Vertical spreads (defined risk directional trades)

### Options Selection Criteria

Options are selected based on:

1. Liquidity (volume and open interest)
2. Delta (proximity to the preferred delta for your risk level)
3. Days to expiration (varies by risk level)

### Adjusting Options Parameters

You can modify the options selection logic in the `options_trader.py` file:

```python
# src/turingtrader/options_trader.py
def _find_best_options(self, options_data, volatility_metrics, preferred_delta):
    # Customize options selection logic here
    # ...
```

## Interactive Brokers Connection

TuringTrader connects to Interactive Brokers (IB) using the ib_insync library.

### Connection Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| host | IB Gateway/TWS hostname | 127.0.0.1 |
| port | IB Gateway/TWS port | 7497 (paper) |
| client_id | Client ID for IB connection | 1 |

### Modifying Connection Parameters

You can specify connection parameters when initializing the `IBClient` class:

```python
from src.turingtrader.ib_client import IBClient

# Custom connection parameters
ib_client = IBClient(
    host="192.168.1.100",  # Custom hostname
    port=7496,             # Live trading port
    client_id=2            # Custom client ID
)
```

## Data Storage Configuration

TuringTrader saves trading data, backtest results, and performance reports in specific directories.

### Default Directories

| Directory | Purpose | Default Location |
|-----------|---------|------------------|
| Data | Trading data | ~/.turingtrader/data |
| Backtest | Backtest results | ~/.turingtrader/backtest |
| Reports | Performance reports | ~/.turingtrader/reports |

### Customizing Data Directories

You can specify custom directories when initializing the relevant classes:

```python
# Trading agent
from src.turingtrader.agent import TuringTrader

trader = TuringTrader(
    data_dir="/path/to/custom/data"
)

# Backtester
from scripts.backtest import OptionsBacktester

backtester = OptionsBacktester(
    data_dir="/path/to/custom/backtest/data"
)

# Performance reporter
from scripts.generate_report import PerformanceReporter

reporter = PerformanceReporter(
    data_dir="/path/to/custom/data",
    output_dir="/path/to/custom/reports"
)
```

## Advanced Configuration

For advanced users, here are some additional configuration options:

### Adding Custom Strategies

To add a custom options strategy:

1. Add a new method in `options_trader.py` (e.g., `_create_butterfly_spread`)
2. Update `_select_options_strategy` to include your new strategy
3. Add an execution method (e.g., `_execute_butterfly_spread`)
4. Update the `execute_trades` method to handle the new strategy

### Custom Indicators

To add custom market indicators:

1. Extend the `VolatilityAnalyzer` class in `volatility.py` with new methods
2. Incorporate your indicators into the volatility assessment logic

### Configuration Files

For more dynamic configuration, you could implement a config file system:

```python
import json

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# Example in main code
config = load_config('config.json')
trader = TuringTrader(
    risk_level=config['risk_level'],
    # Other parameters
)
```

This would allow you to change settings without modifying source code.