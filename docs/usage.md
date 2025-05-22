# Usage Guide

This guide covers how to use the TuringTrader algorithmic trading system.

## Basic Commands

### Running the Trader

The main entry point is `main.py`, which supports different modes:

```bash
# Paper trading mode
python main.py --mode paper

# Live trading mode
python main.py --mode live

# Test connection
python main.py --mode test

# Backtest mode
python main.py --mode backtest --start-date 2022-01-01 --end-date 2022-12-31
```

### Risk Level Selection

You can specify a risk level (1-10, where 10 is highest risk):

```bash
python main.py --mode paper --risk-level 5
```

### Backtest Specific Commands

The dedicated backtesting script offers more options:

```bash
# Basic backtest
python backtest.py --start-date 2022-01-01 --end-date 2022-12-31 --risk-level 5

# Test all risk levels
python backtest.py --start-date 2022-01-01 --end-date 2022-12-31 --all-risk-levels

# Custom initial balance
python backtest.py --start-date 2022-01-01 --end-date 2022-12-31 --initial-balance 500000
```

## Configuration

### Configuration File

The `config.ini` file controls the behavior of the trading system. Key sections include:

#### IBKR Section
Controls Interactive Brokers connection:
```ini
[IBKR]
host = 127.0.0.1
port = 7497  # 7497 for paper trading, 7496 for live
client_id = 1
timeout = 60
read_only = False
```

#### Risk Section
Controls risk parameters:
```ini
[Risk]
risk_level = 5
# Uncomment to override auto-adjusted parameters
# max_daily_risk_pct = 2.0
# min_volatility_threshold = 15.0
# max_position_size_pct = 20.0
```

#### Trading Section
Controls trading behavior:
```ini
[Trading]
index_symbol = SPY
options_only = True
trading_period_minutes = 15
day_start_offset_hours = 0.5
day_end_offset_hours = 0.5
```

### Custom Configuration Path

You can specify a custom configuration file:

```bash
python main.py --mode paper --config my_config.ini
```

## Trading Schedule

The system:
1. Starts with cash every day
2. Trades only during market hours
3. Automatically closes all positions before market close
4. Respects day start and end offsets from configuration

## Risk Management

The risk level (1-10) affects multiple parameters:

| Parameter | Low Risk (1) | Medium Risk (5) | High Risk (10) |
|-----------|-------------|----------------|---------------|
| Max Daily Risk | 0.5% | 2.5% | 5% |
| Min Volatility Threshold | 28% | 20% | 10% |
| Max Position Size | 8% | 20% | 35% |
| Max Delta Exposure | 18 | 50 | 90 |
| Stop Loss | 7% | 15% | 25% |
| Target Profit | 13% | 25% | 40% |

## Generating Reports

Reports are automatically generated after backtests:

```bash
# Generate a report from saved backtest results
python -m utils.reporting --input results.pkl --output-dir ./my_reports
```

## Advanced Usage

### Custom Trading Hours

Modify day start and end offsets in config.ini:
```ini
day_start_offset_hours = 1.0  # Start trading 1 hour after market open
day_end_offset_hours = 1.0    # Stop trading 1 hour before market close
```

### Market Data

Historical data is automatically cached in the `./data` directory to speed up subsequent backtests. To clear this cache:

```python
from historical_data.data_fetcher import HistoricalDataFetcher
fetcher = HistoricalDataFetcher()
fetcher.clear_cache()  # Clear all cache
fetcher.clear_cache('SPY')  # Clear cache for specific symbol
```

### Logging

Control logging verbosity:

```bash
python main.py --mode paper --log-level DEBUG
```

Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL