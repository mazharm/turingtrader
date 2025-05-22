# TuringTrader Usage Guide

This guide explains how to use the TuringTrader algorithmic trading system for S&P500 options trading.

## Trading Agent

The trading agent is the core component of TuringTrader. It connects to Interactive Brokers, assesses market conditions, and executes trades based on volatility and your risk profile.

### Running the Agent

To run the trading agent with default settings:

```bash
python -m src.turingtrader.agent
```

This will start the agent with a MEDIUM risk profile and connect to IB on localhost using default ports.

### Command-Line Options

The agent supports several command-line options:

```bash
python -m src.turingtrader.agent [options]
```

Available options:

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | IB Gateway/TWS hostname | 127.0.0.1 |
| `--port` | IB Gateway/TWS port | 7497 (paper) |
| `--client-id` | Client ID for IB connection | 1 |
| `--risk-level` | Risk level (LOW, MEDIUM, HIGH, AGGRESSIVE) | MEDIUM |
| `--data-dir` | Directory for saving trading data | ~/.turingtrader/data |

Example with custom settings:

```bash
python -m src.turingtrader.agent --host 192.168.1.100 --port 7496 --risk-level HIGH --data-dir /path/to/data
```

### Trading Logic

The agent follows these steps during a trading day:

1. Connects to Interactive Brokers
2. Starts the day in cash (no overnight positions)
3. Analyzes market volatility using S&P500 historical data and VIX
4. During high volatility periods:
   - Selects appropriate options strategies
   - Executes trades based on risk profile
5. Ends the day by closing all positions (back to cash)

## Backtesting

TuringTrader includes a backtesting system to evaluate strategies on historical data.

### Running Backtests

To run a basic backtest:

```bash
python scripts/backtest.py --start-date 2022-01-01 --end-date 2022-12-31
```

This will backtest the strategy on historical data from January 1, 2022, to December 31, 2022, using default settings.

### Backtest Options

```bash
python scripts/backtest.py [options]
```

Available options:

| Option | Description | Default |
|--------|-------------|---------|
| `--start-date` | Start date (YYYY-MM-DD) | (Required) |
| `--end-date` | End date (YYYY-MM-DD) | Today |
| `--risk-level` | Risk level | MEDIUM |
| `--initial-capital` | Initial capital amount | 100000.0 |
| `--strategies` | Strategies to test | iron_condor vertical_spread |
| `--high-vol-only` | Only trade on high volatility days | False |
| `--data-dir` | Directory for backtest data | ~/.turingtrader/backtest |
| `--output-file` | Output file for performance report | None |

Example with custom settings:

```bash
python scripts/backtest.py --start-date 2021-01-01 --end-date 2022-01-01 --risk-level HIGH --initial-capital 50000 --strategies iron_condor --high-vol-only --output-file backtest_report.png
```

### Interpreting Backtest Results

Backtest results are saved as JSON files in the data directory and include metrics such as:

- Total return
- Annual return
- Volatility
- Sharpe ratio
- Maximum drawdown
- Win rate

A visual report will also be generated if an output file is specified.

## Performance Reporting

TuringTrader includes tools to generate performance reports from trading data and backtest results.

### Generating Reports

To generate performance reports:

```bash
python scripts/generate_report.py [options]
```

Available options:

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Directory with trading data | ~/.turingtrader/data |
| `--backtest-dir` | Directory with backtest results | ~/.turingtrader/backtest |
| `--output-dir` | Directory for saving reports | ~/.turingtrader/reports |
| `--start-date` | Start date for report | None |
| `--end-date` | End date for report | None |
| `--trading-report` | Generate trading performance report | False |
| `--risk-report` | Generate risk comparison report | False |
| `--trading-report-file` | Output file for trading report | trading_report.png |
| `--risk-report-file` | Output file for risk report | risk_comparison.png |
| `--show-plot` | Display plots | False |

Example usage:

```bash
python scripts/generate_report.py --trading-report --risk-report --start-date 2022-01-01 --end-date 2022-12-31 --show-plot
```

### Report Types

1. **Trading Report**: Shows account balance, daily P&L, and drawdown over time
2. **Risk Comparison Report**: Compares performance metrics across different risk levels

## Best Practices

### Paper Trading First

Always test your strategies with paper trading before using real money. In IB, use port 7497 for paper trading.

### Setting Appropriate Risk Levels

- **LOW**: For conservative traders or during uncertain market conditions
- **MEDIUM**: For balanced risk/reward
- **HIGH**: For traders comfortable with higher volatility
- **AGGRESSIVE**: For experienced traders or well-capitalized accounts

### Monitoring Performance

Regularly generate performance reports to evaluate your trading results and adjust strategies as needed.

### Market Hours

The agent is designed to trade during regular market hours. Make sure IB Gateway/TWS is running before market open.