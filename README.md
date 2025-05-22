# TuringTrader

An AI-driven algorithmic trading agent for S&P500 options using Interactive Brokers API.

## Overview

TuringTrader is a Python-based algorithmic trading system designed to trade S&P500 options based on market volatility. The system:

- Starts and ends each trading day in cash (no overnight positions)
- Capitalizes on periods of high market volatility
- Provides configurable risk profiles to suit different trader preferences
- Trades exclusively S&P500 options

## Features

- Volatility assessment using historical price data and VIX
- Risk management with configurable risk profiles (Low, Medium, High, Aggressive)
- Options strategy selection based on market conditions
- Backtesting engine for strategy evaluation
- Performance reporting and visualization

## Installation

### Prerequisites

- Python 3.8+
- Interactive Brokers account
- IB Gateway or Trader Workstation (TWS) installed

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/mazharm/turingtrader.git
   cd turingtrader
   ```

2. Install dependencies:
   ```
   pip install -e .
   ```
   
   Or using requirements.txt:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Trading Agent

Run the trading agent with:

```
python -m src.turingtrader.agent --risk-level MEDIUM
```

Options:
- `--host`: IB Gateway/TWS hostname (default: 127.0.0.1)
- `--port`: IB Gateway/TWS port (default: 7497)
- `--client-id`: Client ID for IB connection
- `--risk-level`: Risk level (LOW, MEDIUM, HIGH, AGGRESSIVE)
- `--data-dir`: Directory for saving trading data

### Backtesting

Backtest your strategies with:

```
python scripts/backtest.py --start-date 2022-01-01 --end-date 2022-12-31 --risk-level MEDIUM
```

Options:
- `--start-date`: Start date for backtesting (YYYY-MM-DD)
- `--end-date`: End date for backtesting (YYYY-MM-DD)
- `--risk-level`: Risk level for backtesting
- `--initial-capital`: Initial capital for backtest
- `--strategies`: List of strategies to backtest
- `--high-vol-only`: Only trade on high volatility days
- `--output-file`: Output file for performance report

### Performance Reports

Generate performance reports with:

```
python scripts/generate_report.py --trading-report --risk-report
```

Options:
- `--data-dir`: Directory containing trading data
- `--backtest-dir`: Directory containing backtest results
- `--output-dir`: Directory for saving reports
- `--start-date`: Start date for report
- `--end-date`: End date for report
- `--trading-report`: Generate trading performance report
- `--risk-report`: Generate risk comparison report
- `--show-plot`: Display plots

## Risk Profiles

The system supports four risk levels:

1. **LOW**: Conservative approach with smaller position sizes and tighter stop losses
2. **MEDIUM**: Balanced approach with moderate position sizes
3. **HIGH**: More aggressive approach with larger position sizes
4. **AGGRESSIVE**: Highest risk level with largest positions and wider stop losses

## License

This project is licensed under the MIT License - see the LICENSE file for details.
