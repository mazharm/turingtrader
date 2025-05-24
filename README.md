# TuringTrader

An algorithmic trading system built on Interactive Brokers API, designed to capitalize on market volatility through S&P500 options trading.

## Features

- **Daily Cash Management**: Starts and ends each trading day in cash
- **Volatility-Based Strategy**: Generates profits during high volatility periods while staying cash during low volatility
- **Parameterized Risk Profiles**: Adjustable risk parameters for various risk/reward balances
- **S&P500 Options Trading**: Specialized in options trading on the S&P500 index
- **Comprehensive Backtesting**: Test performance on historical S&P500 data
- **Detailed Reporting**: Performance analysis across different risk profiles

## Installation

```bash
# Clone the repository
git clone https://github.com/mazharm/turingtrader.git
cd turingtrader

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `config.ini` file in the root directory based on the provided template
2. Set your Interactive Brokers account credentials and API configuration
3. Adjust risk parameters according to your preferences

## Usage

### Live Trading

```bash
python main.py --mode live
```

### Backtesting

```bash
python backtest.py --risk-level [1-10] --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

### Generate Reports

```bash
python -m utils.reporting --output-dir ./reports
```

### Algorithm Evaluation

Evaluate the complete trading algorithm's performance across all risk levels using historical data:

```bash
# Evaluate with default settings (past year, $100k initial investment)
python evaluate_algorithm.py

# Custom evaluation period and investment amount
python evaluate_algorithm.py --period 180 --initial-investment 250000 --output-dir ./my_evaluation

# Full options list
python evaluate_algorithm.py --help
```

The evaluation generates:
- Performance reports for each risk level (1-10)
- Comparative analysis of all risk levels
- Investment growth chart
- Summary statistics in CSV format

## Project Structure

- `ibkr_trader/`: Core trading components
- `historical_data/`: Historical data management
- `backtesting/`: Backtesting engine
- `utils/`: Utility functions and tools
- `tests/`: Test suite
- `docs/`: Detailed documentation

## License

[MIT License](LICENSE)
