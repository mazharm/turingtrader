# TuringTrader

AI-driven algorithmic trader focusing on volatility-based strategies using Interactive Brokers API.

## Features

- Intelligent volatility-based trading of S&P 500 options (SPX/SPY)
- Cash-only positions at the beginning and end of each trading day
- Adjustable risk profiles (Low, Medium, High)
- Real-time market data and volatility metrics
- Automated trade execution with stop-loss and take-profit settings
- Comprehensive performance reporting and logging

## Installation

1. Clone this repository
```
git clone https://github.com/mazharm/turingtrader.git
cd turingtrader
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Configure settings in `src/turing_trader/config/settings.py`

## Usage

```
python src/turing_trader/main.py --risk medium
```

## Requirements

- Python 3.8+
- Interactive Brokers TWS API or IB Gateway
- Active Interactive Brokers account with market data subscriptions
