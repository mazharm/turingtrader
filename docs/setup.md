# Setup and Installation Guide

This guide will help you set up the TuringTrader algorithmic trading system on your machine.

## Prerequisites

- Python 3.7+
- Interactive Brokers account and TWS or IB Gateway installed
- Operating system: Windows, macOS, or Linux

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/mazharm/turingtrader.git
cd turingtrader
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Interactive Brokers Connection

1. Start TWS or IB Gateway on your machine
2. Enable API connections in TWS/Gateway settings
   - TWS: File > Global Configuration > API > Settings
   - Check "Enable ActiveX and Socket Clients"
   - Set port (default: 7497 for paper trading, 7496 for live trading)

### 5. Configure TuringTrader

1. Create a configuration file:

```bash
python main.py --create-config
```

2. Edit the generated `config.ini` file with your settings:
   - Set IBKR connection parameters (host, port, client_id)
   - Configure risk parameters
   - Set trading parameters

## Running the Trading System

### Paper Trading Mode

```bash
python main.py --mode paper
```

### Live Trading Mode

```bash
python main.py --mode live
```

### Test Connection

```bash
python main.py --mode test
```

### Backtesting

```bash
python backtest.py --risk-level 5 --start-date 2022-01-01 --end-date 2022-12-31
```

## Troubleshooting

### Connection Issues

1. Ensure TWS/Gateway is running before starting TuringTrader
2. Verify the port numbers match in both TWS and your config file
3. Check that API connections are enabled in TWS/Gateway
4. Make sure the client ID is not used by another application

### API Permissions

1. In TWS, you may need to accept the API connection when prompted
2. For live trading, ensure your account has the necessary permissions
3. Paper trading accounts have different restrictions than live accounts

## Next Steps

After successful setup, proceed to:
1. [Usage Guide](usage.md) for detailed usage instructions
2. [Strategy Documentation](strategy.md) for details on the trading strategy