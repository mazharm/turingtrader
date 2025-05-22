# TuringTrader Setup Guide

This guide will help you set up the TuringTrader algorithmic trading system for use with Interactive Brokers.

## Prerequisites

- Python 3.8 or higher
- Interactive Brokers account
- IB Gateway or Trader Workstation (TWS)
- Basic familiarity with command line operations

## Installation Steps

### 1. Install Interactive Brokers Software

1. Download and install [IB Gateway](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php) or [Trader Workstation (TWS)](https://www.interactivebrokers.com/en/trading/tws-updateable-latest.php)
2. Launch the application and log in with your credentials
3. Enable API connections:
   - In TWS: File → Global Configuration → API → Settings
   - Check "Enable Active X and Socket Clients"
   - Set the port (default is 7497 for paper trading, 7496 for live trading)
   - Check "Allow connections from localhost only" for security

### 2. Clone and Install TuringTrader

1. Clone the repository:
   ```bash
   git clone https://github.com/mazharm/turingtrader.git
   cd turingtrader
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```
   
   Alternatively, install dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configure Connection Settings

If you're using a non-standard connection configuration, you'll need to specify the host, port, and client ID when running the trading agent:

```bash
python -m src.turingtrader.agent --host 127.0.0.1 --port 7497 --client-id 1
```

For paper trading, use port 7497. For live trading, use port 7496.

## Verifying the Installation

To verify that your installation is working correctly:

1. Ensure IB Gateway/TWS is running and connected with your account
2. Run the following command to test the connection:
   ```bash
   python -c "from src.turingtrader.ib_client import IBClient; client = IBClient(); print(client.connect())"
   ```
3. If successful, you should see `True` printed

## Configuration Options

### Risk Levels

The trading agent supports different risk levels, which affect position sizing and strategy parameters:

- **LOW**: Conservative approach (smaller positions, lower risk)
- **MEDIUM**: Balanced approach
- **HIGH**: More aggressive approach
- **AGGRESSIVE**: Highest risk level

You can set the risk level when launching the agent:

```bash
python -m src.turingtrader.agent --risk-level MEDIUM
```

### Data Directory

By default, trading data is saved to `~/.turingtrader/data`. You can specify a custom directory:

```bash
python -m src.turingtrader.agent --data-dir /path/to/data
```

## Troubleshooting

### Connection Issues

- Ensure IB Gateway/TWS is running before starting the trading agent
- Verify the port number matches your IB Gateway/TWS configuration
- Check that API connections are enabled in IB Gateway/TWS

### Package Import Errors

- Make sure you've activated the virtual environment if you created one
- Verify that all dependencies are installed (`pip install -r requirements.txt`)

### Permission Issues

- If you encounter permission issues when writing data, ensure the specified data directory is writable

## Next Steps

After setting up TuringTrader, refer to the [Usage Guide](usage.md) for details on how to use the trading agent, run backtests, and generate performance reports.