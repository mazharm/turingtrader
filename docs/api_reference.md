# API Reference

This document provides a reference guide to the key components and APIs of the TuringTrader algorithmic trading system.

## Core Components

### TuringTrader Class
`ibkr_trader.trader.TuringTrader`

**Description**: Main trader class that orchestrates the entire trading strategy.

**Key Methods**:
- `connect()`: Connect to Interactive Brokers
- `disconnect()`: Disconnect from Interactive Brokers
- `update_account_info()`: Update account information
- `check_market_status()`: Check if market is open and in trading hours
- `fetch_market_data()`: Fetch current market data including VIX
- `analyze_market_conditions()`: Analyze current market conditions
- `fetch_option_chain()`: Fetch current option chain for the index
- `execute_trade()`: Execute a trade based on the trade decision
- `close_all_positions()`: Close all open positions
- `run_trading_cycle()`: Run a single trading decision cycle
- `run_trading_loop()`: Run the continuous trading loop
- `run_backtest()`: Run a backtest over a specified date range

### IBConnector Class
`ibkr_trader.ib_connector.IBConnector`

**Description**: Handles communication with Interactive Brokers API.

**Key Methods**:
- `connect()`: Connect to Interactive Brokers TWS or Gateway
- `disconnect()`: Disconnect from Interactive Brokers
- `check_connection()`: Check if connection is active
- `get_account_summary()`: Get account summary information
- `get_market_data()`: Get historical market data for a symbol
- `get_vix_data()`: Get VIX data as a volatility indicator
- `get_option_chain()`: Get the full option chain for a symbol
- `submit_order()`: Submit an order to Interactive Brokers
- `close_all_positions()`: Close all open positions
- `is_market_open()`: Check if the market is currently open

### VolatilityAnalyzer Class
`ibkr_trader.volatility_analyzer.VolatilityAnalyzer`

**Description**: Analyzes market volatility to determine trading decisions.

**Key Methods**:
- `calculate_historical_volatility()`: Calculate historical volatility based on price history
- `calculate_implied_volatility()`: Calculate implied volatility using the Black-Scholes model
- `analyze_vix()`: Analyze VIX data to determine market volatility state
- `analyze_option_chain()`: Analyze option chain for trading opportunities
- `should_trade_today()`: Determine if we should trade today based on volatility
- `get_position_size_multiplier()`: Calculate position size multiplier based on volatility

### OptionsStrategy Class
`ibkr_trader.options_strategy.OptionsStrategy`

**Description**: Handles S&P500 options trading strategy decisions.

**Key Methods**:
- `select_options_for_volatility()`: Select appropriate options based on volatility
- `generate_trade_decision()`: Generate a trading decision based on market conditions
- `should_close_positions()`: Determine if positions should be closed
- `reset_daily_state()`: Reset daily state for new trading day

### RiskManager Class
`ibkr_trader.risk_manager.RiskManager`

**Description**: Manages trading risk parameters and position sizing.

**Key Methods**:
- `update_account_value()`: Update account value and risk limits
- `update_risk_level()`: Update the risk level
- `calculate_position_size()`: Calculate position size for stocks
- `calculate_option_quantity()`: Calculate quantity for options
- `add_position()`: Add a new position to tracking
- `close_position()`: Close a position and calculate final P&L
- `close_all_positions()`: Close all open positions at current market prices
- `should_close_for_day()`: Determine if we should close all positions for the day

## Backtesting Components

### BacktestEngine Class
`backtesting.backtest_engine.BacktestEngine`

**Description**: Engine for backtesting the TuringTrader algorithm.

**Key Methods**:
- `run_backtest()`: Run a backtest over a specified date range
- `_process_trading_day()`: Process a single trading day in the backtest
- `_execute_trade()`: Execute a trade in the backtest
- `_close_positions()`: Close positions at the end of day
- `_calculate_performance_metrics()`: Calculate performance metrics from backtest results

### PerformanceAnalyzer Class
`backtesting.performance_analyzer.PerformanceAnalyzer`

**Description**: Analyze and report on trading performance.

**Key Methods**:
- `analyze_results()`: Analyze backtest results
- `compare_risk_levels()`: Compare results across different risk levels
- `generate_report()`: Generate a performance report
- `generate_comparison_report()`: Generate a report comparing risk levels

## Data Management Components

### HistoricalDataFetcher Class
`historical_data.data_fetcher.HistoricalDataFetcher`

**Description**: Fetch and manage historical market data.

**Key Methods**:
- `fetch_data()`: Fetch historical market data for a symbol
- `fetch_vix_data()`: Fetch VIX data
- `fetch_sp500_data()`: Fetch S&P 500 data
- `fetch_option_chain()`: Fetch option chain data for a symbol
- `clear_cache()`: Clear cached data

### DataProcessor Class
`historical_data.data_processor.DataProcessor`

**Description**: Process and prepare historical market data.

**Key Methods**:
- `calculate_historical_volatility()`: Calculate historical volatility from price data
- `calculate_technical_indicators()`: Calculate technical indicators
- `merge_market_data()`: Merge underlying and VIX data
- `prepare_backtest_data()`: Prepare data for backtesting
- `generate_option_prices()`: Generate synthetic option prices using Black-Scholes

## Configuration

### Config Class
`ibkr_trader.config.Config`

**Description**: Configuration class for TuringTrader.

**Key Components**:
- `IBKRConfig`: Interactive Brokers connection configuration
- `RiskParameters`: Risk management parameters
- `TradingConfig`: Overall trading configuration

**Key Methods**:
- `_load_from_file()`: Load configuration from file
- `to_dict()`: Convert configuration to dictionary

## Utility Functions

### Reporting Module
`utils.reporting`

**Key Functions**:
- `generate_report()`: Generate a performance report for a backtest
- `generate_multi_risk_report()`: Generate a report comparing multiple risk levels
- `export_results_to_csv()`: Export backtest results to CSV
- `export_multi_risk_summary()`: Export summary of multiple risk levels

### Logging Utilities
`utils.logging_util`

**Key Functions**:
- `setup_logging()`: Set up logging configuration
- `get_logger()`: Get a logger with the specified name