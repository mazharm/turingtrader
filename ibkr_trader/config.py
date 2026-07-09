"""
Configuration settings for the TuringTrader algorithm.
"""

import configparser
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging # Added for logging

logger = logging.getLogger(__name__) # Added for logging


@dataclass
class PostgresConfig:
    """PostgreSQL/TimescaleDB configuration."""
    host: str = "localhost"
    port: int = 5432
    username: str = "postgres"
    password: str = ""
    database: str = "turingtrader"
    schema: str = "public"
    ssl_mode: str = "prefer"
    min_connections: int = 1
    max_connections: int = 10
    use_timescaledb: bool = True

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string (short connect timeout so an
        unreachable optional database cannot stall trading startup)."""
        return (f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}"
                f"/{self.database}?sslmode={self.ssl_mode}&connect_timeout=3")


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    max_connections: int = 10

    def get_connection_string(self) -> str:
        """Get Redis connection URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"


@dataclass
class IBKRConfig:
    """Interactive Brokers connection configuration."""
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 for paper trading, 7496 for live
    client_id: int = 1
    timeout: int = 60
    read_only: bool = False
    default_options_exchange: str = "BOX" # Added for specifying options exchange


@dataclass
class RiskParameters:
    """Risk management parameters for the trading algorithm."""
    # Risk level (1-10, where 10 is highest risk)
    risk_level: int = 5
    
    # Maximum percentage of account to risk per day (further reduced for capital preservation)
    max_daily_risk_pct: float = 1.2
    
    # Minimum volatility threshold to enter positions (annualized %, increased for more selective entry)
    min_volatility_threshold: float = 20.0
    
    # Maximum position size as percentage of account (significantly reduced for better capital management)
    max_position_size_pct: float = 8.0
    
    # Maximum options delta exposure (substantially reduced to limit risk)
    max_delta_exposure: float = 25.0
    
    # Stop loss percentage per position (tightened to minimize drawdowns)
    stop_loss_pct: float = 12.0
    
    # Target profit percentage per position (reduced for more achievable targets)
    target_profit_pct: float = 18.0
    
    # Minimum volatility change to trigger trading (increased for more selective entry)
    min_volatility_change: float = 3.0

    # For Iron Condors: factor of max risk for stop loss (reduced for tighter stop losses)
    # e.g., 30 means stop loss if loss = 30% of max risk
    condor_stop_loss_factor_of_max_risk: float = 30.0 
    # For Iron Condors: factor of credit for profit target (increased to take profits earlier)
    # e.g., 75 means take profit when 75% of credit can be retained (buy back at 25% of credit)
    condor_profit_target_factor_of_credit: float = 75.0
    
    def adjust_for_risk_level(self, level: int) -> None:
        """Adjust risk parameters based on risk level (1-10)."""
        if not 1 <= level <= 10:
            raise ValueError("Risk level must be between 1 and 10")
            
        # Scale parameters according to risk level
        self.risk_level = level
        
        # Volatility threshold for entry - lower levels need higher VIX to trade
        self.min_volatility_threshold = 15 + (10 - level) * 1.5

        # Daily risk as percentage of portfolio
        self.max_daily_risk_pct = 0.5 + level * 0.3

        # Max position size as percentage of portfolio
        self.max_position_size_pct = 2.0 + level * 1.5

        # Max delta exposure
        self.max_delta_exposure = 5 + level * 3

        # Stop loss percentage
        self.stop_loss_pct = 5 + level * 1.0

        # Target profit percentage
        self.target_profit_pct = 8 + level * 1.2

        # Min volatility change to trigger a trade
        self.min_volatility_change = 3.0 - (level * 0.15)
        
        # Iron Condor risk parameters based on risk level
        # Stop loss as percentage of max risk - tighter at lower risk levels
        self.condor_stop_loss_factor_of_max_risk = 18 + (level * 1.5)
        # Profit target as percentage of credit received - take profits earlier at lower risk levels
        self.condor_profit_target_factor_of_credit = 75 + (10 - level) * 3.0


@dataclass
class TradingConfig:
    """Overall trading configuration."""
    # Symbol for S&P500
    index_symbol: str = "SPY"

    # Whether to trade in options only
    options_only: bool = True

    # Default trading period (in minutes)
    trading_period_minutes: int = 15

    # Start of trading day (hours from market open)
    day_start_offset_hours: float = 0.5

    # End of trading day (hours before market close)
    day_end_offset_hours: float = 0.5

    # Default order type
    default_order_type: str = "MKT"

    # Market timezone (IANA name); US equity/options market hours are quoted in this zone
    market_timezone: str = "America/New_York"

    # Safety limits for the live trading loop
    max_daily_trades: int = 5              # Max new positions opened per day
    max_consecutive_errors: int = 5        # Trading halts after this many consecutive cycle errors
    kill_switch_file: str = "KILL_SWITCH"  # If this file exists, flatten everything and halt
    order_fill_timeout_seconds: int = 15   # How long to wait for a working order before repricing
    order_max_attempts: int = 3            # Price-walk attempts before giving up on an entry


@dataclass
class VolatilityHarvestingConfig:
    """Configuration for the Adaptive Volatility-Harvesting System."""
    # IV/HV ratio threshold for signal generation
    iv_hv_ratio_threshold: float = 1.2

    # Minimum IV required for trade entry
    min_iv_threshold: float = 18.0

    # Use adaptive thresholds based on market conditions
    use_adaptive_thresholds: bool = True

    # Range around calculated strikes for iron condor legs (percentage)
    strike_width_pct: float = 5.0

    # Target delta for short options legs (0.15-0.20 is standard for iron condors)
    target_short_delta: float = 0.18

    # Target delta for long options legs
    target_long_delta: float = 0.08

    # Days to expiration range (25-45 days is the sweet spot for theta decay)
    min_dte: int = 21
    max_dte: int = 45


def _str_to_bool(value: str) -> bool:
    """Convert a config string to a boolean."""
    if isinstance(value, bool):
        return value
    truthy = {'true', 'yes', '1', 'on'}
    falsy = {'false', 'no', '0', 'off', ''}
    lowered = str(value).strip().lower()
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise ValueError(f"Not a boolean: {value!r}")


class AppConfig:
    """Main application configuration, aggregating all specific configurations.

    Can be constructed with an optional path to a .ini file:
        Config()              -> defaults (loads ./config.ini if present)
        Config('my.ini')      -> defaults overridden by my.ini (warns if missing)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.ibkr = IBKRConfig()
        self.risk = RiskParameters()
        self.trading = TradingConfig()
        self.postgres = PostgresConfig()
        self.redis = RedisConfig()
        self.vol_harvesting = VolatilityHarvestingConfig()

        if config_path:
            self.load_config(config_path)
        elif os.path.exists("config.ini"):
            self.load_config("config.ini")

    def _get_config_value(self, parser: configparser.ConfigParser, section: str, key: str, type_converter, default: Any = None) -> Any:
        """Helper to safely get and convert config values."""
        try:
            return type_converter(parser.get(section, key))
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid value for configuration: section='{section}', key='{key}'. Error: {e}. Using default: {default}")
            return default

    def load_config(self, config_file_path: str = "config.ini") -> None:
        """Load configuration from a .ini file, overriding defaults."""
        if not os.path.exists(config_file_path):
            logger.warning(f"Configuration file '{config_file_path}' not found. Using default settings for all configurations.")
            return

        parser = configparser.ConfigParser()

        try:
            parser.read(config_file_path)
        except configparser.Error as e:
            logger.error(f"Error reading configuration file '{config_file_path}': {e}. Using default settings.")
            return

        # Load IBKR settings
        if parser.has_section("IBKR"):
            self.ibkr.host = self._get_config_value(parser, "IBKR", "host", str, self.ibkr.host)
            self.ibkr.port = self._get_config_value(parser, "IBKR", "port", int, self.ibkr.port)
            self.ibkr.client_id = self._get_config_value(parser, "IBKR", "client_id", int, self.ibkr.client_id)
            self.ibkr.timeout = self._get_config_value(parser, "IBKR", "timeout", int, self.ibkr.timeout)
            self.ibkr.read_only = self._get_config_value(parser, "IBKR", "read_only", _str_to_bool, self.ibkr.read_only)
            self.ibkr.default_options_exchange = self._get_config_value(parser, "IBKR", "default_options_exchange", str, self.ibkr.default_options_exchange)


        # Load Risk settings
        if parser.has_section("Risk"):
            # Apply the risk level scaling FIRST, then let explicitly configured
            # per-parameter values override the scaled ones.
            self.risk.risk_level = self._get_config_value(parser, "Risk", "risk_level", int, self.risk.risk_level)
            self.risk.adjust_for_risk_level(self.risk.risk_level)

            self.risk.max_daily_risk_pct = self._get_config_value(parser, "Risk", "max_daily_risk_pct", float, self.risk.max_daily_risk_pct)
            self.risk.min_volatility_threshold = self._get_config_value(parser, "Risk", "min_volatility_threshold", float, self.risk.min_volatility_threshold)
            self.risk.max_position_size_pct = self._get_config_value(parser, "Risk", "max_position_size_pct", float, self.risk.max_position_size_pct)
            self.risk.max_delta_exposure = self._get_config_value(parser, "Risk", "max_delta_exposure", float, self.risk.max_delta_exposure)
            self.risk.stop_loss_pct = self._get_config_value(parser, "Risk", "stop_loss_pct", float, self.risk.stop_loss_pct)
            self.risk.target_profit_pct = self._get_config_value(parser, "Risk", "target_profit_pct", float, self.risk.target_profit_pct)
            self.risk.min_volatility_change = self._get_config_value(parser, "Risk", "min_volatility_change", float, self.risk.min_volatility_change)
            self.risk.condor_stop_loss_factor_of_max_risk = self._get_config_value(parser, "Risk", "condor_stop_loss_factor_of_max_risk", float, self.risk.condor_stop_loss_factor_of_max_risk)
            self.risk.condor_profit_target_factor_of_credit = self._get_config_value(parser, "Risk", "condor_profit_target_factor_of_credit", float, self.risk.condor_profit_target_factor_of_credit)

        # Load Trading settings
        if parser.has_section("Trading"):
            self.trading.index_symbol = self._get_config_value(parser, "Trading", "index_symbol", str, self.trading.index_symbol)
            self.trading.options_only = self._get_config_value(parser, "Trading", "options_only", _str_to_bool, self.trading.options_only)
            self.trading.trading_period_minutes = self._get_config_value(parser, "Trading", "trading_period_minutes", int, self.trading.trading_period_minutes)
            self.trading.day_start_offset_hours = self._get_config_value(parser, "Trading", "day_start_offset_hours", float, self.trading.day_start_offset_hours)
            self.trading.day_end_offset_hours = self._get_config_value(parser, "Trading", "day_end_offset_hours", float, self.trading.day_end_offset_hours)
            self.trading.default_order_type = self._get_config_value(parser, "Trading", "default_order_type", str, self.trading.default_order_type)
            self.trading.market_timezone = self._get_config_value(parser, "Trading", "market_timezone", str, self.trading.market_timezone)
            self.trading.max_daily_trades = self._get_config_value(parser, "Trading", "max_daily_trades", int, self.trading.max_daily_trades)
            self.trading.max_consecutive_errors = self._get_config_value(parser, "Trading", "max_consecutive_errors", int, self.trading.max_consecutive_errors)
            self.trading.kill_switch_file = self._get_config_value(parser, "Trading", "kill_switch_file", str, self.trading.kill_switch_file)
            self.trading.order_fill_timeout_seconds = self._get_config_value(parser, "Trading", "order_fill_timeout_seconds", int, self.trading.order_fill_timeout_seconds)
            self.trading.order_max_attempts = self._get_config_value(parser, "Trading", "order_max_attempts", int, self.trading.order_max_attempts)

        # Load PostgreSQL settings
        if parser.has_section("PostgreSQL"):
            self.postgres.host = self._get_config_value(parser, "PostgreSQL", "host", str, self.postgres.host)
            self.postgres.port = self._get_config_value(parser, "PostgreSQL", "port", int, self.postgres.port)
            self.postgres.username = self._get_config_value(parser, "PostgreSQL", "username", str, self.postgres.username)
            self.postgres.password = self._get_config_value(parser, "PostgreSQL", "password", str, self.postgres.password)
            self.postgres.database = self._get_config_value(parser, "PostgreSQL", "database", str, self.postgres.database)
            self.postgres.schema = self._get_config_value(parser, "PostgreSQL", "schema", str, self.postgres.schema)
            self.postgres.ssl_mode = self._get_config_value(parser, "PostgreSQL", "ssl_mode", str, self.postgres.ssl_mode)
            self.postgres.min_connections = self._get_config_value(parser, "PostgreSQL", "min_connections", int, self.postgres.min_connections)
            self.postgres.max_connections = self._get_config_value(parser, "PostgreSQL", "max_connections", int, self.postgres.max_connections)
            self.postgres.use_timescaledb = self._get_config_value(parser, "PostgreSQL", "use_timescaledb", _str_to_bool, self.postgres.use_timescaledb)

        # Load Redis settings
        if parser.has_section("Redis"):
            self.redis.host = self._get_config_value(parser, "Redis", "host", str, self.redis.host)
            self.redis.port = self._get_config_value(parser, "Redis", "port", int, self.redis.port)
            self.redis.password = self._get_config_value(parser, "Redis", "password", str, self.redis.password)
            self.redis.database = self._get_config_value(parser, "Redis", "database", int, self.redis.database)
            self.redis.max_connections = self._get_config_value(parser, "Redis", "max_connections", int, self.redis.max_connections)
            # Assuming default_expiry is part of RedisConfig or handled elsewhere if needed from .ini
            # self.redis.default_expiry = self._get_config_value(parser, "Redis", "default_expiry", int, self.redis.default_expiry)


        # Load VolatilityHarvesting settings
        if parser.has_section("VolatilityHarvesting"):
            self.vol_harvesting.iv_hv_ratio_threshold = self._get_config_value(parser, "VolatilityHarvesting", "iv_hv_ratio_threshold", float, self.vol_harvesting.iv_hv_ratio_threshold)
            self.vol_harvesting.min_iv_threshold = self._get_config_value(parser, "VolatilityHarvesting", "min_iv_threshold", float, self.vol_harvesting.min_iv_threshold)
            self.vol_harvesting.use_adaptive_thresholds = self._get_config_value(parser, "VolatilityHarvesting", "use_adaptive_thresholds", _str_to_bool, self.vol_harvesting.use_adaptive_thresholds)
            self.vol_harvesting.strike_width_pct = self._get_config_value(parser, "VolatilityHarvesting", "strike_width_pct", float, self.vol_harvesting.strike_width_pct)
            self.vol_harvesting.target_short_delta = self._get_config_value(parser, "VolatilityHarvesting", "target_short_delta", float, self.vol_harvesting.target_short_delta)
            self.vol_harvesting.target_long_delta = self._get_config_value(parser, "VolatilityHarvesting", "target_long_delta", float, self.vol_harvesting.target_long_delta)
            self.vol_harvesting.min_dte = self._get_config_value(parser, "VolatilityHarvesting", "min_dte", int, self.vol_harvesting.min_dte)
            self.vol_harvesting.max_dte = self._get_config_value(parser, "VolatilityHarvesting", "max_dte", int, self.vol_harvesting.max_dte)

        logger.info("Configuration loaded successfully.")

def create_default_config_file(filename: str = 'config.ini') -> None:
    """Create a default configuration file."""
    config = configparser.ConfigParser()
    
    config['IBKR'] = {
        'host': '127.0.0.1',
        'port': '7497',  # Paper trading
        'client_id': '1',
        'timeout': '60',
        'read_only': 'False',
        'default_options_exchange': 'BOX'
    }
    
    config['Risk'] = {
        'risk_level': '5',
        'comment': 'Risk level range: 1 (lowest) to 10 (highest)',
        'condor_stop_loss_factor_of_max_risk': '40.0', # Example: SL if 40% of max risk is lost
        'condor_profit_target_factor_of_credit': '85.0' # Example: TP if 85% of credit can be kept (i.e. buy back for 15% of credit)
    }
    
    config['Trading'] = {
        'index_symbol': 'SPY',
        'options_only': 'True',
        'trading_period_minutes': '15',
        'day_start_offset_hours': '0.5',
        'day_end_offset_hours': '0.5',
        'default_order_type': 'MKT',
        'market_timezone': 'America/New_York',
        'max_daily_trades': '5',
        'max_consecutive_errors': '5',
        'kill_switch_file': 'KILL_SWITCH',
        'order_fill_timeout_seconds': '15',
        'order_max_attempts': '3'
    }
    
    config['PostgreSQL'] = {
        'host': 'localhost',
        'port': '5432',
        'username': 'postgres',
        'password': '',
        'database': 'turingtrader',
        'use_timescaledb': 'True'
    }
    
    config['Redis'] = {
        'host': 'localhost',
        'port': '6379',
        'password': '',
        'database': '0',
        'default_expiry': '3600'
    }
    
    config['VolatilityHarvesting'] = {
        'iv_hv_ratio_threshold': '1.5',
        'min_iv_threshold': '30.0',
        'use_adaptive_thresholds': 'True',
        'strike_width_pct': '7.5',
        'target_short_delta': '0.25',
        'target_long_delta': '0.10',
        'min_dte': '18',
        'max_dte': '35'
    }
    
    with open(filename, 'w') as f:
        config.write(f)
        
    print(f"Created default configuration file: {filename}")


if __name__ == "__main__":
    # If run directly, create a default config file
    create_default_config_file()

# For backward compatibility
Config = AppConfig