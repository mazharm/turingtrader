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
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    max_connections: int = 10


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
    
    # Maximum percentage of account to risk per day
    max_daily_risk_pct: float = 2.0
    
    # Minimum volatility threshold to enter positions (annualized %)
    min_volatility_threshold: float = 15.0
    
    # Maximum position size as percentage of account
    max_position_size_pct: float = 20.0
    
    # Maximum options delta exposure
    max_delta_exposure: float = 50.0
    
    # Stop loss percentage per position
    stop_loss_pct: float = 20.0
    
    # Target profit percentage per position
    target_profit_pct: float = 30.0
    
    # Minimum volatility change to trigger trading (percentage points)
    min_volatility_change: float = 2.0

    # For Iron Condors: factor of max risk for stop loss (e.g., 50 means SL if loss = 50% of max risk)
    condor_stop_loss_factor_of_max_risk: float = 50.0 
    # For Iron Condors: factor of credit for profit target (e.g., 80 means TP if 80% of credit is retained)
    condor_profit_target_factor_of_credit: float = 80.0
    
    def adjust_for_risk_level(self, level: int) -> None:
        """Adjust risk parameters based on risk level (1-10)."""
        if not 1 <= level <= 10:
            raise ValueError("Risk level must be between 1 and 10")
            
        # Scale parameters according to risk level
        self.risk_level = level
        # Lower risk means higher volatility threshold for entry
        self.min_volatility_threshold = 10 + (10 - level) * 2
        # Lower risk means lower daily risk
        self.max_daily_risk_pct = level * 0.5
        # Lower risk means smaller position size
        self.max_position_size_pct = 5 + level * 3  
        # Lower risk means lower delta exposure
        self.max_delta_exposure = 10 + level * 8
        # Lower risk means tighter stop loss
        self.stop_loss_pct = 5 + level * 2
        # Lower risk means lower but more achievable target profit
        self.target_profit_pct = 10 + level * 3
        # Lower risk means requiring larger volatility changes to trade
        self.min_volatility_change = 5 - (level * 0.4)


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


@dataclass
class VolatilityHarvestingConfig:
    """Configuration for the Adaptive Volatility-Harvesting System."""
    # IV/HV ratio threshold for signal generation
    iv_hv_ratio_threshold: float = 1.2
    
    # Minimum IV required for trade entry
    min_iv_threshold: float = 25.0
    
    # Use adaptive thresholds based on market conditions
    use_adaptive_thresholds: bool = True
    
    # Range around calculated strikes for iron condor legs (percentage)
    strike_width_pct: float = 1.0
    
    # Target delta for short options legs
    target_short_delta: float = 0.3
    
    # Target delta for long options legs
    target_long_delta: float = 0.1
    
    # Default days to expiration range
    min_dte: int = 14
    max_dte: int = 45


@dataclass
class AppConfig:
    """Main application configuration, aggregating all specific configurations."""
    
    ibkr: IBKRConfig = None
    risk: RiskParameters = None
    trading: TradingConfig = None
    postgres: PostgresConfig = None
    redis: RedisConfig = None
    vol_harvesting: VolatilityHarvestingConfig = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.ibkr is None:
            self.ibkr = IBKRConfig()
        if self.risk is None:
            self.risk = RiskParameters()
        if self.trading is None:
            self.trading = TradingConfig()
        if self.postgres is None:
            self.postgres = PostgresConfig()
        if self.redis is None:
            self.redis = RedisConfig()
        if self.vol_harvesting is None:
            self.vol_harvesting = VolatilityHarvestingConfig()
        
    def _get_config_value(self, parser: configparser.ConfigParser, section: str, key: str, type_converter, default: Any = None) -> Any:
        """Helper to safely get and convert config values."""
        try:
            return type_converter(parser.get(section, key))
        except (configparser.NoSectionError, configparser.NoOptionError):
            logger.warning(f"Configuration key '{key}' not found in section '{section}'. Using default: {default}")
            if default is None and type_converter is bool: # Handle boolean defaults specifically if parser.getboolean is not used directly
                 return False # Or handle as per desired default boolean logic
            return default
        except ValueError as e:
            logger.error(f"Invalid value for configuration: section='{section}', key='{key}'. Error: {e}. Using default: {default}")
            return default

    def load_config(self, config_file_path: str = "config.ini") -> None:
        """Load configuration from a .ini file, overriding defaults."""
        if not os.path.exists(config_file_path):
            logger.warning(f"Configuration file '{config_file_path}' not found. Using default settings for all configurations.")
            # Ensure risk parameters are adjusted even when using defaults
            if hasattr(self.risk, 'risk_level'):
                self.risk.adjust_for_risk_level(self.risk.risk_level)
            return

        parser = configparser.ConfigParser()
        # Add default values for boolean conversion if not present
        parser.BOOLEAN_STATES.update({'': False, 'false': False, 'no': False, '0': False, 'true': True, 'yes': True, '1': True})
        
        try:
            parser.read(config_file_path)
        except configparser.Error as e:
            logger.error(f"Error reading configuration file '{config_file_path}': {e}. Using default settings.")
            if hasattr(self.risk, 'risk_level'):
                self.risk.adjust_for_risk_level(self.risk.risk_level)
            return

        # Load IBKR settings
        if parser.has_section("IBKR"):
            self.ibkr.host = self._get_config_value(parser, "IBKR", "host", str, self.ibkr.host)
            self.ibkr.port = self._get_config_value(parser, "IBKR", "port", int, self.ibkr.port)
            self.ibkr.client_id = self._get_config_value(parser, "IBKR", "client_id", int, self.ibkr.client_id)
            self.ibkr.timeout = self._get_config_value(parser, "IBKR", "timeout", int, self.ibkr.timeout)
            self.ibkr.read_only = self._get_config_value(parser, "IBKR", "read_only", parser.getboolean, self.ibkr.read_only)
            self.ibkr.default_options_exchange = self._get_config_value(parser, "IBKR", "default_options_exchange", str, self.ibkr.default_options_exchange)


        # Load Risk settings
        if parser.has_section("Risk"):
            self.risk.risk_level = self._get_config_value(parser, "Risk", "risk_level", int, self.risk.risk_level)
            self.risk.max_daily_risk_pct = self._get_config_value(parser, "Risk", "max_daily_risk_pct", float, self.risk.max_daily_risk_pct)
            self.risk.min_volatility_threshold = self._get_config_value(parser, "Risk", "min_volatility_threshold", float, self.risk.min_volatility_threshold)
            self.risk.max_position_size_pct = self._get_config_value(parser, "Risk", "max_position_size_pct", float, self.risk.max_position_size_pct)
            self.risk.max_delta_exposure = self._get_config_value(parser, "Risk", "max_delta_exposure", float, self.risk.max_delta_exposure)
            self.risk.stop_loss_pct = self._get_config_value(parser, "Risk", "stop_loss_pct", float, self.risk.stop_loss_pct) # Assuming this was intended
            self.risk.target_profit_pct = self._get_config_value(parser, "Risk", "target_profit_pct", float, self.risk.target_profit_pct) # Assuming this was intended
            self.risk.min_volatility_change = self._get_config_value(parser, "Risk", "min_volatility_change", float, self.risk.min_volatility_change) # Assuming this was intended
            self.risk.condor_stop_loss_factor_of_max_risk = self._get_config_value(parser, "Risk", "condor_stop_loss_factor_of_max_risk", float, self.risk.condor_stop_loss_factor_of_max_risk)
            self.risk.condor_profit_target_factor_of_credit = self._get_config_value(parser, "Risk", "condor_profit_target_factor_of_credit", float, self.risk.condor_profit_target_factor_of_credit)
            
            # Adjust risk parameters based on the loaded (or default) risk_level
            self.risk.adjust_for_risk_level(self.risk.risk_level)

        # Load Trading settings
        if parser.has_section("Trading"):
            self.trading.index_symbol = self._get_config_value(parser, "Trading", "index_symbol", str, self.trading.index_symbol)
            self.trading.options_only = self._get_config_value(parser, "Trading", "options_only", parser.getboolean, self.trading.options_only)
            self.trading.trading_period_minutes = self._get_config_value(parser, "Trading", "trading_period_minutes", int, self.trading.trading_period_minutes)
            self.trading.day_start_offset_hours = self._get_config_value(parser, "Trading", "day_start_offset_hours", float, self.trading.day_start_offset_hours)
            self.trading.day_end_offset_hours = self._get_config_value(parser, "Trading", "day_end_offset_hours", float, self.trading.day_end_offset_hours)
            self.trading.default_order_type = self._get_config_value(parser, "Trading", "default_order_type", str, self.trading.default_order_type)

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
            self.postgres.use_timescaledb = self._get_config_value(parser, "PostgreSQL", "use_timescaledb", parser.getboolean, self.postgres.use_timescaledb)

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
            self.vol_harvesting.use_adaptive_thresholds = self._get_config_value(parser, "VolatilityHarvesting", "use_adaptive_thresholds", parser.getboolean, self.vol_harvesting.use_adaptive_thresholds)
            # Add other VolatilityHarvesting parameters as needed

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
        'condor_stop_loss_factor_of_max_risk': '50.0', # Example: SL if 50% of max risk is lost
        'condor_profit_target_factor_of_credit': '80.0' # Example: TP if 80% of credit can be kept (i.e. buy back for 20% of credit)
    }
    
    config['Trading'] = {
        'index_symbol': 'SPY',
        'options_only': 'True',
        'trading_period_minutes': '15',
        'day_start_offset_hours': '0.5',
        'day_end_offset_hours': '0.5',
        'default_order_type': 'MKT'
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
        'iv_hv_ratio_threshold': '1.2',
        'min_iv_threshold': '25.0',
        'use_adaptive_thresholds': 'True',
        'strike_width_pct': '1.0',
        'target_short_delta': '0.3',
        'target_long_delta': '0.1',
        'min_dte': '14',
        'max_dte': '45'
    }
    
    with open(filename, 'w') as f:
        config.write(f)
        
    print(f"Created default configuration file: {filename}")


if __name__ == "__main__":
    # If run directly, create a default config file
    create_default_config_file()

# For backward compatibility
Config = AppConfig