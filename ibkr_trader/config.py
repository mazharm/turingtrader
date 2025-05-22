"""
Configuration settings for the TuringTrader algorithm.
"""

import configparser
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class IBKRConfig:
    """Interactive Brokers connection configuration."""
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 for paper trading, 7496 for live
    client_id: int = 1
    timeout: int = 60
    read_only: bool = False


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


class Config:
    """Main configuration class for TuringTrader."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from a config file.
        
        Args:
            config_path: Path to the config file. If None, looks for 'config.ini'
                         in the current directory.
        """
        self.ibkr = IBKRConfig()
        self.risk = RiskParameters()
        self.trading = TradingConfig()
        
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
        
        if os.path.exists(config_path):
            self._load_from_file(config_path)
    
    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from a file."""
        parser = configparser.ConfigParser()
        parser.read(config_path)
        
        # Load IBKR configuration
        if 'IBKR' in parser:
            ibkr_section = parser['IBKR']
            self.ibkr.host = ibkr_section.get('host', self.ibkr.host)
            self.ibkr.port = ibkr_section.getint('port', self.ibkr.port)
            self.ibkr.client_id = ibkr_section.getint('client_id', self.ibkr.client_id)
            self.ibkr.timeout = ibkr_section.getint('timeout', self.ibkr.timeout)
            self.ibkr.read_only = ibkr_section.getboolean('read_only', self.ibkr.read_only)
        
        # Load risk parameters
        if 'Risk' in parser:
            risk_section = parser['Risk']
            self.risk.risk_level = risk_section.getint('risk_level', self.risk.risk_level)
            self.risk.adjust_for_risk_level(self.risk.risk_level)
            
            # Override auto-adjusted parameters if explicitly set
            if 'max_daily_risk_pct' in risk_section:
                self.risk.max_daily_risk_pct = risk_section.getfloat('max_daily_risk_pct')
            if 'min_volatility_threshold' in risk_section:
                self.risk.min_volatility_threshold = risk_section.getfloat('min_volatility_threshold')
            if 'max_position_size_pct' in risk_section:
                self.risk.max_position_size_pct = risk_section.getfloat('max_position_size_pct')
        
        # Load trading configuration
        if 'Trading' in parser:
            trading_section = parser['Trading']
            self.trading.index_symbol = trading_section.get('index_symbol', self.trading.index_symbol)
            self.trading.options_only = trading_section.getboolean('options_only', self.trading.options_only)
            self.trading.trading_period_minutes = trading_section.getint('trading_period_minutes', 
                                                                     self.trading.trading_period_minutes)
            self.trading.day_start_offset_hours = trading_section.getfloat('day_start_offset_hours', 
                                                                       self.trading.day_start_offset_hours)
            self.trading.day_end_offset_hours = trading_section.getfloat('day_end_offset_hours', 
                                                                     self.trading.day_end_offset_hours)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'ibkr': {
                'host': self.ibkr.host,
                'port': self.ibkr.port,
                'client_id': self.ibkr.client_id,
                'timeout': self.ibkr.timeout,
                'read_only': self.ibkr.read_only
            },
            'risk': {
                'risk_level': self.risk.risk_level,
                'max_daily_risk_pct': self.risk.max_daily_risk_pct,
                'min_volatility_threshold': self.risk.min_volatility_threshold,
                'max_position_size_pct': self.risk.max_position_size_pct,
                'max_delta_exposure': self.risk.max_delta_exposure,
                'stop_loss_pct': self.risk.stop_loss_pct,
                'target_profit_pct': self.risk.target_profit_pct,
                'min_volatility_change': self.risk.min_volatility_change
            },
            'trading': {
                'index_symbol': self.trading.index_symbol,
                'options_only': self.trading.options_only,
                'trading_period_minutes': self.trading.trading_period_minutes,
                'day_start_offset_hours': self.trading.day_start_offset_hours,
                'day_end_offset_hours': self.trading.day_end_offset_hours,
                'default_order_type': self.trading.default_order_type
            }
        }


def create_default_config_file(filename: str = 'config.ini') -> None:
    """Create a default configuration file."""
    config = configparser.ConfigParser()
    
    config['IBKR'] = {
        'host': '127.0.0.1',
        'port': '7497',  # Paper trading
        'client_id': '1',
        'timeout': '60',
        'read_only': 'False'
    }
    
    config['Risk'] = {
        'risk_level': '5',
        'comment': 'Risk level range: 1 (lowest) to 10 (highest)'
    }
    
    config['Trading'] = {
        'index_symbol': 'SPY',
        'options_only': 'True',
        'trading_period_minutes': '15',
        'day_start_offset_hours': '0.5',
        'day_end_offset_hours': '0.5',
        'default_order_type': 'MKT'
    }
    
    with open(filename, 'w') as f:
        config.write(f)
        
    print(f"Created default configuration file: {filename}")


if __name__ == "__main__":
    # If run directly, create a default config file
    create_default_config_file()