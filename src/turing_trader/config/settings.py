"""
Settings and Configuration Module

This module manages all application settings and configurations:
- Interactive Brokers connection parameters
- Trading parameters and thresholds
- Risk management settings
- Logging configuration
"""
import os
import logging
import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path
import configparser

# Try to load .env file if exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class RiskLevel(Enum):
    """Risk tolerance levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Settings:
    """Application settings manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize settings manager
        
        Args:
            config_file: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Default settings
        self._set_defaults()
        
        # Load from config file if provided
        if config_file:
            self.load_config_file(config_file)
            
        # Override with environment variables
        self.load_environment_variables()
    
    def _set_defaults(self):
        """Set default settings values"""
        # Interactive Brokers API settings
        self.ib_host = "127.0.0.1"
        self.ib_port = 7497  # 7496 for TWS, 7497 for IB Gateway paper trading
        self.ib_client_id = 1
        self.ib_account = ""  # Will be set from IB
        
        # Trading settings
        self.trading_mode = "paper"  # "paper" or "live"
        self.risk_level = RiskLevel.MEDIUM
        self.enable_auto_trading = False
        self.underlying_symbol = "SPY"  # SPY or SPX
        
        # Schedule settings
        self.market_open_time = datetime.time(9, 30)  # 9:30 AM ET
        self.market_close_time = datetime.time(16, 0)  # 4:00 PM ET
        self.liquidation_time = datetime.time(15, 45)  # 3:45 PM ET
        
        # Volatility settings
        self.min_vix_threshold = 15.0
        self.vix_spike_threshold = 10.0  # % increase for spike detection
        self.volatility_lookback_period = 20  # days
        
        # Risk management
        self.max_daily_drawdown_pct = 2.0  # Maximum daily drawdown
        self.max_position_pct = 10.0  # Maximum position size as % of account
        
        # Paths
        self.data_dir = "data"
        self.log_dir = "logs"
    
    def load_config_file(self, config_file: str) -> bool:
        """
        Load settings from configuration file
        
        Args:
            config_file: Path to config file
            
        Returns:
            bool: True if loaded successfully
        """
        if not os.path.exists(config_file):
            self.logger.warning(f"Config file not found: {config_file}")
            return False
            
        try:
            config = configparser.ConfigParser()
            config.read(config_file)
            
            # IB settings
            if 'InteractiveBrokers' in config:
                ib_section = config['InteractiveBrokers']
                self.ib_host = ib_section.get('host', self.ib_host)
                self.ib_port = ib_section.getint('port', self.ib_port)
                self.ib_client_id = ib_section.getint('client_id', self.ib_client_id)
                self.ib_account = ib_section.get('account', self.ib_account)
                
            # Trading settings
            if 'Trading' in config:
                trading_section = config['Trading']
                self.trading_mode = trading_section.get('mode', self.trading_mode)
                risk_level_str = trading_section.get('risk_level', self.risk_level.value)
                self.risk_level = RiskLevel(risk_level_str)
                self.enable_auto_trading = trading_section.getboolean('enable_auto_trading', 
                                                                    self.enable_auto_trading)
                self.underlying_symbol = trading_section.get('underlying_symbol', 
                                                          self.underlying_symbol)
                
            # Schedule settings
            if 'Schedule' in config:
                schedule_section = config['Schedule']
                
                # Parse time strings in format "HH:MM"
                if 'market_open' in schedule_section:
                    h, m = map(int, schedule_section.get('market_open').split(':'))
                    self.market_open_time = datetime.time(h, m)
                    
                if 'market_close' in schedule_section:
                    h, m = map(int, schedule_section.get('market_close').split(':'))
                    self.market_close_time = datetime.time(h, m)
                    
                if 'liquidation_time' in schedule_section:
                    h, m = map(int, schedule_section.get('liquidation_time').split(':'))
                    self.liquidation_time = datetime.time(h, m)
                    
            # Volatility settings
            if 'Volatility' in config:
                vol_section = config['Volatility']
                self.min_vix_threshold = vol_section.getfloat('min_vix_threshold', 
                                                           self.min_vix_threshold)
                self.vix_spike_threshold = vol_section.getfloat('vix_spike_threshold', 
                                                             self.vix_spike_threshold)
                self.volatility_lookback_period = vol_section.getint('lookback_period',
                                                                  self.volatility_lookback_period)
                
            # Risk management
            if 'RiskManagement' in config:
                risk_section = config['RiskManagement']
                self.max_daily_drawdown_pct = risk_section.getfloat('max_daily_drawdown_pct',
                                                                  self.max_daily_drawdown_pct)
                self.max_position_pct = risk_section.getfloat('max_position_pct',
                                                          self.max_position_pct)
                
            # Paths
            if 'Paths' in config:
                paths_section = config['Paths']
                self.data_dir = paths_section.get('data_dir', self.data_dir)
                self.log_dir = paths_section.get('log_dir', self.log_dir)
                
            self.logger.info(f"Loaded configuration from {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading config file: {str(e)}")
            return False
    
    def load_environment_variables(self):
        """Load settings from environment variables"""
        # IB settings
        self.ib_host = os.environ.get('TURING_IB_HOST', self.ib_host)
        self.ib_port = int(os.environ.get('TURING_IB_PORT', self.ib_port))
        self.ib_client_id = int(os.environ.get('TURING_IB_CLIENT_ID', self.ib_client_id))
        self.ib_account = os.environ.get('TURING_IB_ACCOUNT', self.ib_account)
        
        # Trading settings
        self.trading_mode = os.environ.get('TURING_TRADING_MODE', self.trading_mode)
        risk_level_str = os.environ.get('TURING_RISK_LEVEL', self.risk_level.value)
        try:
            self.risk_level = RiskLevel(risk_level_str)
        except ValueError:
            self.logger.warning(f"Invalid risk level in environment: {risk_level_str}")
            
        enable_auto_trading = os.environ.get('TURING_ENABLE_AUTO_TRADING', 
                                           str(self.enable_auto_trading).lower())
        self.enable_auto_trading = enable_auto_trading.lower() in ('true', '1', 'yes')
        
        self.underlying_symbol = os.environ.get('TURING_UNDERLYING_SYMBOL', 
                                             self.underlying_symbol)
        
        # Volatility settings
        self.min_vix_threshold = float(os.environ.get('TURING_MIN_VIX_THRESHOLD', 
                                                   self.min_vix_threshold))
                                                   
        # Paths
        self.data_dir = os.environ.get('TURING_DATA_DIR', self.data_dir)
        self.log_dir = os.environ.get('TURING_LOG_DIR', self.log_dir)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all settings to a dictionary
        
        Returns:
            dict: All settings
        """
        return {
            'ib_host': self.ib_host,
            'ib_port': self.ib_port,
            'ib_client_id': self.ib_client_id,
            'ib_account': self.ib_account,
            'trading_mode': self.trading_mode,
            'risk_level': self.risk_level.value,
            'enable_auto_trading': self.enable_auto_trading,
            'underlying_symbol': self.underlying_symbol,
            'market_open_time': self.market_open_time.strftime('%H:%M'),
            'market_close_time': self.market_close_time.strftime('%H:%M'),
            'liquidation_time': self.liquidation_time.strftime('%H:%M'),
            'min_vix_threshold': self.min_vix_threshold,
            'vix_spike_threshold': self.vix_spike_threshold,
            'volatility_lookback_period': self.volatility_lookback_period,
            'max_daily_drawdown_pct': self.max_daily_drawdown_pct,
            'max_position_pct': self.max_position_pct,
            'data_dir': self.data_dir,
            'log_dir': self.log_dir
        }
    
    def create_example_config(self, output_file: str) -> bool:
        """
        Create an example configuration file
        
        Args:
            output_file: Path to output file
            
        Returns:
            bool: True if file created successfully
        """
        try:
            config = configparser.ConfigParser()
            
            config['InteractiveBrokers'] = {
                'host': self.ib_host,
                'port': str(self.ib_port),
                'client_id': str(self.ib_client_id),
                'account': '',
            }
            
            config['Trading'] = {
                'mode': 'paper',
                'risk_level': 'medium',
                'enable_auto_trading': 'false',
                'underlying_symbol': 'SPY',
            }
            
            config['Schedule'] = {
                'market_open': '09:30',
                'market_close': '16:00',
                'liquidation_time': '15:45',
            }
            
            config['Volatility'] = {
                'min_vix_threshold': str(self.min_vix_threshold),
                'vix_spike_threshold': str(self.vix_spike_threshold),
                'lookback_period': str(self.volatility_lookback_period),
            }
            
            config['RiskManagement'] = {
                'max_daily_drawdown_pct': str(self.max_daily_drawdown_pct),
                'max_position_pct': str(self.max_position_pct),
            }
            
            config['Paths'] = {
                'data_dir': 'data',
                'log_dir': 'logs',
            }
            
            with open(output_file, 'w') as f:
                config.write(f)
                
            self.logger.info(f"Created example config at {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating example config: {str(e)}")
            return False


# Global settings instance
settings = Settings()