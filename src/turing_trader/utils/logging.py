"""
Logging Utility Module

This module provides configurable logging functionality:
- Console logging
- File logging with rotation
- Different log levels for different components
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

def setup_logging(log_level=logging.INFO, 
                 log_file=None, 
                 console=True,
                 log_dir="logs",
                 component=None):
    """
    Configure logging for the application
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Log file path or None for auto-generation
        console: Whether to log to console
        log_dir: Directory for log files
        component: Optional component name for the logger
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    if component:
        logger = logging.getLogger(f"turing_trader.{component}")
    else:
        logger = logging.getLogger("turing_trader")
        
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # File handler
    if log_file is None and log_dir:
        # Create log directory if needed
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log filename based on date and component
        date_str = datetime.now().strftime("%Y%m%d")
        component_str = f"{component}_" if component else ""
        log_file = os.path.join(log_dir, f"{component_str}{date_str}.log")
        
    if log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler (10MB max, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(component=None, log_level=None):
    """
    Get a logger for a specific component
    
    Args:
        component: Component name
        log_level: Optional override for log level
        
    Returns:
        logging.Logger: Logger instance
    """
    logger_name = f"turing_trader.{component}" if component else "turing_trader"
    logger = logging.getLogger(logger_name)
    
    if log_level is not None:
        logger.setLevel(log_level)
        
    return logger