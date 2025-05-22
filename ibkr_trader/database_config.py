"""
Database configuration module for the TuringTrader algorithm.
Handles configuration for TimescaleDB/PostgreSQL and Redis connections.
"""

from dataclasses import dataclass
from typing import Optional


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
    
    # Connection pool settings
    min_connections: int = 1
    max_connections: int = 10
    
    # TimescaleDB specific settings
    use_timescaledb: bool = True
    
    def get_connection_string(self) -> str:
        """
        Get PostgreSQL connection string.
        
        Returns:
            str: Connection string for PostgreSQL
        """
        conn_str = (
            f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )
        return conn_str


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    
    # Connection pool settings
    max_connections: int = 10
    
    # Expiration time for cached items (in seconds)
    default_expiry: int = 3600  # 1 hour
    
    def get_connection_string(self) -> str:
        """
        Get Redis connection string.
        
        Returns:
            str: Connection string for Redis
        """
        auth = f":{self.password}@" if self.password else ""
        conn_str = f"redis://{auth}{self.host}:{self.port}/{self.database}"
        return conn_str


def create_postgres_tables(conn_string: str) -> bool:
    """
    Create required PostgreSQL tables if they don't exist.
    
    Args:
        conn_string: PostgreSQL connection string
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import psycopg2
        
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        # Check if TimescaleDB is installed
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb');")
        has_timescale = cursor.fetchone()[0]
        
        # Create market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                time TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(32) NOT NULL,
                open DOUBLE PRECISION NULL,
                high DOUBLE PRECISION NULL,
                low DOUBLE PRECISION NULL,
                close DOUBLE PRECISION NULL,
                volume BIGINT NULL,
                source VARCHAR(16) NOT NULL
            );
        """)
        
        # Create options data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS options_data (
                time TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(32) NOT NULL,
                expiry DATE NOT NULL,
                strike DOUBLE PRECISION NOT NULL,
                option_type CHAR(1) NOT NULL,
                bid DOUBLE PRECISION NULL,
                ask DOUBLE PRECISION NULL,
                last DOUBLE PRECISION NULL,
                volume BIGINT NULL,
                open_interest BIGINT NULL,
                implied_volatility DOUBLE PRECISION NULL
            );
        """)
        
        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                time TIMESTAMPTZ NOT NULL,
                symbol VARCHAR(32) NOT NULL,
                action VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                price DOUBLE PRECISION NOT NULL,
                order_id VARCHAR(64) NULL,
                strategy VARCHAR(32) NOT NULL,
                pnl DOUBLE PRECISION NULL,
                notes TEXT NULL
            );
        """)
        
        # Convert tables to TimescaleDB hypertables if TimescaleDB is installed
        if has_timescale:
            # Convert market_data to hypertable
            try:
                cursor.execute("""
                    SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
                """)
                
                # Convert options_data to hypertable
                cursor.execute("""
                    SELECT create_hypertable('options_data', 'time', if_not_exists => TRUE);
                """)
            except Exception as e:
                print(f"Warning: Could not convert tables to hypertables: {e}")
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol, time DESC);
            CREATE INDEX IF NOT EXISTS idx_options_data_symbol ON options_data (symbol, expiry, strike);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating PostgreSQL tables: {e}")
        return False


if __name__ == "__main__":
    # Test database connection
    import sys
    
    # Default config
    pg_config = PostgresConfig()
    redis_config = RedisConfig()
    
    # Test PostgreSQL connection
    try:
        import psycopg2
        print(f"Connecting to PostgreSQL: {pg_config.get_connection_string()}")
        conn = psycopg2.connect(pg_config.get_connection_string())
        print("PostgreSQL connection successful")
        conn.close()
    except ImportError:
        print("psycopg2 not installed. Run: pip install psycopg2-binary")
    except Exception as e:
        print(f"PostgreSQL connection error: {e}")
    
    # Test Redis connection
    try:
        import redis
        print(f"Connecting to Redis: {redis_config.get_connection_string()}")
        r = redis.from_url(redis_config.get_connection_string())
        r.ping()
        print("Redis connection successful")
    except ImportError:
        print("redis not installed. Run: pip install redis")
    except Exception as e:
        print(f"Redis connection error: {e}")