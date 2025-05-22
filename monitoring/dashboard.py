"""
Dashboard module for the TuringTrader algorithm.
Provides web-based monitoring and visualization of algorithm performance.
"""

import logging
from typing import Dict, List, Optional, Any
import os
import sys
from datetime import datetime, timedelta
import threading
import json
import time

try:
    import dash
    from dash import dcc, html
    import dash_bootstrap_components as dbc
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
except ImportError as e:
    logging.error(f"Required package missing for dashboard: {e}")
    logging.error("Install with: pip install dash dash-bootstrap-components plotly pandas")

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibkr_trader.config import Config


class Dashboard:
    """
    Dashboard for monitoring the TuringTrader algorithmic trading system.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the dashboard.
        
        Args:
            config: Configuration object
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or Config()
        
        # Initialize data storage
        self.trade_history = []
        self.position_data = {}
        self.performance_metrics = {}
        self.system_status = {
            'last_update': datetime.now().isoformat(),
            'status': 'initializing',
            'message': 'Dashboard starting up',
            'uptime': 0,
            'trading_enabled': False,
        }
        
        # Create the Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True
        )
        
        # Initialize the layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
        
        # Start update thread for simulated data (for development)
        self.update_thread = None
        self.running = False
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("TuringTrader Monitoring Dashboard", className="mt-4 mb-4"),
                    html.Div(id="last-update-time", className="text-muted mb-4"),
                ], width=12)
            ]),
            
            dbc.Row([
                # System status card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H3("System Status", className="d-inline"),
                            dbc.Button("Refresh", id="refresh-button", color="primary", 
                                      size="sm", className="float-right")
                        ]),
                        dbc.CardBody([
                            html.Div([
                                html.H5("Status:"),
                                html.P(id="system-status", className="d-inline ms-2"),
                                html.Div(id="status-indicator", 
                                       className="status-indicator float-right")
                            ]),
                            html.Div([
                                html.H5("Message:"),
                                html.P(id="status-message", className="d-inline ms-2")
                            ]),
                            html.Div([
                                html.H5("Uptime:"),
                                html.P(id="system-uptime", className="d-inline ms-2")
                            ]),
                            html.Div([
                                html.H5("Trading Enabled:"),
                                html.P(id="trading-enabled", className="d-inline ms-2")
                            ]),
                        ])
                    ], className="mb-4")
                ], width=4),
                
                # Key metrics card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Performance Metrics")),
                        dbc.CardBody([
                            html.Div(id="performance-metrics")
                        ])
                    ], className="mb-4")
                ], width=8),
            ]),
            
            dbc.Row([
                # Current positions table
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Current Positions")),
                        dbc.CardBody([
                            html.Div(id="positions-table")
                        ])
                    ], className="mb-4")
                ], width=12),
            ]),
            
            dbc.Row([
                # Trade history graph
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Trade History")),
                        dbc.CardBody([
                            dcc.Graph(id="trade-history-graph")
                        ])
                    ], className="mb-4")
                ], width=12),
            ]),
            
            dbc.Row([
                # Volatility indicators
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Volatility Indicators")),
                        dbc.CardBody([
                            dcc.Graph(id="volatility-graph")
                        ])
                    ], className="mb-4")
                ], width=6),
                
                # Trades by strategy
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H3("Trades by Strategy")),
                        dbc.CardBody([
                            dcc.Graph(id="strategy-breakdown-graph")
                        ])
                    ], className="mb-4")
                ], width=6),
            ]),
            
            # Store for storing intermediate data
            dcc.Store(id='data-store'),
            
            # Interval for refreshing data
            dcc.Interval(
                id='interval-component',
                interval=5000,  # 5 seconds
                n_intervals=0
            )
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        # Callback to refresh data periodically
        @self.app.callback(
            Output('data-store', 'data'),
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks')]
        )
        def update_data(*args):
            """Update the stored data."""
            return {
                'trade_history': self.trade_history,
                'positions': self.position_data,
                'metrics': self.performance_metrics,
                'status': self.system_status
            }
        
        # Callback to update the system status
        @self.app.callback(
            [Output('system-status', 'children'),
             Output('status-indicator', 'className'),
             Output('status-message', 'children'),
             Output('system-uptime', 'children'),
             Output('trading-enabled', 'children'),
             Output('last-update-time', 'children')],
            [Input('data-store', 'data')]
        )
        def update_system_status(data):
            """Update the system status display."""
            if not data or 'status' not in data:
                return "Unknown", "status-indicator status-unknown", "No data", "N/A", "N/A", "Last update: N/A"
                
            status = data['status']
            status_text = status.get('status', 'unknown')
            message = status.get('message', 'No message')
            uptime = status.get('uptime', 0)
            trading_enabled = "Yes" if status.get('trading_enabled', False) else "No"
            
            # Format uptime
            uptime_str = f"{uptime} seconds"
            if uptime >= 86400:
                days = uptime // 86400
                uptime_str = f"{days} days, {(uptime % 86400) // 3600} hours"
            elif uptime >= 3600:
                uptime_str = f"{uptime // 3600} hours, {(uptime % 3600) // 60} minutes"
            elif uptime >= 60:
                uptime_str = f"{uptime // 60} minutes, {uptime % 60} seconds"
                
            # Set indicator color
            indicator_class = "status-indicator "
            if status_text == 'running':
                indicator_class += "status-running"
            elif status_text == 'warning':
                indicator_class += "status-warning"
            elif status_text == 'error':
                indicator_class += "status-error"
            else:
                indicator_class += "status-unknown"
                
            # Format last update time
            last_update = status.get('last_update', datetime.now().isoformat())
            try:
                update_time = datetime.fromisoformat(last_update)
                time_diff = datetime.now() - update_time
                if time_diff.total_seconds() < 60:
                    update_text = f"Last update: {time_diff.seconds} seconds ago"
                elif time_diff.total_seconds() < 3600:
                    update_text = f"Last update: {time_diff.seconds // 60} minutes ago"
                else:
                    update_text = f"Last update: {update_time.strftime('%Y-%m-%d %H:%M:%S')}"
            except:
                update_text = f"Last update: {last_update}"
                
            return (
                status_text.capitalize(),
                indicator_class,
                message,
                uptime_str,
                trading_enabled,
                update_text
            )
        
        # Callback to update performance metrics
        @self.app.callback(
            Output('performance-metrics', 'children'),
            [Input('data-store', 'data')]
        )
        def update_performance_metrics(data):
            """Update the performance metrics display."""
            if not data or 'metrics' not in data or not data['metrics']:
                return html.P("No performance data available")
                
            metrics = data['metrics']
            
            # Create metrics display using grid layout
            return dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Total P&L:"),
                        html.P(f"${metrics.get('total_pnl', 0):.2f}", 
                              className=f"{'text-success' if metrics.get('total_pnl', 0) >= 0 else 'text-danger'}")
                    ], className="metric-item")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H5("Win Rate:"),
                        html.P(f"{metrics.get('win_rate', 0):.1f}%")
                    ], className="metric-item")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H5("Return:"),
                        html.P(f"{metrics.get('return_pct', 0):.2f}%", 
                              className=f"{'text-success' if metrics.get('return_pct', 0) >= 0 else 'text-danger'}")
                    ], className="metric-item")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H5("Max Drawdown:"),
                        html.P(f"{metrics.get('max_drawdown', 0):.2f}%", 
                              className="text-warning")
                    ], className="metric-item")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H5("Total Trades:"),
                        html.P(f"{metrics.get('total_trades', 0)}")
                    ], className="metric-item")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H5("Today's P&L:"),
                        html.P(f"${metrics.get('daily_pnl', 0):.2f}", 
                              className=f"{'text-success' if metrics.get('daily_pnl', 0) >= 0 else 'text-danger'}")
                    ], className="metric-item")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H5("Sharpe Ratio:"),
                        html.P(f"{metrics.get('sharpe_ratio', 0):.2f}")
                    ], className="metric-item")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H5("Account Value:"),
                        html.P(f"${metrics.get('account_value', 0):,.2f}")
                    ], className="metric-item")
                ], width=3)
            ])
        
        # Callback to update positions table
        @self.app.callback(
            Output('positions-table', 'children'),
            [Input('data-store', 'data')]
        )
        def update_positions_table(data):
            """Update the positions table."""
            if not data or 'positions' not in data or not data['positions']:
                return html.P("No open positions")
                
            positions = data['positions']
            
            # Create table
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Strategy"),
                    html.Th("Quantity"),
                    html.Th("Entry Price"),
                    html.Th("Current Price"),
                    html.Th("P&L"),
                    html.Th("P&L %"),
                    html.Th("Entry Date"),
                    html.Th("Days Held")
                ]))
            ]
            
            rows = []
            for symbol, pos in positions.items():
                # Calculate days held
                try:
                    entry_time = datetime.fromisoformat(pos.get('entry_time'))
                    days_held = (datetime.now() - entry_time).days
                except:
                    days_held = 0
                    
                # Format P&L
                pnl = pos.get('pnl', 0)
                pnl_pct = pos.get('pnl_pct', 0)
                
                rows.append(html.Tr([
                    html.Td(symbol),
                    html.Td(pos.get('strategy', 'Unknown')),
                    html.Td(pos.get('quantity', 0)),
                    html.Td(f"${pos.get('entry_price', 0):.2f}"),
                    html.Td(f"${pos.get('current_price', 0):.2f}"),
                    html.Td(f"${pnl:.2f}", className=f"{'text-success' if pnl >= 0 else 'text-danger'}"),
                    html.Td(f"{pnl_pct:.2f}%", className=f"{'text-success' if pnl_pct >= 0 else 'text-danger'}"),
                    html.Td(entry_time.strftime('%Y-%m-%d %H:%M') if 'entry_time' in pos else 'Unknown'),
                    html.Td(days_held)
                ]))
                
            table_body = [html.Tbody(rows)]
            table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True)
            
            return table
        
        # Callback to update trade history graph
        @self.app.callback(
            Output('trade-history-graph', 'figure'),
            [Input('data-store', 'data')]
        )
        def update_trade_history_graph(data):
            """Update the trade history graph."""
            if not data or 'trade_history' not in data or not data['trade_history']:
                # Return empty figure
                return go.Figure().update_layout(
                    title="No trade history data available",
                    template="plotly_dark"
                )
                
            try:
                # Convert trade history to DataFrame
                df = pd.DataFrame(data['trade_history'])
                
                # Make sure we have required columns
                if 'time' not in df.columns or 'pnl' not in df.columns:
                    return go.Figure().update_layout(
                        title="Missing required columns in trade history",
                        template="plotly_dark"
                    )
                
                # Convert time column to datetime
                df['time'] = pd.to_datetime(df['time'])
                
                # Sort by time
                df = df.sort_values('time')
                
                # Calculate cumulative P&L
                df['cumulative_pnl'] = df['pnl'].cumsum()
                
                # Create figure
                fig = go.Figure()
                
                # Add cumulative P&L line
                fig.add_trace(go.Scatter(
                    x=df['time'],
                    y=df['cumulative_pnl'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(width=3)
                ))
                
                # Add individual trade markers
                colors = df['pnl'].apply(lambda x: 'green' if x >= 0 else 'red')
                fig.add_trace(go.Scatter(
                    x=df['time'],
                    y=df['pnl'],
                    mode='markers',
                    marker=dict(color=colors, size=10),
                    name='Individual Trades'
                ))
                
                # Update layout
                fig.update_layout(
                    title="Trade History",
                    xaxis_title="Date",
                    yaxis_title="P&L ($)",
                    hovermode="x unified",
                    template="plotly_dark"
                )
                
                return fig
                
            except Exception as e:
                self.logger.error(f"Error creating trade history graph: {e}")
                return go.Figure().update_layout(
                    title=f"Error creating trade history graph: {str(e)}",
                    template="plotly_dark"
                )
        
        # Callback to update volatility graph
        @self.app.callback(
            Output('volatility-graph', 'figure'),
            [Input('data-store', 'data')]
        )
        def update_volatility_graph(data):
            """Update the volatility indicators graph."""
            # Check if data contains volatility metrics
            if (not data or 'metrics' not in data or not data['metrics'] or 
                'volatility_history' not in data['metrics']):
                # Create empty figure with message
                return go.Figure().update_layout(
                    title="No volatility data available",
                    template="plotly_dark"
                )
            
            try:
                # Get volatility history data
                vol_history = data['metrics']['volatility_history']
                
                # Convert to DataFrame
                df = pd.DataFrame(vol_history)
                
                # Make sure we have required columns
                if 'time' not in df.columns or 'vix' not in df.columns:
                    return go.Figure().update_layout(
                        title="Missing required columns in volatility data",
                        template="plotly_dark"
                    )
                
                # Convert time column to datetime
                df['time'] = pd.to_datetime(df['time'])
                
                # Sort by time
                df = df.sort_values('time')
                
                # Create figure
                fig = go.Figure()
                
                # Add VIX line
                fig.add_trace(go.Scatter(
                    x=df['time'],
                    y=df['vix'],
                    mode='lines',
                    name='VIX',
                    line=dict(width=2, color='orange')
                ))
                
                # Add IV line if available
                if 'iv' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['time'],
                        y=df['iv'],
                        mode='lines',
                        name='IV',
                        line=dict(width=2, color='yellow')
                    ))
                
                # Add HV line if available
                if 'hv' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['time'],
                        y=df['hv'],
                        mode='lines',
                        name='HV',
                        line=dict(width=2, color='cyan')
                    ))
                
                # Add IV/HV ratio if available
                if 'iv_hv_ratio' in df.columns:
                    # Create secondary y-axis for ratio
                    fig.add_trace(go.Scatter(
                        x=df['time'],
                        y=df['iv_hv_ratio'],
                        mode='lines',
                        name='IV/HV Ratio',
                        line=dict(width=2, color='magenta'),
                        yaxis='y2'
                    ))
                    
                    # Update layout for dual y-axis
                    fig.update_layout(
                        yaxis2=dict(
                            title="IV/HV Ratio",
                            overlaying="y",
                            side="right",
                            showgrid=False
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title="Volatility Indicators",
                    xaxis_title="Date",
                    yaxis_title="Volatility (%)",
                    hovermode="x unified",
                    template="plotly_dark"
                )
                
                return fig
                
            except Exception as e:
                self.logger.error(f"Error creating volatility graph: {e}")
                return go.Figure().update_layout(
                    title=f"Error creating volatility graph: {str(e)}",
                    template="plotly_dark"
                )
        
        # Callback to update strategy breakdown graph
        @self.app.callback(
            Output('strategy-breakdown-graph', 'figure'),
            [Input('data-store', 'data')]
        )
        def update_strategy_graph(data):
            """Update the trades by strategy graph."""
            if not data or 'trade_history' not in data or not data['trade_history']:
                return go.Figure().update_layout(
                    title="No trade data available",
                    template="plotly_dark"
                )
                
            try:
                # Convert trade history to DataFrame
                df = pd.DataFrame(data['trade_history'])
                
                # Make sure we have required columns
                if 'strategy' not in df.columns or 'pnl' not in df.columns:
                    return go.Figure().update_layout(
                        title="Missing required columns in trade data",
                        template="plotly_dark"
                    )
                
                # Group by strategy and calculate metrics
                strategy_metrics = df.groupby('strategy').agg(
                    total_pnl=('pnl', 'sum'),
                    avg_pnl=('pnl', 'mean'),
                    count=('pnl', 'count'),
                    win_count=('pnl', lambda x: (x > 0).sum()),
                ).reset_index()
                
                # Calculate win rate
                strategy_metrics['win_rate'] = strategy_metrics['win_count'] / strategy_metrics['count'] * 100
                
                # Create figure with subplots
                fig = go.Figure()
                
                # Create first subplot - Total P&L by strategy
                fig.add_trace(go.Bar(
                    x=strategy_metrics['strategy'],
                    y=strategy_metrics['total_pnl'],
                    name='Total P&L',
                    marker_color=strategy_metrics['total_pnl'].apply(
                        lambda x: 'green' if x >= 0 else 'red'
                    )
                ))
                
                # Update layout
                fig.update_layout(
                    title="P&L by Strategy",
                    xaxis_title="Strategy",
                    yaxis_title="P&L ($)",
                    template="plotly_dark",
                    barmode='group'
                )
                
                return fig
                
            except Exception as e:
                self.logger.error(f"Error creating strategy graph: {e}")
                return go.Figure().update_layout(
                    title=f"Error creating strategy graph: {str(e)}",
                    template="plotly_dark"
                )
    
    def update_positions(self, positions: Dict[str, Dict]) -> None:
        """
        Update the current positions data.
        
        Args:
            positions: Dictionary of current positions
        """
        self.position_data = positions
        self.system_status['last_update'] = datetime.now().isoformat()
    
    def add_trade(self, trade_data: Dict) -> None:
        """
        Add a new trade to the trade history.
        
        Args:
            trade_data: Dictionary with trade details
        """
        self.trade_history.append(trade_data)
        self.system_status['last_update'] = datetime.now().isoformat()
    
    def update_metrics(self, metrics: Dict) -> None:
        """
        Update performance metrics.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        self.performance_metrics = metrics
        self.system_status['last_update'] = datetime.now().isoformat()
    
    def update_status(self, status: str, message: str, trading_enabled: bool = True) -> None:
        """
        Update the system status.
        
        Args:
            status: Status string ('running', 'warning', 'error')
            message: Status message
            trading_enabled: Whether trading is currently enabled
        """
        self.system_status.update({
            'status': status,
            'message': message,
            'last_update': datetime.now().isoformat(),
            'trading_enabled': trading_enabled,
            'uptime': (datetime.now() - datetime.fromisoformat(self.system_status['last_update'])).total_seconds()
        })
    
    def _generate_sample_data(self) -> None:
        """Generate sample data for development purposes."""
        while self.running:
            try:
                # Update system status
                self.update_status('running', 'System operating normally', True)
                
                # Update positions with random data
                positions = {
                    'SPY_IC_1': {
                        'strategy': 'iron_condor',
                        'quantity': 2,
                        'entry_price': 3.45,
                        'current_price': 3.45 + (0.5 - random.random()),
                        'entry_time': (datetime.now() - timedelta(days=3)).isoformat(),
                        'pnl': 120 * (random.random() - 0.3),
                        'pnl_pct': 10 * (random.random() - 0.3)
                    },
                    'SPY_IC_2': {
                        'strategy': 'iron_condor',
                        'quantity': 1,
                        'entry_price': 2.80,
                        'current_price': 2.80 + (0.5 - random.random()),
                        'entry_time': (datetime.now() - timedelta(days=1)).isoformat(),
                        'pnl': 80 * (random.random() - 0.3),
                        'pnl_pct': 8 * (random.random() - 0.3)
                    }
                }
                self.update_positions(positions)
                
                # Update metrics
                metrics = {
                    'total_pnl': 2500 + random.randint(-100, 100),
                    'daily_pnl': 150 + random.randint(-50, 50),
                    'win_rate': 68 + random.randint(-5, 5),
                    'total_trades': 25,
                    'return_pct': 12.5 + random.random() - 0.5,
                    'max_drawdown': 5.2 + random.random() - 0.5,
                    'sharpe_ratio': 1.8 + random.random() * 0.2 - 0.1,
                    'account_value': 102500 + random.randint(-1000, 1000),
                    'volatility_history': self._generate_volatility_history()
                }
                self.update_metrics(metrics)
                
                # Add sample trade if needed
                if random.random() < 0.2:  # 20% chance of new trade
                    trade = {
                        'time': datetime.now().isoformat(),
                        'symbol': 'SPY_IC',
                        'strategy': random.choice(['iron_condor', 'call_spread', 'put_spread']),
                        'quantity': random.randint(1, 3),
                        'price': round(random.uniform(2.5, 4.5), 2),
                        'action': random.choice(['open', 'close']),
                        'pnl': random.choice([1, -1]) * random.uniform(50, 300)
                    }
                    self.add_trade(trade)
                
            except Exception as e:
                self.logger.error(f"Error generating sample data: {e}")
                
            time.sleep(5)  # Update every 5 seconds
    
    def _generate_volatility_history(self, days: int = 30) -> List[Dict]:
        """
        Generate sample volatility history for testing.
        
        Args:
            days: Number of days of history to generate
            
        Returns:
            List of dictionaries with volatility data
        """
        history = []
        base_date = datetime.now() - timedelta(days=days)
        
        # Base values
        base_vix = 18.0
        base_hv = 15.0
        base_iv = 22.0
        
        for i in range(days):
            current_date = base_date + timedelta(days=i)
            
            # Add some randomness and trend
            vix = base_vix + random.uniform(-2, 2) + (i / days) * 5
            hv = base_hv + random.uniform(-1, 1) + (i / days) * 3
            iv = base_iv + random.uniform(-2, 2) + (i / days) * 6
            
            history.append({
                'time': current_date.isoformat(),
                'vix': vix,
                'hv': hv,
                'iv': iv,
                'iv_hv_ratio': iv / hv if hv > 0 else 1.0
            })
            
        return history
    
    def start_sample_data_generator(self) -> None:
        """Start generating sample data for development."""
        import random
        
        self.running = True
        self.update_thread = threading.Thread(target=self._generate_sample_data)
        self.update_thread.daemon = True
        self.update_thread.start()
        self.logger.info("Started sample data generator")
    
    def stop_sample_data_generator(self) -> None:
        """Stop the sample data generator."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
        self.logger.info("Stopped sample data generator")
    
    def run(self, debug: bool = False, port: int = 8050, host: str = '0.0.0.0', use_sample_data: bool = False) -> None:
        """
        Run the dashboard.
        
        Args:
            debug: Whether to run in debug mode
            port: Port number
            host: Host to bind to
            use_sample_data: Whether to generate sample data
        """
        if use_sample_data:
            self.start_sample_data_generator()
            
        try:
            # Add custom CSS
            app_directory = os.path.dirname(os.path.abspath(__file__))
            assets_directory = os.path.join(app_directory, 'assets')
            os.makedirs(assets_directory, exist_ok=True)
            
            with open(os.path.join(assets_directory, 'custom.css'), 'w') as f:
                f.write("""
                .status-indicator {
                    display: inline-block;
                    width: 15px;
                    height: 15px;
                    border-radius: 50%;
                    margin-left: 10px;
                }
                .status-running {
                    background-color: #28a745;
                }
                .status-warning {
                    background-color: #ffc107;
                }
                .status-error {
                    background-color: #dc3545;
                }
                .status-unknown {
                    background-color: #6c757d;
                }
                .metric-item {
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                    background-color: rgba(0, 0, 0, 0.1);
                }
                """)
            
            self.app.run_server(debug=debug, port=port, host=host)
        finally:
            if use_sample_data:
                self.stop_sample_data_generator()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create and run dashboard
    dashboard = Dashboard()
    dashboard.run(debug=True, use_sample_data=True)