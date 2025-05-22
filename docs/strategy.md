# Trading Strategy Documentation

This document explains the volatility-based options trading strategy used by the TuringTrader algorithm.

## Strategy Overview

The TuringTrader implements a volatility-driven options trading strategy focused on S&P500 options. Key features:

1. **Cash-In, Cash-Out Daily**: Starts each day with cash and ends with cash, avoiding overnight positions
2. **Volatility-Based**: Trades only when volatility conditions are favorable
3. **Parameterized Risk**: Offers adjustable risk levels to match individual risk tolerance
4. **S&P500 Focus**: Specializes in trading S&P500 options

## Strategy Components

### 1. Volatility Analysis

The strategy uses two primary volatility indicators:

- **VIX Index**: Measures market's expectation of 30-day forward-looking volatility
- **Historical Volatility**: Calculated from actual price movements

Trading signals are generated when:
- VIX is above a threshold (determined by risk level)
- VIX is rising significantly (relative change exceeds threshold)

### 2. Option Selection

When volatility conditions are favorable, the algorithm:

1. Analyzes the full S&P500 option chain
2. Scores options based on:
   - Implied volatility
   - Days to expiration
   - Strike distance from current price
   - Liquidity (bid-ask spread)
   - Option greeks (primarily delta and theta)

3. Selects the highest-scoring option that aligns with current market conditions

### 3. Position Sizing

Position size is determined dynamically based on:
- Current account value
- Selected risk level
- Current volatility conditions
- Option characteristics (delta exposure)

Higher risk levels allow for larger position sizes relative to account value.

### 4. Entry and Exit

- **Entry Conditions**: 
  - Favorable volatility conditions
  - No existing positions
  - Within daily trading window
  - Option meets selection criteria

- **Exit Conditions**:
  - End of trading day (mandatory)
  - Target profit reached (optional)
  - Stop loss hit (optional)
  - Volatility conditions deteriorate significantly (optional)

## Risk Management

### Risk Profiles

The system offers 10 risk levels, where 1 is most conservative and 10 is most aggressive:

| Parameter | Risk Level 1 | Risk Level 5 | Risk Level 10 |
|-----------|-------------|--------------|--------------|
| Max Daily Risk | 0.5% | 2.5% | 5.0% |
| Min Volatility Threshold | 28% | 20% | 10% |
| Max Position Size | 8% | 20% | 35% |
| Max Delta Exposure | 18 | 50 | 90 |
| Stop Loss | 7% | 15% | 25% |
| Target Profit | 13% | 25% | 40% |
| Min Volatility Change | 4.6% | 3% | 1% |

### Daily Cash Settlement

The strategy eliminates overnight risk by:
- Starting each day with 100% cash position
- Closing all positions before market close
- Never holding positions overnight

This approach provides protection from overnight gaps and after-hours news events.

## Volatility Trading Logic

### Volatility States

The system categorizes market volatility into states:
- **Low Volatility** (VIX < 15): Generally inactive
- **Normal Volatility** (15 ≤ VIX < 20): Selective trading
- **High Volatility** (20 ≤ VIX < 30): Active trading
- **Extreme Volatility** (VIX ≥ 30): Aggressive but cautious trading

### Trade Selection Based on Volatility

- **Rising Volatility**: Favors buying calls or puts depending on direction
- **High Static Volatility**: Favors volatility strategies (straddles/strangles)
- **Falling High Volatility**: Reduces position sizing or abstains from trading

## Performance Characteristics

### Expected Behavior

- **Low Volatility Markets**: Few trades, capital preservation
- **Normal Markets**: Moderate activity when volatility spikes occur
- **High Volatility Markets**: High activity, potential for substantial returns
- **Extreme Volatility**: Highest potential returns but with increased risk

### Metrics to Monitor

1. **Win Rate**: Percentage of profitable trades
2. **Average Gain/Loss Ratio**: Compares average winning trade to average losing trade
3. **Maximum Drawdown**: Largest peak-to-trough decline
4. **Sharpe Ratio**: Risk-adjusted return metric

## Strategy Customization

The strategy can be customized by:
1. Adjusting the risk level (1-10)
2. Modifying volatility thresholds
3. Adjusting trading hours
4. Fine-tuning option selection criteria

## Backtesting Results

Historical backtesting across different market regimes shows:

1. Best performance in volatile markets
2. Capital preservation in low-volatility markets
3. Risk levels 4-6 typically providing optimal risk-adjusted returns
4. Higher win rates at lower risk levels, higher absolute returns at higher risk levels