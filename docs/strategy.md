# Trading Strategy Documentation

This document explains the volatility-based options trading strategy used by the TuringTrader algorithm.

## Strategy Overview

The TuringTrader implements a volatility-driven options trading strategy focused on S&P500 options. Key features:

1. **Cash-In, Cash-Out Daily**: Starts each day with cash and ends with cash, avoiding overnight positions
2. **Volatility-Based**: Trades only when volatility conditions are favorable
3. **Parameterized Risk**: Offers adjustable risk levels to match individual risk tolerance
4. **S&P500 Focus**: Specializes in trading S&P500 options
5. **Multiple Strategy Types**: Adapts between iron condors and vertical spreads based on market conditions

## Strategy Components

### 1. Volatility Analysis

The strategy uses two primary volatility indicators:

- **VIX Index**: Measures market's expectation of 30-day forward-looking volatility
- **Historical Volatility**: Calculated from actual price movements
- **IV/HV Ratio**: Compares implied volatility to historical volatility to identify mispricing

Trading signals are generated when:
- VIX is above a threshold (determined by risk level)
- VIX is rising significantly (relative change exceeds threshold)
- IV/HV ratio indicates favorable premium collection opportunities

### 2. Strategy Selection

The algorithm dynamically selects between multiple option strategies based on current market conditions:

1. **Iron Condors**: Used in balanced volatility environments with sufficient premium available
   - A defined-risk, defined-reward strategy selling both call and put credit spreads
   - Profits when the underlying remains within a price range
   - Best in normal volatility environments with high IV/HV ratios

2. **Bull Put Spreads**: Used in high volatility environments where VIX is beginning to decline
   - A defined-risk, defined-reward strategy selling a put credit spread
   - Profits when the underlying remains above the short strike
   - Better capital efficiency than iron condors
   - Lower overall risk exposure

3. **Bear Call Spreads**: Used in high volatility environments where VIX is still rising
   - A defined-risk, defined-reward strategy selling a call credit spread
   - Profits when the underlying remains below the short strike
   - Better downside protection in volatile markets

When volatility conditions are favorable, the algorithm:

1. Analyzes the full S&P500 option chain
2. Scores options based on:
   - Implied volatility
3. **Option Selection**

When volatility conditions are favorable, the algorithm:

1. Analyzes the full S&P500 option chain
2. Scores options based on:
   - Implied volatility
   - Days to expiration
   - Strike distance from current price
   - Liquidity (bid-ask spread, volume, open interest)
   - Option greeks (primarily delta and theta)

3. For Iron Condors:
   - Selects optimal short strikes based on delta targets
   - Chooses long strikes to manage risk and capital requirements
   - Targets a specific credit-to-width ratio for optimal risk/reward

4. For Vertical Spreads:
   - Selects short strikes with ideal delta (typically 0.20-0.35)
   - Chooses long strikes at appropriate width to manage risk
   - Focuses on strikes with good liquidity and premium

### 4. Position Sizing

Position size is determined dynamically based on:
- Current account value
- Selected risk level
- Current volatility conditions
- Strategy type (more conservative for iron condors)
- Credit-to-risk ratio and probability of profit

Higher risk levels allow for larger position sizes relative to account value, but overall position sizing has been made more conservative to prioritize capital preservation.

### 5. Entry and Exit

- **Entry Conditions**: 
  - Favorable volatility conditions
  - No existing positions
  - Within daily trading window
  - Trade meets minimum thresholds for:
    - Credit received
    - Return on risk
    - Probability of profit
    - Credit-to-width ratio

- **Exit Conditions**:
  - End of trading day (mandatory)
  - Target profit reached (takes profits at 50-70% of maximum)
  - Stop loss hit (limits losses to 25-50% of maximum risk)
  - Volatility conditions deteriorate significantly (optional)

## Risk Management

### Risk Profiles

The system offers 10 risk levels, where 1 is most conservative and 10 is most aggressive. All parameters have been refined to be more conservative across all risk levels:

| Parameter | Risk Level 1 | Risk Level 5 | Risk Level 10 |
|-----------|-------------|--------------|--------------| 
| Max Daily Risk | 0.4% | 1.8% | 3.5% |
| Min Volatility Threshold | 30% | 22% | 14% |
| Max Position Size | 5% | 15% | 25% |
| Max Delta Exposure | 14 | 35 | 65 |
| Stop Loss | 6% | 13% | 20% |
| Target Profit | 13% | 22% | 32% |
| Min Volatility Change | 4.8% | 3.5% | 1.5% |
| IV/HV Ratio Threshold | 1.5 | 1.3 | 1.1 |

### Multi-Strategy Approach

The system now implements multiple strategy types to adapt to different market conditions:

1. **Iron Condors**: 
   - Used in balanced volatility environments
   - Higher capital requirements but potentially better risk distribution
   - Maximum risk clearly defined (width between strikes minus credit received)

2. **Vertical Spreads**:
   - Used in trending or high volatility environments
   - Better capital efficiency (typically 30-50% less capital than iron condors)
   - More directional exposure but still with defined risk

### Strategy Selection Criteria

The strategy selection process uses a robust set of volatility metrics to determine the optimal approach:

- Current VIX level
- VIX rate of change (1-day and 5-day)
- IV/HV ratio
- Volatility trend
- Market regime classification

This multi-strategy approach allows the system to adapt to changing market conditions while maintaining conservative risk parameters.

### Daily Cash Settlement

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