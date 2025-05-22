# Enhanced Adaptive Volatility-Harvesting System Performance Analysis

## Overview

This report presents the performance analysis of the Enhanced Adaptive Volatility-Harvesting System for options trading, focusing on iron condor strategies. The analysis covers different risk settings and key parameter sensitivities to identify optimal configuration settings.

## Risk Level Analysis

The system was tested across risk levels 1-10, with level 1 being the most conservative and level 10 being the most aggressive. Different risk levels impact position sizing, entry/exit thresholds, and strategy parameters.

![Risk Level Analysis](./reports/volatility_harvesting/risk_level_analysis.png)

### Key Findings:

1. **Optimal Risk Level**: Risk level 5-6 provides the best risk-adjusted returns (Sharpe ratio), balancing returns with acceptable drawdowns.

2. **Returns vs. Risk**: As expected, higher risk settings (7-10) generate higher absolute returns but with significantly increased drawdowns and volatility.

3. **Conservative Settings**: Risk levels 1-3 produce more consistent results with smaller drawdowns, but with notably lower returns.

4. **Win Rate Pattern**: Win rates tend to peak in the middle risk range (4-7), as moderate risk allows for balanced trade selection.

5. **Risk/Return Efficiency**: The risk-return scatter plot shows that beyond risk level 7, additional risk doesn't efficiently translate to proportional returns.

## Parameter Sensitivity Analysis

Four key parameters were analyzed to understand their impact on strategy performance:

![Parameter Sensitivity](./reports/volatility_harvesting/parameter_sensitivity.png)

### Key Parameter Findings:

1. **IV/HV Ratio Threshold**:
   - This parameter determines how much implied volatility must exceed historical volatility to trigger a trade.
   - Lower thresholds (1.1-1.2) generate higher returns but with increased volatility.
   - Optimal setting: 1.3, balancing returns with reasonable risk.

2. **Minimum IV Threshold**:
   - Sets the absolute minimum implied volatility required for trade entry.
   - Lower thresholds (15-20%) produce higher returns but with higher drawdowns.
   - Higher thresholds (30%+) reduce opportunities but increase trade quality.
   - Optimal setting: 20-25% for balanced performance.

3. **Strike Width Percentage**:
   - Determines how far apart the short and long strikes are positioned.
   - Narrower widths (2-3%) generate higher returns but with increased risk.
   - Wider widths (7-10%) reduce premium but provide better protection.
   - Optimal setting: 5% balances premium collection with adequate protection.

4. **Target Delta**:
   - Controls how far OTM (out-of-the-money) the option legs are positioned.
   - Lower delta targets (0.20-0.25) are more conservative but collect less premium.
   - Higher delta targets (0.35-0.40) collect more premium but with increased risk.
   - Optimal setting: 0.30 for balanced risk/reward.

## Recommendations

Based on the backtesting results, we recommend the following optimal settings for different investor types:

### Conservative Investors:
- Risk Level: 3
- IV/HV Ratio: 1.4
- Min IV Threshold: 25%
- Strike Width: 7%
- Target Delta: 0.25

### Balanced Investors:
- Risk Level: 5
- IV/HV Ratio: 1.3
- Min IV Threshold: 20%
- Strike Width: 5%
- Target Delta: 0.30

### Aggressive Investors:
- Risk Level: 8
- IV/HV Ratio: 1.2
- Min IV Threshold: 15%
- Strike Width: 3%
- Target Delta: 0.35

## Conclusion

The Enhanced Adaptive Volatility-Harvesting System demonstrates effectiveness across different market conditions, especially in identifying volatility disparities. The system performs best with balanced settings that allow it to be responsive to market opportunities while maintaining appropriate risk controls.

The analysis shows that risk level 5-6 provides the optimal balance between returns and risk for most investors. Parameter tuning should be based on individual risk tolerance and market conditions, with special attention to the IV/HV ratio and strike width parameters, which have the most significant impact on performance.

Future work should focus on enhancing the system's adaptability to rapidly changing volatility environments and further optimizing the trade entry and exit criteria based on machine learning techniques.