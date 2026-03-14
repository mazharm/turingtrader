export interface RiskLevelSummary {
  risk_level: number;
  total_return_pct: number;
  annualized_return_pct: number;
  annualized_volatility_pct: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  best_day_pct: number;
  worst_day_pct: number;
  profitable_months_pct: number;
  final_balance: number;
  max_winning_streak: number;
  max_losing_streak: number;
}

export interface SummaryData {
  generated_at: string;
  start_date: string;
  end_date: string;
  initial_investment: number;
  period_days: number;
  optimal_risk_level: number;
  best_return_risk_level: number;
  risk_levels: RiskLevelSummary[];
}

export interface DailyValue {
  date: string;
  balance: number;
}

export interface RiskParameters {
  risk_level: number;
  max_daily_risk_pct: number;
  min_volatility_threshold: number;
  max_position_size_pct: number;
  max_delta_exposure: number;
  stop_loss_pct: number;
  target_profit_pct: number;
  min_volatility_change: number;
  condor_stop_loss_factor_of_max_risk: number;
  condor_profit_target_factor_of_credit: number;
}

export interface RiskLevelMetrics {
  total_return_pct: number;
  annualized_return_pct: number;
  annualized_volatility_pct: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  best_day_pct: number;
  worst_day_pct: number;
  best_month_pct: number;
  worst_month_pct: number;
  profitable_months_pct: number;
  max_winning_streak: number;
  max_losing_streak: number;
  initial_balance: number;
  final_balance: number;
}

export interface RiskLevelData {
  risk_level: number;
  metrics: RiskLevelMetrics;
  daily_values: DailyValue[];
  risk_parameters: RiskParameters;
}
