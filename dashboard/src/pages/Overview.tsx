import Plot from 'react-plotly.js';
import { useData } from '../hooks/useData.ts';
import MetricCard from '../components/MetricCard.tsx';
import ComparisonTable from '../components/ComparisonTable.tsx';
import type { SummaryData, RiskLevelData } from '../types/index.ts';
import { useState, useEffect } from 'react';

const COLORS = [
  '#00d2ff', '#7c4dff', '#ff6e40', '#64ffda', '#ea80fc',
  '#00e676', '#ffd740', '#ff5252', '#448aff', '#ffffff',
];

export default function Overview() {
  const { data: summary, loading, error } = useData<SummaryData>('summary.json');
  const [equityData, setEquityData] = useState<Record<number, RiskLevelData>>({});

  useEffect(() => {
    if (!summary) return;
    const base = import.meta.env.BASE_URL;
    // Load top 5 risk levels by sharpe for the mini chart
    const top5 = [...summary.risk_levels]
      .sort((a, b) => b.sharpe_ratio - a.sharpe_ratio)
      .slice(0, 5);

    Promise.all(
      top5.map((rl) =>
        fetch(`${base}data/risk_level_${rl.risk_level}.json`)
          .then((r) => r.json())
          .then((d: RiskLevelData) => [rl.risk_level, d] as const)
      )
    ).then((results) => {
      const map: Record<number, RiskLevelData> = {};
      for (const [level, data] of results) map[level] = data;
      setEquityData(map);
    });
  }, [summary]);

  if (loading) return <div className="loading">Loading...</div>;
  if (error || !summary) return <div className="error">{error || 'No data'}</div>;

  const best = summary.risk_levels.reduce((a, b) => a.sharpe_ratio > b.sharpe_ratio ? a : b);
  const bestReturn = summary.risk_levels.reduce((a, b) => a.total_return_pct > b.total_return_pct ? a : b);
  const lowestDD = summary.risk_levels.reduce((a, b) => a.max_drawdown_pct < b.max_drawdown_pct ? a : b);

  return (
    <div className="page">
      <h2>Algorithm Evaluation</h2>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 20, marginTop: -12 }}>
        {summary.start_date} to {summary.end_date} &middot; ${summary.initial_investment.toLocaleString()} initial
        {summary.data_source && (
          <>
            {' '}&middot;{' '}
            <span style={summary.data_source.includes('simulated') || summary.data_source === 'unknown'
              ? { color: '#ffd740', fontWeight: 600 }
              : undefined}>
              {summary.data_source}
            </span>
          </>
        )}
      </div>

      <div className="card-grid">
        <MetricCard
          label="Best Sharpe Ratio"
          value={best.sharpe_ratio.toFixed(2)}
          sublabel={`Risk Level ${best.risk_level}`}
          color="accent"
        />
        <MetricCard
          label="Best Return"
          value={`${bestReturn.total_return_pct >= 0 ? '+' : ''}${bestReturn.total_return_pct.toFixed(2)}%`}
          sublabel={`Risk Level ${bestReturn.risk_level}`}
          color={bestReturn.total_return_pct >= 0 ? 'positive' : 'negative'}
        />
        <MetricCard
          label="Lowest Drawdown"
          value={`${lowestDD.max_drawdown_pct.toFixed(2)}%`}
          sublabel={`Risk Level ${lowestDD.risk_level}`}
          color="accent"
        />
        <MetricCard
          label="Risk Levels Tested"
          value={String(summary.risk_levels.length)}
          sublabel={`${summary.period_days} days each`}
        />
      </div>

      <ComparisonTable data={summary.risk_levels} optimalRiskLevel={summary.optimal_risk_level} />

      {Object.keys(equityData).length > 0 && (
        <div className="card">
          <Plot
            data={Object.entries(equityData)
              .sort(([a], [b]) => Number(a) - Number(b))
              .map(([level, d], i) => ({
                x: d.daily_values.map((v) => v.date),
                y: d.daily_values.map((v) => v.balance),
                type: 'scatter' as const,
                mode: 'lines' as const,
                name: `Risk ${level}`,
                line: { color: COLORS[Number(level) - 1] || COLORS[i], width: 1.5 },
              }))}
            layout={{
              title: { text: 'Equity Curves (Top 5 by Sharpe)', font: { color: '#e0e0e0', size: 14 } },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#8888aa', size: 11 },
              xaxis: { gridcolor: '#2a2e45', linecolor: '#2a2e45' },
              yaxis: {
                gridcolor: '#2a2e45',
                linecolor: '#2a2e45',
                tickformat: '$,.0f',
              },
              legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
              margin: { t: 40, r: 20, b: 60, l: 70 },
              height: 350,
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      )}
    </div>
  );
}
