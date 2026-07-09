import Plot from 'react-plotly.js';
import { useData } from '../hooks/useData.ts';
import MetricCard from '../components/MetricCard.tsx';
import type { StrategiesIndex, StrategyDetail, StrategyRiskLevelSummary } from '../types/index.ts';
import { useState, useEffect } from 'react';

const COLORS = ['#00d2ff', '#64ffda', '#ffd740', '#ff6e40', '#ea80fc', '#448aff'];

const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;

export default function Strategies() {
  const { data: index, loading, error } = useData<StrategiesIndex>('strategies.json');
  const [riskLevel, setRiskLevel] = useState(5);
  const [details, setDetails] = useState<Record<string, StrategyDetail>>({});

  useEffect(() => {
    if (!index) return;
    const base = import.meta.env.BASE_URL;
    Promise.all(
      index.strategies.map((s) =>
        fetch(`${base}data/strategy_${s.slug}.json`)
          .then((r) => r.json())
          .then((d: StrategyDetail) => [s.slug, d] as const)
      )
    ).then((results) => {
      const map: Record<string, StrategyDetail> = {};
      for (const [slug, data] of results) map[slug] = data;
      setDetails(map);
    });
  }, [index]);

  if (loading) return <div className="loading">Loading...</div>;
  if (error || !index) return <div className="error">{error || 'No strategy data'}</div>;

  const atLevel = (s: { risk_levels: StrategyRiskLevelSummary[] }) =>
    s.risk_levels.find((r) => r.risk_level === riskLevel);

  const withMetrics = index.strategies
    .map((s) => ({ strategy: s, metrics: atLevel(s) }))
    .filter((x): x is { strategy: typeof x.strategy; metrics: StrategyRiskLevelSummary } => !!x.metrics);

  const bestSharpe = withMetrics.reduce((a, b) => (a.metrics.sharpe_ratio > b.metrics.sharpe_ratio ? a : b));
  const bestReturn = withMetrics.reduce((a, b) => (a.metrics.total_return_pct > b.metrics.total_return_pct ? a : b));
  const lowestDD = withMetrics.reduce((a, b) => (a.metrics.max_drawdown_pct < b.metrics.max_drawdown_pct ? a : b));

  return (
    <div className="page">
      <h2>Strategy Comparison</h2>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 16, marginTop: -12 }}>
        {index.start_date} to {index.end_date} &middot; ${index.initial_investment.toLocaleString()} initial
        {index.data_source && (
          <>
            {' '}&middot;{' '}
            <span style={index.data_source.includes('simulated')
              ? { color: '#ffd740', fontWeight: 600 }
              : undefined}>
              {index.data_source}
            </span>
          </>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Risk level:</span>
        {Array.from({ length: 10 }, (_, i) => i + 1).map((lvl) => (
          <button
            key={lvl}
            onClick={() => setRiskLevel(lvl)}
            style={{
              padding: '6px 12px',
              borderRadius: 6,
              border: '1px solid var(--border)',
              background: lvl === riskLevel ? 'var(--accent)' : 'var(--surface)',
              color: lvl === riskLevel ? '#0a0e1a' : 'var(--text-muted)',
              fontWeight: lvl === riskLevel ? 700 : 400,
              cursor: 'pointer',
              fontSize: '0.85rem',
            }}
          >
            {lvl}
          </button>
        ))}
      </div>

      <div className="card-grid">
        <MetricCard
          label="Best Sharpe"
          value={bestSharpe.metrics.sharpe_ratio.toFixed(2)}
          sublabel={bestSharpe.strategy.name}
          color="accent"
        />
        <MetricCard
          label="Best Return"
          value={fmtPct(bestReturn.metrics.total_return_pct)}
          sublabel={bestReturn.strategy.name}
          color={bestReturn.metrics.total_return_pct >= 0 ? 'positive' : 'negative'}
        />
        <MetricCard
          label="Lowest Drawdown"
          value={`${lowestDD.metrics.max_drawdown_pct.toFixed(2)}%`}
          sublabel={lowestDD.strategy.name}
          color="accent"
        />
        <MetricCard
          label="Strategies Compared"
          value={String(withMetrics.length)}
          sublabel="same data, same risk framework"
        />
      </div>

      <div className="card" style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ color: 'var(--text-muted)', textAlign: 'right' }}>
              <th style={{ textAlign: 'left', padding: '8px 12px' }}>Strategy</th>
              <th style={{ padding: '8px 12px' }}>Tenor</th>
              <th style={{ padding: '8px 12px' }}>Hold</th>
              <th style={{ padding: '8px 12px' }}>Return</th>
              <th style={{ padding: '8px 12px' }}>Sharpe</th>
              <th style={{ padding: '8px 12px' }}>Max DD</th>
              <th style={{ padding: '8px 12px' }}>Win %</th>
              <th style={{ padding: '8px 12px' }}>Trades</th>
              <th style={{ padding: '8px 12px' }}>Final Balance</th>
            </tr>
          </thead>
          <tbody>
            {withMetrics
              .slice()
              .sort((a, b) => b.metrics.sharpe_ratio - a.metrics.sharpe_ratio)
              .map(({ strategy, metrics }) => (
                <tr key={strategy.slug} style={{ borderTop: '1px solid var(--border)', textAlign: 'right' }}>
                  <td style={{ textAlign: 'left', padding: '10px 12px' }}>
                    <div style={{ fontWeight: 600 }}>{strategy.name}</div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem', maxWidth: 380 }}>
                      {strategy.description}
                    </div>
                  </td>
                  <td style={{ padding: '10px 12px', whiteSpace: 'nowrap' }}>
                    {strategy.params.min_dte}&ndash;{strategy.params.max_dte}d
                  </td>
                  <td style={{ padding: '10px 12px', whiteSpace: 'nowrap' }}>
                    {strategy.params.holding_days}d
                  </td>
                  <td style={{
                    padding: '10px 12px',
                    color: metrics.total_return_pct >= 0 ? 'var(--positive, #00e676)' : 'var(--negative, #ff5252)',
                    fontWeight: 600,
                  }}>
                    {fmtPct(metrics.total_return_pct)}
                  </td>
                  <td style={{ padding: '10px 12px' }}>{metrics.sharpe_ratio.toFixed(2)}</td>
                  <td style={{ padding: '10px 12px' }}>{metrics.max_drawdown_pct.toFixed(2)}%</td>
                  <td style={{ padding: '10px 12px' }}>{metrics.win_rate.toFixed(1)}%</td>
                  <td style={{ padding: '10px 12px' }}>{metrics.total_trades}</td>
                  <td style={{ padding: '10px 12px' }}>${metrics.final_balance.toLocaleString()}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      {Object.keys(details).length > 0 && (
        <div className="card">
          <Plot
            data={index.strategies
              .filter((s) => details[s.slug]?.daily_values_by_level?.[String(riskLevel)])
              .map((s, i) => {
                const dv = details[s.slug].daily_values_by_level[String(riskLevel)];
                return {
                  x: dv.map((v) => v.date),
                  y: dv.map((v) => v.balance),
                  type: 'scatter' as const,
                  mode: 'lines' as const,
                  name: s.name,
                  line: { color: COLORS[i % COLORS.length], width: 1.8 },
                };
              })}
            layout={{
              title: {
                text: `Equity Curves at Risk Level ${riskLevel}`,
                font: { color: '#e0e0e0', size: 14 },
              },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#8888aa', size: 11 },
              xaxis: { gridcolor: '#2a2e45', linecolor: '#2a2e45' },
              yaxis: { gridcolor: '#2a2e45', linecolor: '#2a2e45', tickformat: '$,.0f' },
              legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
              margin: { t: 40, r: 20, b: 60, l: 70 },
              height: 380,
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {withMetrics.length > 0 && (
        <div className="card">
          <Plot
            data={index.strategies.map((s, i) => ({
              x: s.risk_levels.map((r) => r.risk_level),
              y: s.risk_levels.map((r) => r.total_return_pct),
              type: 'scatter' as const,
              mode: 'lines+markers' as const,
              name: s.name,
              line: { color: COLORS[i % COLORS.length], width: 1.8 },
            }))}
            layout={{
              title: { text: 'Total Return by Risk Level', font: { color: '#e0e0e0', size: 14 } },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { color: '#8888aa', size: 11 },
              xaxis: { gridcolor: '#2a2e45', linecolor: '#2a2e45', dtick: 1, title: { text: 'Risk Level', font: { size: 11 } } },
              yaxis: { gridcolor: '#2a2e45', linecolor: '#2a2e45', ticksuffix: '%' },
              legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
              margin: { t: 40, r: 20, b: 80, l: 60 },
              height: 340,
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      )}
    </div>
  );
}
