import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import type Plotly from 'plotly.js';
import { useData } from '../hooks/useData.ts';
import type { SummaryData, RiskLevelData } from '../types/index.ts';

export default function RiskAnalysis() {
  const { data: summary, loading, error } = useData<SummaryData>('summary.json');
  const [riskData, setRiskData] = useState<Record<number, RiskLevelData>>({});
  const [histogramLevel, setHistogramLevel] = useState(5);

  useEffect(() => {
    if (!summary) return;
    const base = import.meta.env.BASE_URL;
    Promise.all(
      summary.risk_levels.map((rl) =>
        fetch(`${base}data/risk_level_${rl.risk_level}.json`)
          .then((r) => r.json())
          .then((d: RiskLevelData) => [rl.risk_level, d] as const)
      )
    ).then((results) => {
      const map: Record<number, RiskLevelData> = {};
      for (const [level, data] of results) map[level] = data;
      setRiskData(map);
    });
  }, [summary]);

  if (loading) return <div className="loading">Loading...</div>;
  if (error || !summary) return <div className="error">{error || 'No data'}</div>;

  const levels = summary.risk_levels.map((r) => r.risk_level);

  // Parameters table data
  const paramRows = levels
    .filter((l) => riskData[l])
    .map((l) => riskData[l].risk_parameters);

  // Daily returns for histogram
  const histData = riskData[histogramLevel];
  let dailyReturns: number[] = [];
  if (histData) {
    for (let i = 1; i < histData.daily_values.length; i++) {
      const prev = histData.daily_values[i - 1].balance;
      const curr = histData.daily_values[i].balance;
      if (prev > 0) dailyReturns.push(((curr / prev) - 1) * 100);
    }
  }

  const barLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#8888aa', size: 11 },
    xaxis: { gridcolor: '#2a2e45', linecolor: '#2a2e45', title: { text: 'Risk Level', font: { size: 11 } } },
    yaxis: { gridcolor: '#2a2e45', linecolor: '#2a2e45' },
    margin: { t: 40, r: 20, b: 50, l: 60 },
    height: 300,
    barcornerradius: 3,
  };

  return (
    <div className="page">
      <h2>Risk Analysis</h2>

      {/* Risk Parameters Table */}
      {paramRows.length > 0 && (
        <div className="card" style={{ overflowX: 'auto' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: 12, color: 'var(--text-muted)' }}>
            Risk Parameters by Level
          </h3>
          <table>
            <thead>
              <tr>
                <th>Level</th>
                <th>Max Daily Risk %</th>
                <th>Min Vol Threshold</th>
                <th>Max Position %</th>
                <th>Max Delta</th>
                <th>Stop Loss %</th>
                <th>Profit Target %</th>
              </tr>
            </thead>
            <tbody>
              {paramRows.map((p) => (
                <tr key={p.risk_level}>
                  <td>{p.risk_level}</td>
                  <td>{p.max_daily_risk_pct}%</td>
                  <td>{p.min_volatility_threshold}%</td>
                  <td>{p.max_position_size_pct}%</td>
                  <td>{p.max_delta_exposure}</td>
                  <td>{p.stop_loss_pct}%</td>
                  <td>{p.target_profit_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Bar charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div className="card">
          <Plot
            data={[{
              x: levels,
              y: summary.risk_levels.map((r) => r.sharpe_ratio),
              type: 'bar',
              marker: { color: levels.map((l) => l === summary.optimal_risk_level ? '#00e676' : '#7c4dff') },
              text: summary.risk_levels.map((r) => r.sharpe_ratio.toFixed(2)),
              textposition: 'outside',
              textfont: { color: '#e0e0e0', size: 10 },
            }]}
            layout={{ ...barLayout, title: { text: 'Sharpe Ratio', font: { color: '#e0e0e0', size: 13 } } }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
        <div className="card">
          <Plot
            data={[{
              x: levels,
              y: summary.risk_levels.map((r) => r.sortino_ratio),
              type: 'bar',
              marker: { color: '#64ffda' },
              text: summary.risk_levels.map((r) => r.sortino_ratio.toFixed(2)),
              textposition: 'outside',
              textfont: { color: '#e0e0e0', size: 10 },
            }]}
            layout={{ ...barLayout, title: { text: 'Sortino Ratio', font: { color: '#e0e0e0', size: 13 } } }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
        <div className="card">
          <Plot
            data={[{
              x: levels,
              y: summary.risk_levels.map((r) => r.max_drawdown_pct),
              type: 'bar',
              marker: { color: '#ff5252' },
              text: summary.risk_levels.map((r) => `${r.max_drawdown_pct.toFixed(1)}%`),
              textposition: 'outside',
              textfont: { color: '#e0e0e0', size: 10 },
            }]}
            layout={{ ...barLayout, title: { text: 'Max Drawdown %', font: { color: '#e0e0e0', size: 13 } } }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
        <div className="card">
          <Plot
            data={[{
              x: levels,
              y: summary.risk_levels.map((r) => r.win_rate),
              type: 'bar',
              marker: { color: '#00e676' },
              text: summary.risk_levels.map((r) => `${r.win_rate.toFixed(1)}%`),
              textposition: 'outside',
              textfont: { color: '#e0e0e0', size: 10 },
            }]}
            layout={{
              ...barLayout,
              title: { text: 'Win Rate %', font: { color: '#e0e0e0', size: 13 } },
              yaxis: { ...barLayout.yaxis, range: [0, 100] },
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Daily return distribution */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <h3 style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>Daily Return Distribution</h3>
          <select
            value={histogramLevel}
            onChange={(e) => setHistogramLevel(Number(e.target.value))}
            style={{
              background: 'var(--surface-hover)',
              border: '1px solid var(--border)',
              color: 'var(--text)',
              padding: '4px 8px',
              borderRadius: 4,
              fontSize: '0.85rem',
            }}
          >
            {levels.map((l) => (
              <option key={l} value={l}>Risk Level {l}</option>
            ))}
          </select>
        </div>
        <Plot
          data={dailyReturns.length > 0 ? [
            {
              x: dailyReturns.filter((r) => r >= 0),
              type: 'histogram' as const,
              marker: { color: 'rgba(0, 230, 118, 0.5)' },
              name: 'Gains',
            },
            {
              x: dailyReturns.filter((r) => r < 0),
              type: 'histogram' as const,
              marker: { color: 'rgba(255, 82, 82, 0.5)' },
              name: 'Losses',
            },
          ] as Plotly.Data[] : []}
          layout={{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#8888aa', size: 11 },
            xaxis: {
              gridcolor: '#2a2e45',
              linecolor: '#2a2e45',
              title: { text: 'Daily Return (%)', font: { size: 11 } },
            },
            yaxis: {
              gridcolor: '#2a2e45',
              linecolor: '#2a2e45',
              title: { text: 'Frequency', font: { size: 11 } },
            },
            barmode: 'overlay',
            margin: { t: 20, r: 20, b: 50, l: 60 },
            height: 300,
            showlegend: true,
            legend: { orientation: 'h', y: -0.2, font: { size: 10 } },
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
