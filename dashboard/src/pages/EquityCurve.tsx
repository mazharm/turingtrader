import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { useData } from '../hooks/useData.ts';
import type { SummaryData, RiskLevelData } from '../types/index.ts';

const COLORS = [
  '#00d2ff', '#7c4dff', '#ff6e40', '#64ffda', '#ea80fc',
  '#00e676', '#ffd740', '#ff5252', '#448aff', '#ffffff',
];

export default function EquityCurve() {
  const { data: summary, loading, error } = useData<SummaryData>('summary.json');
  const [selected, setSelected] = useState<Set<number>>(new Set([1, 3, 5, 7, 10]));
  const [mode, setMode] = useState<'absolute' | 'relative'>('absolute');
  const [riskData, setRiskData] = useState<Record<number, RiskLevelData>>({});

  useEffect(() => {
    if (!summary) return;
    const base = import.meta.env.BASE_URL;
    const toLoad = summary.risk_levels
      .map((rl) => rl.risk_level)
      .filter((l) => !riskData[l]);

    if (toLoad.length === 0) return;

    Promise.all(
      toLoad.map((level) =>
        fetch(`${base}data/risk_level_${level}.json`)
          .then((r) => r.json())
          .then((d: RiskLevelData) => [level, d] as const)
      )
    ).then((results) => {
      setRiskData((prev) => {
        const next = { ...prev };
        for (const [level, data] of results) next[level] = data;
        return next;
      });
    });
  }, [summary, riskData]);

  if (loading) return <div className="loading">Loading...</div>;
  if (error || !summary) return <div className="error">{error || 'No data'}</div>;

  const toggleLevel = (level: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(level)) next.delete(level);
      else next.add(level);
      return next;
    });
  };

  const initial = summary.initial_investment;

  // Build equity traces
  const equityTraces = [...selected]
    .sort((a, b) => a - b)
    .filter((l) => riskData[l])
    .map((level) => {
      const d = riskData[level];
      return {
        x: d.daily_values.map((v) => v.date),
        y: mode === 'absolute'
          ? d.daily_values.map((v) => v.balance)
          : d.daily_values.map((v) => ((v.balance / initial) - 1) * 100),
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: `Risk ${level}`,
        line: { color: COLORS[level - 1], width: 1.5 },
      };
    });

  // Build drawdown traces
  const drawdownTraces = [...selected]
    .sort((a, b) => a - b)
    .filter((l) => riskData[l])
    .map((level) => {
      const d = riskData[level];
      let peak = d.daily_values[0]?.balance ?? 0;
      const dd = d.daily_values.map((v) => {
        if (v.balance > peak) peak = v.balance;
        return ((v.balance - peak) / peak) * 100;
      });
      return {
        x: d.daily_values.map((v) => v.date),
        y: dd,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: `Risk ${level}`,
        line: { color: COLORS[level - 1], width: 1.2 },
        fill: 'tozeroy' as const,
        fillcolor: `${COLORS[level - 1]}10`,
      };
    });

  return (
    <div className="page">
      <h2>Equity Curves</h2>

      <div className="controls">
        {summary.risk_levels.map((rl) => (
          <label key={rl.risk_level}>
            <input
              type="checkbox"
              checked={selected.has(rl.risk_level)}
              onChange={() => toggleLevel(rl.risk_level)}
            />
            {rl.risk_level}
          </label>
        ))}
        <div className="toggle-group">
          <button className={mode === 'absolute' ? 'active' : ''} onClick={() => setMode('absolute')}>$</button>
          <button className={mode === 'relative' ? 'active' : ''} onClick={() => setMode('relative')}>%</button>
        </div>
      </div>

      <div className="card">
        <Plot
          data={equityTraces}
          layout={{
            title: { text: 'Portfolio Value Over Time', font: { color: '#e0e0e0', size: 14 } },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#8888aa', size: 11 },
            xaxis: { gridcolor: '#2a2e45', linecolor: '#2a2e45' },
            yaxis: {
              gridcolor: '#2a2e45',
              linecolor: '#2a2e45',
              tickformat: mode === 'absolute' ? '$,.0f' : '+.1f',
              ticksuffix: mode === 'relative' ? '%' : '',
              title: { text: mode === 'absolute' ? 'Balance ($)' : 'Return (%)', font: { size: 11 } },
            },
            legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
            margin: { t: 40, r: 20, b: 60, l: 80 },
            height: 420,
            hovermode: 'x unified',
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="card">
        <Plot
          data={drawdownTraces}
          layout={{
            title: { text: 'Drawdown', font: { color: '#e0e0e0', size: 14 } },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#8888aa', size: 11 },
            xaxis: { gridcolor: '#2a2e45', linecolor: '#2a2e45' },
            yaxis: {
              gridcolor: '#2a2e45',
              linecolor: '#2a2e45',
              ticksuffix: '%',
              title: { text: 'Drawdown (%)', font: { size: 11 } },
            },
            legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
            margin: { t: 40, r: 20, b: 60, l: 70 },
            height: 300,
            hovermode: 'x unified',
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  );
}
