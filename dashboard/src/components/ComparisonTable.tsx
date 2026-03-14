import { useState } from 'react';
import type { RiskLevelSummary } from '../types/index.ts';

interface Props {
  data: RiskLevelSummary[];
  optimalRiskLevel: number;
}

type SortKey = keyof RiskLevelSummary;

const columns: { key: SortKey; label: string; format: (v: number) => string }[] = [
  { key: 'risk_level', label: 'Risk', format: (v) => String(v) },
  { key: 'total_return_pct', label: 'Return %', format: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` },
  { key: 'annualized_return_pct', label: 'Ann. Return', format: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%` },
  { key: 'sharpe_ratio', label: 'Sharpe', format: (v) => v.toFixed(2) },
  { key: 'sortino_ratio', label: 'Sortino', format: (v) => v.toFixed(2) },
  { key: 'max_drawdown_pct', label: 'Max DD', format: (v) => `${v.toFixed(2)}%` },
  { key: 'win_rate', label: 'Win Rate', format: (v) => `${v.toFixed(1)}%` },
  { key: 'total_trades', label: 'Trades', format: (v) => String(v) },
  { key: 'final_balance', label: 'Final $', format: (v) => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
];

export default function ComparisonTable({ data, optimalRiskLevel }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('risk_level');
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = [...data].sort((a, b) => {
    const av = a[sortKey] as number;
    const bv = b[sortKey] as number;
    return sortAsc ? av - bv : bv - av;
  });

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(key === 'risk_level');
    }
  }

  return (
    <div className="card" style={{ overflowX: 'auto' }}>
      <table>
        <thead>
          <tr>
            {columns.map(({ key, label }) => (
              <th key={key} onClick={() => handleSort(key)}>
                {label} {sortKey === key ? (sortAsc ? '\u25b2' : '\u25bc') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.risk_level} style={
              row.risk_level === optimalRiskLevel
                ? { background: 'rgba(0, 210, 255, 0.08)' }
                : undefined
            }>
              {columns.map(({ key, format }) => {
                const v = row[key] as number;
                let colorClass = '';
                if (key === 'total_return_pct' || key === 'annualized_return_pct') {
                  colorClass = v >= 0 ? 'positive' : 'negative';
                }
                return (
                  <td key={key} className={colorClass}>
                    {format(v)}
                    {key === 'risk_level' && row.risk_level === optimalRiskLevel && (
                      <span style={{ marginLeft: 6, fontSize: '0.7rem', color: 'var(--accent)' }}>BEST SHARPE</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
