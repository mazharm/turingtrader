interface MetricCardProps {
  label: string;
  value: string;
  sublabel?: string;
  color?: 'positive' | 'negative' | 'accent' | 'default';
}

export default function MetricCard({ label, value, sublabel, color = 'default' }: MetricCardProps) {
  const colorClass = color === 'default' ? '' : color;
  return (
    <div className="metric-card">
      <div className="label">{label}</div>
      <div className={`value ${colorClass}`}>{value}</div>
      {sublabel && <div className="sublabel">{sublabel}</div>}
    </div>
  );
}
