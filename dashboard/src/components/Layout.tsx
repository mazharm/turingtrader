import { NavLink, Outlet } from 'react-router-dom';

const navItems = [
  { to: '/', label: 'Overview' },
  { to: '/equity', label: 'Equity Curves' },
  { to: '/risk', label: 'Risk Analysis' },
];

export default function Layout() {
  return (
    <>
      <header style={{
        background: 'var(--surface)',
        borderBottom: '1px solid var(--border)',
        padding: '0 20px',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        <div style={{
          maxWidth: 1280,
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          height: 56,
          gap: 32,
        }}>
          <span style={{
            fontWeight: 700,
            fontSize: '1.1rem',
            color: 'var(--accent)',
            letterSpacing: '-0.02em',
          }}>
            TuringTrader
          </span>
          <nav style={{ display: 'flex', gap: 4 }}>
            {navItems.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                style={({ isActive }) => ({
                  padding: '8px 16px',
                  borderRadius: 6,
                  textDecoration: 'none',
                  fontSize: '0.9rem',
                  fontWeight: isActive ? 600 : 400,
                  color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                  background: isActive ? 'var(--surface-hover)' : 'transparent',
                  transition: 'all 0.15s',
                })}
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <Outlet />
    </>
  );
}
