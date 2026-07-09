import { HashRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout.tsx';
import Overview from './pages/Overview.tsx';
import EquityCurve from './pages/EquityCurve.tsx';
import RiskAnalysis from './pages/RiskAnalysis.tsx';
import Strategies from './pages/Strategies.tsx';
import './App.css';

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="equity" element={<EquityCurve />} />
          <Route path="risk" element={<RiskAnalysis />} />
          <Route path="strategies" element={<Strategies />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
