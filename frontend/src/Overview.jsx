import { useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";
import { api, useAsync, fmt, COLORS, tooltipStyle } from "./api";

export default function Overview({ crop, year, meta, onPick }) {
  const state = useAsync(() => api.state(crop, year), [crop, year]);
  const hist = useAsync(() => api.stateHistory(crop), [crop]);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState("forecast");

  const err = state.error || hist.error;
  if (err) return <p className="error">{err.message}</p>;
  if (!state.data || !hist.data) return <p className="muted">Loading…</p>;

  const info = meta.crops[crop];
  const last = hist.data[hist.data.length - 1];
  const change = (state.data.state_forecast / last.yield - 1) * 100;
  const series = [
    ...hist.data.map((d) => ({ year: d.year, observed: d.yield, forecast: d.year === last.year ? d.yield : null })),
    { year, forecast: state.data.state_forecast },
  ];
  const rows = state.data.districts
    .filter((r) => r.district.toLowerCase().includes(q.toLowerCase()))
    .sort((a, b) => (sort === "name" ? a.district.localeCompare(b.district) : b.forecast - a.forecast));
  const max = Math.max(...state.data.districts.map((r) => r.forecast));

  return (
    <div>
      <div className="kpis">
        <div className="card kpi"><span className="muted">State forecast {year}</span><strong>{fmt(state.data.state_forecast)}</strong><span className="muted">{meta.yield_unit}, area-weighted</span></div>
        <div className="card kpi"><span className="muted">Last observed ({last.year})</span><strong>{fmt(last.yield)}</strong><span className="muted">{change >= 0 ? "+" : ""}{fmt(change, 1)}% to forecast</span></div>
        <div className="card kpi"><span className="muted">Season</span><strong className="cap">{info.season}</strong><span className="muted">{info.districts} districts</span></div>
      </div>

      <section className="card">
        <h2>State yield, {info.season} season</h2>
        <p className="muted small">{info.method_label}</p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={series} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
            <XAxis dataKey="year" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} />
            <YAxis domain={["auto", "auto"]} width={42} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v)} />
            <Legend />
            <Line name="Observed" dataKey="observed" stroke={COLORS.actual} strokeWidth={2} dot={false} connectNulls />
            <Line name="Forecast" dataKey="forecast" stroke={COLORS.forecast} strokeWidth={2} strokeDasharray="5 4" dot={{ r: 4 }} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="card">
        <div className="row">
          <h2>Districts, {year}</h2>
          <div className="row gap">
            <input placeholder="Search district" value={q} onChange={(e) => setQ(e.target.value)} />
            <select value={sort} onChange={(e) => setSort(e.target.value)}>
              <option value="forecast">Highest forecast</option>
              <option value="name">Name</option>
            </select>
          </div>
        </div>
        <table className="tbl">
          <thead><tr><th>District</th><th className="num">Forecast</th><th className="num">80% range</th><th /></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.district} onClick={() => onPick(r.district)}>
                <td>{r.district}</td>
                <td className="num">{fmt(r.forecast)}</td>
                <td className="num muted">{fmt(r.low)} – {fmt(r.high)}</td>
                <td className="barcell"><div className="bar"><span style={{ width: `${(r.forecast / max) * 100}%` }} /></div></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">Click a district for its history and backtest.</p>
      </section>
    </div>
  );
}
