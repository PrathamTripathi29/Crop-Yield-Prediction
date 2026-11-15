import { useEffect } from "react";
import { ResponsiveContainer, ComposedChart, LineChart, Line, Area, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine } from "recharts";
import { api, useAsync, fmt, COLORS, tooltipStyle } from "./api";

const tip = (v) => (Array.isArray(v) ? `${fmt(v[0])} – ${fmt(v[1])}` : fmt(v));

export default function District({ crop, year, meta, district, setDistrict }) {
  const list = useAsync(() => api.districts(crop), [crop]);
  const names = list.data ? list.data.map((d) => d.district) : [];
  const current = names.includes(district) ? district : names[0];

  useEffect(() => { if (current && current !== district) setDistrict(current); }, [current, district, setDistrict]);

  const data = useAsync(
    () => current
      ? Promise.all([api.history(crop, current), api.backtest(crop, current), ...meta.forecast_years.map((y) => api.forecast(crop, current, y))])
      : Promise.resolve(null),
    [crop, current],
  );

  const err = list.error || data.error;
  if (err) return <p className="error">{err.message}</p>;
  if (!list.data || !data.data) return <p className="muted">Loading…</p>;

  const [hist, bt, ...fcs] = data.data;
  const info = meta.crops[crop];
  const lastObs = hist[hist.length - 1];
  const sel = fcs.find((f) => f.year === year);
  const rows = [
    ...hist.map((h) => ({ year: h.year, observed: h.yield, trend: h.fitted, forecast: h.year === lastObs.year ? h.yield : null })),
    ...fcs.map((f) => ({ year: f.year, forecast: f.forecast, band: [f.low, f.high] })),
  ];
  const baseName = info.method === "trend" ? "District trend" : "District average";

  return (
    <div>
      <div className="row">
        <label className="sel">District
          <select value={current} onChange={(e) => setDistrict(e.target.value)}>
            {names.map((n) => <option key={n}>{n}</option>)}
          </select>
        </label>
      </div>

      <div className="kpis">
        <div className="card kpi"><span className="muted">Forecast {year}</span><strong>{fmt(sel.forecast)}</strong><span className="muted">{meta.yield_unit}</span></div>
        <div className="card kpi"><span className="muted">80% range</span><strong>{fmt(sel.low)} – {fmt(sel.high)}</strong><span className="muted">from past backtests</span></div>
        <div className="card kpi"><span className="muted">Last observed ({lastObs.year})</span><strong>{fmt(lastObs.yield)}</strong><span className="muted">{lastObs.area_ha.toLocaleString()} ha sown</span></div>
      </div>

      <section className="card">
        <h2>{current}: history and forecast</h2>
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
            <XAxis dataKey="year" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} />
            <YAxis domain={["auto", "auto"]} width={42} />
            <Tooltip contentStyle={tooltipStyle} formatter={tip} />
            <Legend />
            <Area name="80% range" dataKey="band" stroke="none" fill={COLORS.forecast} fillOpacity={0.2} />
            <Line name="Observed" dataKey="observed" stroke={COLORS.actual} strokeWidth={2} dot={{ r: 2 }} connectNulls />
            <Line name={baseName} dataKey="trend" stroke={COLORS.fitted} strokeDasharray="4 4" dot={false} />
            <Line name="Forecast" dataKey="forecast" stroke={COLORS.forecast} strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} connectNulls />
            <ReferenceLine x={year} stroke={COLORS.forecast} strokeDasharray="2 3" />
          </ComposedChart>
        </ResponsiveContainer>
      </section>

      <section className="card">
        <h2>Past forecasts, one year ahead</h2>
        <p className="muted small">Each point is a forecast made using only earlier years, compared with what was recorded.</p>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={bt} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
            <XAxis dataKey="year" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} />
            <YAxis domain={["auto", "auto"]} width={42} />
            <Tooltip contentStyle={tooltipStyle} formatter={tip} />
            <Legend />
            <Line name="Recorded" dataKey="actual" stroke={COLORS.actual} strokeWidth={2} dot={{ r: 3 }} />
            <Line name={baseName} dataKey="baseline_pred" stroke={COLORS.forecast} strokeDasharray="5 4" dot={false} />
            <Line name="Satellite model" dataKey="ml_pred" stroke={COLORS.satellite} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
}
