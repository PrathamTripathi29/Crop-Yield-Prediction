import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from "recharts";
import { fmt, COLORS, tooltipStyle } from "./api";

export default function Accuracy({ meta }) {
  const crops = Object.entries(meta.crops);
  const chart = crops.map(([c, v]) => ({
    crop: c,
    "Forecast method": v.backtest_one_year_ahead.chosen_baseline.rmse,
    "Satellite model": v.backtest_one_year_ahead.satellite_model.rmse,
  }));

  return (
    <div>
      <section className="card">
        <h2>How accurate is this?</h2>
        <p>
          Every forecast was tested the way it would really be used: train on earlier years, predict the next one.
          Lower error is better (RMSE, in {meta.yield_unit}). The satellite and weather model is roughly level with the
          simple trend, so the app forecasts with the trend and shows the satellite model for comparison.
        </p>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chart} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
            <XAxis dataKey="crop" />
            <YAxis width={42} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(v, 3)} />
            <Legend />
            <Bar dataKey="Forecast method" fill={COLORS.forecast} radius={[4, 4, 0, 0]} />
            <Bar dataKey="Satellite model" fill={COLORS.satellite} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <div className="grid3">
        {crops.map(([c, v]) => {
          const b = v.backtest_one_year_ahead;
          const rows = [["RMSE", "rmse", 3], ["MAE", "mae", 3], ["R²", "r2", 2], ["Accuracy %", "accuracy_pct", 1]];
          return (
            <section className="card" key={c}>
              <h3>{c}</h3>
              <p className="muted small">{v.method_label}</p>
              <table className="tbl compact">
                <thead><tr><th /><th className="num">Forecast</th><th className="num">Satellite</th></tr></thead>
                <tbody>
                  {rows.map(([label, key, d]) => (
                    <tr key={key}><td>{label}</td><td className="num">{fmt(b.chosen_baseline[key], d)}</td><td className="num">{fmt(b.satellite_model[key], d)}</td></tr>
                  ))}
                </tbody>
              </table>
              <p className="muted small">Tested {b.years}, {b.chosen_baseline.rows} district-years. Satellite model: {b.satellite_model.name}.</p>
            </section>
          );
        })}
      </div>

      <section className="card">
        <h3>How to read this</h3>
        <ul>
          <li><b>Accuracy %</b> is 100 minus the average percentage error. It is not a classification score.</li>
          <li><b>The 80% range</b> comes from past forecast errors at the same distance ahead, so it widens for later years.</li>
          <li><b>Far-ahead years</b> extend a trend and can overshoot if yields level off.</li>
          <li><b>Next step:</b> crop-specific satellite masks, which should help the satellite model beat the trend.</li>
        </ul>
      </section>
    </div>
  );
}
