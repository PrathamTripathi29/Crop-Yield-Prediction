import { useState } from "react";
import { api, useAsync } from "./api";
import Overview from "./Overview.jsx";
import District from "./District.jsx";
import Accuracy from "./Accuracy.jsx";

const TABS = ["Overview", "District", "Accuracy"];

export default function App() {
  const meta = useAsync(() => api.meta(), []);
  const [crop, setCrop] = useState("Wheat");
  const [year, setYear] = useState(null);
  const [tab, setTab] = useState("Overview");
  const [district, setDistrict] = useState(null);

  if (meta.error)
    return (
      <div className="shell">
        <p className="error">
          Could not reach the API ({meta.error.message}). Start the backend with{" "}
          <code>uvicorn main:app --port 8000</code>.
        </p>
      </div>
    );
  if (!meta.data) return <div className="shell"><p className="muted">Loading…</p></div>;

  const m = meta.data;
  const years = m.forecast_years;
  const now = new Date().getFullYear();
  const y = year ?? (years.includes(now) ? now : years[years.length - 1]);
  const openDistrict = (d) => { setDistrict(d); setTab("District"); };

  return (
    <div className="shell">
      <header className="top">
        <div>
          <h1>Madhya Pradesh crop yield</h1>
          <p className="muted">District forecasts for Soybean, Wheat and Gram · {m.yield_unit}</p>
        </div>
        <div className="controls">
          <div className="seg">
            {Object.keys(m.crops).map((c) => (
              <button key={c} className={c === crop ? "on" : ""} onClick={() => setCrop(c)}>{c}</button>
            ))}
          </div>
          <label className="sel">
            Forecast year
            <select value={y} onChange={(e) => setYear(Number(e.target.value))}>
              {years.map((v) => <option key={v}>{v}</option>)}
            </select>
          </label>
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "on" : ""} onClick={() => setTab(t)}>{t}</button>
        ))}
      </nav>

      {tab === "Overview" && <Overview crop={crop} year={y} meta={m} onPick={openDistrict} />}
      {tab === "District" && <District crop={crop} year={y} meta={m} district={district} setDistrict={setDistrict} />}
      {tab === "Accuracy" && <Accuracy meta={m} />}

      <footer className="foot muted">{m.note}</footer>
    </div>
  );
}
