import { useEffect, useState } from "react";

// Empty in development (the Vite proxy handles /api). For a deployed build, set
// VITE_API_URL=https://your-backend-host in a .env file.
const BASE = import.meta.env.VITE_API_URL ?? "";

async function get(path, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const res = await fetch(`${BASE}${path}${qs ? `?${qs}` : ""}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  meta: () => get("/api/meta"),
  districts: (crop) => get("/api/districts", { crop }),
  history: (crop, district) => get("/api/history", { crop, district }),
  forecast: (crop, district, year) => get("/api/forecast", { crop, district, year }),
  state: (crop, year) => get("/api/state", { crop, year }),
  stateHistory: (crop) => get("/api/state-history", { crop }),
  backtest: (crop, district) => get("/api/backtest", { crop, district }),
};

// Runs an async function whenever deps change and tracks loading / error state.
export function useAsync(fn, deps) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn()
      .then((data) => !cancelled && setState({ data, loading: false, error: null }))
      .catch((error) => !cancelled && setState({ data: null, loading: false, error }));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

export const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

export const COLORS = { actual: "#3E8E5E", fitted: "#8A9A8F", forecast: "#D09A32", satellite: "#5B8FB9" };
export const tooltipStyle = { background: "var(--card)", border: "1px solid var(--line)", borderRadius: 8, color: "var(--ink)" };
