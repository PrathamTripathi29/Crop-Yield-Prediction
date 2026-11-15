"""
main.py  --  FastAPI backend for the MP crop yield dashboard.

    pip install fastapi uvicorn pandas
    python train_export.py            (once, creates artifacts/)
    uvicorn main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs

No authentication: the API only serves public district statistics and
model results, and takes no personal data.
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ART = Path("artifacts")
meta = json.loads((ART / "meta.json").read_text())
params = json.loads((ART / "trend_params.json").read_text())
history = pd.read_csv(ART / "history.csv")
backtest = pd.read_csv(ART / "backtest.csv")

LAST_YEAR = meta["last_year"]
MAX_H = meta["max_horizon"]

app = FastAPI(title="Madhya Pradesh Crop Yield API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


# ---------------------------------------------------------------- helpers
def check_crop(crop):
    if crop not in meta["crops"]:
        raise HTTPException(404, f"Unknown crop '{crop}'. Choose from {list(meta['crops'])}.")


def check_district(crop, district):
    check_crop(crop)
    if district not in params[crop]["level"]:
        raise HTTPException(404, f"No data for district '{district}' and crop '{crop}'.")


def check_year(year):
    if not LAST_YEAR < year <= LAST_YEAR + MAX_H:
        raise HTTPException(
            422, f"Forecasts are available for {LAST_YEAR + 1} to {LAST_YEAR + MAX_H} "
                 f"(last observed year is {LAST_YEAR}).")


def trend_value(crop, district, year):
    p = params[crop]
    return p["level"][district] + p["slope"] * (year - p["year_mean"][district])


def forecast_row(crop, district, year):
    h = year - LAST_YEAR
    mid = trend_value(crop, district, year)
    iv = meta["crops"][crop]["intervals"][str(h)]
    return {
        "district": district, "year": year, "horizon_years": h,
        "forecast": round(mid, 3),
        "low": round(max(mid + iv["low_offset"], 0), 3),
        "high": round(mid + iv["high_offset"], 3),
    }


def recent_area(crop):
    """Mean sown area over each district's last three observed years (weights for state totals)."""
    h = history[history["crop"] == crop].sort_values("year")
    return h.groupby("district").tail(3).groupby("district")["area"].mean()


# ---------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"status": "ok", "last_observed_year": LAST_YEAR}


@app.get("/api/meta")
def get_meta():
    """Crops, methods, forecast years and one-year-ahead backtest accuracy."""
    return {**meta, "forecast_years": list(range(LAST_YEAR + 1, LAST_YEAR + MAX_H + 1)),
            "note": ("Forecasts extrapolate each district's own yield trend. The satellite/weather "
                     "models are shown for comparison; in backtests they tie the trend rather than beat it.")}


@app.get("/api/districts")
def districts(crop: str):
    check_crop(crop)
    h = history[history["crop"] == crop].sort_values("year")
    last = h.groupby("district").tail(1).set_index("district")
    area = recent_area(crop)
    return sorted(
        [{"district": d, "latest_year": int(last.loc[d, "year"]),
          "latest_yield": round(float(last.loc[d, "yield"]), 3),
          "recent_area_ha": round(float(area[d]))} for d in params[crop]["level"]],
        key=lambda r: r["district"])


@app.get("/api/history")
def get_history(crop: str, district: str):
    """Observed yield by year, with the fitted trend line."""
    check_district(crop, district)
    h = history[(history["crop"] == crop) & (history["district"] == district)].sort_values("year")
    return [{"year": int(r["year"]), "yield": round(float(r["yield"]), 3),
             "area_ha": round(float(r["area"])),
             "fitted": round(trend_value(crop, district, int(r["year"])), 3)}
            for r in h.to_dict("records")]


@app.get("/api/forecast")
def forecast(crop: str, district: str, year: int = Query(..., description="Year to forecast")):
    check_district(crop, district)
    check_year(year)
    h = history[(history["crop"] == crop) & (history["district"] == district)].sort_values("year")
    row = forecast_row(crop, district, year)
    row.update({
        "crop": crop, "season": meta["crops"][crop]["season"],
        "method": meta["crops"][crop]["method_label"],
        "interval": "80% range from one-to-five-year-ahead backtests at the same horizon",
        "last_observed": {"year": int(h["year"].iloc[-1]), "yield": round(float(h["yield"].iloc[-1]), 3)},
        "unit": meta["yield_unit"],
    })
    return row


@app.get("/api/state")
def state(crop: str, year: int):
    """Forecast for every district plus an area-weighted state average."""
    check_crop(crop)
    check_year(year)
    rows = [forecast_row(crop, d, year) for d in params[crop]["level"]]
    w = recent_area(crop)
    total = float(sum(w[r["district"]] for r in rows))
    state_mean = sum(r["forecast"] * float(w[r["district"]]) for r in rows) / total
    return {"crop": crop, "year": year, "unit": meta["yield_unit"],
            "state_forecast": round(state_mean, 3),
            "weighting": "mean sown area over each district's last three observed years",
            "districts": sorted(rows, key=lambda r: -r["forecast"])}


@app.get("/api/state-history")
def state_history(crop: str):
    """Area-weighted state-average yield by year."""
    check_crop(crop)
    h = history[history["crop"] == crop].copy()
    h["w"] = h["yield"] * h["area"]
    g = h.groupby("year").agg(w=("w", "sum"), a=("area", "sum"))
    return [{"year": int(y), "yield": round(float(r.w / r.a), 3)} for y, r in g.iterrows()]


@app.get("/api/backtest")
def get_backtest(crop: str, district: str | None = None):
    """One-year-ahead past predictions: actual vs baseline vs satellite model."""
    check_crop(crop)
    b = backtest[backtest["crop"] == crop]
    if district:
        check_district(crop, district)
        b = b[b["district"] == district]
    return b.drop(columns="crop").round(3).to_dict(orient="records")


@app.get("/api/accuracy")
def accuracy():
    return {c: v["backtest_one_year_ahead"] for c, v in meta["crops"].items()}