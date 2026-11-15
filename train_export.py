"""
train_export.py

Turns the benchmark into files the API can serve. Run it whenever the
training table changes:

    python train_export.py

Needs benchmark_v3.py and model_training_data_v2.csv in the same folder.

For each crop it
  1. picks the better baseline (district mean vs district trend) by
     forward-in-time RMSE: train on earlier years, predict the next;
  2. measures how wrong that forecast is 1 to 5 years ahead, so the API
     can attach honest error bands that widen with the horizon;
  3. records the best satellite+weather model's past predictions, so the
     dashboard can show it next to the baseline and the actual yield;
  4. fits the chosen baseline on every year and saves its parameters.

Writes to artifacts/: meta.json, trend_params.json, history.csv, backtest.csv
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.base import clone

from benchmark_v3 import (FIRST_TEST_YEAR, META, MIN_AREA, MIN_YEARS_PER_DISTRICT,
                          MODELS, STATIC, Trend, feature_sets)

DATA_PATH = "model_training_data_v2.csv"
OUT_DIR = "artifacts"
MAX_HORIZON = 5
MIN_TRAIN_YEARS = 8

SEASON = {"Soybean": "kharif", "Wheat": "rabi", "Gram": "rabi"}
# Best satellite+weather model per crop from the forward benchmark.
BEST_ML = {
    "Soybean": ("core+stage", "Extra Trees"),
    "Wheat": ("core", "Extra Trees"),
    "Gram": ("core", "Ridge a=100"),
}
METHOD_LABEL = {
    "trend": "District trend (each district's own level plus one common yearly slope)",
    "mean": "District average (each district's mean yield over the training years)",
}


def load_crop(df, crop):
    c = df[df["crop"] == crop].dropna(axis=1, how="all")
    c = c[c["area"] >= MIN_AREA[crop]]
    years = c.groupby("district")["year"].transform("nunique")
    return c[years >= MIN_YEARS_PER_DISTRICT].dropna(subset=["yield"]).reset_index(drop=True)


def baseline_predict(train, test, kind):
    t = Trend(train, ["yield"])
    pred = t.predict(test)["yield"] if kind == "trend" else t.level(test, "yield")
    return pred.values


def horizon_backtest(cdf, kind, h):
    """Predict year T from data up to T-h, for every T with enough history."""
    parts = []
    for T in sorted(cdf["year"].unique()):
        if T < FIRST_TEST_YEAR - 1 + h:
            continue
        train, test = cdf[cdf["year"] <= T - h], cdf[cdf["year"] == T]
        if train["year"].nunique() < MIN_TRAIN_YEARS:
            continue
        parts.append(pd.DataFrame({
            "district": test["district"].values, "year": T,
            "actual": test["yield"].values, "pred": baseline_predict(train, test, kind)}))
    return pd.concat(parts, ignore_index=True)


def summarize(actual, pred):
    err = actual - pred
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((actual - actual.mean()) ** 2)),
        "accuracy_pct": float(100 - 100 * np.mean(np.abs(err) / actual)),
        "rows": int(len(actual)),
    }


def ml_forward(cdf, feat_set, model_name):
    """One-year-ahead predictions of the satellite+weather model."""
    dynamic = [c for c in cdf.columns if c not in META + STATIC]
    feats = feature_sets(cdf["crop"].iloc[0], dynamic)[feat_set]
    dyn = [c for c in feats if c not in STATIC]
    sta = [c for c in feats if c in STATIC]
    parts = []
    for T in sorted(cdf["year"].unique()):
        if T < FIRST_TEST_YEAR:
            continue
        train, test = cdf[cdf["year"] < T], cdf[cdf["year"] == T]
        trend = Trend(train, ["yield"] + dyn)
        tr_hat, te_hat = trend.predict(train), trend.predict(test)
        X_tr = (train[dyn] - tr_hat[dyn]).join(train[sta]).fillna(0)
        X_te = (test[dyn] - te_hat[dyn]).join(test[sta]).fillna(0)
        fitted = clone(MODELS[model_name]).fit(X_tr, train["yield"] - tr_hat["yield"])
        parts.append(pd.DataFrame({
            "district": test["district"].values, "year": T,
            "ml_pred": fitted.predict(X_te) + te_hat["yield"].values}))
    return pd.concat(parts, ignore_index=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    last_year = int(df["year"].max())
    meta = {"data_file": DATA_PATH, "last_year": last_year, "max_horizon": MAX_HORIZON,
            "yield_unit": "t/ha", "crops": {}}
    params, history, backtests = {}, [], []

    for crop in ["Soybean", "Wheat", "Gram"]:
        cdf = load_crop(df, crop)
        print(f"{crop}: {len(cdf)} rows, {cdf['district'].nunique()} districts")

        # 1. choose the baseline by one-year-ahead RMSE
        scores = {k: summarize(*[horizon_backtest(cdf, k, 1)[c].values for c in ("actual", "pred")])
                  for k in ("mean", "trend")}
        kind = min(scores, key=lambda k: scores[k]["rmse"])
        other = "mean" if kind == "trend" else "trend"

        # 2. error bands per horizon (empirical 10th and 90th percentile of actual - forecast)
        intervals = {}
        for h in range(1, MAX_HORIZON + 1):
            bt = horizon_backtest(cdf, kind, h)
            err = bt["actual"] - bt["pred"]
            intervals[str(h)] = {"low_offset": float(err.quantile(0.10)),
                                 "high_offset": float(err.quantile(0.90)), "rows": int(len(bt))}

        # 3. satellite model, one year ahead, for the comparison view
        feat_set, ml_name = BEST_ML[crop]
        base1 = horizon_backtest(cdf, kind, 1).rename(columns={"pred": "baseline_pred"})
        ml = ml_forward(cdf, feat_set, ml_name)
        bt = base1.merge(ml, on=["district", "year"])
        bt.insert(0, "crop", crop)
        backtests.append(bt)

        # 4. final fit on every year
        t = Trend(cdf, ["yield"])
        params[crop] = {
            "kind": kind,
            "slope": float(t.slope["yield"]) if kind == "trend" else 0.0,
            "level": {d: float(v) for d, v in t.mean["yield"].items()},
            "year_mean": {d: float(v) for d, v in t.year_mean.items()},
        }
        keep = cdf[["district", "year", "yield", "area"]].copy()
        keep.insert(0, "crop", crop)
        history.append(keep)

        meta["crops"][crop] = {
            "season": SEASON[crop],
            "method": kind,
            "method_label": METHOD_LABEL[kind],
            "districts": int(cdf["district"].nunique()),
            "min_area_ha": MIN_AREA[crop],
            "mean_yield": float(cdf["yield"].mean()),
            "slope_per_year": params[crop]["slope"],
            "intervals": intervals,
            "backtest_one_year_ahead": {
                "years": f"{FIRST_TEST_YEAR}-{last_year}",
                "chosen_baseline": {"name": kind, **scores[kind]},
                "other_baseline": {"name": other, **scores[other]},
                "satellite_model": {"name": ml_name, "feature_set": feat_set,
                                    **summarize(bt["actual"].values, bt["ml_pred"].values)},
            },
        }

    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(f"{OUT_DIR}/trend_params.json", "w") as f:
        json.dump(params, f)
    pd.concat(history).round(4).to_csv(f"{OUT_DIR}/history.csv", index=False)
    pd.concat(backtests).round(4).to_csv(f"{OUT_DIR}/backtest.csv", index=False)
    print(f"Saved artifacts to {OUT_DIR}/")


if __name__ == "__main__":
    main()