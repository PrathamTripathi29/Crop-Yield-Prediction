"""
benchmark_v3.py

Honest benchmark for the 2003-2022 table.

Why it differs from the earlier scripts
  * Yields and satellite features both trend upward over 20 years, so
    everything is detrended first: per-district level + one common
    yearly slope, fitted on TRAINING years only. Models learn the
    deviation from that trend, and the trend is added back to predict.
  * Two baselines to beat, both fitted on training years only:
      - district mean
      - district trend (district level + common slope)
  * Two validation schemes:
      forward : train on all years before T, predict T (a real forecast)
      loyo    : leave one year out (uses future years, so it is kinder)
  * Rows from districts that barely grow the crop are dropped (MIN_AREA),
    because the satellite features describe other crops there.
  * Small, named feature sets instead of 40+ columns for 20 seasons.

Metrics (pooled over all test years)
  RMSE, MAE (t/ha), R2 on raw yield, Accuracy % = 100 - MAPE,
  and Skill vs the two baselines (1 = perfect, 0 = same as baseline,
  negative = worse than the baseline).

Full results are saved to benchmark_results_v3.csv.
"""

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_PATH = "model_training_data_v2.csv"
FIRST_TEST_YEAR = 2011          # forward scheme: first year to be predicted
MIN_AREA = {"Soybean": 10000, "Gram": 5000, "Wheat": 0}   # hectares
MIN_YEARS_PER_DISTRICT = 8
TOP_N = 8

META = ["yield", "district", "year", "season", "crop", "area"]
STATIC = ["ph", "organic_carbon", "clay_pct", "sand_pct", "bulk_density"]

MODELS = {
    "Ridge a=10": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "Ridge a=100": make_pipeline(StandardScaler(), Ridge(alpha=100)),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                           random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                       random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8,
                            colsample_bytree=0.8, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8,
                              subsample_freq=1, min_child_samples=15, random_state=42, verbose=-1),
}


class Trend:
    """District level + one common yearly slope per column, fitted on training rows."""

    def __init__(self, train, cols):
        self.cols = cols
        self.year_mean = train.groupby("district")["year"].mean()
        self.year_mean_all = train["year"].mean()
        yc = train["year"] - train["district"].map(self.year_mean)
        self.mean, self.mean_all, self.slope = {}, {}, {}
        for c in cols:
            x = train[c]
            self.mean[c] = x.groupby(train["district"]).mean()
            self.mean_all[c] = x.mean()
            xc = x - train["district"].map(self.mean[c])
            ok = xc.notna()
            den = (yc[ok] ** 2).sum()
            self.slope[c] = (xc[ok] * yc[ok]).sum() / den if den > 0 else 0.0

    def level(self, df, c):
        return df["district"].map(self.mean[c]).fillna(self.mean_all[c])

    def predict(self, df):
        ym = df["district"].map(self.year_mean).fillna(self.year_mean_all)
        out = pd.DataFrame(index=df.index)
        for c in self.cols:
            out[c] = self.level(df, c) + self.slope[c] * (df["year"] - ym)
        return out


def splits(df, scheme):
    years = sorted(df["year"].unique())
    for t in years:
        if scheme == "forward":
            if t < FIRST_TEST_YEAR:
                continue
            yield df[df["year"] < t], df[df["year"] == t]
        else:
            yield df[df["year"] != t], df[df["year"] == t]


def feature_sets(crop, dynamic_available):
    core = ["ndvi_max", "fapar_mean", "lswi_mean", "rainfall_total", "rzsm_mean"]
    if crop == "Soybean":
        stage = ["rain_m07", "rain_m08", "rain_m09", "rain_m10", "ndvi_m08", "ndvi_m09",
                 "heavy_rain_days", "rain_max_day", "wet_days"]
    else:
        stage = ["ndvi_m01", "ndvi_m02", "ndvi_m03", "tmax_m01", "tmax_m02", "tmax_m03",
                 "hot35_m02", "hot35_m03", "hot_days_35", "tmin_mean"]
    keep = lambda cols: [c for c in cols if c in dynamic_available]
    return {
        "core": keep(core),
        "core+stage": keep(core + stage),
        "core+stage+soil": keep(core + stage) + STATIC,
        "all dynamic": list(dynamic_available),
    }


def evaluate(crop_df, scheme, sets):
    y_all, base_trend, base_mean = [], [], []
    preds = {(s, m): [] for s in sets for m in MODELS}
    for train, test in splits(crop_df, scheme):
        y_all.append(test["yield"].values)
        t_y = Trend(train, ["yield"])
        base_trend.append(t_y.predict(test)["yield"].values)
        base_mean.append(t_y.level(test, "yield").values)
        for set_name, feats in sets.items():
            dyn = [c for c in feats if c not in STATIC]
            sta = [c for c in feats if c in STATIC]
            trend = Trend(train, ["yield"] + dyn)
            tr_hat, te_hat = trend.predict(train), trend.predict(test)
            X_tr = (train[dyn] - tr_hat[dyn]).join(train[sta]).fillna(0)
            X_te = (test[dyn] - te_hat[dyn]).join(test[sta]).fillna(0)
            y_tr = train["yield"] - tr_hat["yield"]
            for name, model in MODELS.items():
                fitted = clone(model).fit(X_tr, y_tr)
                preds[(set_name, name)].append(fitted.predict(X_te) + te_hat["yield"].values)
    cat = np.concatenate
    return cat(y_all), cat(base_trend), cat(base_mean), {k: cat(v) for k, v in preds.items()}


def metrics(y, pred, b_trend, b_mean):
    err = y - pred
    sse = np.sum(err ** 2)
    return {
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAE": np.mean(np.abs(err)),
        "R2": 1 - sse / np.sum((y - y.mean()) ** 2),
        "Accuracy %": 100 - 100 * np.mean(np.abs(err) / y),
        "Skill vs mean": 1 - sse / np.sum((y - b_mean) ** 2),
        "Skill vs trend": 1 - sse / np.sum((y - b_trend) ** 2),
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    all_results = []

    for crop in ["Soybean", "Wheat", "Gram"]:
        cdf = df[df["crop"] == crop].dropna(axis=1, how="all")
        n0 = len(cdf)
        cdf = cdf[cdf["area"] >= MIN_AREA[crop]]
        yrs = cdf.groupby("district")["year"].transform("nunique")
        cdf = cdf[yrs >= MIN_YEARS_PER_DISTRICT].dropna(subset=["yield"]).reset_index(drop=True)
        dynamic = [c for c in cdf.columns if c not in META + STATIC]
        sets = feature_sets(crop, dynamic)

        print(f"\n{'=' * 84}\n{crop.upper()}: {len(cdf)} rows kept of {n0} "
              f"({cdf['district'].nunique()} districts, {cdf['year'].nunique()} years, "
              f"min area {MIN_AREA[crop]} ha); mean yield {cdf['yield'].mean():.2f} t/ha\n{'=' * 84}")

        for scheme in ("forward", "loyo"):
            y, b_trend, b_mean, preds = evaluate(cdf, scheme, sets)
            rows = [
                {"feature_set": "-", "model": "BASELINE district mean", **metrics(y, b_mean, b_trend, b_mean)},
                {"feature_set": "-", "model": "BASELINE district trend", **metrics(y, b_trend, b_trend, b_mean)},
            ]
            rows += [
                {"feature_set": s, "model": m, **metrics(y, p, b_trend, b_mean)}
                for (s, m), p in preds.items()
            ]
            res = pd.DataFrame(rows)
            res.insert(0, "scheme", scheme)
            res.insert(0, "crop", crop)
            all_results.append(res)

            ranked = res.sort_values("RMSE").reset_index(drop=True)
            shown = pd.concat([ranked[ranked["feature_set"] == "-"], ranked[ranked["feature_set"] != "-"].head(TOP_N)])
            shown = shown.sort_values("RMSE")
            label = ("forward-in-time (train on earlier years, predict the next)" if scheme == "forward"
                     else "leave-one-year-out")
            print(f"\n--- {label}: {len(y)} test rows ---")
            print(shown.drop(columns=["crop", "scheme"]).to_string(
                index=False, float_format=lambda v: f"{v:7.3f}"))

            best = ranked[ranked["feature_set"] != "-"].groupby("feature_set").head(1)
            print("\nBest model per feature set:")
            print(best[["feature_set", "model", "RMSE", "Skill vs trend"]].to_string(
                index=False, float_format=lambda v: f"{v:7.3f}"))

    pd.concat(all_results).to_csv("benchmark_results_v3.csv", index=False)
    print("\nSaved full results to benchmark_results_v3.csv")