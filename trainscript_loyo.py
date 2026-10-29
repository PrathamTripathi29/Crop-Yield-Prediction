"""
trainscript_loyo.py

Diagnostic benchmark for the yield models. Validation matches how the
app will really be used: predict a year the model has never seen.

For each crop it holds out one year at a time, trains on the other
years, and reports three things per model:
  - R2 / RMSE / MAE on raw yield
  - "Skill vs district mean": how much better the model is than simply
    predicting each district's average yield from the training years.
    0 = no better than the baseline, 1 = perfect, negative = worse.

Two modes are run for every crop:
  raw     -> features as they are (what the old script did, but with
             honest year-based validation)
  anomaly -> subtract each district's training-year average from both
             the yield and the dynamic features, so the model has to
             explain WHY a year was better or worse than usual for that
             district, instead of recognising which district it is.

No models are saved here on purpose: an anomaly model needs the
district baselines stored next to it. We save once we know which mode
wins.
"""

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_PATH = "model_training_data_final.csv"
FIRST_YEAR = 2016
# vh_total is a SUM of dB values, so it depends on how many Sentinel-1
# images existed that season (S1B stopped in late 2021). Run once with
# False and once with True and compare.
USE_VH = False

META = ["yield", "district", "year", "season", "crop"]
STATIC = ["ph", "organic_carbon", "clay_pct", "sand_pct", "bulk_density"]

MODELS = {
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, gamma="scale")),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=3,
                            subsample=0.8, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=3,
                              subsample=0.8, random_state=42, verbose=-1),
}


def loyo_predictions(crop_df, feature_cols, mode):
    """Leave-one-year-out predictions for every model, plus the
    district-mean baseline, all aligned row for row."""
    dynamic = [c for c in feature_cols if c not in STATIC]
    cols = ["yield"] + dynamic
    y_true, base_pred = [], []
    preds = {name: [] for name in MODELS}

    for held_out in sorted(crop_df["year"].unique()):
        train = crop_df[crop_df["year"] != held_out].reset_index(drop=True)
        test = crop_df[crop_df["year"] == held_out].reset_index(drop=True)

        # Baselines come from TRAINING years only, so nothing leaks.
        district_mean = train.groupby("district")[cols].mean()
        overall_mean = train[cols].mean()
        test_base = (
            district_mean.reindex(test["district"]).reset_index(drop=True).fillna(overall_mean)
        )
        y_true.append(test["yield"].values)
        base_pred.append(test_base["yield"].values)

        if mode == "raw":
            X_tr, y_tr, X_te, offset = train[feature_cols], train["yield"], test[feature_cols], 0.0
        else:
            train_base = district_mean.reindex(train["district"]).reset_index(drop=True)
            X_tr = train[dynamic] - train_base[dynamic]
            y_tr = train["yield"] - train_base["yield"]
            X_te = test[dynamic] - test_base[dynamic]
            offset = test_base["yield"].values

        for name, model in MODELS.items():
            fitted = clone(model).fit(X_tr, y_tr)
            preds[name].append(fitted.predict(X_te) + offset)

    return (
        np.concatenate(y_true),
        np.concatenate(base_pred),
        {name: np.concatenate(p) for name, p in preds.items()},
    )


def score(y, pred, base):
    return {
        "R2": r2_score(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
        "Skill vs district mean": 1 - np.sum((y - pred) ** 2) / np.sum((y - base) ** 2),
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df = df[df["year"] >= FIRST_YEAR]

    feature_cols = [c for c in df.columns if c not in META]
    if not USE_VH:
        feature_cols = [c for c in feature_cols if c != "vh_total"]
    df = df.dropna(subset=feature_cols + ["yield"])

    print("Rows per crop and year after cleaning:")
    print(df.groupby(["crop", "year"]).size().unstack(fill_value=0))
    print("\nFeatures used:", feature_cols)

    for crop in sorted(df["crop"].unique()):
        crop_df = df[df["crop"] == crop]
        print(f"\n{'=' * 78}")
        print(f"{crop.upper()}: {len(crop_df)} rows, {crop_df['district'].nunique()} districts, "
              f"yield mean {crop_df['yield'].mean():.3f}, std {crop_df['yield'].std():.3f}")
        print("=" * 78)

        for mode in ("raw", "anomaly"):
            y, base, preds = loyo_predictions(crop_df, feature_cols, mode)
            rows = [{"Model": "BASELINE: district mean", **score(y, base, base)}]
            rows += [{"Model": name, **score(y, p, base)} for name, p in preds.items()]
            table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
            print(f"\n--- mode: {mode} (leave-one-year-out) ---")
            print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))