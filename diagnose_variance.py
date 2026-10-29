"""
diagnose_variance.py

Answers two questions before any more modelling:

1. Where does the variation in yield actually live?
   - between districts (some districts are just better)
   - between years    (a good or bad season hits the whole state)
   - neither          (noise / things we can't observe)
   Reported as adjusted R2, so adding 52 district dummies doesn't
   flatter the district number.

2. At which level does each feature line up with yield?
   - state-level:  year-average feature vs year-average yield
                   (only ~7 points, treat as a hint, not proof)
   - district anomaly: feature and yield both minus the district's own
                   average, pooled over all rows
   - within-year:  feature and yield both minus that year's state
                   average, i.e. "are districts that are greener than
                   the rest of the state this year also higher-yielding?"

Run:  python diagnose_variance.py
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = "model_training_data_final.csv"
FIRST_YEAR = 2016

META = ["yield", "district", "year", "season", "crop"]
STATIC = ["ph", "organic_carbon", "clay_pct", "sand_pct", "bulk_density"]


def adjusted_r2(df, cols):
    X = pd.get_dummies(df[cols].astype(str), drop_first=True).astype(float)
    r2 = LinearRegression().fit(X, df["yield"]).score(X, df["yield"])
    n, p = X.shape
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df = df[df["year"] >= FIRST_YEAR]
    dynamic = [c for c in df.columns if c not in META + STATIC]

    for crop in sorted(df["crop"].unique()):
        cdf = df[df["crop"] == crop].dropna(subset=dynamic + ["yield"]).copy()
        print(f"\n{'=' * 74}\n{crop.upper()}: {len(cdf)} rows, "
              f"{cdf['year'].nunique()} years, {cdf['district'].nunique()} districts\n{'=' * 74}")

        district_only = adjusted_r2(cdf, ["district"])
        year_only = adjusted_r2(cdf, ["year"])
        both = adjusted_r2(cdf, ["district", "year"])
        print("Share of yield variance explained (adjusted R2):")
        print(f"  district identity alone : {district_only:6.3f}")
        print(f"  year alone              : {year_only:6.3f}")
        print(f"  district + year         : {both:6.3f}")
        print(f"  left over (noise/unseen): {1 - both:6.3f}")

        cols = ["yield"] + dynamic
        dist_anom = cdf[cols] - cdf.groupby("district")[cols].transform("mean")
        year_anom = cdf[cols] - cdf.groupby("year")[cols].transform("mean")
        year_means = cdf.groupby("year")[cols].mean()

        table = pd.DataFrame({
            "state-level (year means)": year_means[dynamic].corrwith(year_means["yield"]),
            "district anomaly (pooled)": dist_anom[dynamic].corrwith(dist_anom["yield"]),
            "within-year, across districts": year_anom[dynamic].corrwith(year_anom["yield"]),
        })
        print("\nCorrelation of each feature with yield, at three levels:")
        print(table.to_string(float_format=lambda v: f"{v:6.2f}"))

        print("\nYear-by-year averages (eyeball the relationship):")
        show = ["yield"] + [c for c in ["rainfall_total", "ndvi_max", "fapar_total", "rzsm_mean"] if c in cdf]
        print(year_means[show].to_string(float_format=lambda v: f"{v:8.3f}"))