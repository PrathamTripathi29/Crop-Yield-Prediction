"""
mergescript_v2.py

Joins the 2003-2022 feature table (mp_feature_table_v2.csv) with the APY
yield labels, and reports every place where rows get lost, so nothing
disappears silently. Keeps the APY `area` column for the crop-purity
test later.

Run after geescriptv2.py has finished the full pull.
"""

import pandas as pd

FEATURES_CSV = "mp_feature_table_v2.csv"
APY_CSV = "crop-wise-area-production-yield.csv"
OUTPUT_CSV = "model_training_data_v2.csv"
YEARS = range(2003, 2023)

CROP_MAP = {"Soyabean": "Soybean", "Wheat": "Wheat", "Gram": "Gram"}

# APY spelling -> GAUL 2025 spelling. Extend this using the unmatched
# names the script prints below (older years have more variants).
DISTRICT_MAP = {
    "Agar-Malwa": "Agar Malwa",
    "Khargone (West Nimar)": "West Nimar",
    "Neemuch": "Nimach",
    "Khandwa (East Nimar)": "East Nimar",
    "Narmadapuram": "Hoshangabad",
    "Narsimhapur": "Narshimapura",
    "Niwari": "Nivari",
}

# ---------------------------------------------------------------------
# 1. Labels
# ---------------------------------------------------------------------
apy = pd.read_csv(APY_CSV, low_memory=False)
apy.columns = [c.strip().lower() for c in apy.columns]
apy = apy[apy["state_name"].str.contains("Madhya Pradesh", case=False, na=False)].copy()
apy["crop_name"] = apy["crop_name"].str.strip()
apy = apy[apy["crop_name"].isin(CROP_MAP)].copy()
apy["year"] = apy["year"].astype(str).str.split("-").str[0].astype(int)
apy = apy[apy["year"].isin(YEARS)].copy()
apy["crop"] = apy["crop_name"].map(CROP_MAP)
apy["season"] = apy["season"].str.strip().str.lower()
CROP_SEASON = {"Soybean": "kharif", "Wheat": "rabi", "Gram": "rabi"}
whole_year = apy["season"] == "whole year"
print(f"Rows labelled 'whole year', remapped: {whole_year.sum()} (years {sorted(apy.loc[whole_year, 'year'].unique())})")
apy.loc[whole_year, "season"] = apy.loc[whole_year, "crop"].map(CROP_SEASON)
apy["district"] = apy["district_name"].str.strip().replace(DISTRICT_MAP)
apy["yield"] = pd.to_numeric(apy["yield"], errors="coerce")
apy["area"] = pd.to_numeric(apy["area"], errors="coerce")

keys = ["year", "district", "crop", "season"]
n_raw = len(apy)
labels = apy.dropna(subset=["yield"])
labels = labels[labels["yield"] > 0]
print(f"APY rows for MP, 3 crops, {YEARS.start}-{YEARS.stop - 1}: {n_raw}")
print(f"  dropped for missing or zero yield: {n_raw - len(labels)}")

dup_mask = labels.duplicated(keys, keep=False)
if dup_mask.any():
    print(f"  WARNING: {dup_mask.sum()} rows share a (year, district, crop, season) key. Sample:")
    print(labels[dup_mask].sort_values(keys)[keys + ["area", "yield"]].head(10).to_string(index=False))
    labels = labels.groupby(keys, as_index=False).agg({"yield": "mean", "area": "sum"})
    print("  -> averaged yield and summed area per key. Check this is what you want.")
labels = labels[keys + ["area", "yield"]]

# ---------------------------------------------------------------------
# 2. Merge, and report what doesn't match
# ---------------------------------------------------------------------
features = pd.read_csv(FEATURES_CSV)
merged = features.merge(labels, on=keys, how="outer", indicator=True)

counts = merged["_merge"].value_counts()
print("\nMerge result:")
print(f"  matched:            {counts.get('both', 0)}")
print(f"  features, no label: {counts.get('left_only', 0)}")
print(f"  label, no features: {counts.get('right_only', 0)}")

no_features = merged[merged["_merge"] == "right_only"]
if len(no_features):
    print("\nUnmatched label rows by year / season / crop:")
    print(no_features.groupby(["year", "season", "crop"]).size().to_string())
    print("\nDistricts with at least one unmatched label row:")
    print(sorted(no_features["district"].unique()))

no_labels = merged[merged["_merge"] == "left_only"]
if len(no_labels):
    print("\nGAUL districts with features but missing labels (district, number of crop-years):")
    print(no_labels.groupby("district").size().sort_values(ascending=False).to_string())

final = merged[merged["_merge"] == "both"].drop(columns="_merge")

# ---------------------------------------------------------------------
# 3. Coverage report
# ---------------------------------------------------------------------
print("\nRows per crop and year:")
print(final.groupby(["crop", "year"]).size().unstack(fill_value=0).to_string())

years_per_district = final.groupby(["crop", "district"])["year"].nunique().reset_index(name="years")
short = years_per_district[years_per_district["years"] < 15]
if len(short):
    print("\nDistricts with fewer than 15 labelled years (likely renamed or split districts):")
    print(short.sort_values(["crop", "years"]).to_string(index=False))

final.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(final)} rows to {OUTPUT_CSV}")