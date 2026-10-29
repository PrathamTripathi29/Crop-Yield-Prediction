
import ee
import pandas as pd

ee.Initialize(project="buoyant-facet-510509-a7")

# ---------------------------------------------------------------------
# 1. Districts: same source as pull_features_mp.py
# ---------------------------------------------------------------------
districts_fc = ee.FeatureCollection("FAO/GAUL/2025/level2").filter(
    ee.Filter.eq("GAUL1_NAME", "Madhya Pradesh")
)
DISTRICT_NAME_PROPERTY = "GAUL2_NAME"

# ---------------------------------------------------------------------
# 2. Season definitions -- MUST match pull_features_mp.py exactly
# ---------------------------------------------------------------------
SEASONS = {
    "kharif": {"start": "06-01", "end": "10-31", "spans_year_end": False},
    "rabi": {"start": "11-01", "end": "04-30", "spans_year_end": True},
}
CROP_SEASON = {"Soybean": "kharif", "Wheat": "rabi", "Gram": "rabi"}


def season_date_range(year: int, season: str):
    s = SEASONS[season]
    start = f"{year}-{s['start']}"
    end = f"{year + 1}-{s['end']}" if s["spans_year_end"] else f"{year}-{s['end']}"
    return start, end


# ---------------------------------------------------------------------
# 3. Extraction: seasonal cumulative rainfall per district
# ---------------------------------------------------------------------

def extract_rainfall(start, end):
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end)
    rainfall_total = chirps.sum().rename("rainfall_total")
    return rainfall_total.reduceRegions(
        collection=districts_fc, reducer=ee.Reducer.mean(), scale=5000, tileScale=2
    )


def fc_to_df(fc: ee.FeatureCollection, value_cols: list) -> pd.DataFrame:
    rows = []
    for f in fc.getInfo()["features"]:
        props = f["properties"]
        row = {"district": props.get(DISTRICT_NAME_PROPERTY)}
        for col in value_cols:
            row[col] = props.get(col)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------

def build_rainfall_table(years, crops=("Soybean", "Wheat", "Gram")) -> pd.DataFrame:
    all_rows = []
    for crop in crops:
        season = CROP_SEASON[crop]
        for year in years:
            start, end = season_date_range(year, season)
            print(f"{crop} ({season}) -- {year}: {start} to {end}")
            
            # Request "mean" from Earth Engine, then rename to rainfall_total in pandas
            df = fc_to_df(extract_rainfall(start, end), ["mean"])
            df.rename(columns={"mean": "rainfall_total"}, inplace=True)
            
            df["crop"] = crop
            df["year"] = year
            df["season"] = season
            all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


if __name__ == "__main__":
    YEARS = range(2020, 2023)  # match the rest of the pipeline's test range
    rainfall_table = build_rainfall_table(YEARS)
    rainfall_table.to_csv("mp_rainfall_table.csv", index=False)
    print(f"\nSaved {len(rainfall_table)} rows to mp_rainfall_table.csv")