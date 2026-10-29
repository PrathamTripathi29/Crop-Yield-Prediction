
import os
import time
from datetime import date

import ee
import pandas as pd

EE_PROJECT = "buoyant-facet-510509-a7"
ee.Initialize(project=EE_PROJECT)

# First run with SMOKE_TEST = True: pulls 2019 only and prints the columns
# so you can check everything works. Then set it to False for the full pull.
SMOKE_TEST = True
YEARS = range(2003, 2023)
CACHE_DIR = "feature_cache_v2"
OUTPUT_CSV = "mp_feature_table_v2.csv"
os.makedirs(CACHE_DIR, exist_ok=True)

SEASON_CROPS = {"kharif": ["Soybean"], "rabi": ["Wheat", "Gram"]}
# (calendar month, year offset). Rabi runs Nov of year Y to Apr of Y+1.
SEASON_MONTHS = {
    "kharif": [(6, 0), (7, 0), (8, 0), (9, 0), (10, 0)],
    "rabi": [(11, 0), (12, 0), (1, 1), (2, 1), (3, 1), (4, 1)],
}
HEAVY_RAIN_MM = 25
WET_DAY_MM = 2.5
HOT_DAY_C = 35

# ---------------------------------------------------------------------
# 1. Districts and cropland mask (same as before)
# ---------------------------------------------------------------------
districts_fc = ee.FeatureCollection("FAO/GAUL/2025/level2").filter(
    ee.Filter.eq("GAUL1_NAME", "Madhya Pradesh")
)
DISTRICT_NAME_PROPERTY = "GAUL2_NAME"

worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
cropland_binary_mp = worldcover.eq(40).clip(districts_fc.geometry().bounds())


def cropland_fraction():
    return cropland_binary_mp.reduceResolution(
        reducer=ee.Reducer.mean(), maxPixels=1024, bestEffort=True
    )


# ---------------------------------------------------------------------
# 2. Season and month windows
# ---------------------------------------------------------------------
def stage_windows(year, season):
    """[(label, start, end_exclusive), ...] one entry per month."""
    windows = []
    for month, offset in SEASON_MONTHS[season]:
        y = year + offset
        start = date(y, month, 1)
        end = date(y + (month == 12), month % 12 + 1, 1)
        windows.append((f"m{month:02d}", start.isoformat(), end.isoformat()))
    return windows


def season_bounds(year, season):
    windows = stage_windows(year, season)
    return windows[0][1], windows[-1][2]


# ---------------------------------------------------------------------
# 3. Feature builders. Each returns (multi-band image, band names).
#    All of them return 2+ bands so reduceRegions names the properties
#    after the bands.
# ---------------------------------------------------------------------
def modis_features(year, season):
    start, end = season_bounds(year, season)
    collection = ee.ImageCollection("MODIS/061/MOD13Q1")
    crop_mask = cropland_fraction().gt(0.5)

    def prep(img):
        # SummaryQA: 0 good, 1 marginal, 2 snow/ice, 3 cloudy
        good = img.select("SummaryQA").lte(1)
        ndvi = img.select("NDVI").multiply(0.0001)
        nir = img.select("sur_refl_b02").multiply(0.0001)
        swir = img.select("sur_refl_b07").multiply(0.0001)
        lswi = nir.subtract(swir).divide(nir.add(swir)).rename("LSWI")
        return ndvi.addBands(lswi).updateMask(crop_mask).updateMask(good)

    season_col = collection.filterDate(start, end).map(prep)
    bands = [
        season_col.select("NDVI").mean().rename("ndvi_mean"),
        season_col.select("NDVI").max().rename("ndvi_max"),
        season_col.select("LSWI").mean().rename("lswi_mean"),
        season_col.select("LSWI").max().rename("lswi_max"),
    ]
    names = ["ndvi_mean", "ndvi_max", "lswi_mean", "lswi_max"]
    for label, s, e in stage_windows(year, season):
        month_ndvi = collection.filterDate(s, e).map(prep).select("NDVI").mean()
        bands.append(month_ndvi.rename(f"ndvi_{label}"))
        names.append(f"ndvi_{label}")
    return ee.Image.cat(bands), names


def fapar_features(year, season):
    start, end = season_bounds(year, season)
    crop_mask = cropland_fraction().gt(0.5)

    def prep(img):
        fpar = img.select("Fpar_500m")
        # valid range is 0-100; 249-255 are fill codes (water, barren, no retrieval)
        return fpar.updateMask(fpar.lte(100)).multiply(0.01).updateMask(crop_mask)

    col = ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(start, end).map(prep)
    image = ee.Image.cat([col.mean().rename("fapar_mean"), col.max().rename("fapar_max")])
    return image, ["fapar_mean", "fapar_max"]


def chirps_features(year, season):
    start, end = season_bounds(year, season)
    daily = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    season_col = daily.filterDate(start, end)
    bands = [
        season_col.sum().rename("rainfall_total"),
        season_col.max().rename("rain_max_day"),
        season_col.map(lambda i: i.gte(HEAVY_RAIN_MM)).sum().rename("heavy_rain_days"),
        season_col.map(lambda i: i.gte(WET_DAY_MM)).sum().rename("wet_days"),
    ]
    names = ["rainfall_total", "rain_max_day", "heavy_rain_days", "wet_days"]
    for label, s, e in stage_windows(year, season):
        bands.append(daily.filterDate(s, e).sum().rename(f"rain_{label}"))
        names.append(f"rain_{label}")
    return ee.Image.cat(bands), names


def era5_features(year, season):
    start, end = season_bounds(year, season)
    era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    hot_k = HOT_DAY_C + 273.15  # compare in Kelvin, so no per-image conversion

    season_era = era.filterDate(start, end)
    season_tmax = season_era.select("temperature_2m_max")
    bands = [
        season_tmax.mean().subtract(273.15).rename("tmax_mean"),
        season_era.select("temperature_2m_min").mean().subtract(273.15).rename("tmin_mean"),
        season_tmax.map(lambda i: i.gte(hot_k)).sum().rename("hot_days_35"),
    ]
    names = ["tmax_mean", "tmin_mean", "hot_days_35"]
    for label, s, e in stage_windows(year, season):
        month_tmax = era.filterDate(s, e).select("temperature_2m_max")
        bands.append(month_tmax.mean().subtract(273.15).rename(f"tmax_{label}"))
        bands.append(month_tmax.map(lambda i: i.gte(hot_k)).sum().rename(f"hot35_{label}"))
        names += [f"tmax_{label}", f"hot35_{label}"]
    return ee.Image.cat(bands), names


def gldas_features(year, season):
    start, end = season_bounds(year, season)
    col = (
        ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
        .filterDate(start, end)
        .select("RootMoist_inst")
    )
    image = ee.Image.cat([col.mean().rename("rzsm_mean"), col.min().rename("rzsm_min")])
    return image, ["rzsm_mean", "rzsm_min"]


def soil_features():
    layers = {
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02": "ph",
        "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02": "organic_carbon",
        "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02": "clay_pct",
        "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02": "sand_pct",
        "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02": "bulk_density",
    }
    bands = [ee.Image(a).select("b0").rename(n) for a, n in layers.items()]
    return ee.Image.cat(bands), list(layers.values())


# ---------------------------------------------------------------------
# 4. Reduce to one row per district, with retries
# ---------------------------------------------------------------------
def reduce_to_df(image, names, scale, crs=None):
    fc = image.reduceRegions(
        collection=districts_fc,
        reducer=ee.Reducer.mean(),
        scale=scale,
        crs=crs,
        tileScale=4,
    )
    # Drop geometry so the getInfo() payload stays small.
    fc = fc.map(lambda f: ee.Feature(None, f.toDictionary()))
    rows = []
    for feat in fc.getInfo()["features"]:
        props = feat["properties"]
        row = {"district": props.get(DISTRICT_NAME_PROPERTY)}
        for n in names:
            row[n] = props.get(n)
        rows.append(row)
    return pd.DataFrame(rows)


def pull(builder, year, season, scale, tries=4):
    for attempt in range(1, tries + 1):
        try:
            image, names = builder(year, season)
            return reduce_to_df(image, names, scale)
        except Exception as e:
            print(f"   {builder.__name__} {season} {year}: attempt {attempt}/{tries} "
                  f"failed ({type(e).__name__}: {str(e)[:150]})")
            time.sleep(8 * attempt)
    raise RuntimeError(f"{builder.__name__} failed for {season} {year}")


def build_season_year(year, season, soil_df):
    cache_path = f"{CACHE_DIR}/{season}_{year}.csv"
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    print(f"{season} {year}: pulling features...")
    parts = [
        pull(modis_features, year, season, 250),
        pull(fapar_features, year, season, 500),
        pull(chirps_features, year, season, 5566),
        pull(era5_features, year, season, 11132),
        pull(gldas_features, year, season, 27830),
    ]
    df = parts[0]
    for part in parts[1:]:
        df = df.merge(part, on="district")
    df = df.merge(soil_df, on="district")
    df["year"], df["season"] = year, season
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    era_bands = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").first().bandNames().getInfo()
    print("ERA5-Land temperature bands:", [b for b in era_bands if b.startswith("temperature_2m")])

    soil_image, soil_names = soil_features()
    soil_df = reduce_to_df(soil_image, soil_names, 250, crs="EPSG:4326")

    years = [2019] if SMOKE_TEST else YEARS
    frames = []
    for season, crops in SEASON_CROPS.items():
        for year in years:
            base = build_season_year(year, season, soil_df)
            for crop in crops:
                frames.append(base.assign(crop=crop))
    table = pd.concat(frames, ignore_index=True)

    if SMOKE_TEST:
        print("\nShape:", table.shape)
        print("Columns:", table.columns.tolist())
        print("\nFirst row:\n", table.iloc[0])
        nulls = table.isna().sum()
        print("\nColumns with nulls:\n", nulls[nulls > 0])
        raise SystemExit("\nSmoke test finished. If the columns look right, set "
                         "SMOKE_TEST = False and run again for the full pull.")

    table.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(table)} rows to {OUTPUT_CSV}")