"""
geescript.py
Extracts satellite and environmental data for Madhya Pradesh (2016-2022).
Optimized: Season-based pulling & SAR Mean normalization.
"""
import ee
import pandas as pd

# Initialize Earth Engine (Make sure to run ee.Authenticate() once if needed)
ee.Initialize(project="buoyant-facet-510509-a7")

# ---------------------------------------------------------------------
# 1. Districts & Masks
# ---------------------------------------------------------------------
districts_fc = ee.FeatureCollection("FAO/GAUL/2025/level2").filter(
    ee.Filter.eq("GAUL1_NAME", "Madhya Pradesh")
)
DISTRICT_NAME_PROPERTY = "GAUL2_NAME"

worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
cropland_binary = worldcover.eq(40)

MP_REGION = districts_fc.geometry().bounds()
cropland_binary_mp = cropland_binary.clip(MP_REGION)

def cropland_fraction():
    return cropland_binary_mp.reduceResolution(
        reducer=ee.Reducer.mean(), maxPixels=1024, bestEffort=True
    )

# ---------------------------------------------------------------------
# 2. Season Mapping
# ---------------------------------------------------------------------
SEASONS = {
    "kharif": {"start": "06-01", "end": "10-31", "spans_year_end": False},
    "rabi": {"start": "11-01", "end": "04-30", "spans_year_end": True},
}

def season_date_range(year: int, season: str):
    s = SEASONS[season]
    start = f"{year}-{s['start']}"
    end = f"{year + 1}-{s['end']}" if s["spans_year_end"] else f"{year}-{s['end']}"
    return start, end

# ---------------------------------------------------------------------
# 3. Extraction Functions
# ---------------------------------------------------------------------
def extract_ndvi_lswi(start, end):
    modis = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end)
    crop_mask_250 = cropland_fraction().gt(0.5)

    def compute_indices(img):
        ndvi = img.select("NDVI").multiply(0.0001)
        nir = img.select("sur_refl_b02").multiply(0.0001)
        swir = img.select("sur_refl_b07").multiply(0.0001)
        lswi = nir.subtract(swir).divide(nir.add(swir)).rename("LSWI")
        return (
            ndvi.rename("NDVI")
            .addBands(lswi)
            .updateMask(crop_mask_250)
            .copyProperties(img, ["system:time_start"])
        )

    indexed = modis.map(compute_indices)
    combined = (
        indexed.select("NDVI").mean().rename("ndvi_mean")
        .addBands(indexed.select("NDVI").max().rename("ndvi_max"))
        .addBands(indexed.select("LSWI").mean().rename("lswi_mean"))
        .addBands(indexed.select("LSWI").max().rename("lswi_max"))
    )
    return combined.reduceRegions(collection=districts_fc, reducer=ee.Reducer.mean(), scale=250, tileScale=2)

def extract_fapar(start, end):
    modis = ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(start, end)
    crop_mask_500 = cropland_fraction().gt(0.5)

    def mask_fapar(img):
        fapar = img.select("Fpar_500m").multiply(0.01)
        return fapar.updateMask(crop_mask_500)

    fapar_total = modis.map(mask_fapar).sum().rename("fapar_total")
    return fapar_total.reduceRegions(collection=districts_fc, reducer=ee.Reducer.mean(), scale=500, tileScale=2)

def extract_sar(start, end):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select("VH")
    )
    def mask_vh(img):
        return img.updateMask(cropland_binary_mp)
        
    # MATHEMATICAL FIX: Using mean() for decibels to prevent pass-frequency bias
    vh_mean = s1.map(mask_vh).mean().rename("vh_mean")
    return vh_mean.reduceRegions(collection=districts_fc, reducer=ee.Reducer.mean(), scale=250, tileScale=8)

def extract_rzsm(start, end):
    gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H").filterDate(start, end).select("RootMoist_inst")
    rzsm_mean = gldas.mean().rename("rzsm_mean")
    return rzsm_mean.reduceRegions(collection=districts_fc, reducer=ee.Reducer.mean(), scale=27830, tileScale=2)

def extract_rainfall(start, end):
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(start, end).select("precipitation")
    rainfall_total = chirps.sum().rename("rainfall_total")
    return rainfall_total.reduceRegions(collection=districts_fc, reducer=ee.Reducer.mean(), scale=5566, tileScale=2)

def extract_soil_properties():
    soil_layers = {
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02": "ph",
        "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02": "organic_carbon",
        "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02": "clay_pct",
        "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02": "sand_pct",
        "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02": "bulk_density",
    }
    soil_image = None
    for asset_id, out_name in soil_layers.items():
        band = ee.Image(asset_id).select("b0").rename(out_name)
        soil_image = band if soil_image is None else soil_image.addBands(band)

    return soil_image.reduceRegions(
        collection=districts_fc, reducer=ee.Reducer.mean(), scale=250, crs="EPSG:4326", tileScale=2
    )

# ---------------------------------------------------------------------
# 4. Smart DataFrame Parser
# ---------------------------------------------------------------------
def fc_to_df(fc: ee.FeatureCollection, value_cols: list) -> pd.DataFrame:
    rows = []
    for f in fc.getInfo()["features"]:
        props = f["properties"]
        row = {"district": props.get(DISTRICT_NAME_PROPERTY)}
        
        for col in value_cols:
            if col in props:
                row[col] = props[col]
            elif f"{col}_mean" in props:
                row[col] = props[f"{col}_mean"]
            elif "mean" in props and len(value_cols) == 1:
                row[col] = props["mean"]
            else:
                row[col] = pd.NA
                
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# 5. Main Pipeline (Optimized Per Season)
# ---------------------------------------------------------------------
def build_feature_table(years) -> pd.DataFrame:
    print("Pulling static soil properties (one-time)...")
    soil_df = fc_to_df(
        extract_soil_properties(),
        ["ph", "organic_carbon", "clay_pct", "sand_pct", "bulk_density"],
    )

    all_rows = []
    seasons_to_pull = ["kharif", "rabi"]
    
    for year in years:
        for season in seasons_to_pull:
            start, end = season_date_range(year, season)
            print(f"Season: {season.upper()} -- {year}: {start} to {end}")

            # Pull features per season (No duplicate pulls for Wheat/Gram)
            ndvi_lswi_df = fc_to_df(extract_ndvi_lswi(start, end), ["ndvi_mean", "ndvi_max", "lswi_mean", "lswi_max"])
            fapar_df = fc_to_df(extract_fapar(start, end), ["fapar_total"])
            sar_df = fc_to_df(extract_sar(start, end), ["vh_mean"]) # Note the name change to vh_mean
            rzsm_df = fc_to_df(extract_rzsm(start, end), ["rzsm_mean"])
            rain_df = fc_to_df(extract_rainfall(start, end), ["rainfall_total"])

            merged = (
                ndvi_lswi_df.merge(fapar_df, on="district")
                .merge(sar_df, on="district")
                .merge(rzsm_df, on="district")
                .merge(rain_df, on="district")
                .merge(soil_df, on="district")
            )
            
            # Smartly duplicate the Rabi data for both winter crops
            target_crops = ["Soybean"] if season == "kharif" else ["Wheat", "Gram"]
            
            for crop in target_crops:
                crop_df = merged.copy()
                crop_df["crop"] = crop
                crop_df["year"] = year
                crop_df["season"] = season
                all_rows.append(crop_df)

    return pd.concat(all_rows, ignore_index=True)

if __name__ == "__main__":
    # Clean 7-year timeline without missing SAR gaps
    YEARS = range(2016, 2023)
    
    feature_table = build_feature_table(YEARS)
    feature_table.to_csv("mp_feature_table_2016_2022.csv", index=False)
    print(f"\nSaved {len(feature_table)} rows to mp_feature_table_2016_2022.csv")