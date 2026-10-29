
import ee
import pandas as pd

ee.Initialize(project="buoyant-facet-510509-a7")

districts_fc = ee.FeatureCollection("FAO/GAUL/2025/level2").filter(
    ee.Filter.eq("GAUL1_NAME", "Madhya Pradesh")
)
DISTRICT_NAME_PROPERTY = "GAUL2_NAME"

# One SoilGrids250m asset per property, each a 6-band image (one band
# per depth interval, ordered 0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm,
# 100-200cm per ISRIC's own GEE documentation).
SOILGRIDS_ASSETS = {
    "ph": "phh2o_mean",
    "organic_carbon": "soc_mean",
    "clay_pct": "clay_mean",
    "sand_pct": "sand_mean",
    "bulk_density": "bdod_mean",
}


def inspect_band_names():
    """Prints the ACTUAL band names in each SoilGrids asset. Run this
    first (it already runs automatically below) -- if the main
    extraction still comes back empty, this tells you exactly what the
    real band names are, so you can hardcode one if needed instead of
    relying on 'take the first band'."""
    for out_name, asset_suffix in SOILGRIDS_ASSETS.items():
        img = ee.Image(f"projects/soilgrids-isric/{asset_suffix}")
        bands = img.bandNames().getInfo()
        print(f"  {asset_suffix}: {bands}")


def extract_soil_properties():
    soil_image = None
    for out_name, asset_suffix in SOILGRIDS_ASSETS.items():
        img = ee.Image(f"projects/soilgrids-isric/{asset_suffix}")
        # Take the FIRST band (0-5cm / topsoil) by position rather than
        # by a hardcoded name string -- avoids the exact-string-mismatch
        # bug that likely caused these columns to come back empty before.
        first_band_name = img.bandNames().get(0)
        band = img.select([first_band_name]).rename(out_name)
        soil_image = band if soil_image is None else soil_image.addBands(band)

    return soil_image.reduceRegions(
        collection=districts_fc, reducer=ee.Reducer.mean(), scale=250, tileScale=2
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


if __name__ == "__main__":
    print("Band names found in each SoilGrids asset (sanity check):")
    inspect_band_names()
    print()

    soil_df = fc_to_df(extract_soil_properties(), list(SOILGRIDS_ASSETS.keys()))
    print(soil_df.head())

    if soil_df.drop(columns="district").isna().all().all():
        print(
            "\nWARNING: every value came back null. Compare the band "
            "names printed above against SOILGRIDS_ASSETS -- something "
            "about the asset paths themselves (not just band names) may "
            "be off, e.g. a renamed or moved project."
        )

    soil_df.to_csv("mp_soil_properties.csv", index=False)
    print(f"\nSaved {len(soil_df)} rows to mp_soil_properties.csv")