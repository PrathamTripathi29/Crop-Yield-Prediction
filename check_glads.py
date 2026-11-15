import ee

ee.Initialize(project="buoyant-facet-510509-a7")
print("Connected to Earth Engine", flush=True)

districts = ee.FeatureCollection("FAO/GAUL/2025/level2").filter(
    ee.Filter.eq("GAUL1_NAME", "Madhya Pradesh")
)
img = (
    ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
    .filterDate("2019-07-01", "2019-07-02")
    .select("RootMoist_inst")
    .first()
)

for scale in (27830, 10000, 5000, 2500):
    fc = img.reduceRegions(collection=districts, reducer=ee.Reducer.mean(), scale=scale)
    fc = fc.map(lambda f: ee.Feature(None, f.toDictionary()))
    rows = fc.getInfo()["features"]
    bad = [r["properties"]["GAUL2_NAME"] for r in rows if r["properties"].get("mean") is None]
    print(scale, "-> null districts:", bad, flush=True)