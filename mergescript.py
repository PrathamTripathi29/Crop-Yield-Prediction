import pandas as pd

# 1. Load the new season-optimized Earth Engine features
inputs_df = pd.read_csv("mp_feature_table_2016_2022.csv")

# 2. Load and extract the APY labels
apy_df = pd.read_csv("crop-wise-area-production-yield.csv", low_memory=False)
apy_df.columns = [c.strip().lower() for c in apy_df.columns]

apy_mp = apy_df[apy_df['state_name'].str.contains('Madhya Pradesh', case=False, na=False)].copy()
target_crops = ['Soyabean', 'Wheat', 'Gram']
apy_mp = apy_mp[apy_mp['crop_name'].str.strip().isin(target_crops)]

# Convert '2020-2021' to integer year
apy_mp['year'] = apy_mp['year'].str.split('-').str[0].astype(int)

# DYNAMIC FILTER: Keep all years from 2016 to 2022
apy_mp = apy_mp[apy_mp['year'].isin(range(2016, 2023))]

# Standardize text to match Earth Engine features
crop_map = {'Soyabean': 'Soybean', 'Wheat': 'Wheat', 'Gram': 'Gram'}
apy_mp['crop_name'] = apy_mp['crop_name'].str.strip().map(crop_map)
apy_mp['season'] = apy_mp['season'].str.strip().str.lower()

district_map = {
    'Agar-Malwa': 'Agar Malwa',
    'Khargone (West Nimar)': 'West Nimar',
    'Neemuch': 'Nimach',
    'Khandwa (East Nimar)': 'East Nimar',
    'Narmadapuram': 'Hoshangabad',
    'Narsimhapur': 'Narshimapura',
    'Niwari': 'Nivari'
}
apy_mp['district_name'] = apy_mp['district_name'].str.strip().replace(district_map)

# Extract final labels
apy_labels = apy_mp[['year', 'district_name', 'crop_name', 'season', 'yield']].dropna()
apy_labels.rename(columns={'district_name': 'district', 'crop_name': 'crop'}, inplace=True)

# 3. Final Merge
final_dataset = pd.merge(
    inputs_df, apy_labels, 
    on=['district', 'crop', 'year', 'season'], how='inner'
)

# Save the final dataset
final_dataset.to_csv("model_training_data_final.csv", index=False)
print(f"Successfully created 2016-2022 training dataset with {len(final_dataset)} rows!")