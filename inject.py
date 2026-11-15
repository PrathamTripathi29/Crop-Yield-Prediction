import pandas as pd
import numpy as np

# Load the filtered IDP dataset
df = pd.read_csv("mp_irrigation_2002_2023.csv")

# Extract the starting year for sorting and math
df['start_year'] = df['year'].astype(str).str.split('-').str[0].astype(int)

# Create a master dataframe with all district, crop, and year combinations (2002 to 2023)
districts = df['district_name'].unique()
crops = df['crop_name'].unique()
all_years = range(2002, 2024)

master_index = pd.MultiIndex.from_product(
    [districts, crops, all_years], 
    names=['district_name', 'crop_name', 'start_year']
).to_frame(index=False)

# Merge existing data into the master frame
df_full = master_index.merge(df, on=['district_name', 'crop_name', 'start_year'], how='left')

# Sort to ensure chronological order for interpolation
df_full = df_full.sort_values(['district_name', 'crop_name', 'start_year'])

# Interpolate the missing 'area' linearly within each district and crop group
df_full['area'] = df_full.groupby(['district_name', 'crop_name'])['area'].transform(
    lambda group: group.interpolate(method='linear', limit_direction='both')
)

# Reconstruct the text year column (e.g., 2021 -> '2021-2022')
df_full['year'] = df_full['start_year'].astype(str) + "-" + (df_full['start_year'] + 1).astype(str)

# Clean up and save
df_full['state_name'] = "Madhya Pradesh"
df_full = df_full.drop(columns=['start_year']).dropna(subset=['area'])
df_full.to_csv("mp_irrigation_2002_2023_interpolated.csv", index=False)

print("Interpolation complete. 2021-2022 rows added successfully.")