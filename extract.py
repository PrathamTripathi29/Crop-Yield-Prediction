import pandas as pd

# Load the full dataset
df = pd.read_csv("crop-wise-irrigated-area.csv", low_memory=False)

# Filter for Madhya Pradesh
df_mp = df[df['state_name'].str.contains('Madhya Pradesh', case=False, na=False)]

# Filter for the specific crops
target_crops = ['Wheat', 'Gram', 'Soyabean']
df_filtered = df_mp[df_mp['crop_name'].isin(target_crops)].copy()

# Extract the starting year as an integer (e.g., '2002-2003' -> 2002)
df_filtered['start_year'] = df_filtered['year'].astype(str).str.split('-').str[0].astype(int)

# Filter for 2002 to 2023 (one year before 2003, one year after 2022)
df_final = df_filtered[(df_filtered['start_year'] >= 2002) & (df_filtered['start_year'] <= 2023)]

# Drop the temporary column and save
df_final = df_final.drop(columns=['start_year'])
df_final.to_csv("mp_irrigation_2002_2023.csv", index=False)
print("Extraction complete. Saved to mp_irrigation_2002_2023.csv")