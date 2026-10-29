import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib

# Import Models
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. Load Dataset
df = pd.read_csv("model_training_data_final.csv")
# Drop 2015 to ensure clean SAR data
df = df[df['year'] > 2015].dropna().reset_index(drop=True)

models_to_test = {
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "Support Vector (SVR)": make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, gamma='scale')),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, max_features=0.8, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=300, max_depth=12, max_features=0.8, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42, verbose=-1)
}

# 2. Time-Series Train & Evaluate Per Crop
for crop in df['crop'].unique():
    print(f"\n{'='*75}\nTime-Series Benchmarking for: {crop.upper()} (Train: 2016-2020 | Test: 2021-2022)\n{'='*75}")
    
    crop_df = df[df['crop'] == crop].copy()
    
    # Split Train/Test
    train_df = crop_df[crop_df['year'] <= 2020].copy()
    test_df = crop_df[crop_df['year'] >= 2021].copy()
    
    # DYNAMIC TARGET ENCODING: Calculate historical yield per district using ONLY the train set
    district_means = train_df.groupby('district')['yield'].mean().rename('hist_yield')
    
    # Merge back into train and test
    train_df = train_df.join(district_means, on='district')
    test_df = test_df.join(district_means, on='district')
    
    # Failsafe: If a district appears in test but wasn't in train, fill with global average
    test_df['hist_yield'] = test_df['hist_yield'].fillna(train_df['yield'].mean())
    
    # DROP YEAR (to prevent trend collapse) AND DISTRICT (replaced by hist_yield)
    drop_cols = ['yield', 'district', 'season', 'crop', 'year']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['yield']
    
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df['yield']
    
    results = []
    
    for name, model in models_to_test.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        accuracy = max(0, 100 - mape)
        
        results.append({
            "Model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "Error (%)": mape,
            "Accuracy (%)": accuracy
        })
    
    results_df = pd.DataFrame(results).sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)
    print(results_df.to_string(index=False, formatters={"R2": "{:6.3f}".format, "MAE": "{:6.3f}".format, "RMSE": "{:6.3f}".format, "Error (%)": "{:9.3f}".format, "Accuracy (%)": "{:12.3f}".format}))
    
    # 3. Final Production Training (All 7 Years)
    best_model_name = results_df.iloc[0]['Model']
    best_model_instance = models_to_test[best_model_name]
    
    print(f"\n[+] Retraining winning {crop} model ({best_model_name}) on full 2016-2022 dataset...")
    
    # Apply Target Encoding to full dataset
    full_district_means = crop_df.groupby('district')['yield'].mean().rename('hist_yield')
    crop_df_full = crop_df.join(full_district_means, on='district')
    
    X_full = crop_df_full.drop(columns=drop_cols)
    y_full = crop_df_full['yield']
    
    best_model_instance.fit(X_full, y_full)
    
    filename = f"model_{crop.lower()}_timeseries.joblib"
    joblib.dump(best_model_instance, filename)
    print(f"[+] Saved: {filename}")