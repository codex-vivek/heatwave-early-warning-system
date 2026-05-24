import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def main():
    print("--- Starting STEP 3: Feature Engineering ---")
    
    # Load cleaned dataset
    cleaned_data_path = "heatwave_data_cleaned.csv"
    if not os.path.exists(cleaned_data_path):
        raise FileNotFoundError(f"Missing {cleaned_data_path}. Please run step2_eda.py first.")
        
    df = pd.read_csv(cleaned_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort data by city and date to ensure correct time-series alignment
    df = df.sort_values(['city', 'date']).reset_index(drop=True)
    
    # ---------------------------------------------
    # 1. Define Daily Targets (for reference)
    # ---------------------------------------------
    # Binary: Heatwave (1) if max_temp >= 40°C, else (0)
    df['heatwave_today'] = (df['max_temp'] >= 40).astype(int)
    
    # Multi-class: Severity Level
    # 0 = Safe (< 40°C)
    # 1 = Warning (40-43°C)
    # 2 = Danger (43-46°C)
    # 3 = Extreme (>= 46°C)
    def encode_severity(t):
        if t < 40: return 0
        elif 40 <= t < 43: return 1
        elif 43 <= t < 46: return 2
        else: return 3
        
    df['severity_today'] = df['max_temp'].apply(encode_severity)

    # ---------------------------------------------
    # 2. Define Future targets (Next 7 Days Warning)
    # ---------------------------------------------
    # Target 1: Will a heatwave occur in the next 7 days (t+1 to t+7)?
    # Target 2: What is the maximum severity level reached in the next 7 days?
    df['heatwave_next_7d'] = 0.0
    df['severity_next_7d'] = 0.0
    
    # Shift forward row-by-row (looking ahead 1 to 7 days) and take the maximum
    for i in range(1, 8):
        df['heatwave_next_7d'] = np.maximum(df['heatwave_next_7d'], df.groupby('city')['heatwave_today'].shift(-i))
        df['severity_next_7d'] = np.maximum(df['severity_next_7d'], df.groupby('city')['severity_today'].shift(-i))

    # ---------------------------------------------
    # 3. Create Lag Features (1, 2, and 7 days)
    # ---------------------------------------------
    # Lag features capture immediate weather trends and weekly patterns.
    lag_cols = ['max_temp', 'humidity', 'heat_index', 'aqi']
    print("Generating lag features (1-day, 2-day, 7-day lags)...")
    for lag in [1, 2, 7]:
        for col in lag_cols:
            df[f'{col}_lag_{lag}'] = df.groupby('city')[col].shift(lag)

    # ---------------------------------------------
    # 4. Create Rolling Average Features (3-day and 7-day)
    # ---------------------------------------------
    # Rolling averages capture short-term weather trends.
    roll_cols = ['max_temp', 'humidity', 'heat_index']
    print("Generating rolling average features (3-day and 7-day)...")
    for window in [3, 7]:
        for col in roll_cols:
            # We use rolling mean grouped by city.
            df[f'{col}_roll_mean_{window}'] = df.groupby('city')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())

    # ---------------------------------------------
    # 5. Clean up NaNs created by shifts
    # ---------------------------------------------
    # Lag and future shift features create NaN values at the boundaries.
    # We drop these boundary rows to keep our training matrix clean.
    print(f"Dataset shape before dropping NaNs: {df.shape}")
    df_ml = df.dropna().copy()
    print(f"Dataset shape after dropping NaNs: {df_ml.shape}")

    # Convert targets to integer type
    df_ml['heatwave_next_7d'] = df_ml['heatwave_next_7d'].astype(int)
    df_ml['severity_next_7d'] = df_ml['severity_next_7d'].astype(int)

    # ---------------------------------------------
    # 6. One-Hot Encoding for City
    # ---------------------------------------------
    # Convert city names to one-hot columns
    df_ml = pd.get_dummies(df_ml, columns=['city'], drop_first=False)
    
    # Identify feature and target columns
    # We drop targets, dates, and daily intermediate values from training features.
    cols_to_drop = ['date', 'heatwave_today', 'severity_today', 'heatwave_next_7d', 'severity_next_7d']
    feature_cols = [c for c in df_ml.columns if c not in cols_to_drop]
    
    X = df_ml[feature_cols]
    y_bin = df_ml['heatwave_next_7d']
    y_sev = df_ml['severity_next_7d']

    # Keep track of dates and cities for split mapping
    dates = df_ml['date']
    
    # ---------------------------------------------
    # 7. Time-based Train-Test Split
    # ---------------------------------------------
    # To prevent time-series data leakage, we split by date.
    # Training: Data before 2025-12-01
    # Testing: Data on or after 2025-12-01 (approx. last 20% of dataset)
    split_date = pd.to_datetime('2025-12-01')
    
    train_mask = dates < split_date
    test_mask = dates >= split_date
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train_bin, y_test_bin = y_bin[train_mask], y_bin[test_mask]
    y_train_sev, y_test_sev = y_sev[train_mask], y_sev[test_mask]
    
    print(f"Train set size: {len(X_train)} rows")
    print(f"Test set size: {len(X_test)} rows")
    
    # ---------------------------------------------
    # 8. Address Class Imbalance with SMOTE
    # ---------------------------------------------
    # Heatwaves are seasonal and rare, leading to highly imbalanced target sets.
    print("\nClass distribution before SMOTE (Binary target):")
    print(y_train_bin.value_counts(normalize=True))
    
    print("Class distribution before SMOTE (Severity target):")
    print(y_train_sev.value_counts(normalize=True))
    
    print("\nApplying SMOTE to balance the training sets...")
    
    # SMOTE for Binary classification
    smote_bin = SMOTE(random_state=42)
    X_train_bin_res, y_train_bin_res = smote_bin.fit_resample(X_train, y_train_bin)
    
    # SMOTE for Multi-class classification
    # If some severity classes have very few samples, SMOTE will generate synthetic ones.
    smote_sev = SMOTE(random_state=42)
    X_train_sev_res, y_train_sev_res = smote_sev.fit_resample(X_train, y_train_sev)
    
    print(f"Resampled Binary training set size: {len(X_train_bin_res)} rows")
    print(f"Resampled Severity training set size: {len(X_train_sev_res)} rows")
    
    # Save the processed data
    os.makedirs("processed_data", exist_ok=True)
    
    X_train.to_csv("processed_data/X_train_raw.csv", index=False)
    X_test.to_csv("processed_data/X_test.csv", index=False)
    
    # Save resampled binary datasets
    X_train_bin_res.to_csv("processed_data/X_train_bin.csv", index=False)
    y_train_bin_res.to_csv("processed_data/y_train_bin.csv", index=False)
    y_test_bin.to_csv("processed_data/y_test_bin.csv", index=False)
    
    # Save resampled severity datasets
    X_train_sev_res.to_csv("processed_data/X_train_sev.csv", index=False)
    y_train_sev_res.to_csv("processed_data/y_train_sev.csv", index=False)
    y_test_sev.to_csv("processed_data/y_test_sev.csv", index=False)
    
    print("\n--- STEP 3: Completed successfully ---")

if __name__ == "__main__":
    main()
