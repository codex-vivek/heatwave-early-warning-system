import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def calculate_heat_index(temp_c, rh):
    """
    Calculates Heat Index in Celsius using the standard US NWS Rothfusz regression.
    Input temperature (temp_c) is in Celsius, relative humidity (rh) is in percentage.
    """
    # Convert Celsius to Fahrenheit
    t = (temp_c * 9.0/5.0) + 32.0
    
    # Simple formula first
    hi = 0.5 * (t + 61.0 + ((t - 68.0) * 1.2) + (rh * 0.094))
    
    # If the average of simple HI and T is 80°F or above, use the full Rothfusz equation
    if (hi + t) / 2 >= 80:
        hi = (-42.379 + 
              2.04901523 * t + 
              10.14333127 * rh - 
              0.22475541 * t * rh - 
              0.00683783 * t**2 - 
              0.05481717 * rh**2 + 
              0.00122874 * t**2 * rh + 
              0.00085282 * t * rh**2 - 
              0.00000199 * t**2 * rh**2)
        
        # Adjustments for dry and humid extremes
        if rh < 13 and 80 <= t <= 112:
            adj = -((13 - rh) / 4) * ((17 - abs(t - 95)) / 17)**0.5
            hi += adj
        elif rh > 85 and 80 <= t <= 87:
            adj = ((rh - 85) / 10) * ((87 - t) / 5)
            hi += adj
            
    # Convert Fahrenheit back to Celsius
    hi_c = (hi - 32) * 5.0/9.0
    return hi_c

def main():
    print("--- Starting STEP 2: Data Cleaning & EDA ---")
    
    # Load dataset
    data_path = "heatwave_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Please make sure Step 1 ran successfully.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows and {df.shape[1]} columns.")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # ---------------------------------------------
    # 1. Handle Missing Values
    # ---------------------------------------------
    # We will interpolate missing values within each city group to avoid cross-city data contamination.
    cols_to_interpolate = ['max_temp', 'min_temp', 'humidity', 'wind_speed', 'aqi', 'rainfall']
    print("Handling missing values using group-wise linear interpolation...")
    
    for col in cols_to_interpolate:
        null_count_before = df[col].isnull().sum()
        df[col] = df.groupby('city')[col].transform(lambda x: x.interpolate(method='linear').ffill().bfill())
        null_count_after = df[col].isnull().sum()
        if null_count_before > 0:
            print(f"  Fixed {null_count_before - null_count_after} missing values in '{col}'. Remaining: {null_count_after}")

    # ---------------------------------------------
    # 2. Outlier Analysis & Cleaning
    # ---------------------------------------------
    # Check for physically impossible values and cap or interpolate them.
    print("Checking and cleaning outliers/anomalies...")
    
    # Cap relative humidity to [0, 100]
    df['humidity'] = df['humidity'].clip(0, 100)
    
    # Cap wind speed and rainfall to not be negative
    df['wind_speed'] = df['wind_speed'].clip(lower=0)
    df['rainfall'] = df['rainfall'].clip(lower=0)
    
    # Temperature outliers check: extreme temp > 55°C or < 0°C for Indian cities
    temp_outliers = (df['max_temp'] > 55) | (df['max_temp'] < 0)
    if temp_outliers.any():
        print(f"  Found {temp_outliers.sum()} temperature anomalies. Interpolating...")
        df.loc[temp_outliers, 'max_temp'] = np.nan
        df['max_temp'] = df.groupby('city')['max_temp'].transform(lambda x: x.interpolate(method='linear').ffill().bfill())

    # ---------------------------------------------
    # 3. Create Heat Index Feature
    # ---------------------------------------------
    print("Calculating Rothfusz Heat Index...")
    # Apply standard Heat Index calculation row by row
    df['heat_index'] = df.apply(lambda row: calculate_heat_index(row['max_temp'], row['humidity']), axis=1)

    # ---------------------------------------------
    # 4. Save Cleaned Dataset
    # ---------------------------------------------
    cleaned_data_path = "heatwave_data_cleaned.csv"
    df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned dataset saved successfully to {cleaned_data_path}")

    # ---------------------------------------------
    # 5. Exploratory Data Analysis & Visualizations
    # ---------------------------------------------
    print("Generating EDA plots...")
    os.makedirs("plots", exist_ok=True)
    
    # Plot 1: Temperature trends over time for all cities
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='date', y='max_temp', hue='city', alpha=0.8)
    plt.title('Max Temperature Trends (2024 - 2026) for Indian Cities', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Max Temp (°C)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot1_path = os.path.join("plots", "temp_trends.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot1_path}")

    # Plot 2: Correlation heatmap of weather variables and Heat Index
    plt.figure(figsize=(10, 8))
    numeric_cols = ['max_temp', 'min_temp', 'humidity', 'wind_speed', 'aqi', 'rainfall', 'heat_index']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
    plt.title('Correlation Matrix of Weather Variables & Heat Index', fontsize=14)
    plt.tight_layout()
    plot2_path = os.path.join("plots", "correlation_heatmap.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot2_path}")

    # Plot 3: Distribution of temperature and severity levels
    # Define severity level based on temperature thresholds
    def get_severity(t):
        if t < 40: return 'Safe'
        elif 40 <= t < 43: return 'Warning'
        elif 43 <= t < 46: return 'Danger'
        else: return 'Extreme'
        
    df['severity'] = df['max_temp'].apply(get_severity)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 3a. Temperature Distribution Histogram
    sns.histplot(data=df, x='max_temp', kde=True, ax=axes[0], color='coral')
    axes[0].axvline(40, color='yellow', linestyle='--', label='Warning (40°C)')
    axes[0].axvline(43, color='orange', linestyle='--', label='Danger (43°C)')
    axes[0].axvline(46, color='red', linestyle='--', label='Extreme (46°C)')
    axes[0].set_title('Distribution of Max Temperatures', fontsize=13)
    axes[0].set_xlabel('Max Temp (°C)')
    axes[0].legend()
    
    # 3b. Count Plot of Severity Levels
    order = ['Safe', 'Warning', 'Danger', 'Extreme']
    palette = {'Safe': 'green', 'Warning': 'gold', 'Danger': 'crimson', 'Extreme': 'darkred'}
    sns.countplot(data=df, x='severity', order=order, palette=palette, ax=axes[1])
    axes[1].set_title('Distribution of Heatwave Severity Levels', fontsize=13)
    axes[1].set_xlabel('Severity Level')
    axes[1].set_ylabel('Days Count')
    
    plt.tight_layout()
    plot3_path = os.path.join("plots", "severity_distribution.png")
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot3_path}")
    
    print("\n--- STEP 2: Completed successfully ---")

if __name__ == "__main__":
    main()
