import joblib
import pandas as pd
import numpy as np

def main():
    binary_pkg = joblib.load("models/best_binary_model.joblib")
    severity_pkg = joblib.load("models/best_multiclass_model.joblib")
    
    # Let's print feature names
    print("Features expected by model:", binary_pkg['features'])
    
    # Let's fetch the forecast data for Kolkata
    # (Coordinates: 22.5726, 88.3639)
    from app import fetch_7day_forecast, prepare_forecast_features
    forecast_df = fetch_7day_forecast("Kolkata", 22.5726, 88.3639)
    
    X_forecast, forecast_with_features = prepare_forecast_features(forecast_df, "Kolkata", binary_pkg['features'])
    
    print("\n--- X_forecast Row 0 ---")
    print(X_forecast.iloc[0].to_dict())
    
    bin_preds = binary_pkg['model'].predict(X_forecast)
    sev_preds = severity_pkg['model'].predict(X_forecast)
    
    print("\nBinary predictions:", bin_preds)
    print("Severity predictions:", sev_preds)
    
    # Check if features have NaNs
    print("\nAny NaNs in X_forecast:", X_forecast.isna().any().any())
    
if __name__ == "__main__":
    main()
