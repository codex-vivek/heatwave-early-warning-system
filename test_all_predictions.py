import joblib
import pandas as pd
import numpy as np

def main():
    binary_pkg = joblib.load("models/best_binary_model.joblib")
    severity_pkg = joblib.load("models/best_multiclass_model.joblib")
    features_list = binary_pkg['features']
    
    from app import fetch_7day_forecast, prepare_forecast_features
    
    test_locations = {
        "Delhi (UT)": {"lat": 28.6139, "lon": 77.2090},
        "Rajasthan (Jaipur)": {"lat": 26.9124, "lon": 75.7873},
        "West Bengal (Kolkata)": {"lat": 22.5726, "lon": 88.3639},
        "Maharashtra (Mumbai)": {"lat": 19.0760, "lon": 72.8777}
    }
    
    for name, coords in test_locations.items():
        print(f"\n==================== {name} ====================")
        forecast_df = fetch_7day_forecast(name, coords['lat'], coords['lon'])
        if forecast_df is None:
            print("Failed to fetch forecast.")
            continue
            
        X_forecast, forecast_with_features = prepare_forecast_features(forecast_df, name, features_list)
        
        bin_preds = binary_pkg['model'].predict(X_forecast)
        sev_preds = severity_pkg['model'].predict(X_forecast)
        
        print("Forecast Max Temps:", forecast_with_features['max_temp'].tolist())
        print("Forecast Heat Indexes:", forecast_with_features['heat_index'].tolist())
        print("Binary predictions:", bin_preds.tolist())
        print("Severity predictions:", sev_preds.tolist())
        print("Overall 7-Day Binary Outlook (bin_preds[0]):", bin_preds[0])
        print("Overall 7-Day Severity Outlook (sev_preds[0]):", sev_preds[0])

if __name__ == "__main__":
    main()
