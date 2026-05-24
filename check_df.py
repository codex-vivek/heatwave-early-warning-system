from app import fetch_7day_forecast
import pandas as pd

def main():
    # Fetch data for Kolkata
    df = fetch_7day_forecast("Kolkata", 22.5726, 88.3639)
    print("Columns in forecast_df:", df.columns)
    print("\nRow details:")
    # Print date, max_temp, humidity, and aqi
    print(df[['date', 'max_temp', 'humidity', 'aqi']].to_string())

if __name__ == "__main__":
    main()
