import pandas as pd
import requests
import time
from datetime import datetime, timedelta

# List of cities with their coordinates
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873}
}

def fetch_weather_data(city_name, lat, lon, start_date, end_date):
    """
    Fetches historical weather data from Open-Meteo API.
    """
    print(f"Fetching weather data for {city_name}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
        "hourly": "relative_humidity_2m", # We'll average this to get daily
        "timezone": "Asia/Kolkata"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching weather for {city_name}: {response.text}")
        return None
    
    data = response.json()
    
    # Process daily data
    daily_df = pd.DataFrame(data['daily'])
    daily_df['city'] = city_name
    daily_df.rename(columns={
        'time': 'date',
        'temperature_2m_max': 'max_temp',
        'temperature_2m_min': 'min_temp',
        'precipitation_sum': 'rainfall',
        'wind_speed_10m_max': 'wind_speed'
    }, inplace=True)
    
    # Process hourly humidity to get daily mean
    hourly_df = pd.DataFrame(data['hourly'])
    hourly_df['time'] = pd.to_datetime(hourly_df['time']).dt.date
    daily_humidity = hourly_df.groupby('time')['relative_humidity_2m'].mean().reset_index()
    daily_humidity.columns = ['date', 'humidity']
    daily_humidity['date'] = daily_humidity['date'].astype(str)
    
    # Merge
    merged_df = pd.merge(daily_df, daily_humidity, on='date')
    return merged_df

def fetch_aqi_data(city_name, lat, lon, start_date, end_date):
    """
    Fetches historical air quality data (AQI) from Open-Meteo Air Quality API.
    """
    print(f"Fetching AQI data for {city_name}...")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "us_aqi",
        "timezone": "Asia/Kolkata"
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching AQI for {city_name}: {response.text}")
        return None
    
    data = response.json()
    
    # Process hourly AQI to get daily mean
    hourly_df = pd.DataFrame(data['hourly'])
    hourly_df['time'] = pd.to_datetime(hourly_df['time']).dt.date
    daily_aqi = hourly_df.groupby('time')['us_aqi'].mean().reset_index()
    daily_aqi.columns = ['date', 'aqi']
    daily_aqi['date'] = daily_aqi['date'].astype(str)
    
    return daily_aqi

def main():
    # Define time period (last 2 years)
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    
    all_data = []
    
    for city, coords in CITIES.items():
        weather_df = fetch_weather_data(city, coords['lat'], coords['lon'], start_date, end_date)
        aqi_df = fetch_aqi_data(city, coords['lat'], coords['lon'], start_date, end_date)
        
        if weather_df is not None and aqi_df is not None:
            city_df = pd.merge(weather_df, aqi_df, on='date')
            all_data.append(city_df)
            print(f"Collected {len(city_df)} records for {city}")
        
        # Respect API rate limits
        time.sleep(1)
        
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Reorder columns for clarity
        cols = ['city', 'date', 'max_temp', 'min_temp', 'humidity', 'wind_speed', 'aqi', 'rainfall']
        final_df = final_df[cols]
        
        # Save to CSV
        output_file = "heatwave_data.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\nTotal Data Collected: {len(final_df)} rows")
        print(f"Data saved successfully to {output_file}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()
