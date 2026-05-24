import os
import requests
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# Set page configuration to wide layout
st.set_page_config(
    page_title="Heatwave Early Warning System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design & Glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Overall layout styling */
    .stApp {
        background: radial-gradient(circle at 20% 30%, #1e1e2f 0%, #0f0f15 100%);
        color: #f5f6fa;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Area */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF8C00 0%, #FF0080 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #b0b5c0;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    
    .glass-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Severity Card Styling */
    .severity-safe {
        border-left: 6px solid #2ecc71;
        background: rgba(46, 204, 113, 0.07);
    }
    .severity-warning {
        border-left: 6px solid #f1c40f;
        background: rgba(241, 196, 15, 0.07);
    }
    .severity-danger {
        border-left: 6px solid #e74c3c;
        background: rgba(231, 76, 60, 0.07);
    }
    .severity-extreme {
        border-left: 6px solid #8e44ad;
        background: rgba(142, 68, 173, 0.07);
    }
    
    .status-text {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    .status-lbl-safe { color: #2ecc71; }
    .status-lbl-warning { color: #f1c40f; }
    .status-lbl-danger { color: #e74c3c; }
    .status-lbl-extreme { color: #8e44ad; }
    
    /* Styled HTML table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
        margin: 1rem 0;
        background: rgba(255, 255, 255, 0.01);
        border-radius: 8px;
        overflow: hidden;
    }
    .styled-table th {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ffffff;
        text-align: left;
        padding: 12px 15px;
        font-weight: 600;
    }
    .styled-table td {
        padding: 12px 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .styled-table tr:hover {
        background-color: rgba(255, 255, 255, 0.03);
    }
    
    /* Severity Badges */
    .badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-safe { background-color: rgba(46, 204, 113, 0.2); color: #2ecc71; border: 1px solid #2ecc71; }
    .badge-warning { background-color: rgba(241, 196, 15, 0.2); color: #f1c40f; border: 1px solid #f1c40f; }
    .badge-danger { background-color: rgba(231, 76, 60, 0.2); color: #e74c3c; border: 1px solid #e74c3c; }
    .badge-extreme { background-color: rgba(142, 68, 173, 0.2); color: #8e44ad; border: 1px solid #8e44ad; }

    /* Glassmorphism Plotly & Tabs Styling */
    div[data-testid="stPlotlyChart"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    
    div[data-testid="stTabs"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

</style>
""", unsafe_allow_html=True)

# Coordinates dictionary for all Indian States & Union Territories
STATE_CAPITALS = {
    "Andaman and Nicobar Islands (UT)": {"city": "Port Blair", "lat": 11.6234, "lon": 92.7265},
    "Andhra Pradesh": {"city": "Amaravati", "lat": 16.5745, "lon": 80.3732},
    "Arunachal Pradesh": {"city": "Itanagar", "lat": 27.0844, "lon": 93.6053},
    "Assam": {"city": "Dispur", "lat": 26.1433, "lon": 91.7898},
    "Bihar": {"city": "Patna", "lat": 25.5941, "lon": 85.1376},
    "Chandigarh (UT)": {"city": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    "Chhattisgarh": {"city": "Raipur", "lat": 21.2514, "lon": 81.6296},
    "Dadra and Nagar Haveli and Daman and Diu (UT)": {"city": "Daman", "lat": 20.3974, "lon": 72.8328},
    "Delhi (UT)": {"city": "New Delhi", "lat": 28.6139, "lon": 77.2090},
    "Goa": {"city": "Panaji", "lat": 15.4909, "lon": 73.8278},
    "Gujarat": {"city": "Gandhinagar", "lat": 23.2156, "lon": 72.6369},
    "Haryana": {"city": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    "Himachal Pradesh": {"city": "Shimla", "lat": 31.1048, "lon": 77.1734},
    "Jammu and Kashmir (UT)": {"city": "Srinagar", "lat": 34.0837, "lon": 74.7973},
    "Jharkhand": {"city": "Ranchi", "lat": 23.3441, "lon": 85.3096},
    "Karnataka": {"city": "Bengaluru", "lat": 12.9716, "lon": 77.5946},
    "Kerala": {"city": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366},
    "Ladakh (UT)": {"city": "Leh", "lat": 34.1526, "lon": 77.5771},
    "Lakshadweep (UT)": {"city": "Kavaratti", "lat": 10.5667, "lon": 72.6417},
    "Madhya Pradesh": {"city": "Bhopal", "lat": 23.2599, "lon": 77.4126},
    "Maharashtra": {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    "Manipur": {"city": "Imphal", "lat": 24.8170, "lon": 93.9368},
    "Meghalaya": {"city": "Shillong", "lat": 25.5788, "lon": 91.8831},
    "Mizoram": {"city": "Aizawl", "lat": 23.7307, "lon": 92.7173},
    "Nagaland": {"city": "Kohima", "lat": 25.6751, "lon": 94.1086},
    "Odisha": {"city": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245},
    "Puducherry (UT)": {"city": "Puducherry", "lat": 11.9416, "lon": 79.8083},
    "Punjab": {"city": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    "Rajasthan": {"city": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    "Sikkim": {"city": "Gangtok", "lat": 27.3314, "lon": 88.6138},
    "Tamil Nadu": {"city": "Chennai", "lat": 13.0827, "lon": 80.2707},
    "Telangana": {"city": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    "Tripura": {"city": "Agartala", "lat": 23.8315, "lon": 91.2868},
    "Uttar Pradesh": {"city": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    "Uttarakhand": {"city": "Dehradun", "lat": 30.3165, "lon": 78.0322},
    "West Bengal": {"city": "Kolkata", "lat": 22.5726, "lon": 88.3639}
}

def calculate_heat_index(temp_c, rh):
    """
    Calculates Heat Index in Celsius using the standard US NWS Rothfusz regression.
    """
    t = (temp_c * 9.0/5.0) + 32.0
    hi = 0.5 * (t + 61.0 + ((t - 68.0) * 1.2) + (rh * 0.094))
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
        if rh < 13 and 80 <= t <= 112:
            hi += -((13 - rh) / 4) * ((17 - abs(t - 95)) / 17)**0.5
        elif rh > 85 and 80 <= t <= 87:
            hi += ((rh - 85) / 10) * ((87 - t) / 5)
    return (hi - 32) * 5.0/9.0

@st.cache_data(ttl=3600)  # Cache forecast data for 1 hour
def fetch_7day_forecast(city_name, lat, lon):
    """
    Fetches 14 past days + 7 forecast days of weather and AQI data dynamically.
    """
    # 1. Fetch weather variables
    url_w = "https://api.open-meteo.com/v1/forecast"
    params_w = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
        "hourly": "relative_humidity_2m",
        "timezone": "Asia/Kolkata",
        "past_days": 14
    }
    try:
        response_w = requests.get(url_w, params=params_w, timeout=10)
        if response_w.status_code != 200:
            return None
        data_w = response_w.json()
    except Exception as e:
        return None
    
    daily_df = pd.DataFrame(data_w['daily'])
    daily_df.rename(columns={
        'time': 'date',
        'temperature_2m_max': 'max_temp',
        'temperature_2m_min': 'min_temp',
        'precipitation_sum': 'rainfall',
        'wind_speed_10m_max': 'wind_speed'
    }, inplace=True)
    
    # Calculate daily mean humidity from hourly forecast
    hourly_df = pd.DataFrame(data_w['hourly'])
    hourly_df['time'] = pd.to_datetime(hourly_df['time']).dt.date
    daily_humidity = hourly_df.groupby('time')['relative_humidity_2m'].mean().reset_index()
    daily_humidity.columns = ['date', 'humidity']
    daily_humidity['date'] = daily_humidity['date'].astype(str)
    daily_df['date'] = daily_df['date'].astype(str)
    
    weather_df = pd.merge(daily_df, daily_humidity, on='date')
    
    # 2. Fetch air quality variables (AQI)
    url_aqi = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params_aqi = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "us_aqi",
        "timezone": "Asia/Kolkata",
        "past_days": 14
    }
    try:
        response_aqi = requests.get(url_aqi, params=params_aqi, timeout=10)
        if response_aqi.status_code != 200:
            # Fallback AQI if request fails
            daily_aqi = pd.DataFrame({
                "date": weather_df['date'],
                "aqi": 80.0
            })
        else:
            data_aqi = response_aqi.json()
            hourly_aqi = pd.DataFrame(data_aqi['hourly'])
            hourly_aqi['time'] = pd.to_datetime(hourly_aqi['time']).dt.date
            daily_aqi = hourly_aqi.groupby('time')['us_aqi'].mean().reset_index()
            daily_aqi.columns = ['date', 'aqi']
            daily_aqi['date'] = daily_aqi['date'].astype(str)
    except Exception as e:
        daily_aqi = pd.DataFrame({
            "date": weather_df['date'],
            "aqi": 80.0
        })
    
    # Merge weather and AQI forecast
    forecast_df = pd.merge(weather_df, daily_aqi, on='date', how='left')
    # Fill any missing AQI values
    forecast_df['aqi'] = forecast_df['aqi'].interpolate(method='linear').ffill().bfill()
    forecast_df['city'] = city_name
    return forecast_df

def prepare_forecast_features(forecast_df, city_name, features_list):
    """
    Computes lags and rolling features on the 21-day timeline directly.
    """
    df = forecast_df.copy()
    
    # Calculate Heat Index
    df['heat_index'] = df.apply(lambda r: calculate_heat_index(r['max_temp'], r['humidity']), axis=1)
    
    # Compute Lags
    lag_cols = ['max_temp', 'humidity', 'heat_index', 'aqi']
    for lag in [1, 2, 7]:
        for col in lag_cols:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
    # Compute Rolling Averages
    roll_cols = ['max_temp', 'humidity', 'heat_index']
    for window in [3, 7]:
        for col in roll_cols:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
            
    # Select only the forecast rows (which are the last 7 rows of our 21-day timeline)
    forecast_with_features = df.tail(7).copy()
    
    # Add dummy city columns for one-hot representation.
    # Set matching dummy to 1 if it corresponds to one of the trained cities.
    trained_cities = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Jaipur']
    for c in trained_cities:
        is_match = 0
        if c.lower() in city_name.lower():
            is_match = 1
        forecast_with_features[f'city_{c}'] = is_match
        
    # Reorder features to match exact training format
    X_forecast = forecast_with_features[features_list]
    return X_forecast, forecast_with_features


# ---------------------------------------------
# App Layout Assembly
# ---------------------------------------------
st.markdown("<div class='main-title'>☀️ Heatwave Early Warning System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predictive Climate ML for India's Major Cities</div>", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)
    binary_pkg = joblib.load(os.path.join(base_dir, "models", "best_binary_model.joblib"))
    severity_pkg = joblib.load(os.path.join(base_dir, "models", "best_multiclass_model.joblib"))
    return binary_pkg, severity_pkg

try:
    binary_model_package, severity_model_package = load_models()
    features_list = binary_model_package['features']
except Exception as e:
    st.error("Failed to load models. Please ensure Step 4 completed successfully.")
    st.stop()

# Sidebar Control Panel
st.sidebar.markdown("### 🎛️ Control Panel")
selected_state = st.sidebar.selectbox("Select State / UT", list(STATE_CAPITALS.keys()))
city_info = STATE_CAPITALS[selected_state]
selected_city = city_info['city']
lat = city_info['lat']
lon = city_info['lon']

st.sidebar.markdown(f"**State Capital:** {selected_city}")
st.sidebar.markdown(f"**Coordinates:** {lat}°N, {lon}°E")

# Run Forecast Calculation
forecast_raw = fetch_7day_forecast(selected_city, lat, lon)

if forecast_raw is not None:
    X_forecast, forecast_with_features = prepare_forecast_features(forecast_raw, selected_city, features_list)
    
    # Predictions
    # Predict binary heatwave (next 7 days) and multiclass severity
    bin_preds = binary_model_package['model'].predict(X_forecast)
    sev_preds = severity_model_package['model'].predict(X_forecast)
    
    # Since model is trained to predict if a heatwave will occur in the *next 7 days* from date t,
    # the prediction at index 0 (today) represents the 7-day outlook!
    heatwave_outlook = bin_preds[0]
    severity_outlook = sev_preds[0]
    
    # Calculate daily parameters for display
    # Severity Code mapper: 0=Safe, 1=Warning, 2=Danger, 3=Extreme
    severity_labels = {0: "Safe", 1: "Warning", 2: "Danger", 3: "Extreme"}
    severity_colors = {0: "safe", 1: "warning", 2: "danger", 3: "extreme"}
    
    # ----------------- Dashboard Main Layout -----------------
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='glass-header'>🔮 7-Day Heatwave Outlook</div>", unsafe_allow_html=True)
        
        # Color coding classes
        sev_color_class = severity_colors[severity_outlook]
        sev_label = severity_labels[severity_outlook]
        
        st.markdown(f"""
<div class='glass-card severity-{sev_color_class}'>
    <div>Outlook Severity Level</div>
    <div class='status-text status-lbl-{sev_color_class}'>{sev_label.upper()}</div>
</div>
""", unsafe_allow_html=True)
        
        outlook_text = "YES" if heatwave_outlook == 1 else "NO"
        outlook_color = "status-lbl-danger" if heatwave_outlook == 1 else "status-lbl-safe"
        st.markdown(f"""
<div class='glass-card'>
    <div>Heatwave Expected (Next 7 Days)?</div>
    <div class='status-text {outlook_color}'>{outlook_text}</div>
</div>
""", unsafe_allow_html=True)
        
        # Current Weather Parameters
        today_data = forecast_with_features.iloc[0]
        st.markdown(f"""
<div class='glass-card'>
    <div class='glass-header'>🌡️ Current Conditions</div>
    <div style='margin-bottom: 0.5rem;'><strong>Max Temperature:</strong> {today_data['max_temp']:.1f}°C</div>
    <div style='margin-bottom: 0.5rem;'><strong>Relative Humidity:</strong> {today_data['humidity']:.1f}%</div>
    <div style='margin-bottom: 0.5rem;'><strong>Rothfusz Heat Index:</strong> {today_data['heat_index']:.1f}°C</div>
    <div style='margin-bottom: 0.5rem;'><strong>Wind Speed:</strong> {today_data['wind_speed']:.1f} km/h</div>
    <div style='margin-bottom: 0.5rem;'><strong>US AQI:</strong> {today_data['aqi']:.1f}</div>
    <div style='margin-bottom: 0.5rem;'><strong>Rainfall:</strong> {today_data['rainfall']:.1f} mm</div>
</div>
""", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-header'>📈 7-Day Temperature & Heat Index Forecast</div>", unsafe_allow_html=True)
        
        # Create Plotly Chart
        fig = go.Figure()
        
        # Max Temperature line
        fig.add_trace(go.Scatter(
            x=forecast_with_features['date'],
            y=forecast_with_features['max_temp'],
            mode='lines+markers',
            name='Max Temperature (°C)',
            line=dict(color='#FF4B2B', width=3),
            marker=dict(size=8, symbol='circle')
        ))
        
        # Heat Index line
        fig.add_trace(go.Scatter(
            x=forecast_with_features['date'],
            y=forecast_with_features['heat_index'],
            mode='lines+markers',
            name='Heat Index (°C)',
            line=dict(color='#FF8C00', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f5f6fa', family='Outfit'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Date"),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Temperature / Heat Index (°C)"),
            margin=dict(l=40, r=40, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # 7-Day Forecast Data Table (HTML table)
    # Calculate daily heatwave and severity labels using the current day's feature set
    # (Note: daily warning represents predicting that specific day's status using model predictions)
    daily_bin_preds = bin_preds
    daily_sev_preds = sev_preds
    
    html_rows = ""
    for i, (idx, row) in enumerate(forecast_with_features.iterrows()):
        # Get predictions for that day
        daily_bin = "Yes" if daily_bin_preds[i] == 1 else "No"
        daily_sev = severity_labels[daily_sev_preds[i]]
        color_class = severity_colors[daily_sev_preds[i]]
        
        dt_obj = datetime.strptime(row['date'], '%Y-%m-%d')
        formatted_date = dt_obj.strftime('%A, %b %d')
        
        html_rows += f"<tr><td><strong>{formatted_date}</strong></td><td>{row['max_temp']:.1f}°C</td><td>{row['min_temp']:.1f}°C</td><td>{row['humidity']:.1f}%</td><td>{row['heat_index']:.1f}°C</td><td>{row['aqi']:.0f}</td><td>{row['wind_speed']:.1f} km/h</td><td>{row['rainfall']:.1f} mm</td><td><span class='badge badge-{color_class}'>{daily_sev}</span></td></tr>"
        
    table_html = f"""
<div class='glass-card'>
    <div class='glass-header'>📋 Detailed 7-Day Weather & Warning Details</div>
    <table class='styled-table'>
        <thead>
            <tr>
                <th>Date</th>
                <th>Max Temp</th>
                <th>Min Temp</th>
                <th>Humidity</th>
                <th>Heat Index</th>
                <th>US AQI</th>
                <th>Wind Speed</th>
                <th>Rainfall</th>
                <th>Predicted Warning</th>
            </tr>
        </thead>
        <tbody>
            {html_rows}
        </tbody>
    </table>
</div>
"""
    st.markdown(table_html, unsafe_allow_html=True)

    # ----------------- SHAP Explainability Section -----------------
    st.markdown("<div class='glass-header' style='margin-top: 2rem;'>🧠 Model Interpretability & Explainability (SHAP)</div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Feature Importance", "📈 Dependence Plot"])
    
    base_dir = os.path.dirname(__file__)
    with tab1:
        st.markdown("##### SHAP Summary Plot")
        st.markdown("This beeswarm plot shows the global feature importance of our Random Forest model and the direction of impact. Higher values of features (red) or lower values (blue) shift predictions toward or away from a heatwave.")
        shap_summary_path = os.path.join(base_dir, "plots", "shap_summary_plot.png")
        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path, use_container_width=True)
        else:
            st.warning("SHAP summary plot image not found. Ensure step5_explainability.py ran successfully.")
            
    with tab2:
        st.markdown("##### Feature Dependence Analysis")
        st.markdown("This plot shows the relationship between our most important feature (`max_temp`) and its SHAP values. A SHAP value above 0 indicates that the feature value increases the likelihood of a heatwave occurring.")
        shap_dependence_path = os.path.join(base_dir, "plots", "shap_dependence_plot.png")
        if os.path.exists(shap_dependence_path):
            st.image(shap_dependence_path, use_container_width=True)
        else:
            st.warning("SHAP dependence plot image not found. Ensure step5_explainability.py ran successfully.")

else:
    st.error("Failed to fetch weather forecast data from the Open-Meteo API. Please verify your internet connection.")
