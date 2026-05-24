# ☀️ Heatwave Early Warning System (Indian Cities)

A complete Machine Learning-based Heatwave Early Warning System that predicts whether a heatwave will occur in the next 7 days and classifies its severity level for five major Indian cities: **Delhi, Mumbai, Chennai, Kolkata, and Jaipur**.

The system utilizes historical meteorological data and real-time weather forecasts from the **Open-Meteo API** and applies a custom-trained Machine Learning classifier to output early warning alerts.

---

## 🏛️ System Architecture

The pipeline consists of the following modular steps:
1. **Data Collection (`step1_data_collection.py`):** Fetches 2 years of daily weather and air quality (AQI) historical data via Open-Meteo API.
2. **Data Cleaning & EDA (`step2_eda.py`):** Imputes missing values, caps outliers, computes the US National Weather Service **Rothfusz Heat Index**, and saves visualizations (temperature trends, correlation matrices, and severity distributions).
3. **Feature Engineering (`step3_feature_engineering.py`):** Creates temporal lag features (1, 2, and 7 days), rolling averages (3-day and 7-day), applies a time-based train-test split to avoid leakage, and balances classes using **SMOTE**.
4. **Model Training & Selection (`step4_model_training.py`):** Trains Logistic Regression, Random Forest, and XGBoost models for both binary warning and multi-class severity prediction.
5. **SHAP Explainability (`step5_explainability.py`):** Interprets predictions of the best-performing model using SHAP values (beeswarm summary and dependence plots).
6. **Streamlit Dashboard (`app.py`):** Interactive, high-end Glassmorphism dashboard fetching live 7-day weather and AQI forecast APIs, and rendering predictions, interactive charts, detailed metrics, and SHAP plots.

---

## 📊 Model Performance Comparison

The models were trained on data before **2025-12-01** and tested on data from **2025-12-01 onwards** (time-based split). The following results were achieved:

| Task | Model | Accuracy | Weighted F1 | Macro F1 | Weighted Precision | Weighted Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Binary (Heatwave)** | Logistic Regression | 91.55% | 0.9277 | 0.7588 | 0.9509 | 91.55% |
| **Binary (Heatwave)** | **Random Forest (Best)** | **95.24%** | **0.9552** | **0.8284** | **0.9596** | **95.24%** |
| **Binary (Heatwave)** | XGBoost | 94.05% | 0.9466 | 0.8077 | 0.9580 | 94.05% |
| **Severity Level** | Logistic Regression | 90.48% | 0.9201 | 0.6310 | 0.9448 | 90.48% |
| **Severity Level** | **Random Forest (Best)** | **94.40%** | **0.9478** | **0.6883** | **0.9539** | **94.40%** |
| **Severity Level** | XGBoost | 93.81% | 0.9436 | 0.6570 | 0.9520 | 93.81% |

*Note: Since the dataset is highly imbalanced (heatwaves are seasonal), **Macro F1-score** was used to select the best models. Random Forest outperformed other models on both tasks.*

---

## 🛠️ Setup & Running Instructions

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. In your project directory, run:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipelines (Optional, already computed)
You can run the pipeline stages individually to recreate the models and plots:
```bash
# Clean data and compute Heat Index + plots
python step2_eda.py

# Generate lags, rolling averages, train-test split, and SMOTE
python step3_feature_engineering.py

# Train models and save best classifiers
python step4_model_training.py

# Generate SHAP explainability plots
python step5_explainability.py
```

### 3. Run the Streamlit Dashboard
Launch the interactive web application:
```bash
streamlit run app.py
```

---

## 🎨 Severity Classification Rules
The severity categories are mapped dynamically based on maximum daily temperature:
- 🟢 **Safe:** below 40°C
- 🟡 **Warning:** 40°C – 43°C
- 🔴 **Danger:** 43°C – 46°C
- 🟣 **Extreme:** above 46°C
