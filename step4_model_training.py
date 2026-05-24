import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, model_name, task_name):
    """
    Computes performance metrics (accuracy, precision, recall, f1-score) for a model.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, f1-score (macro and weighted averages)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'Accuracy': accuracy,
        'Weighted F1': f1_w,
        'Macro F1': f1_m,
        'Weighted Precision': precision_w,
        'Weighted Recall': recall_w
    }
    
    print(f"\n--- {model_name} ({task_name}) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1_w:.4f} | Macro F1: {f1_m:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return metrics

def plot_and_save_confusion_matrix(y_true, y_pred, classes, filename, title):
    """
    Plots a confusion matrix heatmap and saves it as a PNG file.
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotations: count + percentage
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_norm[i, j] * 100
            annot[i, j] = f"{c}\n({p:.1f}%)"
            
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {filename}")

def main():
    print("--- Starting STEP 4: Model Training ---")
    
    # Paths to processed datasets
    data_dir = "processed_data"
    
    # Load feature sets
    X_train_raw = pd.read_csv(f"{data_dir}/X_train_raw.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    
    # Load binary targets
    X_train_bin = pd.read_csv(f"{data_dir}/X_train_bin.csv")
    y_train_bin = pd.read_csv(f"{data_dir}/y_train_bin.csv").values.ravel()
    y_test_bin = pd.read_csv(f"{data_dir}/y_test_bin.csv").values.ravel()
    
    # Load severity (multi-class) targets
    X_train_sev = pd.read_csv(f"{data_dir}/X_train_sev.csv")
    y_train_sev = pd.read_csv(f"{data_dir}/y_train_sev.csv").values.ravel()
    y_test_sev = pd.read_csv(f"{data_dir}/y_test_sev.csv").values.ravel()
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # ---------------------------------------------
    # 1. Feature Scaling (for Logistic Regression)
    # ---------------------------------------------
    # Logistic Regression requires scaled inputs. Tree-based models (Random Forest, XGBoost) do not.
    print("Fitting Scalers for Logistic Regression...")
    scaler_bin = StandardScaler()
    X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
    X_test_bin_scaled = scaler_bin.transform(X_test)
    
    scaler_sev = StandardScaler()
    X_train_sev_scaled = scaler_sev.fit_transform(X_train_sev)
    X_test_sev_scaled = scaler_sev.transform(X_test)

    # Dictionary to keep track of performance metrics
    results = []

    # =========================================================================
    # TASK 1: BINARY HEATWAVE PREDICTION (Yes/No)
    # =========================================================================
    print("\n" + "="*50)
    print("TASK 1: Binary Heatwave Prediction (Yes/No)")
    print("="*50)
    
    # Define Binary Models
    models_bin = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True), # True indicates scaled data
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), False)
    }
    
    best_bin_f1 = -1
    best_bin_model = None
    best_bin_name = ""
    best_bin_scaler = None
    
    for name, (model, use_scaled) in models_bin.items():
        # Select appropriate training/testing data
        X_tr = X_train_bin_scaled if use_scaled else X_train_bin
        X_te = X_test_bin_scaled if use_scaled else X_test
        
        # Train model
        model.fit(X_tr, y_train_bin)
        
        # Predict on test set
        y_pred = model.predict(X_te)
        
        # Evaluate
        metrics = evaluate_model(y_test_bin, y_pred, name, 'Binary')
        metrics['Model'] = name
        metrics['Task'] = 'Binary'
        results.append(metrics)
        
        # Plot confusion matrix
        plot_and_save_confusion_matrix(
            y_test_bin, y_pred, 
            classes=['No Heatwave', 'Heatwave'], 
            filename=f"plots/cm_binary_{name.lower().replace(' ', '_')}.png",
            title=f"Confusion Matrix: {name} (Binary Heatwave)"
        )
        
        # Keep track of best model by Macro F1 (since we have imbalanced classes, Macro F1 is key)
        if metrics['Macro F1'] > best_bin_f1:
            best_bin_f1 = metrics['Macro F1']
            best_bin_model = model
            best_bin_name = name
            best_bin_scaler = scaler_bin if use_scaled else None

    # =========================================================================
    # TASK 2: SEVERITY LEVEL PREDICTION (Safe / Warning / Danger / Extreme)
    # =========================================================================
    print("\n" + "="*50)
    print("TASK 2: Severity Level Prediction (Safe / Warning / Danger)")
    print("="*50)
    
    # Define Severity Models
    # Note: open-meteo dataset severity mapping: 0=Safe, 1=Warning, 2=Danger
    models_sev = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), False)
    }
    
    best_sev_f1 = -1
    best_sev_model = None
    best_sev_name = ""
    best_sev_scaler = None
    
    # Class names present in the test set. Safely get classes that exist in y_test_sev.
    # Severity codes mapping: 0: Safe, 1: Warning, 2: Danger
    classes_sev = ['Safe', 'Warning', 'Danger']
    
    for name, (model, use_scaled) in models_sev.items():
        X_tr = X_train_sev_scaled if use_scaled else X_train_sev
        X_te = X_test_sev_scaled if use_scaled else X_test
        
        # Train model
        model.fit(X_tr, y_train_sev)
        
        # Predict on test set
        y_pred = model.predict(X_te)
        
        # Evaluate
        metrics = evaluate_model(y_test_sev, y_pred, name, 'Severity')
        metrics['Model'] = name
        metrics['Task'] = 'Severity'
        results.append(metrics)
        
        # Plot confusion matrix
        plot_and_save_confusion_matrix(
            y_test_sev, y_pred, 
            classes=classes_sev, 
            filename=f"plots/cm_severity_{name.lower().replace(' ', '_')}.png",
            title=f"Confusion Matrix: {name} (Severity Level)"
        )
        
        # Keep track of best model by Macro F1
        if metrics['Macro F1'] > best_sev_f1:
            best_sev_f1 = metrics['Macro F1']
            best_sev_model = model
            best_sev_name = name
            best_sev_scaler = scaler_sev if use_scaled else None

    # ---------------------------------------------
    # 3. Model Comparison & Summary Table
    # ---------------------------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df[['Task', 'Model', 'Accuracy', 'Weighted F1', 'Macro F1', 'Weighted Precision', 'Weighted Recall']]
    results_df.to_csv("plots/model_comparison_metrics.csv", index=False)
    
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(results_df.to_string(index=False))

    # ---------------------------------------------
    # 4. Save Best Models and Scalers
    # ---------------------------------------------
    print(f"\nSaving Best Binary Model: {best_bin_name} (Macro F1: {best_bin_f1:.4f})")
    binary_model_package = {
        'model': best_bin_model,
        'scaler': best_bin_scaler,
        'features': list(X_train_raw.columns),
        'model_name': best_bin_name
    }
    joblib.dump(binary_model_package, "models/best_binary_model.joblib")
    
    print(f"Saving Best Severity Model: {best_sev_name} (Macro F1: {best_sev_f1:.4f})")
    severity_model_package = {
        'model': best_sev_model,
        'scaler': best_sev_scaler,
        'features': list(X_train_raw.columns),
        'model_name': best_sev_name
    }
    joblib.dump(severity_model_package, "models/best_multiclass_model.joblib")
    
    print("\n--- STEP 4: Completed successfully ---")

if __name__ == "__main__":
    main()
