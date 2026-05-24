import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

def main():
    print("--- Starting STEP 5: SHAP Explainability ---")
    
    # Define paths
    model_path = "models/best_binary_model.joblib"
    test_data_path = "processed_data/X_test.csv"
    
    # Load best model package
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Please run step4_model_training.py first.")
    
    model_package = joblib.load(model_path)
    model = model_package['model']
    feature_names = model_package['features']
    print(f"Loaded best binary model: {model_package['model_name']}")
    
    # Load test dataset
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Missing {test_data_path}. Please run step3_feature_engineering.py first.")
        
    X_test = pd.read_csv(test_data_path)
    print(f"Loaded test dataset with shape: {X_test.shape}")
    
    # ---------------------------------------------
    # 1. Initialize SHAP Explainer
    # ---------------------------------------------
    # We use TreeExplainer for tree-based models like Random Forest
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # To keep computation fast and prevent timeouts, we sample the test set
    # 200 samples is a representative size for generating summary and dependence plots
    sample_size = min(200, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    print(f"Calculating SHAP values for {sample_size} samples...")
    
    # Calculate SHAP values
    shap_results = explainer(X_sample)
    
    # ---------------------------------------------
    # 2. Extract SHAP Values for the Positive Class (Heatwave)
    # ---------------------------------------------
    # For Scikit-learn RandomForestClassifier, the output shape can vary based on version.
    # In newer SHAP versions, explainer(X) returns an Explanation object.
    # If the output contains multi-class values (shape: samples, features, classes), 
    # we select class 1 (Heatwave) for our binary explanations.
    if len(shap_results.shape) == 3:  # shape is (samples, features, classes)
        shap_values_class = shap_results.values[:, :, 1]
        base_value_class = shap_results.base_values[:, 1]
    else:  # shape is (samples, features) or older SHAP API
        # Handle older SHAP returns where shap_values is a list of arrays
        shap_values_legacy = explainer.shap_values(X_sample)
        if isinstance(shap_values_legacy, list) and len(shap_values_legacy) == 2:
            shap_values_class = shap_values_legacy[1]
            base_value_class = explainer.expected_value[1]
        else:
            shap_values_class = shap_values_legacy
            base_value_class = explainer.expected_value
            
    # Create an Explanation object for plotting if we processed raw arrays
    if not isinstance(shap_results, shap.Explanation) or len(shap_results.shape) == 3:
        explanation = shap.Explanation(
            values=shap_values_class,
            base_values=base_value_class,
            data=X_sample.values,
            feature_names=feature_names
        )
    else:
        explanation = shap_results

    # ---------------------------------------------
    # 3. Create SHAP Summary Plot
    # ---------------------------------------------
    # The summary plot shows feature importances and their direction of impact on the target.
    print("Generating SHAP Summary Plot...")
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=12, show=False)
    plt.title("SHAP Feature Importance & Impact (Binary Heatwave Prediction)", fontsize=14, pad=20)
    plt.tight_layout()
    
    summary_plot_path = "plots/shap_summary_plot.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {summary_plot_path}")

    # ---------------------------------------------
    # 4. Create SHAP Dependence Plot
    # ---------------------------------------------
    # The dependence plot shows how the SHAP value of a feature varies with its feature value.
    # We find the most important feature by taking the mean absolute SHAP value.
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    best_feat_idx = np.argmax(mean_abs_shap)
    top_feature = feature_names[best_feat_idx]
    print(f"Top feature identified by SHAP: {top_feature}")
    
    print(f"Generating SHAP Dependence Plot for '{top_feature}'...")
    plt.figure(figsize=(8, 6))
    shap.plots.scatter(explanation[:, top_feature], show=False)
    plt.title(f"SHAP Dependence Plot: {top_feature}", fontsize=14, pad=15)
    plt.tight_layout()
    
    dependence_plot_path = "plots/shap_dependence_plot.png"
    plt.savefig(dependence_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {dependence_plot_path}")
    
    print("\n--- STEP 5: Completed successfully ---")

if __name__ == "__main__":
    main()
