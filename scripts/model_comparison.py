import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

def main():
    base_dir = Path(__file__).resolve().parent
    dataset_dir = (base_dir / ".." / "dataset").resolve()
    models_dir = (base_dir / ".." / "models").resolve()
    results_dir = (base_dir / ".." / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    x_test_path = dataset_dir / "X_test.csv"
    y_test_path = dataset_dir / "y_test.csv"

    if not x_test_path.exists():
        print("Test data not found. Run train_test_split.py first.")
        return

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    y_test_series = y_test.iloc[:, 0]

    # Load models
    rf_model_path = models_dir / "best_model_random_forest.joblib"
    xgb_model_path = models_dir / "best_model_xgboost.joblib"
    dt_model_path = models_dir / "best_model_decision_tree.joblib"

    if not rf_model_path.exists() or not xgb_model_path.exists() or not dt_model_path.exists():
        print("Trained models not found. Train all models first.")
        return

    rf_bundle = joblib.load(rf_model_path)
    xgb_bundle = joblib.load(xgb_model_path)
    dt_bundle = joblib.load(dt_model_path)

    rf_model = rf_bundle['pipeline']
    rf_encoder = rf_bundle['label_encoder']

    xgb_model = xgb_bundle['pipeline']
    xgb_encoder = xgb_bundle['label_encoder']
    
    dt_model = dt_bundle['pipeline']
    dt_encoder = dt_bundle['label_encoder']

    # Make predictions
    rf_pred_encoded = rf_model.predict(X_test)
    xgb_pred_encoded = xgb_model.predict(X_test)
    dt_pred_encoded = dt_model.predict(X_test)

    # Decode predictions
    rf_pred = rf_encoder.inverse_transform(rf_pred_encoded)
    xgb_pred = xgb_encoder.inverse_transform(xgb_pred_encoded)
    dt_pred = dt_encoder.inverse_transform(dt_pred_encoded)

    # Calculate accuracies
    rf_accuracy = accuracy_score(y_test_series, rf_pred)
    xgb_accuracy = accuracy_score(y_test_series, xgb_pred)
    dt_accuracy = accuracy_score(y_test_series, dt_pred)

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    models = ['Decision Tree', 'Random Forest', 'XGBoost']
    accuracies = [dt_accuracy, rf_accuracy, xgb_accuracy]

    bars = ax1.bar(models, accuracies, color=['peachpuff', 'skyblue', 'lightgreen'])
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.95, 1.0)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    # Performance difference
    diff_rf = xgb_accuracy - rf_accuracy
    diff_dt = xgb_accuracy - dt_accuracy
    diffs = [diff_rf, diff_dt]
    
    colors = ['green' if d >= 0 else 'red' for d in diffs]
    bars2 = ax2.bar(['XGB - RF', 'XGB - DT'], diffs, color=colors)
    ax2.set_title('Accuracy Advantage of XGBoost')
    ax2.set_ylabel('Difference')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add value label
    for bar, d in zip(bars2, diffs):
        ax2.text(bar.get_x() + bar.get_width()/2., d + (0.00002 if d >= 0 else -0.00002),
            f'{d:.6f}', ha='center', va='bottom' if d >= 0 else 'top',
            fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print results
    print("=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"XGBoost Accuracy:       {xgb_accuracy:.4f}")
    print(f"Difference (XGB - RF):  {diff_rf:.6f}")
    print(f"Difference (XGB - DT):  {diff_dt:.6f}")
    print(f"Comparison plot saved:  {results_dir / 'model_comparison.png'}")
    print("=" * 60)
    
    # Determine better model
    best_acc = max(dt_accuracy, rf_accuracy, xgb_accuracy)
    if best_acc == xgb_accuracy:
        print("Result: XGBoost performs the best!")
    elif best_acc == rf_accuracy:
        print("Result: Random Forest performs the best!")
    else:
        print("Result: Decision Tree performs the best!")

if __name__ == "__main__":
    main()