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

    if not rf_model_path.exists() or not xgb_model_path.exists():
        print("Trained models not found. Train the models first.")
        return

    rf_bundle = joblib.load(rf_model_path)
    xgb_bundle = joblib.load(xgb_model_path)

    rf_model = rf_bundle['pipeline']
    rf_encoder = rf_bundle['label_encoder']

    xgb_model = xgb_bundle['pipeline']
    xgb_encoder = xgb_bundle['label_encoder']

    # Make predictions
    rf_pred_encoded = rf_model.predict(X_test)
    xgb_pred_encoded = xgb_model.predict(X_test)

    # Decode predictions
    rf_pred = rf_encoder.inverse_transform(rf_pred_encoded)
    xgb_pred = xgb_encoder.inverse_transform(xgb_pred_encoded)

    # Calculate accuracies
    rf_accuracy = accuracy_score(y_test_series, rf_pred)
    xgb_accuracy = accuracy_score(y_test_series, xgb_pred)

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    models = ['Random Forest', 'XGBoost']
    accuracies = [rf_accuracy, xgb_accuracy]

    bars = ax1.bar(models, accuracies, color=['skyblue', 'lightgreen'])
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.95, 1.0)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    # Performance difference
    diff = xgb_accuracy - rf_accuracy
    colors = ['red' if diff < 0 else 'green']
    ax2.bar(['XGBoost - Random Forest'], [diff], color=colors)
    ax2.set_title('Accuracy Difference')
    ax2.set_ylabel('Difference')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add value label
    ax2.text(0, diff + (0.00002 if diff >= 0 else -0.00002),
            f'{diff:.6f}', ha='center', va='bottom' if diff >= 0 else 'top',
            fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print results
    print("=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"XGBoost Accuracy:       {xgb_accuracy:.4f}")
    print(f"Difference (XGB - RF):  {diff:.6f}")
    print(f"Comparison plot saved:  {results_dir / 'model_comparison.png'}")
    print("=" * 60)
    
    # Determine better model
    if abs(diff) < 0.001:
        print("Result: Models perform similarly!")
    elif diff > 0:
        print("Result: XGBoost performs better!")
    else:
        print("Result: Random Forest performs better!")

if __name__ == "__main__":
    main()