from __future__ import annotations
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_dir = (base_dir / ".." / "dataset").resolve()
    models_dir = (base_dir / ".." / "models").resolve()
    results_dir = (base_dir / ".." / "results").resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    x_train_path = dataset_dir / "X_train.csv"
    y_train_path = dataset_dir / "y_train.csv"
    x_test_path = dataset_dir / "X_test.csv"
    y_test_path = dataset_dir / "y_test.csv"

    if not x_train_path.exists():
        raise FileNotFoundError(
            f"Missing {x_train_path}. Run `train_test_split.py` first."
        )

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    y_train_series = y_train.iloc[:, 0]
    y_test_series = y_test.iloc[:, 0]
    feature_columns = list(X_train.columns)

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(y_train_series.values)
    y_test_int = label_encoder.transform(y_test_series.values)
    class_names = list(label_encoder.classes_)

    # Build RF pipeline (imputation + model)
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", rf),
        ]
    )

    # Train on full training set
    rf_pipeline.fit(X_train, y_train_int)

    # Test evaluation
    y_test_pred = rf_pipeline.predict(X_test)

    test_acc = accuracy_score(y_test_int, y_test_pred)
    test_precision = precision_score(y_test_int, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test_int, y_test_pred, average='weighted')

    print("\n" + "="*60)
    print("Random Forest Evaluation Metrics")
    print("="*60)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision (Weighted): {test_precision:.4f}")
    print(f"F1-Score (Weighted):  {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_int, y_test_pred, target_names=class_names))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test_int, y_test_pred)
    print(cm)
    print("="*60)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Random Forest Confusion Matrix\nAccuracy: {test_acc:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(results_dir / "random_forest_evaluation.png", dpi=300)
    plt.close()
    print(f"Evaluation metrics saved to {results_dir / 'random_forest_evaluation.png'}")

    # Save trained model bundle for later predictions
    bundle = {
        "model_name": "random_forest",
        "pipeline": rf_pipeline,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "class_names": class_names,
    }

    model_path = models_dir / "best_model_random_forest.joblib"
    joblib.dump(bundle, model_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
    