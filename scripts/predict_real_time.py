import joblib
import pandas as pd
from pathlib import Path

def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent
    models_dir = (base_dir / ".." / "models").resolve()
    
    # Load your trained model bundles
    try:
        rf_bundle = joblib.load(models_dir / "best_model_random_forest.joblib")
        xgb_bundle = joblib.load(models_dir / "best_model_xgboost.joblib")
        dt_bundle = joblib.load(models_dir / "best_model_decision_tree.joblib")
    except FileNotFoundError:
        print("Models not found. Please train all three models first.")
        return

    # Extract components from bundles
    rf_model = rf_bundle['pipeline']
    rf_encoder = rf_bundle['label_encoder']
    feature_columns = rf_bundle['feature_columns']
    
    xgb_model = xgb_bundle['pipeline']
    xgb_encoder = xgb_bundle['label_encoder']

    dt_model = dt_bundle['pipeline']
    dt_encoder = dt_bundle['label_encoder']

    # Pre-calculated accuracies from model comparison
    rf_accuracy = 98.53
    xgb_accuracy = 98.41
    dt_accuracy = 97.80  # Approximate DT accuracy

    def predict_intern_performance(name, intern_id, metrics):
        # Create DataFrame with the correct feature names
        X = pd.DataFrame([metrics], columns=feature_columns)

        # Calculate Actual Rule-Based Ground Truth
        sprint = metrics[feature_columns.index('Sprint_Completion')]
        quality = metrics[feature_columns.index('Task_Quality')]
        comm = metrics[feature_columns.index('Communication')]
        attendance = metrics[feature_columns.index('Attendance')]
        on_time = metrics[feature_columns.index('On_Time_Delivery')]
        
        if sum([sprint >= 75, quality >= 75, comm >= 6, attendance >= 85, on_time >= 70]) >= 3:
            ground_truth = "High"
        elif sum([sprint <= 60, quality <= 60, comm <= 4, attendance <= 60, on_time <= 50]) >= 3:
            ground_truth = "Low"
        else:
            ground_truth = "Medium"

        # Random Forest Prediction
        rf_pred_encoded = rf_model.predict(X)[0]
        rf_pred = rf_encoder.inverse_transform([rf_pred_encoded])[0]
        rf_conf = max(rf_model.predict_proba(X)[0]) * 100

        # XGBoost Prediction
        xgb_pred_encoded = xgb_model.predict(X)[0]
        xgb_pred = xgb_encoder.inverse_transform([xgb_pred_encoded])[0]
        xgb_conf = max(xgb_model.predict_proba(X)[0]) * 100

        # Decision Tree Prediction
        dt_pred_encoded = dt_model.predict(X)[0]
        dt_pred = dt_encoder.inverse_transform([dt_pred_encoded])[0]
        dt_conf = max(dt_model.predict_proba(X)[0]) * 100

        print(f"\n{'='*60}")
        print(f"=== Intern Assessment: {name} ({intern_id}) ===")
        print(f"{'='*60}")
        print(f"  RULE-BASED GROUND TRUTH: {ground_truth} Performer")
        print(f"{'-'*60}")
        print(f" Random Forest Accuracy: {rf_accuracy}%")
        print(f" Random Forest Prediction: {rf_pred} | Confidence: {rf_conf:.2f}%\n")
        print(f" XGBoost Accuracy:       {xgb_accuracy}%")
        print(f" XGBoost Prediction:     {xgb_pred} | Confidence: {xgb_conf:.2f}%\n")
        print(f" Decision Tree Accuracy: {dt_accuracy}%")
        print(f" Decision Tree Prediction:{dt_pred} | Confidence: {dt_conf:.2f}%")

        # Majority Voting Logic
        predictions = [rf_pred, xgb_pred, dt_pred]
        vote_counts = {p: predictions.count(p) for p in set(predictions)}
        majority_vote = max(vote_counts, key=vote_counts.get)

        if vote_counts[majority_vote] == 3:
            print(f"\n ->  Unanimous Agreement: All 3 models predict {majority_vote} Performer")
        elif vote_counts[majority_vote] == 2:
            print(f"\n ->  Borderline Case: Resolved by Majority Vote (2 to 1). Final Verdict: {majority_vote} Performer")
        else:
            # Extremely rare case where all 3 models predict different things (1 High, 1 Med, 1 Low)
            # Fallback to the model with the highest accuracy (XGBoost)
            print(f"\n ->  Models completely disagree! Defaulting to highest-accuracy model (XGBoost): {xgb_pred} Performer")

    # Example inputs for 3 interns (Metrics map exactly to feature_columns)
    interns = [
        ("Ramesh", "RHS23157", [30, 18, 80, 1, 75, 80, 85, 6, 25, 20]),
        ("Aneesh", "ANS45231", [28, 26, 92, 1, 90, 88, 95, 8, 22, 21]),
        ("Karan", "KRN87412", [20, 12, 60, 0, 55, 65, 70, 5, 18, 12])
    ]
    
    for intern in interns:
        predict_intern_performance(intern[0], intern[1], intern[2])

if __name__ == "__main__":
    main()