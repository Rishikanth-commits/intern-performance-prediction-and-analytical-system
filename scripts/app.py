from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
from pathlib import Path
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Load trained models
base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent

# Models are saved in project-root/models by training scripts.
models_dir = project_root / "models"
if not models_dir.exists():
    models_dir = base_dir / "models"

try:
    rf_bundle = joblib.load(models_dir / "best_model_random_forest.joblib")
    xgb_bundle = joblib.load(models_dir / "best_model_xgboost.joblib")
    dt_bundle = joblib.load(models_dir / "best_model_decision_tree.joblib")

    rf_model = rf_bundle['pipeline']
    rf_encoder = rf_bundle['label_encoder']
    feature_columns = rf_bundle.get('feature_columns', [])

    xgb_model = xgb_bundle['pipeline']
    xgb_encoder = xgb_bundle['label_encoder']
    
    dt_model = dt_bundle['pipeline']
    dt_encoder = dt_bundle['label_encoder']

    print(" Models loaded successfully!")
except Exception as e:
    print(f" Error loading models: {e}")
    rf_model = xgb_model = dt_model = None
    feature_columns = []

# Feature order (must match training)
FEATURE_COLUMNS = feature_columns

# Load HTML frontend
def load_frontend_html():
    frontend_path = project_root / "frontend.html"
    if frontend_path.exists():
        try:
            with open(frontend_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return None
    return None

FRONTEND_HTML = load_frontend_html()

@app.route("/", methods=["GET"])
def serve_frontend():
    if FRONTEND_HTML:
        return FRONTEND_HTML
    return jsonify({
        "message": "Intern Performance Prediction API (Flask)",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "model-info": "/model-info (GET)"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not rf_model or not xgb_model:
        return jsonify({"error": "Models not loaded"}), 500   
    try:
        data = request.get_json()        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        required_fields = [
            'meetings_scheduled', 'meetings_attended', 'attendance', 'punctuality',
            'sprint_completion', 'task_quality', 'on_time_delivery', 'communication',
            'tasks_assigned', 'tasks_completed'
        ]        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        
        # Create feature vector
        features = {
            'Meetings_Scheduled': float(data.get('meetings_scheduled')),
            'Meetings_Attended': float(data.get('meetings_attended')),
            'Attendance': float(data.get('attendance')),
            'Punctuality': float(data.get('punctuality')),
            'Sprint_Completion': float(data.get('sprint_completion')),
            'Task_Quality': float(data.get('task_quality')),
            'On_Time_Delivery': float(data.get('on_time_delivery')),
            'Communication': float(data.get('communication')) / 10 if float(data.get('communication')) > 10 else float(data.get('communication')),  # Scale if >10
            'Tasks_Assigned': float(data.get('tasks_assigned')),
            'Tasks_Completed': float(data.get('tasks_completed')),
        }        
        X_raw = pd.DataFrame([features])
        X = X_raw[FEATURE_COLUMNS] if len(FEATURE_COLUMNS) > 0 else X_raw
        
        # Get predictions from both models
        rf_pred_encoded = rf_model.predict(X)[0]
        xgb_pred_encoded = xgb_model.predict(X)[0]
        
        # Get probabilities for confidence levels
        rf_proba = rf_model.predict_proba(X)[0]
        xgb_proba = xgb_model.predict_proba(X)[0]
        
        rf_confidence = round(float(max(rf_proba) * 100), 2)
        xgb_confidence = round(float(max(xgb_proba) * 100), 2)
        
        # Decode predictions
        rf_prediction = rf_encoder.inverse_transform([rf_pred_encoded])[0]
        xgb_prediction = xgb_encoder.inverse_transform([xgb_pred_encoded])[0]
        
        # Debug print
        print(f"Input features: {features}")
        print(f"RF encoded pred: {rf_pred_encoded}, decoded: {rf_prediction}")
        print(f"RF confidence: {rf_confidence}%")
        print(f"XGBoost encoded pred: {xgb_pred_encoded}, decoded: {xgb_prediction}")
        print(f"XGBoost confidence: {xgb_confidence}%")
        
        # Determine winner (which model predicts better)
        winner = "Both Agree" if rf_prediction == xgb_prediction else "Models Differ"        
        return jsonify({
            "random_forest": rf_prediction,
            "rf_confidence": rf_confidence,
            "xgboost": xgb_prediction,
            "xgb_confidence": xgb_confidence,
            "winner": winner,
            "engineered_features_used": X.iloc[0].to_dict()
        })   
    except ValueError as e:
        return jsonify({"error": f"Invalid data type: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/predict-by-id", methods=["GET"])
def predict_by_id():
    if not rf_model or not xgb_model or not dt_model:
        return jsonify({"error": "Models not loaded"}), 500
        
    intern_id = request.args.get("id")
    if not intern_id:
        return jsonify({"error": "Please provide an intern ID (e.g., ?id=INT001)"}), 400
        
    # Read the intern database
    db_path = project_root / "dataset" / "intern_database.csv"
    if not db_path.exists():
        return jsonify({"error": "Intern database not found. Please run generate_intern_db.py first."}), 404
        
    df = pd.read_csv(db_path)
    intern_data = df[df["Intern_ID"] == intern_id]
    
    if intern_data.empty:
        return jsonify({"error": f"Intern with ID {intern_id} not found."}), 404
        
    # Extract data for the model
    row = intern_data.iloc[0]
    features = {
        'Meetings_Scheduled': float(row['Meetings_Scheduled']),
        'Meetings_Attended': float(row['Meetings_Attended']),
        'Attendance': float(row['Attendance']),
        'Punctuality': float(row['Punctuality']),
        'Sprint_Completion': float(row['Sprint_Completion']),
        'Task_Quality': float(row['Task_Quality']),
        'On_Time_Delivery': float(row['On_Time_Delivery']),
        'Communication': float(row['Communication']),
        'Tasks_Assigned': float(row['Tasks_Assigned']),
        'Tasks_Completed': float(row['Tasks_Completed']),
    }
    
    X_raw = pd.DataFrame([features])
    X = X_raw[FEATURE_COLUMNS] if len(FEATURE_COLUMNS) > 0 else X_raw
    
    rf_prediction = rf_encoder.inverse_transform([rf_model.predict(X)[0]])[0]
    xgb_prediction = xgb_encoder.inverse_transform([xgb_model.predict(X)[0]])[0]
    dt_prediction = dt_encoder.inverse_transform([dt_model.predict(X)[0]])[0]
    
    # Weighted Soft (Confidence) Voting Logic
    rf_probs = rf_model.predict_proba(X)[0]
    xgb_probs = xgb_model.predict_proba(X)[0]
    dt_probs = dt_model.predict_proba(X)[0]
    
    # We give XGBoost (our most accurate model) double the voting weight
    avg_probs = (rf_probs * 1.0 + dt_probs * 1.0 + xgb_probs * 2.0) / 4.0
    best_class_index = avg_probs.argmax()
    final_verdict = rf_encoder.inverse_transform([best_class_index])[0]
    
    agreement = "Unanimous" if (rf_prediction == xgb_prediction == dt_prediction) else "Soft Vote"
    
    return jsonify({
        "intern_id": intern_id,
        "name": row["Name"],
        "random_forest": rf_prediction,
        "xgboost": xgb_prediction,
        "decision_tree": dt_prediction,
        "final_verdict": final_verdict,
        "agreement": agreement,
        "fetched_metrics": features
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "models_loaded": rf_model is not None and xgb_model is not None
    })

@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "random_forest_accuracy": "0.9853",
        "xgboost_accuracy": "0.9841",
        "classes": ["High", "Medium", "Low"],
        "features": FEATURE_COLUMNS
    })

if __name__ == "__main__":
    # Run Flask development server
    print("\n" + "="*60)
    print(" Flask Intern Performance Prediction API")
    print("="*60)
    print(" Running on: http://127.0.0.1:8000")
    print(" API Docs: http://127.0.0.1:8000/")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=8000, debug=False)