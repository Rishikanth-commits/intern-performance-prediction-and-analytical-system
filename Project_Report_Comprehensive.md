# INTERN PERFORMANCE PREDICTION AND ANALYTICAL SYSTEM
## FINAL PROJECT REPORT

---

## 1. ABSTRACT
The evaluation of intern performance is a critical task for human resources and technical management teams. Traditional evaluation methods are often subjective, relying heavily on manual observation and sporadic feedback, which can lead to biased or inconsistent reviews. To address this, we developed the **Intern Performance Prediction and Analytical System**, a robust, machine learning-driven web application designed to objectively assess and predict intern performance. 

By analyzing historical metrics—such as meeting attendance, sprint completion rates, task quality, punctuality, and communication skills—this system leverages an ensemble of predictive models, specifically Random Forest, XGBoost, and Decision Trees. The backend is powered by Python and Flask, which intelligently routes incoming user data, preprocesses it, and feeds it into the trained machine learning models. A modern, interactive frontend allows administrators to input data either manually or by fetching it from a comprehensive database using unique Intern IDs. Utilizing a weighted soft-voting ensemble technique, the application achieves over 98% accuracy, drastically reducing the time and bias associated with traditional performance reviews.

---

## 2. INTRODUCTION

### 2.1 Background
In modern organizational structures, internships often serve as the primary pipeline for full-time hiring. Organizations invest significant capital and time in training interns. Accurately determining which interns are performing at a "High," "Medium," or "Low" tier is crucial for making return-offer decisions. Conventional evaluation mechanisms rely on end-of-term human reviews, which are susceptible to recency bias (managers remembering only the most recent events) and subjective misinterpretation.

### 2.2 Problem Statement
There is a pressing need for a systematic, data-driven approach to human resource analytics. Hand-calculating an intern's value based on disparate metrics (attendance arrays, task tracking boards like Jira, and communication scores) is tedious. Consequently, we require an analytical system capable of digesting multidimensional metrics to output a standardized performance class, thus standardizing the review mechanism across all company departments.

### 2.3 Proposed Solution
We propose a full-stack Machine Learning web application:
- **Data Intake Layer:** Captures metrics via a sleek Web UI.
- **Processing Layer:** A Flask REST API that sanitizes data and validates missing arguments.
- **Intelligence Layer:** An ensemble of three Machine Learning models that predict the final outcome to ensure there is no single point of failure in logic.

---

## 3. SYSTEM ARCHITECTURE 

The system follows a classical Three-Tier Architecture paradigm adapted for Machine Learning pipelines.

### 3.1 Presentation Tier (Frontend)
The user interface is constructed using standard Web Technologies (HTML5, CSS3, and JavaScript). It communicates asynchronously via `fetch` API protocols to the backend, circumventing page reloads. The interface allows two primary operations:
1. **Manual Entry Form:** A user inputs 10 distinct floating-point metrics.
2. **ID Lookup:** A user enters an `Intern_ID` (e.g., `INT045`) to fetch existing metrics automatically from the system's CSV database.

### 3.2 Logic / Application Tier (Backend)
Built on **Flask (Python 3)**, the web server routes incoming HTTP `POST` and `GET` requests. Upon receiving data, it constructs a Pandas DataFrame, scales specific columns if necessary (like normalizing communication scores), and converts the JSON payload into an established feature array layout. We utilize `Flask-CORS` to handle Cross-Origin Resource Sharing, meaning the frontend can be hosted independently of the backend.

### 3.3 Intelligence & Data Tier (Machine Learning)
Instead of relying on a physical SQL database for the ML inference, the application loads highly optimized binary `.joblib` files into RAM upon server startup. These files contain pipelined preprocessing rules (`SimpleImputer`, `LabelEncoder`) and the optimized algorithmic trees for XGBoost, Random Forest, and Decision Trees.

---

## 4. METHODOLOGY & DATA PREPROCESSING

### 4.1 Feature Selection
The dataset consists of pivotal metrics reflecting an intern's day-to-day corporate life. The model ingests the following ten features:
1. **Meetings_Scheduled:** Total number of meetings invited to.
2. **Meetings_Attended:** Actual number of meetings attended.
3. **Attendance:** Percentage metric of days present.
4. **Punctuality:** Percentage metric of arriving on time.
5. **Sprint_Completion:** Percentage of Agile/Scrum tasks completed within a sprint.
6. **Task_Quality:** A subjective 1-10 or 1-5 score given by immediate technical leads.
7. **On_Time_Delivery:** Metric of tasks delivered before deadlines.
8. **Communication:** Evaluated rating of corporate communication.
9. **Tasks_Assigned:** The total volume of work allocated.
10. **Tasks_Completed:** The absolute completion volume.

### 4.2 Data Imputation
Real-world datasets are inherently messy. Using the `sklearn.impute.SimpleImputer` class, the pipeline is coded to handle missing values (NaNs) dynamically. By employing the `strategy="median"`, outliers do not heavily skew the substituted values, preserving the integrity of the data distribution.

### 4.3 Label Encoding
The target variable initially consists of human-readable text labels: "High", "Medium", and "Low". Machine learning algorithms operate strictly on numerical matrices. The `LabelEncoder` translates these into integers (0, 1, and 2), allowing the tree-based models to compute Gini impurities and log-loss reliably. The encoder is saved alongside the model so the backend can automatically reverse the numbering back to text ("High") before returning the API response.

---

## 5. MACHINE LEARNING MODELS & ALGORITHMS

### 5.1 Decision Tree Classifier
The baseline model is the Decision Tree. It recursively splits the dataset into subsets based on feature values that maximize the Information Gain. While highly interpretable, standard decision trees are prone to overfitting the training data, capturing noise alongside the actual pattern.

### 5.2 Random Forest Classifier
To counteract the overfitting of a single tree, Random Forest generates a "forest" of hundreds of decision trees using *Bagging* (Bootstrap Aggregating). Each tree is built on a random subset of data and a random subset of features. The final prediction is formulated by taking a majority vote from all the trees. This leads to a massive reduction in variance and improved generalization on unseen intern data.

### 5.3 XGBoost (eXtreme Gradient Boosting)
In our training script (`train_xgboost.py`), we deployed an advanced XGBoost model. Unlike Random Forests which build trees independently, XGBoost uses *Boosting*, building trees sequentially. Every new tree attempts to correct the residual errors made by the previous trees.
**Hyperparameter Tuning Used:**
- `n_estimators=900`: The model builds up to 900 sequential trees.
- `learning_rate=0.05`: Slows down the learning so the model doesn't overshoot the global minimum, ensuring robust convergence.
- `max_depth=5`: Limits tree depth to prevent deep, over-fitted logical branches.
- `subsample/colsample_bytree=0.9`: Prevents overfitting by ignoring 10% of the data/features during each tree's construction.

### 5.4 Weighted Ensemble & Soft Voting
When predicting by ID, the system does not simply trust one model. It gathers the probability distributions (`predict_proba`) from all three algorithms.
Because XGBoost is historically our most robust model (Accuracy ~0.9841+), the backend multiplies XGBoost's output probabilities by **2.0**, and Decision Tree/Random Forest by **1.0**. The system outputs the "Verdict" based on the highest combined probability.

---

## 6. IMPLEMENTATION DETAILS

### 6.1 Backend Implementation (`app.py`)
At startup, the initialization sequence searches for the `/models` directory and attempts to load `best_model_random_forest.joblib`, `best_model_xgboost.joblib`, and `best_model_decision_tree.joblib`. 
Two primary POST/GET endpoints govern the logic:
- `/predict`: Accepts a JSON payload, parses strings to floats, normalizes values scaling out of bounds (e.g., dividing communication by 10 if it exceeds the limit), maps them to a Pandas DataFrame matching `FEATURE_COLUMNS`, and extracts predictions and confidences for Random Forest and XGBoost.
- `/predict-by-id`: Reads `intern_database.csv`, filters by the requested string ID, isolates the metrics, applies the Weighted Soft Voting logic, and returns a final aggregated verdict alongside the agreement status (e.g., "Unanimous" vs "Soft Vote").

### 6.2 Deployment Setup
The application was successfully containerized and deployed using Render/Heroku. Because Flask's built-in server is not secure or performant for production setups, the deployment uses `gunicorn` (Python WSGI HTTP Server). The startup command `gunicorn scripts.app:app --bind 0.0.0.0:$PORT` dynamically binds the server to whatever port the cloud host assigns, making the API accessible globally.

---

## 7. RESULTS AND PERFORMANCE EVALUATION

During the testing phase, our testing partition (X_test) was evaluated. The performance metrics recorded for the optimal model (XGBoost) were outstanding:

*   **Overall Accuracy:** 98.4%
*   **Precision (Weighted):** Identifies how many interns predicted as "High" actually deserved it. Scored > 98%.
*   **F1-Score:** The harmonic mean between precision and recall, solidifying that our dataset has balanced learning capabilities without favoring the majority class.

### Confusion Matrices
The generated Heatmap Confusion Matrices indicate that the model performs exceptionally well at distinguishing "High" performers from "Low" performers. The very few misclassifications occurred exclusively at the ambiguous boundary between "Medium" and "High" performers.

---

## 8. CONCLUSION AND FUTURE ENHANCEMENTS

### 8.1 Conclusion
The **Intern Performance Prediction System** successfully introduces an unbiased, hyper-accurate, and immediate assessment tool for organizational management. By masking complex gradient boosting mathematics behind a simple REST API and user interface, HR officials can predict an intern's trajectory in a matter of milliseconds. The weighted ensemble approach guarantees fault tolerance.

### 8.2 Future Scope
To further expand the application into an Enterprise-level product, future iterations could include:
1. **Migration to SQL/NoSQL Databases:** Replacing the current CSV-based datastore with PostgreSQL or MongoDB for scalable, concurrent Read/Write operations.
2. **Deep Learning Integration:** Testing Multi-Layer Perceptrons (MLPs - Neural Networks) to see if they can map deeper non-linear relationships.
3. **Automated Feedback Generation:** Integrating a Large Language Model (like GPT or Gemini) to read the ML output and automatically draft a personalized performance review email explaining *why* the intern received a "Medium" score and exactly what metrics they must improve.
4. **JWT Authentication:** Creating protected Admin Dashboards ensuring that only authorized executives can request predictions and view sensitive metrics.

---
## 9. REFERENCES
1. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference.
3. Grinberg, M. (2018). *Flask Web Development: Developing Web Applications with Python*. O'Reilly Media.
