import pandas as pd
from sklearn.model_selection import train_test_split

# Read cleaned dataset
df = pd.read_csv("../dataset/intern_performance_cleaned.csv")

# Apply Feature Engineering dynamically
import feature_engineering
print("\nApplying Feature Engineering logic to dataset...")
df = feature_engineering.apply_feature_engineering(df)

print("=" * 80)
print("TRAIN-TEST SPLIT (80-20)")
print("=" * 80)

print(f"\nOriginal Dataset Size: {len(df)} records")

# Extract features and target
X = df.drop('Performance', axis=1)  
y = df['Performance']                

# Apply 80-20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,          
    random_state=42,         
    stratify=y               
)
print(f"\n Training Set: {len(X_train)} records (80%)")
print(f" Testing Set: {len(X_test)} records (20%)")

# Save files
X_train.to_csv("../dataset/X_train.csv", index=False)
y_train.to_csv("../dataset/y_train.csv", index=False)
X_test.to_csv("../dataset/X_test.csv", index=False)
y_test.to_csv("../dataset/y_test.csv", index=False)

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print("""
 X_train.csv  -> Training features
 y_train.csv  -> Training target
 X_test.csv   -> Testing features
 y_test.csv   -> Testing target
""")

# Class distribution in training set
print("\n" + "=" * 80)
print("CLASS DISTRIBUTION - TRAINING SET (80%)")
print("=" * 80)
train_dist = y_train.value_counts()
print(f"\n{train_dist}")
print(f"\nPercentages:")
for label, count in train_dist.items():
    percentage = (count / len(y_train)) * 100
    print(f"  {label}: {count} records ({percentage:.1f}%)")

# Class distribution in testing set
print("\n" + "=" * 80)
print("CLASS DISTRIBUTION - TESTING SET (20%)")
print("=" * 80)
test_dist = y_test.value_counts()
print(f"\n{test_dist}")
print(f"\nPercentages:")
for label, count in test_dist.items():
    percentage = (count / len(y_test)) * 100
    print(f"  {label}: {count} records ({percentage:.1f}%)")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Dataset Split: 80-20
Training set: {len(X_train)} records (features + targets)
Testing set: {len(X_test)} records (features + targets)
Total: {len(df)} records

Features per record: {len(X_train.columns)}
Target variable: Performance (HIGH, MEDIUM, LOW)

Class Distribution: STRATIFIED 
Maintains exact proportions in both train & test

Random State: 42 (Reproducible)
Same split every time script runs
""")

print("=" * 80)
print(" TRAIN-TEST SPLIT COMPLETE!")
print("=" * 80)
