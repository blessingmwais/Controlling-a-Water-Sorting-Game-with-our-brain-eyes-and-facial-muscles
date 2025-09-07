import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load precomputed feature CSV (expects a 'label' column)
df = pd.read_csv("multimodal_features.csv")
# Split into features and target label
X = df.drop("label", axis=1).values
y = df["label"].values

# Split data into training and testing sets (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardise features: fit on train, apply to test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the fitted scaler for later inference
joblib.dump(scaler, "scaler.pkl")

# Helper function to train a model and time training/inference
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Measure training time
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Measure inference time
    start_infer = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - start_infer

    # Compute accuracy on the test set
    acc = accuracy_score(y_test, y_pred)
    return acc, train_time, infer_time

# Baseline model using default float64 features
clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
acc_base, train_base, infer_base = evaluate_model(
    clf_baseline, X_train_scaled, X_test_scaled, y_train, y_test
)

# Save the baseline model and record its file size (KB)
joblib.dump(clf_baseline, "baseline_model.pkl")
baseline_size_kb = os.path.getsize("baseline_model.pkl") / 1024

# "Quantised" variant by casting features to float32 (reduces memory footprint)
X_train_f32 = X_train_scaled.astype(np.float32)
X_test_f32 = X_test_scaled.astype(np.float32)

# Same RandomForest setup for a fair comparison
clf_quant = RandomForestClassifier(n_estimators=100, random_state=42)
acc_quant, train_quant, infer_quant = evaluate_model(
    clf_quant, X_train_f32, X_test_f32, y_train, y_test
)

# Save the float32-trained model and record size (KB)
joblib.dump(clf_quant, "quantised_model.pkl")
quant_size_kb = os.path.getsize("quantised_model.pkl") / 1024

# Compressed artifact: save baseline model with gzip compression
joblib.dump(clf_baseline, "compressed_model.pkl", compress=("gzip", 3))
compressed_size_kb = os.path.getsize("compressed_model.pkl") / 1024

# Reload the compressed model to confirm it loads and predicts correctly
clf_compressed = joblib.load("compressed_model.pkl")
start_infer = time.time()
y_pred_comp = clf_compressed.predict(X_test_scaled)
infer_comp = time.time() - start_infer
acc_comp = accuracy_score(y_test, y_pred_comp)

# Print a compact comparison table for accuracy, timings, and model sizes
print("\n=== Optimisation Comparison ===")
print(f"{'Model':<15} {'Accuracy':<10} {'Train Time (s)':<15} {'Infer Time (s)':<15} {'Size (KB)':<10}")
print("-" * 70)
print(f"{'Baseline':<15} {acc_base:<10.4f} {train_base:<15.4f} {infer_base:<15.4f} {baseline_size_kb:<10.2f}")
print(f"{'Quantised':<15} {acc_quant:<10.4f} {train_quant:<15.4f} {infer_quant:<15.4f} {quant_size_kb:<10.2f}")
print(f"{'Compressed':<15} {acc_comp:<10.4f} {'N/A':<15} {infer_comp:<15.4f} {compressed_size_kb:<10.2f}")

# Show percentage size reduction from compression vs. baseline
reduction = 100 * (baseline_size_kb - compressed_size_kb) / baseline_size_kb
print(f"\nCompression reduced size by {reduction:.2f}%")
