import pandas as pd
import numpy as np
import joblib
import os
import time
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the prefiltered multimodal dataset
df = pd.read_csv("filtered_multimodal_data.csv")

# Define channels for each signal modality
emg_channels = ["ch_1_filtered"]           # EMG for muscle activity
eeg_channels = ["ch_2_filtered", "ch_7_filtered"]  # EEG for brain signals
eog_channels = ["ch_6_filtered"]           # EOG for eye movement

# Sampling rate and 1-second window size
fs = 250
window_size = fs

# Prepare lists for computed features and their labels
features = []
labels = []

# Compute bandpower in a specific frequency band
def compute_bandpower(signal, fs=250, band=(8, 30)):
    freqs, psd = welch(signal, fs=fs)
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[idx_band])

# Divide dataset into consecutive windows for feature extraction
num_windows = len(df) // window_size
for i in range(num_windows):
    # Extract data segment and its label
    window = df.iloc[i * window_size : (i + 1) * window_size]
    label = window["label"].mode()[0]
    feature_vector = []

    # EMG feature extraction
    for ch in emg_channels:
        signal = window[ch].values
        feature_vector.extend([
            np.mean(np.abs(signal)),
            np.std(signal),
            np.max(np.abs(signal)),
            np.sum(np.square(signal)),
            np.count_nonzero(np.diff(np.sign(signal)))
        ])

    # EOG feature extraction
    for ch in eog_channels:
        signal = window[ch].values
        feature_vector.extend([
            np.mean(signal),
            np.std(signal),
            np.ptp(signal),
            np.sum(np.abs(np.diff(signal))),
            np.count_nonzero(np.diff(np.sign(signal)))
        ])

    # EEG feature extraction (alpha/beta band powers)
    for ch in eeg_channels:
        signal = window[ch].values
        feature_vector.extend([
            np.mean(signal),
            np.std(signal),
            np.median(signal),
            compute_bandpower(signal, fs=fs, band=(8, 12)),
            compute_bandpower(signal, fs=fs, band=(13, 30))
        ])

    features.append(feature_vector)
    labels.append(label)

# Build a DataFrame of features with appropriate column names
feature_names = []

# EMG feature names
for ch in emg_channels:
    feature_names += [
        f"{ch}_abs_mean", f"{ch}_std", f"{ch}_abs_max",
        f"{ch}_energy", f"{ch}_zero_crossings"
    ]

# EOG feature names
for ch in eog_channels:
    feature_names += [
        f"{ch}_mean", f"{ch}_std", f"{ch}_range",
        f"{ch}_total_change", f"{ch}_zero_crossings"
    ]

# EEG feature names
for ch in eeg_channels:
    feature_names += [
        f"{ch}_mean", f"{ch}_std", f"{ch}_median",
        f"{ch}_alpha_power", f"{ch}_beta_power"
    ]

# Create feature DataFrame and save to CSV
features_df = pd.DataFrame(features, columns=feature_names)
features_df["label"] = labels
features_df.to_csv("multimodal_features.csv", index=False)
print("Saved enriched features to 'multimodal_features.csv'")

# Prepare data for training (features + labels)
X = features_df.drop("label", axis=1).values
y = features_df["label"].values

# Standardise features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split with stratified sampling for class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Define multiple classifiers for comparison
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "k-NN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(),
    "MLP (Neural Net)": MLPClassifier(max_iter=1000, random_state=42)
}

# Evaluate each classifier: accuracy, timing, model size
print("\n=== Classifier Comparison ===")
summary = []

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")

    # Measure training time
    start_train = time.time()
    clf.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train

    # Measure inference time per sample
    start_infer = time.time()
    y_pred = clf.predict(X_test)
    end_infer = time.time()
    inference_time = (end_infer - start_infer) / len(X_test) * 1000  # ms per sample

    # Save model temporarily to compute file size
    model_filename = f"temp_{name.replace(' ', '_')}.joblib"
    joblib.dump(clf, model_filename)
    model_size_kb = os.path.getsize(model_filename) / 1024
    os.remove(model_filename)

    # Performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = np.mean([
        float(score) for score in classification_report(y_test, y_pred, output_dict=True, zero_division=0)["macro avg"].values()
    ])

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Store results for summary table
    summary.append({
        "Classifier": name,
        "Accuracy": f"{accuracy * 100:.2f}%",
        "F1-Score": f"{f1:.2f}",
        "Training Time (s)": f"{train_time:.2f}",
        "Inference Time (ms/sample)": f"{inference_time:.2f}",
        "Model Size (KB)": f"{model_size_kb:.0f}"
    })

# Print summary table comparing all classifiers
print("\n=== Summary Table ===")
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# Save summary table to CSV for record-keeping
summary_df.to_csv("classifier_summary.csv", index=False)
