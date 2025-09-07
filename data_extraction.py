import pandas
import numpy 
import joblib
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load CSV
df = pandas.read_csv("filtered_multimodal_data.csv")

# Define channels based on modality
emg_channels = ["ch_1_filtered"]  # Jaw clench
eeg_channels = ["ch_2_filtered", "ch_7_filtered"]  # Relaxation EEG
eog_channels = ["ch_6_filtered"]  # Eye movement

# 1 second as sampling at 250 Hz
window_size = 250  
# Lists to store the feature values and their labels
features = []
labels = []

# Calculate how many 1 second windows there are
num_windows = len(df) // window_size

# Using welch method to calculate bandwidth 
def compute_bandpower(signal, fs=250, band=(8, 30)):
    freqs, psd = welch(signal, fs=fs)
    idx_band = numpy.logical_and(freqs >= band[0], freqs <= band[1])
    # Returning the sum of the signal power
    return numpy.sum(psd[idx_band])

# Feature extraction loop
for i in range(num_windows):
    # Extracting a window of data for the current segment
    window = df.iloc[i * window_size : (i + 1) * window_size]
    # Gathering the name of the lable
    label = window["label"].mode()[0]

    # Skip "rest" label to improve accuracy
    #if label == "rest":
    #    continue

    feature_vector = []

    # Extracting EMG features from the EMG channel
    for ch in emg_channels:
        signal = window[ch].values
        feature_vector.extend([
            numpy.mean(numpy.abs(signal)),
            numpy.std(signal),
            numpy.max(numpy.abs(signal)),
            numpy.sum(numpy.square(signal)),
            numpy.count_nonzero(numpy.diff(numpy.sign(signal)))
        ])

    # Extracting EOG features 
    for ch in eog_channels:
        signal = window[ch].values
        feature_vector.extend([
            numpy.mean(signal),
            numpy.std(signal),
            numpy.ptp(signal),
            numpy.sum(numpy.abs(numpy.diff(signal))),
            numpy.count_nonzero(numpy.diff(numpy.sign(signal)))
        ])

    # Extracting EEG features from each EEG channel 
    for ch in eeg_channels:
        signal = window[ch].values
        feature_vector.extend([
            numpy.mean(signal),
            numpy.std(signal),
            numpy.median(signal),
            compute_bandpower(signal, band=(8, 12)),
            compute_bandpower(signal, band=(13, 30))
        ])

    # Adding the computed features and their labels
    features.append(feature_vector)
    labels.append(label)

# Save features
feature_names = []

# List containing EMG features
for ch in emg_channels:
    feature_names += [
        f"{ch}_abs_mean", f"{ch}_std", f"{ch}_abs_max",
        f"{ch}_energy", f"{ch}_zero_crossings"
    ]

# List containing EOG features
for ch in eog_channels:
    feature_names += [
        f"{ch}_mean", f"{ch}_std", f"{ch}_range",
        f"{ch}_total_change", f"{ch}_zero_crossings"
    ]

# List containing EEG features
for ch in eeg_channels:
    feature_names += [
        f"{ch}_mean", f"{ch}_std", f"{ch}_median",
        f"{ch}_alpha_power", f"{ch}_beta_power"
    ]

# Create a pandas dataframe with their values and their labels
features_df = pandas.DataFrame(features, columns=feature_names)
features_df["label"] = labels

# Save to CSV
features_df.to_csv("multimodal_features.csv", index=False)
print("Saved enriched multimodal features to multimodal_features.csv")

# Machine Learning 

# Split data into columns and target labels
X = features_df.drop("label", axis=1).values
y = features_df["label"].values

# Standardise features to have a zero mean 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testsing, using stratified sampling to maintain class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Random Forect Classifier, with 100 trees and fixed randomness 
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Generate predictions on the test set
y_pred = clf.predict(X_test)

# Evaluation 

# Calculating the overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Displaying the confusion matrix to actually see where errors occur
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Full report, containing precision, recall, f1 for each action
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Saving the model and scaler
joblib.dump(clf, 'trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')