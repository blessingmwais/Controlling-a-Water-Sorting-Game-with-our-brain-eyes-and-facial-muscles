import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# Load Data
df = pd.read_csv('multimodal_eog_emg_eeg_data.csv')

# Filtering Functions
def notch_filter(signal, fs=250, freq=50, Q=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, signal)

def bandpass_filter(signal, lowcut, highcut, fs=250, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Define Filtering Settings
fs = 250  # Sampling rate in Hz

channels_info = {
    'ch_1': {'type': 'EMG', 'lowcut': 20, 'highcut': 100},
    'ch_2': {'type': 'EEG', 'lowcut': 0.5, 'highcut': 30},
    'ch_7': {'type': 'EEG', 'lowcut': 0.5, 'highcut': 30},
    'ch_6': {'type': 'EOG', 'lowcut': 0.1, 'highcut': 5}
}

# Apply Filtering
for ch, info in channels_info.items():
    print(f"Filtering {ch} ({info['type']})...")
    signal = df[ch].values

    # DC Offset Removal
    signal = signal - np.mean(signal)

    # Notch Filtering (50 Hz)
    signal = notch_filter(signal, fs=fs)

    # Bandpass Filtering (modality-specific)
    signal = bandpass_filter(signal, info['lowcut'], info['highcut'], fs=fs)

    # Save filtered signal
    df[f"{ch}_filtered"] = signal

# Save Filtered Data
df.to_csv('filtered_multimodal_data.csv', index=False)
print("\nFiltered signals saved to 'filtered_multimodal_data.csv'")
