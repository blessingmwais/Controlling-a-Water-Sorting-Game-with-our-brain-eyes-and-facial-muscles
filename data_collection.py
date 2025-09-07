import time
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams

# Setup BrainFlow and connect to board
BoardShim.enable_dev_board_logger()  # Enabling logger for debugging

# Initialising parameters for Cyton board
board_id = 0  
params = BrainFlowInputParams()
# Specify serial port being used
params.serial_port = "COM7"  

# Preparing and starting board session
board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream()
print(" Board connected and streaming")

# Setup recording parameters - Channel indecies and sampling rate
channels = BoardShim.get_eeg_channels(board_id)  
sampling_rate = BoardShim.get_sampling_rate(board_id) 


# Duration in seconds per action for Recording
duration_per_action = 5
# Number of prepetitions for robust data
num_repeats = 15  

# Action list for recording
actions = [
    ("rest", duration_per_action),
    ("look_left", duration_per_action),
    ("look_right", duration_per_action),
    ("jaw_clench", duration_per_action),
    ("mental_relax", duration_per_action)
]

# Lists to store the raw signal values and their corresponding labels 
labels = []
collected_data = []

# Data collection loop
try:
    print("\nStarting data collection. Get ready...")
    #Pause before starting to sample
    time.sleep(5)

    # Loop for gathering data for each action  
    for action, seconds in actions * num_repeats:
        input(f"\nPrepare to perform '{action}'. Press ENTER when ready.")
        print(f"Recording '{action}' for {seconds} seconds...")

        # Start time record for 5 second sampling
        t_start = time.time()
        while (time.time() - t_start) < seconds:
            # getting the most recent data from the board
            data = board.get_current_board_data(sampling_rate)
            # Extracting signals from relevant channels
            signal_data = data[channels].T  # samples stored in rows

            # Adding each sample's channel data and its label to the dataset
            for row in signal_data:
                collected_data.append(list(row))
                labels.append(action)

            # capture once per second
            time.sleep(1)  

        print(f"Finished recording '{action}'.")

finally:
    # Stop the data stream and disconnect from the cyton board
    board.stop_stream()
    board.release_session()
    print("\nStopped data stream and released session.")

# Saving the data
# Creating column names for each channel
columns = [f"ch_{i+1}" for i in range(len(channels))]

# Combining the collected signal data and labels into the dataset
df = pd.DataFrame(collected_data, columns=columns)
df["label"] = labels

# Saving the dataset to a CSV file
df.to_csv("multimodal_eog_emg_eeg_data.csv", index=False)

print("\nSaved dataset as 'multimodal_eog_emg_eeg_data.csv'.")
