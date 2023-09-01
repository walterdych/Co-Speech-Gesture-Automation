import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter

# Constants for Savitzky-Golay filter
WINDOW_SIZE = 11  
POLYNOMIAL_ORDER = 3  

# Constants for file names
INPUT_DIR = "MOTION_TRACKING_FILES"
OUTPUT_DIR = "SPEED_FILES"
FILE_SUFFIX = "_processed.pkl"
SPEED_FILE_SUFFIX = "_speed.pkl"

def calculate_speed(time_series): # Calculate speed between consecutive keypoints in a time series.
    speeds = [0]  # initial speed is 0

    for i in range(1, len(time_series)):
        timestamp_prev, x_coord_prev, y_coord_prev, z_coord_prev = time_series[i-1]
        timestamp_curr, x_coord_curr, y_coord_curr, z_coord_curr = time_series[i]  # Define current coordinates
        distance = euclidean((x_coord_prev, y_coord_prev), (x_coord_curr, y_coord_curr))

        distance = euclidean((x_coord_prev, y_coord_prev), (x_coord_curr, y_coord_curr))
        time_diff = timestamp_curr - timestamp_prev
        speed = distance / time_diff
        speeds.append(speed)

    return speeds

def smooth_speed(speeds): # Smooth speeds using a Savitzky-Golay filter.
    smoothed = savgol_filter(speeds, WINDOW_SIZE, POLYNOMIAL_ORDER)
    return np.clip(smoothed, 0, None)  # clip values at 0

def generate_plot(timestamps, speeds, smoothed_speeds): # Generate a plot comparing smoothed and unsmoothed speeds.
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, speeds, label='Unsmoothed speed')
    plt.plot(timestamps, smoothed_speeds, label='Smoothed speed')
    plt.xlabel('Time (ms)')
    plt.ylabel('Speed')
    plt.title('Speed vs Time')
    plt.legend()
    plt.show()

def load_data(input_file): # Load data from a pickle file.
    with open(input_file, 'rb') as f:
        time_series = pickle.load(f)
    return time_series

def save_data(data, output_file): # Save data to a pickle file.
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

def process_file(input_file):
    time_series = load_data(input_file)
    timestamps, x_coords, y_coords, z_coords = zip(*time_series)

    keypoints = list(zip(x_coords, y_coords, z_coords))
    speeds = calculate_speed(time_series)
    smoothed_speeds = smooth_speed(speeds)

    data = {
        'timestamps': timestamps,
        'keypoints': keypoints,
        'speed_unsmooth': speeds,
        'speed_smooth': smoothed_speeds,
    }

    base_filename = os.path.basename(input_file)
    filename_without_ext = os.path.splitext(base_filename)[0]
    output_filename = f"{filename_without_ext}{SPEED_FILE_SUFFIX}"

    save_data(data, os.path.join(OUTPUT_DIR, output_filename))

    generate_plot(timestamps, speeds, smoothed_speeds)

# Process each file in the input directory
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(FILE_SUFFIX):
        process_file(os.path.join(INPUT_DIR, filename))