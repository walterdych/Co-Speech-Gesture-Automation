import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter

# Constants for Savitzky-Golay filter
WINDOW_SIZE = 11  # choose an odd number, the larger it is the smoother the result
POLYNOMIAL_ORDER = 3  # order of the polynomial used to fit the samples

def calculate_speed(time_series):
    speeds = [0]  # initial speed is 0

    # Calculate speed between consecutive keypoints
    for i in range(1, len(time_series)):
        t1, kp1 = time_series[i-1]
        t2, kp2 = time_series[i]

        # Calculate the distance between the keypoints
        distance = euclidean((kp1.x, kp1.y), (kp2.x, kp2.y))

        # Calculate the time difference between the keypoints
        time_diff = t2 - t1

        # Calculate the speed
        speed = distance / time_diff

        # Append the speed to the list
        speeds.append(speed)

    return speeds

def smooth_speed(speeds):
    smoothed = savgol_filter(speeds, WINDOW_SIZE, POLYNOMIAL_ORDER)
    return np.clip(smoothed, 0, None)  # clip values at 0

def generate_plot(timestamps, speeds, smoothed_speeds):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, speeds, label='Unsmoothed speed')
    plt.plot(timestamps, smoothed_speeds, label='Smoothed speed')
    plt.xlabel('Time (ms)')
    plt.ylabel('Speed')
    plt.title('Speed vs Time')
    plt.legend()
    plt.show()


def process_file(input_file, output_dir):
    # Load the time series from the pickle file
    with open(input_file, 'rb') as f:
        time_series = pickle.load(f)

    # Split time_series into separate lists for timestamps and keypoints
    timestamps, keypoints = zip(*time_series)

    # Calculate the speeds
    speeds = calculate_speed(time_series)

    # Smooth the speeds
    smoothed_speeds = smooth_speed(speeds)

    # Save the data to a pickle file
    data = {
        'timestamps': timestamps,
        'keypoints': keypoints,
        'speed_unsmooth': speeds,
        'speed_smooth': smoothed_speeds,
    }
    base_filename = os.path.basename(input_file)
    filename_without_ext = os.path.splitext(base_filename)[0]
    
    print(smoothed_speeds)

    # Generate plot comparing smoothed and unsmoothed speeds
    generate_plot(timestamps, speeds, smoothed_speeds)

    output_filename = f"{filename_without_ext}_speed.pkl"
    with open(os.path.join(output_dir, output_filename), 'wb') as f:
        pickle.dump(data, f)

# Specify the input and output directories
input_dir = "Motion_Tracking_Annotations"
output_dir = "Speed_Files"

# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith("_processed.pkl"):
        process_file(os.path.join(input_dir, filename), output_dir)

