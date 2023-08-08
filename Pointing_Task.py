# Modules
import os
import cv2
import pandas as pd
import numpy as np
import multiprocessing
import mediapipe as mp
from scipy.signal import savgol_filter

# Global variables
INPUT_DIR = "VIDEO_FILES"
OUTPUT_DIR = "ANNOTATIONS"
DESIRED_FPS = 30
WINDOW_SIZE = 27
POLYNOMIAL_ORDER = 3
NUM_CORES = 8
MAX_S_LENGTH = 60
MAX_STEADY_LENGTH = 50
POSE_LANDMARK = mp.solutions.pose.PoseLandmark.RIGHT_WRIST
MODEL = 1

# Instantiate mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_video(video_path, frame_interval, landmark):
    """
    Processes the video and extracts relevant keypoints and timestamps
    """
    # Initialize variables
    timestamps = []
    keypoints = []

    # Load video
    cap = cv2.VideoCapture(video_path)

    frame_counter = 0  # initialize frame counter

    with mp_pose.Pose(min_detection_confidence=.6, min_tracking_confidence=.7, model_complexity=MODEL) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Process every nth frame based on frame interval
            if frame_counter % frame_interval == 0:
                image, landmark_coords = process_frame(image, pose, landmark)

                if landmark_coords is not None:
                    keypoints.append(landmark_coords)

                    # Calculate the timestamp in milliseconds and rounded
                    timestamp = round(frame_counter * (1000 / DESIRED_FPS))
                    timestamps.append(timestamp)
                    
                    # Display the resulting image
                    cv2.imshow('MediaPipe Pose', image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

            frame_counter += 1  # increment frame counter

    cv2.destroyAllWindows()
    return timestamps, keypoints

def process_frame(image, pose, landmark):
    """
    Processes an image frame and returns the processed frame and landmark coordinates
    """
    # Get the original resolution
    height, width, _ = image.shape

    # Convert the image from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Process the image and get the pose landmarks
    results = pose.process(image)

    if results.pose_landmarks:
        # Extract the coordinates of the right wrist and scale them
        x = round((results.pose_landmarks.landmark[landmark].x), 4)
        y = round((results.pose_landmarks.landmark[landmark].y), 4)

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Resize the image back to the original resolution
        image = cv2.resize(image, (width, height))

        return image, (x, y)
    
    return image, None

def calculate_speed(timestamps, keypoints):
    """
    Calculates the speed of movement between keypoints and smooths the speed
    """
    speeds = [0]  # add initial speed as 0 for the first frame
    speeds_unsmooth = [0]  # add initial unsmoothed speed as 0 for the first frame

    # Calculate speed and smooth it
    for i in range(1, len(keypoints)):
        speed_unsmooth = np.sqrt(np.sum(np.square(np.subtract(keypoints[i], keypoints[i-1])))) / (timestamps[i] - timestamps[i-1])
        speeds.append(speed_unsmooth)
        speeds_unsmooth.append(speed_unsmooth)

    speeds = [0 if speed < 0.0003 else speed for speed in speeds]
    speed_smooth = (np.round(abs(savgol_filter(speeds, WINDOW_SIZE, POLYNOMIAL_ORDER)), 8))
    speed_smooth = [0 if speed < 0.00005 else speed for speed in speed_smooth]

    return speeds, speeds_unsmooth, speed_smooth

def generate_annotations(speed_smooth):
    """
    Generates the annotations based on the smoothed speed
    """
    threshold = np.percentile(speed_smooth, 70)

    # Initialize the variables we need to keep track of the strokes
    state = [''] * len(speed_smooth)
    last_stroke_type = None
    start_index = None
    steady_start = None
    steady_end = None
    apexes = [''] * len(speed_smooth)

    # Iterate over the speed_smooth array to generate the annotations
    for i in range(1, len(speed_smooth)):
        if speed_smooth[i] > threshold and speed_smooth[i-1] <= threshold:  # Upward crossing
            if last_stroke_type is None or last_stroke_type == 'Steady' or last_stroke_type == 'R':
                last_stroke_type = 'Approach'
                start_index = i
            elif last_stroke_type == 'Approach' and i - start_index >= MAX_S_LENGTH:
                last_stroke_type = None
        elif speed_smooth[i] <= threshold and speed_smooth[i-1] > threshold:  # Downward crossing
            if last_stroke_type == 'Approach':
                last_stroke_type = 'Steady'
                start_index = i
                steady_start = i
            elif last_stroke_type == 'R':
                last_stroke_type = None
                start_index = None
        elif last_stroke_type == 'Steady' and speed_smooth[i] > threshold and speed_smooth[i-1] <= threshold:  # Upward crossing after 'Steady'
            if steady_end is not None:  # make sure that 'R' only starts after 'Steady'
                last_stroke_type = 'R'
                start_index = i

        if last_stroke_type == 'Steady' and steady_start is None:
            steady_start = i
        elif steady_start is not None and last_stroke_type != 'Steady':
            steady_end = i
            min_speed_index = np.argmin(speed_smooth[steady_start:i]) + steady_start
            apexes[min_speed_index] = 'AX'
            steady_start = None
            if speed_smooth[i] > threshold and speed_smooth[i-1] <= threshold:  # Transition from 'Steady' to 'R'
                last_stroke_type = 'R'
                start_index = i
        elif last_stroke_type == 'Steady' and i - start_index >= MAX_STEADY_LENGTH:
            last_stroke_type = None
            start_index = None
            steady_start = None
        state[i] = last_stroke_type if last_stroke_type is not None else ''
    
    return state, apexes

def create_dataframe(timestamps, keypoints, speeds_unsmooth, speed_smooth, state, apexes):
    """
    Creates a pandas dataframe with the relevant data
    """
    # Create pandas DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Keypoints': keypoints,
        'Speed Unsmoothed': speeds_unsmooth,
        'Speed Smoothed': speed_smooth,
        'State': state, 
        'Apex': apexes
    })

    df['Gesture Phase'] = df['State'].apply(lambda x: 'S' if x in ['Steady', 'Approach'] else ('R' if x == 'R' else ''))
    return df

def save_dataframe(df, filename, output_dir=OUTPUT_DIR):
    """
    Saves the dataframe as a CSV file
    """
    output_filename = os.path.splitext(filename)[0] + "_MT.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

def create_annotation_files(df, filename):
    """
    Creates ELAN importable annotation files
    """
    # Create State Annotations
    state_df = create_state_annotations(df)
    save_dataframe(state_df, os.path.splitext(filename)[0] + "_Gesture_State.csv")

    # Create Apex Annotations
    apex_df = create_apex_annotations(df)
    save_dataframe(apex_df, os.path.splitext(filename)[0] + "_Apex.csv")

    # Create Gesture Phase Annotations
    gphase_df = create_gesture_phase_annotations(df)
    save_dataframe(gphase_df, os.path.splitext(filename)[0] + "_Gesture_Phase.csv")

def create_state_annotations(df):
    """
    Creates state annotations for ELAN import
    """
    # Identify change points for gesture_t
    df['change_points'] = df['State'].shift(1) != df['State']
    df['time_ms'] = df['Timestamp'] / 1000

    # Group by change points and calculate 'Begin Time' and 'End Time'
    grouped_df = df.groupby((df['change_points']).cumsum())
    state_df = pd.DataFrame({
        'Begin Time': grouped_df['time_ms'].first(),
        'End Time': grouped_df['time_ms'].last(),
        'State': grouped_df['State'].first(),
    })

    # Add 10ms to 'End Time'
    state_df['End Time'] = state_df['End Time'] + 0.033  # convert 10ms to seconds

    return state_df

def create_apex_annotations(df):
    """
    Creates apex annotations for ELAN import
    """
    # Identify change points for gesture_t
    df['change_points'] = df['Apex'].shift(1) != df['Apex']

    # Create time interval between the current and next point
    df['next_time_s'] = df['time_ms'].shift(-1)
    df['interval_s'] = (df['next_time_s'] - df['time_ms']) / 2

    # Calculate begin and end times of the intervals
    df['Begin Time'] = df['time_ms'] - df['interval_s']
    df['End Time'] = df['time_ms'] + df['interval_s']

    # Select only the rows where apex is not NaN
    df_apex = df[~df['Apex'].isna()]

    # Create the final dataframe with necessary columns only
    apex_df = df_apex[['Begin Time', 'End Time', 'Apex']]

    return apex_df

def create_gesture_phase_annotations(df):
    """
    Creates gesture phase annotations for ELAN import
    """
    # Identify change points for gesture_t
    df['change_points'] = df['Gesture Phase'].shift(1) != df['Gesture Phase']
    df['time_ms'] = df['Timestamp'] / 1000

    # Group by change points and calculate 'Begin Time' and 'End Time'
    grouped_df = df.groupby((df['change_points']).cumsum())
    gphase_df = pd.DataFrame({
        'Begin Time': grouped_df['time_ms'].first(),
        'End Time': grouped_df['time_ms'].last(),
        'Gesture Phase': grouped_df['Gesture Phase'].first(),
    })

    # Add 10ms to 'End Time'
    gphase_df['End Time'] = gphase_df['End Time'] + 0.033  # convert 10ms to seconds

    return gphase_df

def process_file(filename):
    """
    Processes the given file and generates output data files
    """
    print(f"Processing {filename}...")
    if filename.endswith((".mp4", ".MOV")):
        video_path = os.path.join(INPUT_DIR, filename)

        # Get the actual frames per second (fps) of the video
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the frame interval based on desired and actual fps
        frame_interval = int(round(actual_fps / DESIRED_FPS))

        # Process the video
        timestamps, keypoints = process_video(video_path, frame_interval, POSE_LANDMARK)

        # Calculate speed
        speeds, speeds_unsmooth, speed_smooth = calculate_speed(timestamps, keypoints)

        # Generate annotations
        state, apexes = generate_annotations(speed_smooth)

        # Create and save dataframe
        df = create_dataframe(timestamps, keypoints, speeds_unsmooth, speed_smooth, state, apexes)
        save_dataframe(df, filename)

        # Create and save annotation files
        create_annotation_files(df, filename)
    
    print(f"{filename} is done processing.")

def main():
    """
    The main function that processes all videos
    """
    with multiprocessing.Pool(NUM_CORES) as pool:
        filenames = [filename for filename in os.listdir(INPUT_DIR) if filename.endswith((".mp4", ".MOV"))]
        pool.map(process_file, filenames)

if __name__ == '__main__':
    main()

