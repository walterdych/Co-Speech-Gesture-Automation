import cv2
import pandas as pd
import os
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter
import  multiprocessing

# Instantiate mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Specify the directories
input_dir = "VIDEOS"
output_dir = "Motion_Tracking_Annotations"

# Instantiate the variables for the Savitzky-Golay filter
window_size = 27  # choose an odd number, the larger it is the smoother the result, though, more data loss
polynomial_order = 3  # order of the polynomial used to fit the samples

# Define the desired frames per second (fps)
desired_fps = 30

num_cores = 8  # Change this to the number of cores you want to use
max_s_length = 60  # Adjust this value to suit your needs
max_steady_length = 50  # Set this to your desired value

pose_landmark = mp_pose.PoseLandmark.RIGHT_WRIST # change RIGHT_WRIST to corresponding landmark
############# List of landmarks #############
    #NOSE
    #LEFT_EYE
    #RIGHT_EYE
    #LEFT_EAR
    #RIGHT_EAR
    #MOUTH_LEFT
    #MOUTH_RIGHT
    #LEFT_SHOULDER
    #RIGHT_SHOULDER
    #LEFT_ELBOW
    #RIGHT_ELBOW
    #LEFT_WRIST
    #RIGHT_WRIST
    #LEFT_PINKY
    #RIGHT_PINKY
    #LEFT_INDEX
    #RIGHT_INDEX
    #LEFT_THUMB
    #RIGHT_THUMB

speeds = [0]  # add initial speed as 0 for the first frame
speeds_unsmooth = [0]  # add initial unsmoothed speed as 0 for the first frame
annotations = []
apexes = []  # separate list for apex annotations

# Iterate over all .mp4 files in the input directory
def process_file(filename):
    print(f"Processing {filename}...")
    if filename.endswith((".mp4", ".MOV")):
        video_path = os.path.join(input_dir, filename)

        # Initialize variables
        timestamps = []
        keypoints = []

        # Load video
        cap = cv2.VideoCapture(video_path)

        # Get the actual frames per second (fps) of the video
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the frame interval based on desired and actual fps
        frame_interval = int(round(actual_fps / desired_fps))

        frame_counter = 0  # initialize frame counter

        with mp_pose.Pose(min_detection_confidence=.6, min_tracking_confidence=.7, model_complexity=2) as pose:
        #"min_detection_confidence" is set to ensure the program maps to skeletons correctly, without forcing it to not track, .6 seems to find the right balance
        #"min_tracking_confidence" is set to ensure the program TRACKS movements correctly
        #"model_complexity" sets the pretrained model to use: "0" = Lite_Model, "1" = Normal_Model, "2" = Heavy_Model
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # Get the original resolution
                height, width, _ = image.shape

                # Process every nth frame based on frame interval
                if frame_counter % frame_interval == 0:

                    # Convert the image from RGB to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Process the image and get the pose landmarks
                    results = pose.process(image)

                    if results.pose_landmarks:
                        # Extract the coordinates of the right wrist and scale them
                        x = round((results.pose_landmarks.landmark[pose_landmark].x), 4)
                        y = round((results.pose_landmarks.landmark[pose_landmark].y), 4)
                        keypoints.append((x, y))

                        # Calculate the timestamp in milliseconds and rounded
                        timestamp = round(frame_counter * (1000 / desired_fps))
                        timestamps.append(timestamp)
                    
                        # Draw pose landmarks on the image
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Create a window
                    cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)

                    # Resize the image back to the original resolution
                    image = cv2.resize(image, (width, height))
                    
                    # Display the resulting image
                    cv2.imshow('MediaPipe Pose', image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

                frame_counter += 1  # increment frame counter

        cv2.destroyAllWindows()

        # Calculate speed and smooth it
        for i in range(1, len(keypoints)):
            speed_unsmooth = np.sqrt(np.sum(np.square(np.subtract(keypoints[i], keypoints[i-1])))) / (timestamps[i] - timestamps[i-1])
            speeds.append(speed_unsmooth)
            speeds_unsmooth.append(speed_unsmooth)

        speeds = [0 if speed < 0.0003 else speed for speed in speeds]
        speed_smooth = (np.round(abs(savgol_filter(speeds, window_size, polynomial_order)), 8))
        speed_smooth = [0 if speed_smooth < 0.00005 else speed_smooth for speed_smooth in speed_smooth]
        # Create a threshold to mark when the person performs a point
        threshold = np.percentile(speed_smooth, 70)
        
        # Initialize the variables we need to keep track of the strokes
        last_stroke_type = None

        # Initialise the necessary variables
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
                elif last_stroke_type == 'Approach' and i - start_index >= max_s_length:
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
            elif last_stroke_type == 'Steady' and i - start_index >= max_steady_length:
                last_stroke_type = None
                start_index = None
                steady_start = None
            state[i] = last_stroke_type if last_stroke_type is not None else ''
    
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

        # Create output 
        output_filename = os.path.splitext(filename)[0] + "_MT.csv"
        output_path = os.path.join(output_dir, output_filename)
        df.to_csv(output_path, index=False)

        #Create ELAN importable annotations for States
        state_df = df
        # Identify change points for gesture_t
        state_df['change_points'] = state_df['State'].shift(1) != state_df['State']
        state_df['time_ms'] = state_df['Timestamp'] / 1000
        # Group by change points and calculate 'Begin Time' and 'End Time'
        grouped_state_df = df.groupby((df['change_points']).cumsum())
        gesture_t_state_df = pd.DataFrame({
            'Begin Time': grouped_state_df['time_ms'].first(),
            'End Time': grouped_state_df['time_ms'].last(),
            'State': grouped_state_df['State'].first(),
        })
        # Add 10ms to 'End Time'
        gesture_t_state_df['End Time'] = gesture_t_state_df['End Time'] + 0.033  # convert 10ms to seconds
        # Create output
        output_filename = os.path.splitext(filename)[0] + "_Gesture_State.csv"
        output_path = os.path.join(output_dir, output_filename)
        gesture_t_state_df.to_csv(output_path, index=False)

        #Create ELAN importable annotations for Apex
        Apex_df = df
        # Identify change points for gesture_t
        Apex_df['change_points'] = Apex_df['Apex'].shift(1) != Apex_df['Apex']
        # Create time interval between the current and next point
        Apex_df['next_time_s'] = Apex_df['time_ms'].shift(-1)
        Apex_df['interval_s'] = (Apex_df['next_time_s'] - Apex_df['time_ms']) / 2
        # Calculate begin and end times of the intervals
        Apex_df['Begin Time'] = Apex_df['time_ms'] - Apex_df['interval_s']
        Apex_df['End Time'] = Apex_df['time_ms'] + Apex_df['interval_s']
        # Select only the rows where apex is not NaN
        df_apex = Apex_df[~Apex_df['Apex'].isna()]
        # Create the final dataframe with necessary columns only
        gesture_t_Apex_df = df_apex[['Begin Time', 'End Time', 'Apex']]
        # Create output
        output_filename = os.path.splitext(filename)[0] + "_Apex.csv"
        output_path = os.path.join(output_dir, output_filename)
        gesture_t_Apex_df.to_csv(output_path, index=False)

        #Create ELAN importable annotations for States
        gphase_df = df
        # Identify change points for gesture_t
        gphase_df['change_points'] = gphase_df['Gesture Phase'].shift(1) != gphase_df['Gesture Phase']
        gphase_df['time_ms'] = gphase_df['Timestamp'] / 1000
        # Group by change points and calculate 'Begin Time' and 'End Time'
        grouped_gphase_df = df.groupby((df['change_points']).cumsum())
        gesture_t_gphase_df = pd.DataFrame({
            'Begin Time': grouped_gphase_df['time_ms'].first(),
            'End Time': grouped_gphase_df['time_ms'].last(),
            'Gesture Phase': grouped_gphase_df['Gesture Phase'].first(),
        })
        # Add 10ms to 'End Time'
        gesture_t_gphase_df['End Time'] = gesture_t_gphase_df['End Time'] + 0.033  # convert 10ms to seconds
        # Create output
        output_filename = os.path.splitext(filename)[0] + "_Gesture_Phase.csv"
        output_path = os.path.join(output_dir, output_filename)
        gesture_t_gphase_df.to_csv(output_path, index=False)
    
    print(f"{filename} is done processing.")

if __name__ == '__main__':
    with multiprocessing.Pool(num_cores) as pool:
        filenames = [filename for filename in os.listdir(input_dir) if filename.endswith((".mp4", ".MOV"))]
        pool.map(process_file, filenames)
