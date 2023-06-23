# Author: Walter Dych, walterpdych@gmail.com

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Instantiate mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Specify the directories
input_dir = "VIDEOS"
output_dir = "Motion Tracking Annotations"

# Instantiate the variables for the Savitzky-Golay filter
window_size = 15  # choose an odd number, the larger it is the smoother the result
polynomial_order = 2  # order of the polynomial used to fit the samples

# Define the desired frames per second (fps)
desired_fps = 30

# Iterate over all .mp4 files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".mp4"):
        video_path = os.path.join(input_dir, filename)

        # Initialize variables
        timestamps = []
        keypoints = []
        speeds = [0]  # add initial speed as 0 for the first frame
        speeds_unsmooth = [0]  # add initial unsmoothed speed as 0 for the first frame
        annotations = []
        apexes = []  # separate list for apex annotations

        # Load video
        cap = cv2.VideoCapture(video_path)

        # Get the actual frames per second (fps) of the video
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the frame interval based on desired and actual fps
        frame_interval = int(round(actual_fps / desired_fps))

        frame_counter = 0  # initialize frame counter

        with mp_pose.Pose(min_detection_confidence=.6, min_tracking_confidence=.80) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # Get the original resolution
                height, width, _ = image.shape

                # Process every nth frame based on frame interval
                if frame_counter % frame_interval == 0:

                    # Convert the image from BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Process the image and get the pose landmarks
                    results = pose.process(image)

                    if results.pose_landmarks:
                        # Extract the coordinates of the right wrist and scale them
                        x = round((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x), 4)
                        y = round((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y), 4)
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

        # Create a threshold to mark when the person performs a point
        threshold = .00025
        
        # Initialize the variables we need to keep track of the strokes
        inside_stroke = False
        last_stroke_type = None

        # Create the annotations and track the intervals
        annotations = ['']
        last_stroke_type = None
        gesture_intervals = []
        start_index = None

        for i in range(1, len(speed_smooth)):
            if speed_smooth[i] > threshold and speed_smooth[i-1] <= threshold:
                if last_stroke_type is None or last_stroke_type == 'R':
                    last_stroke_type = 'S'
                else:
                    last_stroke_type = 'R'
                start_index = i
            elif speed_smooth[i] <= threshold and speed_smooth[i-1] > threshold:
                if last_stroke_type == 'S':
                    gesture_intervals.append((last_stroke_type, start_index, i))
                elif last_stroke_type == 'R':
                    gesture_intervals.append((last_stroke_type, start_index, i))
                    last_stroke_type = None

            if last_stroke_type is not None:
                annotations.append(last_stroke_type)
            else:
                annotations.append('')

            apexes = [''] * len(speed_smooth)
            for j in range(len(gesture_intervals)):
                # Find the Apex in each stroke
                if gesture_intervals[j][0] == 'S':
                    start, end = gesture_intervals[j][1], gesture_intervals[j][2]
                    max_speed_index = np.argmax(speed_smooth[start:end]) + start
                    apexes[max_speed_index] = 'AX'
        
        # Debugging Length Errors
        print("Length of timestamps:", len(timestamps))
        print("Length of speed_smooth:", len(speed_smooth))
        print("Length of speeds_unsmooth:", len(speeds_unsmooth))
        print("Length of annotations:", len(annotations))
        print("Length of apexes:", len(apexes)) 

        # Create pandas DataFrame
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'Keypoints': keypoints,
            'Speed Unsmoothed': speeds_unsmooth,
            'Speed Smoothed': speed_smooth,
            'Annotation': annotations, 
            'Apex': apexes
        })

        # Create output filename
        output_filename = os.path.splitext(filename)[0] + "_MT_Annotations.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Save to CSV
        df.to_csv(output_path, index=False)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot UnSmoothedSpeed as a time series
        ax.plot(df['Timestamp'], df['Speed Unsmoothed'], label='UnSmoothed Speed', linestyle='solid')

        # Plot smoothed speed as a time series
        ax.plot(df['Timestamp'], df['Speed Smoothed'], label='Smoothed Speed', linestyle='solid')

        # Add labels and title
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Speed')
        ax.set_title('Time Series of Speed and Smoothed Speed')

        # Add a legend
        ax.legend()

        # Display the plot
        plt.show()
