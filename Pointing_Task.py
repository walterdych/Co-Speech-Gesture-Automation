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
window_size = 9  # choose an odd number, the larger it is the smoother the result
polynomial_order = 1  # order of the polynomial used to fit the samples

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

        # Get the video resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the frame interval based on desired and actual fps
        frame_interval = int(round(actual_fps / desired_fps))

        frame_counter = 0  # initialize frame counter

        with mp_pose.Pose(min_detection_confidence=0.50, min_tracking_confidence=0.75) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # Process every nth frame based on frame interval
                if frame_counter % frame_interval == 0:
                    # Convert the image from BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Process the image and get the pose landmarks
                    results = pose.process(image)

                    if results.pose_landmarks:
                        # Extract the coordinates of the right wrist and scale them
                        x = round((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame_width), 4)
                        y = round((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame_height), 4)
                        keypoints.append((x, y))

                        # Calculate the timestamp in milliseconds and rounded
                        timestamp = round(frame_counter * (1000 / desired_fps))
                        timestamps.append(timestamp)

                        # Draw pose landmarks on the image
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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
        speed_smooth = savgol_filter(speeds, window_size, polynomial_order)

        # Create a threshold to mark when the person performs a point
        threshold = np.percentile(speed_smooth, 75)

        # Initialize the variables we need to keep track of the strokes
        inside_stroke = False
        last_stroke_type = None
        gesture_intervals = []
        start_index = None

        # Create the annotations and track the intervals
        annotations = []
        for i, speed in enumerate(speed_smooth):
            if speed > threshold:
                if not inside_stroke:
                    # If we just started a new stroke, switch the annotation type and remember the start index
                    last_stroke_type = 'R' if last_stroke_type == 'S' else 'S'
                    start_index = i
                annotations.append(last_stroke_type)
                inside_stroke = True
            else:
                # We are not inside a stroke
                if inside_stroke:
                    # If we just finished a stroke, save the interval
                    gesture_intervals.append((last_stroke_type, start_index, i))
                inside_stroke = False
                start_index = None
                annotations.append('')

        # If the last frame was part of a stroke, save the interval
        if inside_stroke:
            gesture_intervals.append((last_stroke_type, start_index, len(speed_smooth)))

        # Create the Apex and Full Extension annotations
        apexes = [''] * len(speed_smooth)
        full_extensions = [''] * len(speed_smooth)
        for i in range(len(gesture_intervals)):
            # Find the Apex in each stroke
            if gesture_intervals[i][0] == 'S':
                start, end = gesture_intervals[i][1], gesture_intervals[i][2]
                max_speed_index = np.argmax(speeds[start:end]) + start
                apexes[max_speed_index] = 'AX'
            # Find the Full Extension in the interval between a "S" and a "R" stroke
            if i < len(gesture_intervals) - 1 and gesture_intervals[i][0] == 'S' and gesture_intervals[i+1][0] == 'R':
                start, end = gesture_intervals[i][2], gesture_intervals[i+1][2]
                min_speed_index = np.argmin(speeds[start:end]) + start
                full_extensions[min_speed_index] = 'FE'

        # Create pandas DataFrame
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'Speed Smoothed': speed_smooth,
            'Speed UnSmoothed': speeds_unsmooth,
            'Annotation': annotations,
            'Apex': apexes,
            'Full Extension': full_extensions
        })

        # Create output filename
        output_filename = os.path.splitext(filename)[0] + "_MT_Annotations.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Save to CSV
        df.to_csv(output_path, index=False)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot UnSmoothedSpeed as a time series
        ax.plot(df['Timestamp'], df['Speed UnSmoothed'], label='UnSmoothed Speed', linestyle='solid')

        # Plot smoothed speed as a time series
        ax.plot(df['Timestamp'], df['Speed Smoothed'], label='Smoothed Speed', linestyle='dashed')

        # Add labels and title
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Speed')
        ax.set_title('Time Series of Speed and Smoothed Speed')

        # Add a legend
        ax.legend()

        # Display the plot
        plt.show()
