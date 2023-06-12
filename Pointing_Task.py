import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter

# Instantiate mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Specify the directories
input_dir = "VIDEOS"
output_dir = "Motion Tracking Annotations"

# Instantiate the variables for the Savitzky-Golay filter
window_size = 33  # choose an odd number, the larger it is the smoother the result
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
        annotations = []
        apexes = []  # separate list for apex annotations

        # Load video
        cap = cv2.VideoCapture(video_path)

        # Get the actual frames per second (fps) of the video
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the frame interval based on desired and actual fps
        frame_interval = int(round(actual_fps / desired_fps))

        frame_counter = 0  # initialize frame counter

        with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:
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
                        # Extract the coordinates of the right wrist
                        x, y, z = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z
                        # Calculate the timestamp in milliseconds and rounded
                        timestamp = round(frame_counter * (1000 / desired_fps))
                        timestamps.append(timestamp)
                        keypoints.append((x, y, z))

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
            speed = np.sqrt(np.sum(np.square(np.subtract(keypoints[i], keypoints[i-1])))) / (timestamps[i] - timestamps[i-1])
            speeds.append(speed)
        speeds_smooth = savgol_filter(speeds, window_size, polynomial_order)

        # Create a threshold to mark when the person performs a point and label it as a stroke "S"
        threshold = np.percentile(speeds_smooth, 82)
        annotations = ['S' if speed > threshold else '' for speed in speeds_smooth]

        # Initialize all apexes as ''
        apexes = [''] * len(annotations)

        # Identify "strokes" as continuous 'S' intervals
        stroke_intervals = []
        start_index = None
        for i, annotation in enumerate(annotations):
            if annotation == 'S' and start_index is None:
                start_index = i
            elif annotation == '' and start_index is not None:
                stroke_intervals.append((start_index, i))
                start_index = None

        # Find the point of max speed within each "stroke" and label it as an apex "AX"
        for interval in stroke_intervals:
            start, end = interval
            max_speed_index = np.argmax(speeds[start:end]) + start
            apexes[max_speed_index] = 'AX'

        # Create pandas DataFrame
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'Keypoints': keypoints,
            'Speed': speeds,
            'Annotation': annotations,
            'Apex': apexes
        })

        # Create output filename
        output_filename = os.path.splitext(filename)[0] + "_MT_Annotations.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Save to CSV
        df.to_csv(output_path, index=False)
        