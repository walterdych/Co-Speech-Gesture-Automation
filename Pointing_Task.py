import cv2
import mediapipe as mp
import numpy as np
import math
import pandas as pd
import os
from scipy.signal import medfilt

# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Constants for MediaPipe model keypoints indices
RIGHT_WRIST_INDEX = 16
RIGHT_ELBOW_INDEX = 14
RIGHT_SHOULDER_INDEX = 12
RIGHT_INDEX_FINGER_TIP = 20

# Angle threshold for arm extension in degrees
THRESHOLD_ANGLE = 175

# Speed threshold for stroke detection
SPEED_THRESHOLD = 0.015

# Helper function to calculate distance between two points (used for speed calculation)
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Helper function to calculate angle between three points
def calculate_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

# Folder containing the video files
video_folder = 'C:/Users/cosmo/Desktop/Random Scripts/VIDEOS'

# Folder for saving motion tracking annotations
mt_annotations = 'C:/Users/cosmo/Desktop/Random Scripts/Motion Tracking Annotations'

# Iterate through video files
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)

    # Process video and gather keypoints
    data = []
    cap = cv2.VideoCapture(video_path)
    prev_keypoint = None
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS of the video
    stroke_start_frame = None
    hold_start_frame = None
    rest_start_frame = None
    apex_frame = None
    gesture_phase = ''
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the image to get the keypoints
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            shoulder = [results.pose_landmarks.landmark[RIGHT_SHOULDER_INDEX].x, results.pose_landmarks.landmark[RIGHT_SHOULDER_INDEX].y]
            elbow = [results.pose_landmarks.landmark[RIGHT_ELBOW_INDEX].x, results.pose_landmarks.landmark[RIGHT_ELBOW_INDEX].y]
            wrist = [results.pose_landmarks.landmark[RIGHT_WRIST_INDEX].x, results.pose_landmarks.landmark[RIGHT_WRIST_INDEX].y]
            index_tip = [results.pose_landmarks.landmark[RIGHT_INDEX_FINGER_TIP].x, results.pose_landmarks.landmark[RIGHT_INDEX_FINGER_TIP].y]

            if prev_keypoint is not None:
                speed = calculate_distance(index_tip, prev_keypoint)
            else:
                speed = 0

            angle = calculate_angle(shoulder, elbow, wrist)
            fully_extended = angle > THRESHOLD_ANGLE

            data.append({
                'Frame': frame_count,
                'Time (ms)': round(frame_count * (1000 / fps)),  # Convert frame count to milliseconds
                'Stroke': '',
                'Hold': '',
                'Rest': '',
                'Apex': '',
                'Gesture Phase': ''
            })

            # Draw keypoints on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw line to visualize arm extension
            cv2.line(frame, (int(shoulder[0] * frame.shape[1]), int(shoulder[1] * frame.shape[0])),
                     (int(wrist[0] * frame.shape[1]), int(wrist[1] * frame.shape[0])), (0, 255, 0), 2)

            # Stroke, hold, rest, and apex detection
            if speed > SPEED_THRESHOLD:
                if stroke_start_frame is None:
                    stroke_start_frame = frame_count
                    gesture_phase = 'S'
                    data[stroke_start_frame]['Stroke'] = gesture_phase
            elif fully_extended:
                if hold_start_frame is None:
                    hold_start_frame = frame_count
                    gesture_phase = 'H'
                    data[hold_start_frame]['Hold'] = gesture_phase
            else:
                if rest_start_frame is None:
                    rest_start_frame = frame_count
                    gesture_phase = 'R'
                    data[rest_start_frame]['Rest'] = gesture_phase

            # Apex detection within the stroke
            if gesture_phase == 'S' and fully_extended:
                apex_frame = frame_count
                data[apex_frame]['Apex'] = 'AX'
                data[apex_frame]['Gesture Phase'] = gesture_phase

            # Reset gesture phases when transitioning between stroke, hold, and rest
            if gesture_phase != '':
                if gesture_phase == 'S' and (frame_count - stroke_start_frame) > 1:
                    gesture_phase = ''
                    stroke_start_frame = None
                elif gesture_phase == 'H' and (frame_count - hold_start_frame) > 1:
                    gesture_phase = ''
                    hold_start_frame = None
                elif gesture_phase == 'R' and (frame_count - rest_start_frame) > 1:
                    gesture_phase = ''
                    rest_start_frame = None

        else:
            # Mark this frame as tracking error
            data.append({
                'Frame': frame_count,
                'Time (ms)': round(frame_count * (1000 / fps)),  # Convert frame count to milliseconds
                'Stroke': '',
                'Hold': '',
                'Rest': '',
                'Apex': '',
                'Gesture Phase': ''
            })

        # Display frame with keypoints
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if results.pose_landmarks:
            prev_keypoint = index_tip
            frame_count += 1

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    # Create DataFrame from the collected data
    df = pd.DataFrame(data)

    # Write data to CSV
    output_csv = video_file.replace('.mp4', '_MT_Annotation.csv')
    output_path = os.path.join(mt_annotations, output_csv)
    df.to_csv(output_path, index=False)
