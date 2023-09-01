import cv2
import mediapipe as mp
import argparse
import os
import pickle

# Initialize argparse for input and output paths
parser = argparse.ArgumentParser(description='Track specified keypoint from an input video.')
parser.add_argument('-i','--input', required=True, help='Path to the input video file')
parser.add_argument('-o','--output', required=True, help='Directory to save the output data')
args = parser.parse_args()

# Variables to change within the script
fps = 30  # Frames per second
model_complexity = 1  # Model complexity: 0, 1, or 2
keypoint_index = 16  # Keypoint index to track (Right wrist is 16)

# Initialize Mediapipe components
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize VideoCapture and set FPS
cap = cv2.VideoCapture(args.input)
cap.set(cv2.CAP_PROP_FPS, fps)

# Initialize variables
frame_idx = 0
keypoint_data = []

# Start processing
with mp_holistic.Holistic(min_detection_confidence=0.5, model_complexity=model_complexity) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Extract keypoint data
        if results.pose_landmarks:
            keypoint = results.pose_landmarks.landmark[keypoint_index]
            keypoint_data.append([frame_idx, keypoint.x, keypoint.y, keypoint.z])

        frame_idx += 1

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Extract the name of the input video to use for the output file
input_video_name = os.path.basename(args.input)
output_file_name = os.path.splitext(input_video_name)[0] + '_processed.pkl'

# Save keypoint data to output directory using Pickle
output_path = os.path.join(args.output, output_file_name)
with open(output_path, 'wb') as f:
    pickle.dump(keypoint_data, f)

print(f"Keypoint data has been saved to {output_path}")
