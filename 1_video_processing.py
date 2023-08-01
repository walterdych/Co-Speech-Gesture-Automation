import cv2
import argparse
import pickle
import os
import mediapipe as mp

# Define constants
POSE_LANDMARK = mp.solutions.pose.PoseLandmark.RIGHT_WRIST
DESIRED_FPS = 30

def process_video(input_file):
    # Initialize pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize video capture
    cap = cv2.VideoCapture(input_file)

    # Get the actual frames per second (fps) of the video
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame interval based on desired and actual fps
    frame_interval = int(round(actual_fps / DESIRED_FPS))

    # Initialize frame counter
    frame_counter = 0

    # Initialize list to store timestamp-keypoint pairs
    time_series = []

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            break

        # Process every nth frame based on frame interval
        if frame_counter % frame_interval == 0:

            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image to detect the pose landmarks
            results = pose.process(image)

            # Convert the image back to BGR for final display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw the pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display the image
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            # Append the timestamp-keypoint pair to the list
            time_series.append((frame_counter / actual_fps, results.pose_landmarks.landmark[POSE_LANDMARK]))

        frame_counter += 1

    # Release the video capture
    cap.release()

    # Destroy all the windows
    cv2.destroyAllWindows()

    return time_series

def main(input_file, output_dir):
    # Process the video
    time_series = process_video(input_file)

    # Save the time series to a pickle file
    base_filename = os.path.basename(input_file)
    filename_without_ext = os.path.splitext(base_filename)[0]
    output_filename = f"{filename_without_ext}_processed.pkl"
    with open(os.path.join(output_dir, output_filename), 'wb') as f:
        pickle.dump(time_series, f)
    print(f"Video processing complete for {input_file}")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to input video file")
    parser.add_argument("-o", "--output", required=True, help="path to output directory")
    args = vars(parser.parse_args())

    main(args["input"], args["output"])
