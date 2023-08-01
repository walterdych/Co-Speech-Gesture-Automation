import cv2
import argparse
import pickle
import os
import mediapipe as mp

# Define constants
POSE_LANDMARK = mp.solutions.pose.PoseLandmark.RIGHT_WRIST
DESIRED_FPS = 30

def process_video(input_file): # Process a video to detect pose landmarks and generate a time series.
    cap = initialize_video_capture(input_file)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(actual_fps / DESIRED_FPS))
    frame_counter = 0
    time_series = []

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            break

        # Process every nth frame based on frame interval
        if frame_counter % frame_interval == 0:
            results = process_image(image)
            display_image(image, results)

            # Append the timestamp-keypoint pair to the list
            time_series.append((frame_counter / actual_fps, results.pose_landmarks.landmark[POSE_LANDMARK]))

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

    return time_series

def initialize_video_capture(input_file): # Initialize video capture and pose.
    return cv2.VideoCapture(input_file)

def process_image(image): # Process an image to detect the pose landmarks.
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return results

def display_image(image, results): # Display an image.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        return

def main(input_file, output_dir): # Main function to process a video and save the time series to a pickle file.
    time_series = process_video(input_file)
    save_time_series(time_series, input_file, output_dir)

def save_time_series(time_series, input_file, output_dir): # Save the time series to a pickle file.
    base_filename = os.path.basename(input_file)
    filename_without_ext = os.path.splitext(base_filename)[0]
    output_filename = f"{filename_without_ext}_processed.pkl"
    
    with open(os.path.join(output_dir, output_filename), 'wb') as f:
        pickle.dump(time_series, f)
    print(f"Video processing complete for {input_file}")

if __name__ == '__main__': # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to input video file")
    parser.add_argument("-o", "--output", required=True, help="path to output directory")
    args = vars(parser.parse_args())

    main(args["input"], args["output"])
