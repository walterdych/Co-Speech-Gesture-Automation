import cv2
import mediapipe as mp
import csv
import time



# Set up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Parameters
video_path = 'C:/Users/cosmo/Desktop/Random Scripts/VIDEOS/5003_I_BOARD_ONE.mp4'  # Replace with your video file path
output_csv = 'output.csv'  # Replace with your desired output CSV file path
fps = 30  # Replace with the FPS of your video

# Initialize MediaPipe pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Open video file
    video = cv2.VideoCapture(video_path)

    # Get video resolution
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a grid based on video resolution
    grid_width = width
    grid_height = height
    keypoints = [[(x, y) for x in range(grid_width)] for y in range(grid_height)]

    # Create CSV file and write headers
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ['Time (ms)', 'Right Wrist Visibility', 'Right Wrist X', 'Right Wrist Y', 'Speed (pixels/ms)']
        writer.writerow(headers)

        # Process video frames
        frame_count = 0
        prev_time = time.time()
        prev_wrist_x = 0
        prev_wrist_y = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            # Convert the BGR frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe
            results = pose.process(frame_rgb)
            if results.pose_landmarks is not None:
                # Extract right wrist keypoints
                pose_landmarks = results.pose_landmarks.landmark
                right_wrist = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_wrist_visibility = right_wrist.visibility if right_wrist.visibility > 0 else 0.0
                right_wrist_x = right_wrist.x * width
                right_wrist_y = right_wrist.y * height

                # Calculate speed in pixels/ms
                current_time = time.time()
                elapsed_time = current_time - prev_time
                if elapsed_time > 0:
                    speed = abs(round(((right_wrist_x - prev_wrist_x) + (right_wrist_y - prev_wrist_y)) / (elapsed_time * 2), 3))
                else:
                    speed = 0

                # Write right wrist data to CSV
                row = [round(frame_count / fps * 1000), right_wrist_visibility, right_wrist_x, right_wrist_y, speed]
                writer.writerow(row)

                # Update previous values for next iteration
                prev_time = current_time
                prev_wrist_x = right_wrist_x
                prev_wrist_y = right_wrist_y

            # Render keypoints on the frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Show the frame
            cv2.imshow('MediaPipe', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    # Release resources
    video.release()
    cv2.destroyAllWindows()