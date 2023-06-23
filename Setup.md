# Steps for Running Automated Gesture Annotation

1. Download scripts from [Github](https://github.com/walterdych/Co-Speech-Gesture-Automation).

2. Place videos in `VIDEOS` folder.

3. Replace video extension in script `Pointing_Task.py` (`.mp4` or `.MOV`). The Python script will iterate over all .mp4 files in the input directory:

    ```python
    # Iterate over all .mp4 files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
    ```

4. Once the previous three steps are finished, you should be able to run the script and it will iterate through all of the videos in the `VIDEOS` directory.

5. A visualization will pop up to show that the video processing is running as shown below.

    Note: You can comment out this code segment to not have this visualization, this is mainly for debugging purposes:

    ```python
    # Create a window
    cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)
    # Resize the image back to the original resolution
    image = cv2.resize(image, (width, height))
    # Display the resulting image
    cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    ```

6. Once the video is fully processed, a chart will be created to show the smoothed and unsmoothed speed vectors, this can also be commented out as it will pause the loop. See example below.
