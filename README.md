# Repository Walkthrough

This repository contains a sequence of Jupyter notebooks that guide you through various aspects of video processing, speed alignment, gesture annotation, and ELAN exportable annotation. Follow the steps in the order of the notebook numbers to achieve the final product.

## Table of Contents

1. [Installation of Dependencies](#installation-of-dependencies)

2. [Video Processing](#video-processing)

3. [Speed and Alignment](#speed-and-alignment)

4. [Gesture Annotation](#gesture-annotation)

5. [ELAN Exportable Annotation](#elan-exportable-annotation)

## Installation of Dependencies

Before running any notebook, install the required dependencies:

```shell
pip install pandas mediapipe matplotlib.pyplot plotly.graph_objects os numpy cv2 scipy.signal
```

## 1. <strong>Video Processing Notebook

### <strong>Importing Libraries</strong>
Here, we import essential libraries:
- `cv2`: OpenCV for image and video processing
- `mediapipe`: Google"s MediaPipe for pose estimation
- `os`: For operating system related tasks
- `pandas`: For DataFrame support
### <strong>Setting Parameters</strong>
In this section, you can modify the following parameters:
- `MODEL`: Choose between Lite model (`1`) and Full model (`2`). Lite Model (`1`) is the  `Default`.
- `video_path`: Path to the video file.
### <strong>Initialization</strong>
This part initializes MediaPipe components used in the notebook.
### <strong>Processing Loop</strong>
The core logic of video processing is performed in this loop.
This code block reads in a video file and extracts pose landmarks from each frame of the video using the `mediapipe` and `opencv-python` libraries. The pose landmarks are then stored in a DataFrame along with the corresponding time stamp.
### <strong>Data Output</strong>
Finally, the processed data is stored in a DataFrame and saved as a pickle file and csv file.

---

## 2. <strong>Speed and Alignment Notebook

### <strong>Load Data</strong>
In this section, the data is loaded into the notebook for further processing. You can adjust the source file paths as needed.
### <strong>Choosing The Right Keypoints</strong>

Using this script, we can isolate the keypoints with which we would like to perform the analyses on. Below is the index of Keypoints defined in mediapipe:

<figure>
    <img src="https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png" width=25% height=25%
         alt="Keypoint Index">
    <figcaption>Mediapipe Pose Keypoint Index</figcaption>
</figure>


Luckily, we have created a dictionary of the keypoints and can now refer to them by their name instead of the number its been asigned.

### <center><strong><u>Available Keypoints:</u></strong></center>
<table>
  <tr>
    <td>right_wrist</td>
    <td>left_wrist</td>
    <td>right_elbow</td>
    <td>left_elbow</td>
    <td>nose</td>
  </tr>
  <tr>
    <td>right_shoulder</td>
    <td>left_shoulder</td>
    <td>right_eye</td>
    <td>left_eye</td>
  </tr>
</table>

### <strong>Calculate Unsmoothed Speed</strong>
Here, the speed of the right wrist is calculated without any smoothing. Parameters such as the sampling rate can be adjusted.
### <strong>Apply Savitzky-Golay Filtering</strong>
The Savitzky-Golay filtering method is applied to smooth the speed data. You can adjust the window size and polynomial order.

This code block is used to smooth the values in the `speed_unsmooth` column of a Pandas DataFrame called `keypoints_df`. The first line of code sets any values in the `speed_unsmooth` column that are below the 20th percentile to 0, effectively removing any low-speed outliers from the data. The second line of code applies a Savitzky-Golay filter to the `speed_unsmooth` column with a window size of 9 and a polynomial order of 2, and stores the smoothed values in a new column called `speed_smooth`.

The `np.percentile()` function is used to calculate the 20th percentile of the `speed_unsmooth` column, which is used as the threshold for removing low-speed outliers. The `savgol_filter()` function is used to apply a Savitzky-Golay filter to the `speed_unsmooth` column, which is a type of smoothing filter that can be used to remove noise from data while preserving important features such as peaks and valleys.

Overall, this code block is used to smooth the values in the `speed_unsmooth` column of a DataFrame using a Savitzky-Golay filter, and remove low-speed outliers from the data. Possible ways to improve the code include adding comments to explain the purpose of each line of code, and using more descriptive variable names.
### <strong>Align Speed Data with Pitch Data</strong>
This section aligns the calculated speed data with the pitch data. Any time offsets or scaling factors can be adjusted here.
### <strong>Save Aligned Data as CSV</strong>
Finally, the aligned data is saved as a CSV file. You can specify the destination path for the output file.
### <strong>Plot Data</strong>

The `plot()` function is used to plot the unsmoothed and smoothed speed over time. The `label` parameter is used to set the legend label for each line. The `color` parameter is used to set the color of each line.



---

## 3. <strong>Gesture Annotation Notebook

### <strong>Requirements</strong>
- pandas
- numpy
- matplotlib
- scikit-learn

You can install these Requirments by running this shell command in your terminal

```shell
    pip install pandas numpy matplotlib scikit-learn
```
### <strong>Generate Annotations</strong>

The `generate_annotations` function takes a pandas DataFrame containing the time, unsmoothed speed, and smoothed speed of a gesture as input and returns a new DataFrame with additional columns that represent annotations for the gesture. The annotations are generated based on the smoothed speed of the gesture and include stroke phase annotations and apex annotations.

The function first initializes several variables and then iterates over the smoothed speed data to generate the annotations. The annotations are generated using a series of if statements that check the current value of the smoothed speed and the previous value of the smoothed speed. The function also keeps track of the start and end indices of each steady phase of the gesture and calculates the index of the minimum value of the smoothed speed within each steady phase to generate the apex annotations.

Finally, the function creates a new DataFrame with the annotations and maps the stroke phase annotations to 'Stroke' or 'Rest'.
### <strong>Variables</strong>

- `MAX_S_LENGTH`: an integer that represents the maximum length of an 'S' stroke in the gesture.
- `MAX_STEADY_LENGTH`: an integer that represents the maximum length of a steady phase in the gesture.
- `THRESHOLD`: a float that represents the threshold value for the smoothed speed of the gesture.
- `COLORS`: a dictionary that maps stroke phase annotations to RGBA color values.

These variables are used in the `generate_annotations` function to generate annotations for a gesture based on the smoothed speed of the gesture. The `MAX_S_LENGTH` and `MAX_STEADY_LENGTH` variables are used to determine the maximum length of an 'S' stroke and a steady phase, respectively. The `THRESHOLD` variable is used to determine when the gesture starts or ends a stroke or enters a steady phase. The `COLORS` variable is used to map stroke phase annotations to colors for visualization purposes.

Overall, this excerpt defines some important parameters for generating gesture annotations and demonstrates how variables can be defined and used in Python code.
### <strong>Save Data</strong>
### <strong>Visualize Annotations</strong>
The code first creates a `plotly` figure object and adds a scatter plot of the smoothed speed of the gesture over time. The code then adds rectangles to the plot to represent the stroke phases of the gesture. The rectangles are created by grouping the annotations by phase and finding the start and end times of each phase. The `COLORS` dictionary is used to map the stroke phase annotations to colors for the rectangles.

The code then adds scatter plots to the figure for each stroke phase and for the apex annotations. The `COLORS` dictionary is used to set the colors of the stroke phase markers.

Finally, the code updates the layout of the figure to include axis titles, a title, and a legend. The size of the figure is also set. The figure is then displayed using the `show` method.

---

## 4. <strong>Gesture Annotation Notebook

### <strong>Export Gesture Phase Annotations</strong>
### <strong>Export Apex Annotations</strong>

---
