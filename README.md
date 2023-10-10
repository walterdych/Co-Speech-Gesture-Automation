# **Overview**

This repository contains scripts for automating co-speech gesture analysis. The repository is organized into several sub-folders:

- **SUPER_COMPUTING_SCRIPTS**: Contains the main scripts for the pipeline
- **LOCAL_SCRIPTS**: (Ignore this folder)
- **MOTION_TRACKING_FILES**: Output of script 1
- **SPEED_FILES**: Output of script 2
- **ANNOTATIONS**: Outputs of scripts 3 and 4
- **VIDEO_FILES**: Place your videos here for processing

## Installation

### Prerequisites

To run the scripts, you'll need Python installed along with several libraries. Each script has its own set of required libraries.

### Installation Steps

1. Clone the repository to your local machine.
2. Navigate to the `SUPER_COMPUTING_SCRIPTS` folder.
3. Install the required Python packages.

```shell
    pip install mediapipe opencv-python scikit-learn matplotlib plotly pandas numpy
```

## Walkthrough

### 1. Video Processing Notebook

This notebook focuses on video processing using computational techniques. It employs libraries such as `OpenCV`, `MediaPipe`, `pandas`, and `matplotlib`. The notebook allows users to set various parameters, such as choosing between a ***Lite*** and ***Full*** model, and specifies the path to a video file for processing.

### 2. Speed and Alignment Notebook

This notebook aims to calculate the speed of **selected keypoints** and **upsample** the data. It begins by loading the required data and then guides the user through the process of selecting the appropriate keypoints for analysis. It uses `MediaPipe` for keypoint definitions.

### 3. Gesture Annotation Notebook

This notebook is tailored for generating annotations for gestures. Libraries such as `pandas`, `numpy`, `matplotlib`, and `scikit-learn` are used. The notebook provides a function called **`generate_annotations`** that takes a DataFrame containing **time** and **speed data** to produce ***stroke***, ***phase***, and ***apex*** annotations for gestures.

### 4. ELAN Exportable Annotation Notebook

 Exports gesture ***phase*** and ***apex*** annotationsfor use with **ELAN**.

## Usage

1. Place the video files you wish to process in the `VIDEO_FILES` folder.
2. Run the scripts in the `SUPER_COMPUTING_SCRIPTS` folder in sequential order.
3. Retrieve the processed files from the respective output folders.
