import pandas as pd
import pickle
import os
import numpy as np
import plotly.graph_objects as go

max_s_length = 60
max_steady_length = 50

input_dir = "Speed_Files"  # Update with the correct input directory path
output_dir = "Annotations"

def create_annotations(filename):
    input_filename = os.path.join(input_dir, filename)
    with open(input_filename, 'rb') as f:
        data = pickle.load(f)  # Load all data into a variable

    timestamps = data['timestamps']
    keypoints = data['keypoints']
    speed_unsmoothed = data['speed_unsmooth']
    speed_smooth = data['speed_smooth']  # Access the 'speed_smooth' data

    # Convert speed_smooth to float if it's not already
    speed_smooth = [float(x) for x in speed_smooth]

    threshold = .3
    # Initialize the variables we need to keep track of the strokes
    last_stroke_type = None

    # Initialise the necessary variables
    state = [''] * len(speed_smooth)
    last_stroke_type = None
    start_index = None
    steady_start = None
    steady_end = None
    apexes = [''] * len(speed_smooth)

    # Iterate over the speed_smooth array to generate the annotations
    for i in range(1, len(speed_smooth)):
        if speed_smooth[i] > threshold and speed_smooth[i-1] <= threshold:  # Upward crossing
            if last_stroke_type is None or last_stroke_type == 'Steady' or last_stroke_type == 'R':
                last_stroke_type = 'Approach'
                start_index = i
            elif last_stroke_type == 'Approach' and i - start_index >= max_s_length:
                last_stroke_type = None
        elif speed_smooth[i] <= threshold and speed_smooth[i-1] > threshold:  # Downward crossing
            if last_stroke_type == 'Approach':
                last_stroke_type = 'Steady'
                start_index = i
                steady_start = i
            elif last_stroke_type == 'R':
                last_stroke_type = None
                start_index = None
        elif last_stroke_type == 'Steady' and speed_smooth[i] > threshold and speed_smooth[i-1] <= threshold:  # Upward crossing after 'Steady'
            if steady_end is not None:  # make sure that 'R' only starts after 'Steady'
                last_stroke_type = 'R'
                start_index = i

        if last_stroke_type == 'Steady' and steady_start is None:
            steady_start = i
        elif steady_start is not None and last_stroke_type != 'Steady':
            steady_end = i
            min_speed_index = np.argmin(speed_smooth[steady_start:i]) + steady_start
            apexes[min_speed_index] = 'AX'
            steady_start = None
            if speed_smooth[i] > threshold and speed_smooth[i-1] <= threshold:  # Transition from 'Steady' to 'R'
                last_stroke_type = 'R'
                start_index = i
        elif last_stroke_type == 'Steady' and i - start_index >= max_steady_length:
            last_stroke_type = None
            start_index = None
            steady_start = None
        state[i] = last_stroke_type if last_stroke_type is not None else ''

    # Create pandas DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Keypoints': keypoints,
        'Speed Unsmoothed': speed_unsmoothed,
        'Speed Smoothed': speed_smooth,
        'State': state, 
        'Apex': apexes
    })

    df['Gesture Phase'] = df['State'].apply(lambda x: 'S' if x in ['Steady', 'Approach'] else ('R' if x == 'R' else ''))

    # Create output 
    output_filename = os.path.splitext(filename)[0] + "_MT.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    # Plotly plot
    fig = go.Figure()

    # Add speed curve
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], 
        y=df['Speed Smoothed'], 
        mode='lines',
        name='Speed curve'
    ))

    # Define colors for each state
    colors = {'Approach': 'rgba(231,107,243,0.2)', 'Steady': 'rgba(107,174,214,0.2)', 'R': 'rgba(127,127,127,0.2)', '': 'rgba(255,255,255,0.2)'}

    # Add a column for changes in state
    df['State_Change'] = df['State'].shift() != df['State']

    # Add a cumulative sum of the changes to identify groups of same consecutive states
    df['Group'] = df['State_Change'].cumsum()

    # Create a new dataframe representing the intervals for each state
    interval_df = df.groupby(['Group', 'State']).agg(Start=('Timestamp', 'min'), End=('Timestamp', 'max')).reset_index()

    # Initialize an empty list to hold the shapes
    shapes = []

    for _, row in interval_df.iterrows():
        shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=row['Start'], x1=row['End'],
            y0=0, y1=1,
            fillcolor=colors[row['State']],
            opacity=0.85,
            layer="below",
            line_width=0,
        ))

    # Add invisible scatter plots for the legend
    for state in colors.keys():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=colors[state]),
            showlegend=True,
            name=state,
        ))

    # Add apexes as markers
    apex_df = df[df['Apex'] == 'AX']
    fig.add_trace(go.Scatter(
        x=apex_df['Timestamp'], 
        y=apex_df['Speed Smoothed'], 
        mode='markers',
        name='Apexes'
    ))

    # Label x and y axis
    fig.update_xaxes(title_text='time')
    fig.update_yaxes(title_text='Speed (px/ms)')

    # Add shapes to layout
    fig.update_layout(showlegend=True, shapes=shapes)

    fig.show()

if __name__ == '__main__':
    filenames = [filename for filename in os.listdir(input_dir) if filename.endswith("_processed_speed.pkl") and not filename.endswith("_speeds_speeds.pkl")]
    for filename in filenames:
        create_annotations(filename)
