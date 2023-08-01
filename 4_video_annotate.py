import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

MAX_S_LENGTH = 60
MAX_STEADY_LENGTH = 50
THRESHOLD = 0.3
INPUT_DIR = "Speed_Files"
OUTPUT_DIR = "Annotations"
COLORS = {'Approach': 'rgba(231,107,243,0.2)', 'Steady': 'rgba(107,174,214,0.2)', 'R': 'rgba(127,127,127,0.2)', '': 'rgba(255,255,255,0.2)'}

def load_data(filename): # Load data from the specified file.
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def generate_annotations(data): # Generate annotations based on speed_smooth data.
    speed_smooth = list(map(float, data['speed_smooth']))
    state = [''] * len(speed_smooth)
    last_stroke_type = None
    start_index = None
    steady_start = None
    steady_end = None
    apexes = [''] * len(speed_smooth)

    # Iterate over the speed_smooth array to generate the annotations
    for i in range(1, len(speed_smooth)):
        if speed_smooth[i] > THRESHOLD and speed_smooth[i-1] <= THRESHOLD:
            if last_stroke_type is None or last_stroke_type == 'Steady' or last_stroke_type == 'R':
                last_stroke_type = 'Approach'
                start_index = i
            elif last_stroke_type == 'Approach' and i - start_index >= MAX_S_LENGTH:
                last_stroke_type = None
        elif speed_smooth[i] <= THRESHOLD and speed_smooth[i-1] > THRESHOLD:
            if last_stroke_type == 'Approach':
                last_stroke_type = 'Steady'
                start_index = i
                steady_start = i
            elif last_stroke_type == 'R':
                last_stroke_type = None
                start_index = None
        elif last_stroke_type == 'Steady' and speed_smooth[i] > THRESHOLD and speed_smooth[i-1] <= THRESHOLD:
            if steady_end is not None:
                last_stroke_type = 'R'
                start_index = i

        if last_stroke_type == 'Steady' and steady_start is None:
            steady_start = i
        elif steady_start is not None and last_stroke_type != 'Steady':
            steady_end = i
            min_speed_index = np.argmin(speed_smooth[steady_start:i]) + steady_start
            apexes[min_speed_index] = 'AX'
            steady_start = None
            if speed_smooth[i] > THRESHOLD and speed_smooth[i-1] <= THRESHOLD:
                last_stroke_type = 'R'
                start_index = i
        elif last_stroke_type == 'Steady' and i - start_index >= MAX_STEADY_LENGTH:
            last_stroke_type = None
            start_index = None
            steady_start = None
        state[i] = last_stroke_type if last_stroke_type is not None else ''

    df = pd.DataFrame({
        'Timestamp': data['timestamps'],
        'Keypoints': data['keypoints'],
        'Speed Unsmoothed': data['speed_unsmooth'],
        'Speed Smoothed': speed_smooth,
        'State': state, 
        'Apex': apexes
    })

    df['Gesture Phase'] = df['State'].apply(lambda x: 'S' if x in ['Steady', 'Approach'] else ('R' if x == 'R' else ''))
    return df

def create_plots(df): # Create a plotly plot based on the dataframe.
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], 
        y=df['Speed Smoothed'], 
        mode='lines',
        name='Speed curve'
    ))

    df['State_Change'] = df['State'].shift() != df['State']
    df['Group'] = df['State_Change'].cumsum()
    interval_df = df.groupby(['Group', 'State']).agg(Start=('Timestamp', 'min'), End=('Timestamp', 'max')).reset_index()

    shapes = []
    for _, row in interval_df.iterrows():
        shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=row['Start'], x1=row['End'],
            y0=0, y1=1,
            fillcolor=COLORS[row['State']],
            opacity=0.85,
            layer="below",
            line_width=0,
        ))

    for state in COLORS.keys():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=COLORS[state]),
            showlegend=True,
            name=state,
        ))

    apex_df = df[df['Apex'] == 'AX']
    fig.add_trace(go.Scatter(
        x=apex_df['Timestamp'], 
        y=apex_df['Speed Smoothed'], 
        mode='markers',
        name='Apexes'
    ))

    fig.update_xaxes(title_text='time')
    fig.update_yaxes(title_text='Speed (px/ms)')
    fig.update_layout(showlegend=True, shapes=shapes)

    fig.show()

def save_to_csv(df, filename): # Save dataframe to a csv file.
    output_filename = os.path.splitext(filename)[0] + "_MT.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df.to_csv(output_path, index=False)

def process_file(filename): # Process a single file.
    input_filename = os.path.join(INPUT_DIR, filename)
    data = load_data(input_filename)
    df = generate_annotations(data)
    save_to_csv(df, filename)
    create_plots(df)

def process_all_files(): # Process all files in the input directory.
    filenames = [filename for filename in os.listdir(INPUT_DIR) if filename.endswith("_processed_speed.pkl") and not filename.endswith("_speeds_speeds.pkl")]
    for filename in filenames:
        process_file(filename)

if __name__ == '__main__':
    process_all_files()
