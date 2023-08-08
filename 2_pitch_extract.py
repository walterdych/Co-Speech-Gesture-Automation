import parselmouth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AUDIO_PATH = 'AUDIO_FILES/YOUR_FILE.WAVs'  # replace with your audio file path

def extract_pitch_curve(audio_path): # Extract pitch curve from an audio file.
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    time_values = pitch.xs()

    pitch_values = replace_unvoiced_by_nan(pitch_values)

    return time_values, pitch_values

def replace_unvoiced_by_nan(pitch_values): # Replace unvoiced samples by NaN in the pitch values.
    pitch_values[pitch_values==0] = np.nan
    return pitch_values

def plot_pitch_curve(time_values, pitch_values): # Plot the pitch curve.
    plt.figure()
    plt.plot(time_values, pitch_values, 'o', markersize=5, color='w')
    plt.plot(time_values, pitch_values, 'r')
    plt.grid(True)
    plt.xlabel("Time [msec]")
    plt.ylabel("Pitch [Hz]")
    plt.show()

def save_to_csv(time_values, pitch_values, output_file): # Save time values and pitch values to a CSV file.
    time_values_ms = np.round(time_values * 1000).astype(int)

    df = pd.DataFrame({
        'Time (ms)': time_values_ms,
        'Pitch (Hz)': pitch_values
    })

    df.to_csv(output_file, index=False)

def main(): # Main function to extract, plot, and save pitch curve from an audio file.
    time_values, pitch_values = extract_pitch_curve(AUDIO_PATH)
    plot_pitch_curve(time_values, pitch_values)
    output_file = AUDIO_PATH.rsplit('.', 1)[0] + '_pitch_curve.csv'
    save_to_csv(time_values, pitch_values, output_file)

if __name__ == '__main__':
    main()