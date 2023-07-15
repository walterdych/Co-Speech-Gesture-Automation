import parselmouth
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_pitch_curve(audio_path):
    # Load sound
    sound = parselmouth.Sound(audio_path)

    # Use Praat's default settings for pitch extraction
    pitch = sound.to_pitch()

    # Convert pitch to numpy array (times and corresponding pitch values)
    pitch_values = pitch.selected_array['frequency']
    time_values = pitch.xs()

    # Replace unvoiced samples by NaN to not plot
    pitch_values[pitch_values==0] = np.nan

    return time_values, pitch_values

def plot_pitch_curve(time_values, pitch_values):
    # Plot the pitch track
    plt.figure()
    plt.plot(time_values, pitch_values, 'o', markersize=5, color='w')
    plt.plot(time_values, pitch_values, 'r')
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Pitch [Hz]")
    plt.show()

def save_to_csv(time_values, pitch_values, output_file):
    # Convert time to milliseconds and round
    time_values_ms = np.round(time_values * 1000).astype(int)

    # Create a DataFrame
    df = pd.DataFrame({
        'Time (ms)': time_values_ms,
        'Pitch (Hz)': pitch_values
    })

    # Save to CSV
    df.to_csv(output_file, index=False)

def main():
    audio_path = 'SOUND\Bertrand&Benedicte_INPUT2_Part2.WAV'  # replace with your audio file path
    time_values, pitch_values = extract_pitch_curve(audio_path)
    plot_pitch_curve(time_values, pitch_values)
    output_file = audio_path.rsplit('.', 1)[0] + '_pitch_curve.csv'
    save_to_csv(time_values, pitch_values, output_file)

if __name__ == '__main__':
    main()
